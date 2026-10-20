//! Softmax and normalizations.

use kapel_core::{Result, Tensor};
use kapel_ops::{LayerNorm, LogSoftmax, Operation, RmsNorm, Run, Softmax};

use crate::backend::{CHUNK, Cpu};
use crate::io;
use crate::lines::map_lines;
use crate::math;
use crate::pool::SharedMut;

/// `y = softmax(x)` on one line.
fn softmax_line(isa: crate::Isa, x: &[f32], y: &mut [f32]) {
    let max = math::max(isa, x);
    y.copy_from_slice(x);
    let total = math::exp_shifted_sum(isa, y, max);
    math::scale(isa, y, 1.0 / total);
}

impl Run<Softmax> for Cpu {
    fn forward(&self, op: &Softmax, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Softmax::NAME, inputs[0])?;
        let isa = self.isa;
        let out = map_lines(self, &[&x], &signature.shape, op.dim, |x, y| {
            softmax_line(isa, x[0], y)
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Softmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Softmax::NAME, inputs[0])?;
        let g = io::contiguous(Softmax::NAME, grad)?;
        let isa = self.isa;
        let out = map_lines(self, &[&x, &g], &signature.shape, op.dim, |v, out| {
            softmax_line(isa, v[0], out);
            let dot = math::dot(isa, v[1], out);
            for (y, &g) in out.iter_mut().zip(v[1]) {
                *y *= g - dot;
            }
        });
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            &signature.shape,
            signature.dtype,
        )?)])
    }
}

impl Run<LogSoftmax> for Cpu {
    fn forward(&self, op: &LogSoftmax, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(LogSoftmax::NAME, inputs[0])?;
        let isa = self.isa;
        let out = map_lines(self, &[&x], &signature.shape, op.dim, |x, y| {
            let max = math::max(isa, x[0]);
            let log_total = math::exp_shifted_total(isa, x[0], max).ln();
            for (y, &x) in y.iter_mut().zip(x[0]) {
                *y = x - max - log_total;
            }
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &LogSoftmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(LogSoftmax::NAME, inputs[0])?;
        let g = io::contiguous(LogSoftmax::NAME, grad)?;
        let isa = self.isa;
        let out = map_lines(self, &[&x, &g], &signature.shape, op.dim, |v, out| {
            softmax_line(isa, v[0], out);
            let total = math::sum(isa, v[1]);
            for (y, &g) in out.iter_mut().zip(v[1]) {
                *y = g - *y * total;
            }
        });
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            &signature.shape,
            signature.dtype,
        )?)])
    }
}

/// The optional vector input `index`, or a constant.
fn vector(
    op: &'static str,
    inputs: &[&Tensor],
    index: usize,
    width: usize,
    absent: f32,
) -> Result<Vec<f32>> {
    match inputs.get(index) {
        Some(tensor) => Ok(io::contiguous(op, tensor)?.into_owned()),
        None => Ok(vec![absent; width]),
    }
}

/// How rows are normalized: subtracting the mean (LayerNorm) or not (RMSNorm).
#[derive(Clone, Copy)]
enum Kind {
    Layer,
    Rms,
}

/// Normalizes one row in place into `hat`, and returns `1/σ` (or `1/rms`).
fn normalize(kind: Kind, isa: crate::Isa, x: &[f32], eps: f32, hat: &mut [f32]) -> f32 {
    let width = x.len() as f32;
    match kind {
        Kind::Layer => {
            let mean = math::sum(isa, x) / width;
            for (h, &v) in hat.iter_mut().zip(x) {
                *h = v - mean;
            }
            let inv = 1.0 / (math::dot(isa, hat, hat) / width + eps).sqrt();
            math::scale(isa, hat, inv);
            inv
        }
        Kind::Rms => {
            let inv = 1.0 / (math::dot(isa, x, x) / width + eps).sqrt();
            for (h, &v) in hat.iter_mut().zip(x) {
                *h = v * inv;
            }
            inv
        }
    }
}

/// Rows per task, for rows of `width` elements.
fn rows_per_task(width: usize) -> usize {
    (CHUNK / width.max(1)).max(1)
}

/// The forward pass of both normalizations: `y = x̂ · (offset + w) + b`.
#[allow(clippy::too_many_arguments)]
fn norm_forward(
    cpu: &Cpu,
    kind: Kind,
    x: &[f32],
    width: usize,
    eps: f32,
    w: &[f32],
    offset: f32,
    b: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let isa = cpu.isa;
    cpu.fill(&mut out, rows_per_task(width) * width, |start, part| {
        for (row, y) in part.chunks_mut(width).enumerate() {
            let x = &x[start + row * width..][..width];
            normalize(kind, isa, x, eps, y);
            for c in 0..width {
                y[c] = y[c] * (offset + w[c]) + b[c];
            }
        }
    });
    out
}

/// The backward pass of both normalizations: the gradient of `x`, and the sums
/// over rows of `g · x̂` and of `g`, the gradients of the weight and the bias.
/// Each task sums its own rows; the tasks' sums are then added in task order.
#[allow(clippy::too_many_arguments)]
fn norm_backward(
    cpu: &Cpu,
    kind: Kind,
    x: &[f32],
    g: &[f32],
    width: usize,
    eps: f32,
    w: &[f32],
    offset: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = x.len() / width.max(1);
    let per_task = rows_per_task(width);
    let tasks = rows.div_ceil(per_task);
    let mut gx = vec![0.0; x.len()];
    let mut partial = vec![0.0; tasks * 2 * width];
    let (gx_shared, partial_shared) = (SharedMut::new(&mut gx), SharedMut::new(&mut partial));
    let isa = cpu.isa;
    cpu.pool.run(tasks, |task| {
        // SAFETY: each task owns its rows of `gx` and its slice of `partial`.
        let sums = unsafe { partial_shared.slice(task * 2 * width, 2 * width) };
        let (gw, gb) = sums.split_at_mut(width);
        let mut hat = vec![0.0; width];
        let mut gh = vec![0.0; width];
        for row in task * per_task..((task + 1) * per_task).min(rows) {
            let (x, g) = (&x[row * width..][..width], &g[row * width..][..width]);
            let inv = normalize(kind, isa, x, eps, &mut hat);
            for c in 0..width {
                gh[c] = g[c] * (offset + w[c]);
                gw[c] += g[c] * hat[c];
                gb[c] += g[c];
            }
            let n = width as f32;
            let mean_ghh = math::dot(isa, &gh, &hat) / n;
            let mean_gh = match kind {
                Kind::Layer => math::sum(isa, &gh) / n,
                Kind::Rms => 0.0,
            };
            // SAFETY: as above.
            let out = unsafe { gx_shared.slice(row * width, width) };
            for c in 0..width {
                out[c] = inv * (gh[c] - mean_gh - hat[c] * mean_ghh);
            }
        }
    });
    let (mut gw, mut gb) = (vec![0.0; width], vec![0.0; width]);
    for sums in partial.chunks(2 * width.max(1)) {
        for c in 0..width {
            gw[c] += sums[c];
            gb[c] += sums[width + c];
        }
    }
    (gx, gw, gb)
}

impl Run<LayerNorm> for Cpu {
    fn forward(&self, op: &LayerNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(LayerNorm::NAME, inputs[0])?;
        let width = *signature.shape.last().expect("checked by infer");
        let w = vector(LayerNorm::NAME, inputs, 1, width, 1.0)?;
        let b = vector(LayerNorm::NAME, inputs, 2, width, 0.0)?;
        let out = norm_forward(self, Kind::Layer, &x, width, op.eps as f32, &w, 0.0, &b);
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &LayerNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(LayerNorm::NAME, inputs[0])?;
        let g = io::contiguous(LayerNorm::NAME, grad)?;
        let width = *signature.shape.last().expect("checked by infer");
        let w = vector(LayerNorm::NAME, inputs, 1, width, 1.0)?;
        let (gx, gw, gb) = norm_backward(self, Kind::Layer, &x, &g, width, op.eps as f32, &w, 0.0);
        let mut grads = vec![Some(io::write(
            &self.buffers,
            &gx,
            &signature.shape,
            signature.dtype,
        )?)];
        for (index, values) in [(1, gw), (2, gb)] {
            if index < inputs.len() {
                grads.push(Some(io::write(
                    &self.buffers,
                    &values,
                    &[width],
                    inputs[index].dtype(),
                )?));
            }
        }
        Ok(grads)
    }
}

impl Run<RmsNorm> for Cpu {
    fn forward(&self, op: &RmsNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(RmsNorm::NAME, inputs[0])?;
        let width = *signature.shape.last().expect("checked by infer");
        let offset = op.weight_offset as f32;
        let w = vector(RmsNorm::NAME, inputs, 1, width, 1.0 - offset)?;
        let out = norm_forward(
            self,
            Kind::Rms,
            &x,
            width,
            op.eps as f32,
            &w,
            offset,
            &vec![0.0; width],
        );
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &RmsNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(RmsNorm::NAME, inputs[0])?;
        let g = io::contiguous(RmsNorm::NAME, grad)?;
        let width = *signature.shape.last().expect("checked by infer");
        let offset = op.weight_offset as f32;
        let w = vector(RmsNorm::NAME, inputs, 1, width, 1.0 - offset)?;
        let (gx, gw, _) = norm_backward(self, Kind::Rms, &x, &g, width, op.eps as f32, &w, offset);
        let mut grads = vec![Some(io::write(
            &self.buffers,
            &gx,
            &signature.shape,
            signature.dtype,
        )?)];
        if inputs.len() > 1 {
            grads.push(Some(io::write(
                &self.buffers,
                &gw,
                &[width],
                inputs[1].dtype(),
            )?));
        }
        Ok(grads)
    }
}
