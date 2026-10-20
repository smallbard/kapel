//! Reductions.

use kapel_core::{Result, Tensor};
use kapel_ops::{Max, Mean, Operation, Run, Sum};

use crate::backend::{CHUNK, Cpu};
use crate::io;
use crate::lines::lines;

/// Row-major strides.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Every combination of the coordinates of `dims`, as offsets in `shape`.
fn reduced_offsets(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let mut offsets = vec![0];
    for &d in dims {
        let (size, stride) = (shape[d], s[d]);
        offsets = offsets
            .iter()
            .flat_map(|&o| (0..size).map(move |i| o + i * stride))
            .collect();
    }
    offsets
}

/// For each result of reducing `dims`, the offset of its first element.
fn kept_offsets(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let kept: Vec<usize> = (0..shape.len()).filter(|d| !dims.contains(d)).collect();
    // The kept dimensions in increasing order: the results come out row-major.
    reduced_offsets(shape, &kept)
}

/// The sum over `dims`, each result added in a fixed order.
fn sum(cpu: &Cpu, x: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let (inner, bases) = (reduced_offsets(shape, dims), kept_offsets(shape, dims));
    let mut out = vec![0.0; bases.len()];
    let chunk = (CHUNK / inner.len().max(1)).max(1);
    cpu.fill(&mut out, chunk, |start, part| {
        for (y, &base) in part.iter_mut().zip(&bases[start..]) {
            *y = inner.iter().map(|&o| x[base + o]).sum();
        }
    });
    out
}

/// Each input element receives `factor` times the gradient of its result.
fn spread(cpu: &Cpu, g: &[f32], shape: &[usize], dims: &[usize], factor: f32) -> Vec<f32> {
    let (inner, bases) = (reduced_offsets(shape, dims), kept_offsets(shape, dims));
    let total: usize = shape.iter().product();
    let mut out = vec![0.0; total];
    // Results are spread over disjoint sets of input elements.
    let shared = crate::pool::SharedMut::new(&mut out);
    let chunk = (CHUNK / inner.len().max(1)).max(1);
    cpu.pool.run(bases.len().div_ceil(chunk), |task| {
        for r in task * chunk..((task + 1) * chunk).min(bases.len()) {
            for &o in &inner {
                // SAFETY: each input element belongs to exactly one result, and
                // each result to one task.
                unsafe { *shared.ptr(bases[r] + o) = factor * g[r] };
            }
        }
    });
    out
}

impl Run<Sum> for Cpu {
    fn forward(&self, op: &Sum, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Sum::NAME, inputs[0])?;
        let out = sum(self, &x, inputs[0].shape(), &op.dims);
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(&self, op: &Sum, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let g = io::contiguous(Sum::NAME, grad)?;
        let out = spread(self, &g, inputs[0].shape(), &op.dims, 1.0);
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}

fn count(shape: &[usize], dims: &[usize]) -> f32 {
    dims.iter().map(|&d| shape[d]).product::<usize>() as f32
}

impl Run<Mean> for Cpu {
    fn forward(&self, op: &Mean, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Mean::NAME, inputs[0])?;
        let n = count(inputs[0].shape(), &op.dims);
        let out: Vec<f32> = sum(self, &x, inputs[0].shape(), &op.dims)
            .iter()
            .map(|v| v / n)
            .collect();
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Mean,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let g = io::contiguous(Mean::NAME, grad)?;
        let n = count(inputs[0].shape(), &op.dims);
        let out = spread(self, &g, inputs[0].shape(), &op.dims, 1.0 / n);
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}

/// The position of the first maximum of each line along `dim`.
fn argmax(cpu: &Cpu, x: &[f32], shape: &[usize], dim: usize) -> Vec<usize> {
    let len = shape[dim];
    let (count, line) = lines(shape, dim);
    let mut out = vec![0usize; count];
    let shared = crate::pool::SharedMut::new(&mut out);
    let chunk = (CHUNK / len.max(1)).max(1);
    cpu.pool.run(count.div_ceil(chunk), |task| {
        for l in task * chunk..((task + 1) * chunk).min(count) {
            let (start, step) = line(l);
            let best = (0..len)
                .map(|i| start + i * step)
                .fold(start, |best, at| if x[at] > x[best] { at } else { best });
            // SAFETY: one slot per line, one line per task.
            unsafe { *shared.ptr(l) = best };
        }
    });
    out
}

impl Run<Max> for Cpu {
    fn forward(&self, op: &Max, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Max::NAME, inputs[0])?;
        let out: Vec<f32> = argmax(self, &x, inputs[0].shape(), op.dim)
            .iter()
            .map(|&at| x[at])
            .collect();
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(&self, op: &Max, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Max::NAME, inputs[0])?;
        let g = io::contiguous(Max::NAME, grad)?;
        let mut out = vec![0.0; x.len()];
        for (l, at) in argmax(self, &x, inputs[0].shape(), op.dim)
            .into_iter()
            .enumerate()
        {
            out[at] = g[l];
        }
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}
