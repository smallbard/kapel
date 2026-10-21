//! Gathering and scattering rows.

use kapel_core::{Result, Tensor};
use kapel_ops::{GatherRows, Operation, Run, ScatterAddRows};

use crate::backend::{CHUNK, Cpu};
use crate::io;

/// `out[i] = src[rows[i]]`, rows of `width` elements, in parallel.
fn gather(cpu: &Cpu, src: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows.len() * width];
    let chunk = (CHUNK / width.max(1)).max(1) * width.max(1);
    cpu.fill(&mut out, chunk, |start, part| {
        for (n, row) in part.chunks_mut(width.max(1)).enumerate() {
            let r = rows[start / width.max(1) + n];
            row.copy_from_slice(&src[r * width..(r + 1) * width]);
        }
    });
    out
}

/// `out[rows[i]] += src[i]`. Each output row adds its sources in their order of
/// appearance, whatever the number of threads.
fn scatter(cpu: &Cpu, src: &[f32], rows: &[usize], width: usize, count: usize) -> Vec<f32> {
    let mut sources: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (i, &r) in rows.iter().enumerate() {
        sources[r].push(i);
    }
    let mut out = vec![0.0; count * width];
    let chunk = (CHUNK / width.max(1)).max(1) * width.max(1);
    cpu.fill(&mut out, chunk, |start, part| {
        for (n, row) in part.chunks_mut(width.max(1)).enumerate() {
            for &i in &sources[start / width.max(1) + n] {
                for (y, &v) in row.iter_mut().zip(&src[i * width..(i + 1) * width]) {
                    *y += v;
                }
            }
        }
    });
    out
}

impl Run<GatherRows> for Cpu {
    fn forward(&self, op: &GatherRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let src = io::contiguous(GatherRows::NAME, inputs[0])?;
        let rows = io::indices(GatherRows::NAME, inputs[1], inputs[0].shape()[0])?;
        let width = inputs[0].shape()[1..].iter().product();
        io::write(
            &self.buffers,
            &gather(self, &src, &rows, width),
            &signature.shape,
            signature.dtype,
        )
    }

    fn backward(
        &self,
        op: &GatherRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let shape = inputs[0].shape();
        let rows = io::indices(GatherRows::NAME, inputs[1], shape[0])?;
        let width = shape[1..].iter().product();
        let g = io::contiguous(GatherRows::NAME, grad)?;
        let out = scatter(self, &g, &rows, width, shape[0]);
        Ok(vec![
            Some(io::write(&self.buffers, &out, shape, signature.dtype)?),
            None,
        ])
    }
}

impl Run<ScatterAddRows> for Cpu {
    fn forward(&self, op: &ScatterAddRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let src = io::contiguous(ScatterAddRows::NAME, inputs[0])?;
        let rows = io::indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        let width = signature.shape[1..].iter().product();
        let out = scatter(self, &src, &rows, width, op.rows);
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &ScatterAddRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let rows = io::indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        let width = signature.shape[1..].iter().product();
        let g = io::contiguous(ScatterAddRows::NAME, grad)?;
        let out = gather(self, &g, &rows, width);
        Ok(vec![
            Some(io::write(
                &self.buffers,
                &out,
                inputs[0].shape(),
                signature.dtype,
            )?),
            None,
        ])
    }
}
