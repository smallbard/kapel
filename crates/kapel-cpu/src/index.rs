//! Gathering and scattering rows.

use kapel_core::{DType, Error, F16, Result, Tensor};
use kapel_ops::{GatherRows, Operation, Run, ScatterAddRows};

use crate::backend::{CHUNK, Cpu};
use crate::{io, quantized};

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

/// The rows of a contiguous table that is not `f32`, decoded one by one into
/// `f32`: only the rows gathered are read.
fn gather_decoded(cpu: &Cpu, table: &Tensor, rows: &[usize], width: usize) -> Result<Vec<f32>> {
    let dtype = table.dtype();
    let row_bytes = dtype.storage_bytes(width)?;
    let start = dtype.storage_bytes(table.layout().offset())?;
    let bytes = &io::host_bytes(GatherRows::NAME, table)?[start..];
    let mut out = vec![0.0; rows.len() * width];
    let chunk = (CHUNK / width.max(1)).max(1) * width.max(1);
    cpu.fill(&mut out, chunk, |at, part| {
        for (n, row) in part.chunks_mut(width.max(1)).enumerate() {
            let r = rows[at / width.max(1) + n];
            let code = &bytes[r * row_bytes..(r + 1) * row_bytes];
            match dtype {
                DType::Quantized(format) => quantized::decode_row(cpu.isa(), format, code, row),
                DType::BF16 => {
                    for (y, pair) in row.iter_mut().zip(code.as_chunks::<2>().0) {
                        *y = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
                    }
                }
                _ => {
                    for (y, pair) in row.iter_mut().zip(code.as_chunks::<2>().0) {
                        *y = F16(u16::from_le_bytes(*pair)).to_f64() as f32;
                    }
                }
            }
        }
    });
    Ok(out)
}

impl Run<GatherRows> for Cpu {
    /// An `f32` table is read in place; a `bf16`, `f16` or quantized one has
    /// only its gathered rows decoded.
    fn forward(&self, op: &GatherRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let table = inputs[0];
        let decoded = match table.dtype() {
            DType::Quantized(format) if quantized::supported(format) => true,
            DType::BF16 | DType::F16 => true,
            _ => false,
        };
        if decoded && table.layout().is_contiguous() {
            let rows = io::indices(GatherRows::NAME, inputs[1], table.shape()[0])?;
            let width = table.shape()[1..].iter().product();
            let out = gather_decoded(self, table, &rows, width)?;
            return io::write(&self.buffers, &out, &signature.shape, signature.dtype);
        }
        if table.dtype().is_quantized() {
            return Err(Error::InvalidArgument {
                op: GatherRows::NAME,
                message: format!(
                    "a {} table must be contiguous and of a format the backend decodes",
                    table.dtype()
                ),
            });
        }
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
        if inputs[0].dtype().is_quantized() {
            return Ok(vec![None, None]);
        }
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
