//! Matrix products.

use kapel_core::{DType, Error, Layout, Result, Tensor};
use kapel_ops::{Matmul, Operation, QuantizedMatmul, Run};

use crate::backend::Cpu;
use crate::gemm::{self, Matrix};
use crate::io::{self, Strided};
use crate::{bytes, quantized};

/// Up to this many rows of `x`, a quantized product reads its weights in
/// place; beyond, decoding each block once per [`quantized::ROWS`] rows costs
/// more than dequantizing the matrix once.
const IN_PLACE_ROWS: usize = 16;

/// The matrix of batch `index` of a strided tensor `[…, R, C]`, broadcast to the
/// batch shape `batch`.
fn matrix<'a>(input: &'a Strided<'_>, batch: &[usize], index: usize) -> Result<Matrix<'a>> {
    let layout = &input.layout;
    let rank = layout.rank();
    let (strides, shape) = (layout.strides(), layout.shape());
    // The position of the batch element, read through broadcast strides.
    let lead = batch.len() - (rank - 2);
    let mut offset = layout.offset();
    let mut rest = index;
    for d in (0..batch.len()).rev() {
        let coordinate = rest % batch[d];
        rest /= batch[d];
        if d >= lead && shape[d - lead] != 1 {
            offset += coordinate * strides[d - lead];
        }
    }
    // An empty tensor may have an offset at the end of its data.
    Matrix::strided(
        input.data.get(offset..).unwrap_or_default(),
        shape[rank - 2],
        shape[rank - 1],
        strides[rank - 2],
        strides[rank - 1],
    )
}

/// For each batch element of the output, the batch element of an input of
/// batch shape `own` that it reads.
fn owners(own: &[usize], batch: &[usize]) -> Result<Vec<usize>> {
    Ok(Layout::contiguous(own)?
        .broadcast_to(batch)?
        .offsets()
        .collect())
}

impl Run<Matmul> for Cpu {
    fn forward(&self, op: &Matmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let (a, b) = (
            io::strided(Matmul::NAME, inputs[0])?,
            io::strided(Matmul::NAME, inputs[1])?,
        );
        let rank = signature.shape.len();
        let batch = &signature.shape[..rank - 2];
        let (m, n) = (signature.shape[rank - 2], signature.shape[rank - 1]);
        let count: usize = batch.iter().product();
        let pairs = (0..count)
            .map(|t| Ok((matrix(&a, batch, t)?, matrix(&b, batch, t)?)))
            .collect::<Result<Vec<_>>>()?;
        let bias = match inputs.get(2) {
            Some(b) => Some(io::contiguous(Matmul::NAME, b)?),
            None => None,
        };
        io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
            for (c, &(left, right)) in out.chunks_mut((m * n).max(1)).zip(&pairs) {
                // With a bias, each row starts from it and the product adds
                // to it.
                if let Some(bias) = &bias {
                    for row in c.chunks_mut(n.max(1)) {
                        row.copy_from_slice(bias);
                    }
                }
                gemm::gemm(&self.pool, self.isa, left, right, c, bias.is_some());
            }
        })
    }

    /// `∂a = g · bᵀ`, `∂b = aᵀ · g`; a broadcast input accumulates the products
    /// of every batch element that read it, in batch order.
    fn backward(
        &self,
        op: &Matmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let (a, b) = (
            io::strided(Matmul::NAME, inputs[0])?,
            io::strided(Matmul::NAME, inputs[1])?,
        );
        let g = io::strided(Matmul::NAME, grad)?;
        let rank = signature.shape.len();
        let batch = &signature.shape[..rank - 2];
        let count: usize = batch.iter().product();
        let mut grads = Vec::new();
        for (side, input) in inputs.iter().take(2).enumerate() {
            let shape = input.shape();
            let own = &shape[..shape.len() - 2];
            let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let owner = owners(own, batch)?;
            let mut out = vec![0.0; shape.iter().product()];
            let mut written = vec![false; own.iter().product::<usize>().max(1)];
            for (t, &target) in owner.iter().enumerate().take(count) {
                let c = &mut out[target * rows * cols..(target + 1) * rows * cols];
                let accumulate = std::mem::replace(&mut written[target], true);
                let (left, right) = if side == 0 {
                    (matrix(&g, batch, t)?, matrix(&b, batch, t)?.t())
                } else {
                    (matrix(&a, batch, t)?.t(), matrix(&g, batch, t)?)
                };
                gemm::gemm(&self.pool, self.isa, left, right, c, accumulate);
            }
            grads.push(Some(io::write(
                &self.buffers,
                &out,
                shape,
                signature.dtype,
            )?));
        }
        if let Some(bias) = inputs.get(2) {
            // The output's gradient summed over every row, in order.
            let n = bias.shape()[0];
            let g = io::contiguous(Matmul::NAME, grad)?;
            let mut sum = vec![0.0f32; n];
            for row in g.chunks(n.max(1)) {
                for (s, v) in sum.iter_mut().zip(row) {
                    *s += v;
                }
            }
            grads.push(Some(io::write(&self.buffers, &sum, &[n], bias.dtype())?));
        }
        Ok(grads)
    }
}

/// The number of rows of `x [..., K]`: the product of its leading
/// dimensions, which `x.len() / K` cannot give when a dimension is zero.
fn leading_rows(x: &Tensor) -> usize {
    let shape = x.shape();
    shape[..shape.len() - 1].iter().product()
}

/// The bytes of the rows of a quantized weight, in place.
fn rows_of<'a>(w: &'a Tensor, k: usize) -> Result<quantized::Weights<'a>> {
    let DType::Quantized(format) = w.dtype() else {
        return Err(Error::UnsupportedDType {
            op: QuantizedMatmul::NAME,
            dtype: w.dtype(),
        });
    };
    if !quantized::supported(format) || !w.layout().is_contiguous() {
        return Err(Error::InvalidArgument {
            op: QuantizedMatmul::NAME,
            message: format!(
                "weights in {} must be contiguous and decoded here",
                w.dtype()
            ),
        });
    }
    let row_bytes = w.dtype().storage_bytes(k)?;
    let start = w.dtype().storage_bytes(w.layout().offset())?;
    let all = io::host_bytes(QuantizedMatmul::NAME, w)?;
    Ok(quantized::Weights {
        format,
        bytes: &all[start..start + w.shape()[0] * row_bytes],
        n: w.shape()[0],
    })
}

/// Rows `first .. first + out.len() / k` of `w`, dequantized to `f32` in
/// parallel.
fn decode_into(cpu: &Cpu, w: quantized::Weights<'_>, k: usize, first: usize, out: &mut [f32]) {
    let row_bytes = w.bytes.len() / w.n.max(1);
    // Sixteen rows per task.
    cpu.fill(out, (16 * k).max(1), |at, part| {
        let row = first + at / k.max(1);
        let rows = part.len() / k.max(1);
        let bytes = &w.bytes[row * row_bytes..(row + rows) * row_bytes];
        quantized::decode_row(cpu.isa, w.format, bytes, part);
    });
}

/// The weights `[N, K]` dequantized to `f32`.
fn dequantized(cpu: &Cpu, w: &Tensor) -> Result<Vec<f32>> {
    let k = w.shape()[1];
    let rows = rows_of(w, k)?;
    let mut out = vec![0.0; rows.n * k];
    decode_into(cpu, rows, k, 0, &mut out);
    Ok(out)
}

/// The weight bytes of `f32` dequantized at a time by a product of many
/// rows: a tile that stays in the caches, instead of whole matrices of up to
/// gigabytes.
const TILE_BYTES: usize = 2 << 20;

/// `out = x · [w₁; w₂; …]ᵀ` for many rows of `x`: each tile of rows of the
/// weights is dequantized, then multiplied by the blocked product.
fn tiled(
    cpu: &Cpu,
    weights: &[quantized::Weights<'_>],
    k: usize,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
) -> Result<()> {
    let width: usize = weights.iter().map(|w| w.n).sum();
    let tile = (TILE_BYTES / (4 * k.max(1))).clamp(16, 4096);
    let mut scratch = vec![0.0f32; tile * k];
    let mut part = vec![0.0f32; rows * tile];
    let mut column = 0;
    for w in weights {
        for first in (0..w.n).step_by(tile) {
            let t = tile.min(w.n - first);
            let decoded = &mut scratch[..t * k];
            decode_into(cpu, *w, k, first, decoded);
            let c = &mut part[..rows * t];
            gemm::gemm(
                &cpu.pool,
                cpu.isa,
                Matrix::new(x, rows, k)?,
                Matrix::new(decoded, t, k)?.t(),
                c,
                false,
            );
            for r in 0..rows {
                out[r * width + column + first..][..t].copy_from_slice(&c[r * t..][..t]);
            }
        }
        column += w.n;
    }
    Ok(())
}

/// Weights `[Nᵢ, K]` dequantized to `f32`, their rows stacked.
fn stacked(cpu: &Cpu, weights: &[&Tensor]) -> Result<Vec<f32>> {
    let mut all = Vec::new();
    for w in weights {
        match rows_of(w, w.shape()[1]) {
            Ok(_) => all.extend(dequantized(cpu, w)?),
            Err(_) => all.extend(stacked_reference(&[w])?),
        }
    }
    Ok(all)
}

/// Weights dequantized by `kapel_core::dequantize`, for the formats and
/// layouts without a decoder here.
fn stacked_reference(weights: &[&Tensor]) -> Result<Vec<f32>> {
    let mut all = Vec::new();
    for w in weights {
        all.extend(w.to_f64_vec()?.into_iter().map(|v| v as f32));
    }
    Ok(all)
}

impl Run<QuantizedMatmul> for Cpu {
    /// For a few rows of `x` (generation), the weights are read in place,
    /// block by block, with activations in `f32` or in bytes
    /// ([`Cpu::with_byte_activations`]); for more (the prefill), tiles of the
    /// weights are dequantized, then multiplied by blocks.
    fn forward(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        io::supported(QuantizedMatmul::NAME, signature.dtype)?;
        let x = io::contiguous(QuantizedMatmul::NAME, inputs[0])?;
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        let rows = leading_rows(inputs[0]);
        let in_place = |test: fn(kapel_core::QuantFormat) -> bool| {
            inputs[1..].iter().all(|w| {
                matches!(w.dtype(), DType::Quantized(f) if test(f)) && w.layout().is_contiguous()
            })
        };
        if !in_place(quantized::supported) {
            // A format without its own decoder: the reference's.
            let w = stacked_reference(&inputs[1..])?;
            let (left, right) = (Matrix::new(&x, rows, k)?, Matrix::new(&w, n, k)?.t());
            return io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
                gemm::gemm(&self.pool, self.isa, left, right, out, false);
            });
        }
        let weights = inputs[1..]
            .iter()
            .map(|w| rows_of(w, k))
            .collect::<Result<Vec<_>>>()?;
        if rows <= IN_PLACE_ROWS {
            let bytes_too = self.byte_activations && in_place(bytes::supported);
            // Every element of the output is written.
            return io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
                if bytes_too {
                    bytes::product(&self.pool, self.isa, self.vnni, &weights, k, &x, rows, out);
                } else {
                    quantized::product(&self.pool, self.isa, &weights, k, &x, rows, out);
                }
            });
        }
        let mut failed = None;
        let y = io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
            failed = tiled(self, &weights, k, &x, rows, out).err();
        })?;
        match failed {
            Some(error) => Err(error),
            None => Ok(y),
        }
    }

    fn backward(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let g = io::contiguous(QuantizedMatmul::NAME, grad)?;
        let w = stacked(self, &inputs[1..])?;
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        let rows = leading_rows(inputs[0]);
        let mut out = vec![0.0; rows * k];
        gemm::gemm(
            &self.pool,
            self.isa,
            Matrix::new(&g, rows, n)?,
            Matrix::new(&w, n, k)?,
            &mut out,
            false,
        );
        let mut grads = vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)];
        grads.resize(inputs.len(), None);
        Ok(grads)
    }
}
