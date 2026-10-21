//! Matrix products.

use kapel_core::{Error, Layout, Result, Tensor, dequantize};
use kapel_ops::{Matmul, Operation, QuantizedMatmul, Run};

use crate::backend::Cpu;
use crate::gemm::{self, Matrix};
use crate::io::{self, Strided};

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
        io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
            for (c, &(left, right)) in out.chunks_mut((m * n).max(1)).zip(&pairs) {
                gemm::gemm(&self.pool, self.isa, left, right, c, false);
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
        Ok(grads)
    }
}

/// The number of rows of `x [..., K]`: the product of its leading
/// dimensions, which `x.len() / K` cannot give when a dimension is zero.
fn leading_rows(x: &Tensor) -> usize {
    let shape = x.shape();
    shape[..shape.len() - 1].iter().product()
}

/// The weights `[N, K]` dequantized to `f32`, row by row in parallel.
fn dequantized(cpu: &Cpu, w: &Tensor) -> Result<Vec<f32>> {
    let (n, k) = (w.shape()[0], w.shape()[1]);
    let dtype = w.dtype();
    let row_bytes = dtype.storage_bytes(k)?;
    let bytes = io::host_bytes(QuantizedMatmul::NAME, w)?;
    let start = dtype.storage_bytes(w.layout().offset())?;
    let mut out = vec![0.0; n * k];
    let failed = std::sync::atomic::AtomicBool::new(false);
    cpu.fill(&mut out, k.max(1), |at, row| {
        let r = at / k.max(1);
        match dequantize(
            dtype,
            &bytes[start + r * row_bytes..start + (r + 1) * row_bytes],
        ) {
            Ok(values) => row.iter_mut().zip(values).for_each(|(y, v)| *y = v as f32),
            Err(_) => failed.store(true, std::sync::atomic::Ordering::Relaxed),
        }
    });
    if failed.into_inner() {
        return Err(Error::UnsupportedDType {
            op: QuantizedMatmul::NAME,
            dtype,
        });
    }
    Ok(out)
}

impl Run<QuantizedMatmul> for Cpu {
    /// The weights are dequantized, then multiplied. Kernels that read the codes
    /// directly come with the inference engine.
    fn forward(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        io::supported(QuantizedMatmul::NAME, signature.dtype)?;
        let x = io::contiguous(QuantizedMatmul::NAME, inputs[0])?;
        let w = dequantized(self, inputs[1])?;
        let (n, k) = (inputs[1].shape()[0], inputs[1].shape()[1]);
        let rows = leading_rows(inputs[0]);
        let mut out = vec![0.0; rows * n];
        gemm::gemm(
            &self.pool,
            self.isa,
            Matrix::new(&x, rows, k)?,
            Matrix::new(&w, n, k)?.t(),
            &mut out,
            false,
        );
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let g = io::contiguous(QuantizedMatmul::NAME, grad)?;
        let w = dequantized(self, inputs[1])?;
        let (n, k) = (inputs[1].shape()[0], inputs[1].shape()[1]);
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
