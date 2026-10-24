//! Matrix products.

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, same_device, shape};

/// The batched matrix product: `a` is `[…, M, K]`, `b` is `[…, K, N]`, and the
/// output is `[…, M, N]`, with `y[…, i, j] = Σₖ a[…, i, k] · b[…, k, j]`.
///
/// The batch dimensions (all but the last two) broadcast as in
/// [`shape::broadcast`]. Both inputs may be strided views: a transposed matrix is
/// a `transpose` of the last two dimensions, without a copy.
///
/// A third input, `bias [N]` of the type of `a`, is added to every row of the
/// output before it is rounded to its type: `y[…, i, j] = bias[j] + Σₖ a[…,
/// i, k] · b[…, k, j]`, the bias of a linear layer in the product itself.
/// Its gradient is the sum of the output's over every row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Matmul;

impl Operation for Matmul {
    const NAME: &'static str = "matmul";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2, 3], "2 or 3")?;
        same_device(Self::NAME, inputs)?;
        let (a, b) = (inputs[0], inputs[1]);
        float(Self::NAME, a)?;
        float(Self::NAME, b)?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                op: Self::NAME,
                expected: a.dtype(),
                got: b.dtype(),
            });
        }
        if a.rank() < 2 || b.rank() < 2 {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "both inputs need at least 2 dimensions, got shapes {:?} and {:?}",
                    a.shape(),
                    b.shape()
                ),
            });
        }
        let (ra, rb) = (a.rank(), b.rank());
        let (m, k) = (a.shape()[ra - 2], a.shape()[ra - 1]);
        let (k2, n) = (b.shape()[rb - 2], b.shape()[rb - 1]);
        if k != k2 {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "inner dimensions differ: {:?} · {:?} ({k} ≠ {k2})",
                    a.shape(),
                    b.shape()
                ),
            });
        }
        let mut out = shape::broadcast(Self::NAME, &a.shape()[..ra - 2], &b.shape()[..rb - 2])?;
        out.extend([m, n]);
        if let Some(bias) = inputs.get(2) {
            if bias.dtype() != a.dtype() {
                return Err(Error::DTypeMismatch {
                    op: Self::NAME,
                    expected: a.dtype(),
                    got: bias.dtype(),
                });
            }
            if bias.shape() != [n] {
                return Err(Error::ShapeMismatch {
                    op: Self::NAME,
                    expected: vec![n],
                    got: bias.shape().to_vec(),
                });
            }
        }
        Ok(Signature {
            shape: out,
            dtype: a.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// The product by quantized weights: `x` is `[…, K]` in a float type, `w` is
/// `[N, K]` in a quantized type, its blocks along `K`, and the output is `[…, N]`
/// with `y[…, n] = Σₖ x[…, k] · w[n, k]`, `w` taken at its exact dequantized
/// value. The weights are frozen: only `x` receives a gradient.
///
/// Several weights `w₁ [N₁, K], w₂ [N₂, K], …`, of any quantized types, may
/// follow `x`: they act as one matrix, their rows stacked, and the output is
/// `[…, N₁ + N₂ + …]`. The projections of a layer that read the same input
/// (queries, keys and values) are then one product: a backend reads their
/// weights in one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QuantizedMatmul;

impl Operation for QuantizedMatmul {
    const NAME: &'static str = "quantized_matmul";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        if inputs.len() < 2 {
            return Err(Error::Arity {
                op: Self::NAME,
                expected: "2 or more",
                got: inputs.len(),
            });
        }
        same_device(Self::NAME, inputs)?;
        let x = inputs[0];
        float(Self::NAME, x)?;
        let k = x.shape().last().copied();
        let mut n = 0usize;
        for w in &inputs[1..] {
            if !w.dtype().is_quantized() {
                return Err(Error::UnsupportedDType {
                    op: Self::NAME,
                    dtype: w.dtype(),
                });
            }
            if w.rank() != 2 || k != Some(w.shape()[1]) {
                return Err(Error::InvalidArgument {
                    op: Self::NAME,
                    message: format!(
                        "expected x [..., K] and w [N, K], got {:?} and {:?}",
                        x.shape(),
                        w.shape()
                    ),
                });
            }
            if !w.shape()[1].is_multiple_of(w.dtype().block_len()) {
                return Err(Error::BlockMisaligned {
                    dtype: w.dtype(),
                    elements: w.shape()[1],
                });
            }
            n = n
                .checked_add(w.shape()[0])
                .ok_or(Error::Overflow { op: Self::NAME })?;
        }
        let mut out = x.shape()[..x.rank() - 1].to_vec();
        out.push(n);
        Ok(Signature {
            shape: out,
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}
