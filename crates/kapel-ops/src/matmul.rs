//! Matrix products.

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, same_device, shape};

/// The batched matrix product: `a` is `[…, M, K]`, `b` is `[…, K, N]`, and the
/// output is `[…, M, N]`, with `y[…, i, j] = Σₖ a[…, i, k] · b[…, k, j]`.
///
/// The batch dimensions (all but the last two) broadcast as in
/// [`shape::broadcast`]. Both inputs may be strided views: a transposed matrix is
/// a `transpose` of the last two dimensions, without a copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Matmul;

impl Operation for Matmul {
    const NAME: &'static str = "matmul";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
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
        let mut out = shape::broadcast(&a.shape()[..ra - 2], &b.shape()[..rb - 2])?;
        out.extend([m, n]);
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QuantizedMatmul;

impl Operation for QuantizedMatmul {
    const NAME: &'static str = "quantized_matmul";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
        same_device(Self::NAME, inputs)?;
        let (x, w) = (inputs[0], inputs[1]);
        float(Self::NAME, x)?;
        if !w.dtype().is_quantized() {
            return Err(Error::UnsupportedDType {
                op: Self::NAME,
                dtype: w.dtype(),
            });
        }
        if x.rank() < 1 || w.rank() != 2 || x.shape()[x.rank() - 1] != w.shape()[1] {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "expected x [..., K] and w [N, K], got {:?} and {:?}",
                    x.shape(),
                    w.shape()
                ),
            });
        }
        let mut out = x.shape()[..x.rank() - 1].to_vec();
        out.push(w.shape()[0]);
        Ok(Signature {
            shape: out,
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}
