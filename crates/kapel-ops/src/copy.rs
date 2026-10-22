//! Copies: a tensor into row-major order, tensors joined along a dimension.

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, same_device};

/// `y = x`, its elements stored in row-major order: a strided view (transposed,
/// narrowed, broadcast) becomes a tensor of its own. The values are copied bit
/// for bit, in any type but the quantized ones; a contiguous `x` may be
/// returned as is.
///
/// The gradient passes through unchanged; an integer input receives none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Contiguous;

impl Operation for Contiguous {
    const NAME: &'static str = "contiguous";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[1], "1")?;
        let x = inputs[0];
        if x.dtype().is_quantized() && !x.is_contiguous() {
            return Err(Error::UnsupportedDType {
                op: Self::NAME,
                dtype: x.dtype(),
            });
        }
        Ok(Signature {
            shape: x.shape().to_vec(),
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// The inputs joined along dimension `dim`: they have the same type and the
/// same shape but along `dim`, and the output's size along `dim` is the sum of
/// theirs. The values are copied bit for bit, in any type but the quantized
/// ones.
///
/// Each input's gradient is its slice of the output's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Concat {
    /// The dimension along which the inputs follow each other.
    pub dim: usize,
}

impl Operation for Concat {
    const NAME: &'static str = "concat";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        let Some(first) = inputs.first() else {
            return Err(Error::Arity {
                op,
                expected: "at least 1",
                got: 0,
            });
        };
        same_device(op, inputs)?;
        let rank = first.rank();
        if self.dim >= rank {
            return Err(Error::InvalidDim {
                op,
                dim: self.dim,
                rank,
            });
        }
        let mut shape = first.shape().to_vec();
        shape[self.dim] = 0;
        for x in inputs {
            if x.dtype().is_quantized() {
                return Err(Error::UnsupportedDType {
                    op,
                    dtype: x.dtype(),
                });
            }
            if x.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    op,
                    expected: first.dtype(),
                    got: x.dtype(),
                });
            }
            let agrees = x.rank() == rank
                && (0..rank).all(|d| d == self.dim || x.shape()[d] == first.shape()[d]);
            if !agrees {
                let mut expected = first.shape().to_vec();
                expected[self.dim] = x.shape().get(self.dim).copied().unwrap_or(0);
                return Err(Error::ShapeMismatch {
                    op,
                    expected,
                    got: x.shape().to_vec(),
                });
            }
            shape[self.dim] = shape[self.dim]
                .checked_add(x.shape()[self.dim])
                .ok_or(Error::Overflow { op })?;
        }
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(Error::Overflow { op })?;
        Ok(Signature {
            shape,
            dtype: first.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}
