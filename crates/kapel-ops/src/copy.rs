//! Copying a tensor into row-major order.

use kapel_core::{Result, Tensor};

use crate::{Operation, Signature, arity};

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
            return Err(kapel_core::Error::UnsupportedDType {
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
