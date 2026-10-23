//! Reductions.

use kapel_core::{Result, Tensor};

use crate::{Operation, Signature, arity, float, shape};

/// The sum over `dims`. Without `keepdim`, the reduced dimensions disappear.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sum {
    /// The dimensions to reduce.
    pub dims: Vec<usize>,
    /// Keep the reduced dimensions, with size 1.
    pub keepdim: bool,
}

/// The mean over `dims`. Without `keepdim`, the reduced dimensions disappear.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mean {
    /// The dimensions to reduce.
    pub dims: Vec<usize>,
    /// Keep the reduced dimensions, with size 1.
    pub keepdim: bool,
}

/// The maximum along `dim`. The gradient goes to the first maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Max {
    /// The dimension to reduce.
    pub dim: usize,
    /// Keep the reduced dimension, with size 1.
    pub keepdim: bool,
}

/// The position along `dim` of the maximum, as a `u32`: the first one on a
/// tie. NaN is never the maximum; a line of NaN only gives 0. This is the
/// choice of greedy decoding, made where the logits are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArgMax {
    /// The dimension to reduce.
    pub dim: usize,
    /// Keep the reduced dimension, with size 1.
    pub keepdim: bool,
}

impl Operation for ArgMax {
    const NAME: &'static str = "argmax";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let signature = infer_reduction(Self::NAME, inputs, &[self.dim], self.keepdim)?;
        let len = inputs[0].shape()[self.dim];
        if len == 0 || len > u32::MAX as usize {
            return Err(kapel_core::Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "dimension {} has {len} elements: a position needs 1 to 2³² of them",
                    self.dim
                ),
            });
        }
        Ok(Signature {
            dtype: kapel_core::DType::U32,
            ..signature
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        false
    }
}

fn infer_reduction(
    op: &'static str,
    inputs: &[&Tensor],
    dims: &[usize],
    keepdim: bool,
) -> Result<Signature> {
    arity(op, inputs, &[1], "1")?;
    float(op, inputs[0])?;
    shape::check_dims(op, dims, inputs[0].rank())?;
    Ok(Signature {
        shape: shape::reduced(inputs[0].shape(), dims, keepdim),
        dtype: inputs[0].dtype(),
    })
}

impl Operation for Sum {
    const NAME: &'static str = "sum";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_reduction(Self::NAME, inputs, &self.dims, self.keepdim)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

impl Operation for Mean {
    const NAME: &'static str = "mean";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_reduction(Self::NAME, inputs, &self.dims, self.keepdim)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

impl Operation for Max {
    const NAME: &'static str = "max";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let signature = infer_reduction(Self::NAME, inputs, &[self.dim], self.keepdim)?;
        if inputs[0].shape()[self.dim] == 0 {
            return Err(kapel_core::Error::InvalidArgument {
                op: Self::NAME,
                message: format!("dimension {} is empty: it has no maximum", self.dim),
            });
        }
        Ok(signature)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}
