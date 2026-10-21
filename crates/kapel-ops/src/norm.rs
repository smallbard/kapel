//! Softmax and normalizations.

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, same_device};

/// `y = eˣ / Σ eˣ` along `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Softmax {
    /// The dimension that sums to 1.
    pub dim: usize,
}

/// `y = x − ln Σ eˣ` along `dim`: the logarithm of [`Softmax`], computed
/// without forming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogSoftmax {
    /// The dimension that is normalized.
    pub dim: usize,
}

fn infer_along(op: &'static str, inputs: &[&Tensor], dim: usize) -> Result<Signature> {
    arity(op, inputs, &[1], "1")?;
    float(op, inputs[0])?;
    if dim >= inputs[0].rank() {
        return Err(Error::InvalidDim {
            op,
            dim,
            rank: inputs[0].rank(),
        });
    }
    Ok(Signature {
        shape: inputs[0].shape().to_vec(),
        dtype: inputs[0].dtype(),
    })
}

impl Operation for Softmax {
    const NAME: &'static str = "softmax";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_along(Self::NAME, inputs, self.dim)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

impl Operation for LogSoftmax {
    const NAME: &'static str = "log_softmax";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_along(Self::NAME, inputs, self.dim)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// Checks `x [..., C]` and the optional vectors `[C]` that follow it.
fn infer_norm(op: &'static str, inputs: &[&Tensor], max_inputs: usize) -> Result<Signature> {
    let accepted: Vec<usize> = (1..=max_inputs).collect();
    let expected = if max_inputs == 2 {
        "1 or 2"
    } else {
        "1, 2 or 3"
    };
    arity(op, inputs, &accepted, expected)?;
    same_device(op, inputs)?;
    let x = inputs[0];
    float(op, x)?;
    let width = x
        .shape()
        .last()
        .copied()
        .ok_or_else(|| Error::InvalidArgument {
            op,
            message: "the input needs at least one dimension".into(),
        })?;
    for vector in &inputs[1..] {
        float(op, vector)?;
        if vector.shape() != [width] {
            return Err(Error::ShapeMismatch {
                op,
                expected: vec![width],
                got: vector.shape().to_vec(),
            });
        }
    }
    Ok(Signature {
        shape: x.shape().to_vec(),
        dtype: x.dtype(),
    })
}

/// Layer normalization over the last dimension: with `μ` and `σ²` the mean and
/// the (biased) variance of a row,
/// `y = (x − μ) / √(σ² + eps) · weight + bias`.
///
/// Inputs: `x [..., C]`, then optionally `weight [C]`, then optionally `bias [C]`.
/// The vectors may be in any float type, not necessarily that of `x`; their
/// gradients come in their own type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNorm {
    /// Added to the variance.
    pub eps: f64,
}

impl Operation for LayerNorm {
    const NAME: &'static str = "layer_norm";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_norm(Self::NAME, inputs, 3)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// Root-mean-square normalization over the last dimension:
/// `y = x / √(mean(x²) + eps) · (weight_offset + weight)`.
///
/// Inputs: `x [..., C]`, then optionally `weight [C]`, in any float type.
/// `weight_offset` is 0 for Llama and Qwen, 1 for Gemma, which stores
/// `weight − 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNorm {
    /// Added to the mean of the squares.
    pub eps: f64,
    /// Added to the weight before it multiplies.
    pub weight_offset: f64,
}

impl Operation for RmsNorm {
    const NAME: &'static str = "rms_norm";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        infer_norm(Self::NAME, inputs, 2)
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}
