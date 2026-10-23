//! Element-wise operations and type conversions.

use kapel_core::{DType, Result, Tensor};

use crate::{Operation, Signature, arity, float, same_device, shape};

/// A function applied to each element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryKind {
    /// `−x`
    Neg,
    /// `eˣ`
    Exp,
    /// `ln x`
    Log,
    /// `√x`
    Sqrt,
    /// `1 / √x`
    Rsqrt,
    /// `x²`
    Square,
    /// `|x|`; its derivative at 0 is taken as 0.
    Abs,
    /// `1 / (1 + e⁻ˣ)`
    Sigmoid,
    /// `tanh x`
    Tanh,
    /// `max(x, 0)`; its derivative at 0 is taken as 0.
    Relu,
    /// `x · sigmoid(x)`, also called swish.
    Silu,
    /// The tanh approximation of GELU (GPT-2, Gemma):
    /// `x/2 · (1 + tanh(√(2/π) · (x + 0.044715 x³)))`.
    GeluTanh,
    /// The exact GELU (BERT, and `"gelu"` in `transformers`): `x · Φ(x)`, with
    /// `Φ` the standard normal distribution function ([`crate::special::erf`]).
    GeluErf,
}

/// `y = f(x)` for each element `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Unary {
    /// The function.
    pub kind: UnaryKind,
}

impl Operation for Unary {
    const NAME: &'static str = "unary";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[1], "1")?;
        float(Self::NAME, inputs[0])?;
        Ok(Signature {
            shape: inputs[0].shape().to_vec(),
            dtype: inputs[0].dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// A gated unit: `y = f(x[…, :n]) · x[…, n:]` for an input `x [..., 2n]`, the
/// gate first. It is the MLP of Llama (SwiGLU, `f` = SiLU), of Gemma (GeGLU)
/// and of most decoders, once their gate and up projections are stacked into
/// one product: one operation instead of a [`Unary`] and a [`Binary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Glu {
    /// The function of the gate.
    pub activation: UnaryKind,
}

impl Operation for Glu {
    const NAME: &'static str = "glu";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        arity(op, inputs, &[1], "1")?;
        let x = inputs[0];
        float(op, x)?;
        let width = x.shape().last().copied().unwrap_or(0);
        if x.rank() == 0 || width % 2 != 0 {
            return Err(kapel_core::Error::InvalidArgument {
                op,
                message: format!(
                    "the last dimension holds the gate then the values: even, got {:?}",
                    x.shape()
                ),
            });
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().expect("rank checked") = width / 2;
        Ok(Signature {
            shape,
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// A function of two elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryKind {
    /// `a + b`
    Add,
    /// `a − b`
    Sub,
    /// `a · b`
    Mul,
    /// `a / b`
    Div,
    /// `max(a, b)`; on a tie, the gradient goes to `a`.
    Max,
    /// `min(a, b)`; on a tie, the gradient goes to `a`.
    Min,
}

/// `y = f(a, b)`, with `a` and `b` broadcast to a common shape (see
/// [`shape::broadcast`]). The gradient of a broadcast input is summed over the
/// dimensions it was repeated along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Binary {
    /// The function.
    pub kind: BinaryKind,
}

impl Operation for Binary {
    const NAME: &'static str = "binary";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
        same_device(Self::NAME, inputs)?;
        float(Self::NAME, inputs[0])?;
        float(Self::NAME, inputs[1])?;
        if inputs[0].dtype() != inputs[1].dtype() {
            return Err(kapel_core::Error::DTypeMismatch {
                op: Self::NAME,
                expected: inputs[0].dtype(),
                got: inputs[1].dtype(),
            });
        }
        Ok(Signature {
            shape: shape::broadcast(Self::NAME, inputs[0].shape(), inputs[1].shape())?,
            dtype: inputs[0].dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// `y = scale · x + bias`, with two constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    /// The factor.
    pub scale: f64,
    /// The constant added.
    pub bias: f64,
}

impl Operation for Affine {
    const NAME: &'static str = "affine";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[1], "1")?;
        float(Self::NAME, inputs[0])?;
        Ok(Signature {
            shape: inputs[0].shape().to_vec(),
            dtype: inputs[0].dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        true
    }
}

/// Converts each element to another scalar type, as [`DType::encode`] does:
/// floats round to nearest, ties to even; integers round and saturate. Between
/// float types, the gradient passes through unchanged; an integer input or
/// output receives none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cast {
    /// The target type.
    pub dtype: DType,
}

impl Operation for Cast {
    const NAME: &'static str = "cast";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[1], "1")?;
        for dtype in [inputs[0].dtype(), self.dtype] {
            if dtype.is_quantized() {
                return Err(kapel_core::Error::UnsupportedDType {
                    op: Self::NAME,
                    dtype,
                });
            }
        }
        Ok(Signature {
            shape: inputs[0].shape().to_vec(),
            dtype: self.dtype,
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        self.dtype.is_float()
    }
}
