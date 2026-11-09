//! The catalogue of Kapel operations.
//!
//! Each operation is a small struct holding its parameters, documented with its
//! exact mathematical definition. [`Operation::infer`] checks the inputs and
//! gives the output's shape and data type, so that every backend accepts and
//! rejects the same calls with the same errors.
//!
//! A backend implements [`Run`] for each operation: the forward computation, and
//! the backward one, which turns the gradient of the output into the gradients of
//! the inputs. [`Backend`] gathers the whole catalogue.
//!
//! The reference implementation, in `f64`, is the crate `kapel-ref`; its
//! conformity suite checks any backend against it.

mod attention;
mod copy;
mod elementwise;
mod experts;
mod index;
mod loss;
mod matmul;
mod norm;
mod paged;
mod recurrent;
mod reduce;
mod rope;
pub mod shape;
pub mod special;
mod update;

use std::fmt;

pub use attention::{Attention, documents_mask, is_documents};
pub use copy::{Concat, Contiguous};
pub use elementwise::{Affine, Binary, BinaryKind, Cast, Glu, Unary, UnaryKind};
pub use experts::{ExpertMatmul, GatherColumns, TopK};
pub use index::{GatherRows, ScatterAddRows};
use kapel_core::{DType, Error, Result, Tensor};
pub use loss::CrossEntropy;
pub use matmul::{Matmul, QuantizedMatmul};
pub use norm::{LayerNorm, LogSoftmax, RmsNorm, Softmax};
pub use paged::{
    COPY_FROM_HOST, PagedAttention, Record, Recording, WRITE_ROWS, Write, check_copy, check_write,
    check_written,
};
pub use recurrent::{
    CAUSAL_CONV, DELTA_RULE, HeadLayout, Recurrent, check_conv, check_delta, check_slots,
};
pub use reduce::{ArgMax, Max, Mean, Sum};
pub use rope::{Rope, RopeLayout, RopeScaling};
pub use update::{
    ADAMW, ADAMW8, AdamWStep, BLOCK8, Moments8, Update, check_adamw, check_adamw8bit, dynamic_map,
    nearest_code,
};

/// The shape and data type of an operation's output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The output shape.
    pub shape: Vec<usize>,
    /// The output data type, for a backend that computes in the inputs' type.
    pub dtype: DType,
}

/// An operation of the catalogue.
pub trait Operation: fmt::Debug + Sync {
    /// The name used in errors and profiles.
    const NAME: &'static str;

    /// Checks the inputs and gives the output's shape and data type.
    ///
    /// # Errors
    ///
    /// Whatever is wrong with the inputs: their number, shapes, data types,
    /// devices, or the operation's own parameters.
    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature>;

    /// Whether input `index` receives a gradient. Indices, masks and quantized
    /// weights do not.
    fn differentiable(&self, index: usize) -> bool;

    /// Checks the inputs and the gradient of a backward call: [`infer`], then
    /// a floating-point `grad` with the output's shape, on the inputs' device.
    /// Every [`Run::backward`] starts with it.
    ///
    /// # Errors
    ///
    /// Those of [`infer`], [`Error::ShapeMismatch`], [`Error::UnsupportedDType`]
    /// or [`Error::DeviceMismatch`].
    ///
    /// [`infer`]: Operation::infer
    fn infer_backward(&self, inputs: &[&Tensor], grad: &Tensor) -> Result<Signature> {
        let signature = self.infer(inputs)?;
        check_grad(Self::NAME, inputs, grad, &signature)?;
        Ok(signature)
    }
}

/// Checks that `grad` can be the gradient of an output with `signature`.
fn check_grad(
    op: &'static str,
    inputs: &[&Tensor],
    grad: &Tensor,
    signature: &Signature,
) -> Result<()> {
    if grad.shape() != signature.shape.as_slice() {
        return Err(Error::ShapeMismatch {
            op,
            expected: signature.shape.clone(),
            got: grad.shape().to_vec(),
        });
    }
    float(op, grad)?;
    if let Some(first) = inputs.first()
        && grad.device() != first.device()
    {
        return Err(Error::DeviceMismatch {
            op,
            expected: first.device(),
            got: grad.device(),
        });
    }
    Ok(())
}

/// A backend's implementation of operation `O`.
pub trait Run<O: Operation> {
    /// Computes the output of `op` on `inputs`.
    ///
    /// # Errors
    ///
    /// Those of [`Operation::infer`], and those of the backend.
    fn forward(&self, op: &O, inputs: &[&Tensor]) -> Result<Tensor>;

    /// Given the gradient of a loss with respect to the output, `grad`, the
    /// gradient with respect to each input: `None` for an input that is not
    /// [differentiable](Operation::differentiable).
    ///
    /// # Errors
    ///
    /// Those of [`Operation::infer_backward`], and those of the backend.
    fn backward(&self, op: &O, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>>;
}

/// Moving tensors between host memory and a backend's device.
pub trait Transfer {
    /// A copy of the host tensor `tensor` on the backend's device; the same
    /// tensor for a host backend.
    ///
    /// # Errors
    ///
    /// Those of the device.
    fn upload(&self, tensor: &Tensor) -> Result<Tensor>;

    /// A copy of `tensor` in host memory; the same tensor for a host backend.
    ///
    /// # Errors
    ///
    /// Those of the device.
    fn download(&self, tensor: &Tensor) -> Result<Tensor>;
}

/// A backend that runs the whole catalogue, and writes the caches of
/// generation in place.
pub trait Backend:
    Write
    + Recurrent
    + Record
    + Update
    + Run<Unary>
    + Run<Glu>
    + Run<Binary>
    + Run<Affine>
    + Run<Cast>
    + Run<Contiguous>
    + Run<Concat>
    + Run<Matmul>
    + Run<QuantizedMatmul>
    + Run<Sum>
    + Run<Mean>
    + Run<Max>
    + Run<ArgMax>
    + Run<Softmax>
    + Run<LogSoftmax>
    + Run<CrossEntropy>
    + Run<LayerNorm>
    + Run<RmsNorm>
    + Run<Attention>
    + Run<PagedAttention>
    + Run<Rope>
    + Run<GatherRows>
    + Run<ScatterAddRows>
    + Run<TopK>
    + Run<GatherColumns>
    + Run<ExpertMatmul>
{
}

impl<B> Backend for B where
    B: Write
        + Recurrent
        + Record
        + Update
        + Run<Unary>
        + Run<Glu>
        + Run<Binary>
        + Run<Affine>
        + Run<Cast>
        + Run<Contiguous>
        + Run<Concat>
        + Run<Matmul>
        + Run<QuantizedMatmul>
        + Run<Sum>
        + Run<Mean>
        + Run<Max>
        + Run<ArgMax>
        + Run<Softmax>
        + Run<LogSoftmax>
        + Run<CrossEntropy>
        + Run<LayerNorm>
        + Run<RmsNorm>
        + Run<Attention>
        + Run<PagedAttention>
        + Run<Rope>
        + Run<GatherRows>
        + Run<ScatterAddRows>
        + Run<TopK>
        + Run<GatherColumns>
        + Run<ExpertMatmul>
{
}

/// Checks the number of inputs.
pub(crate) fn arity(
    op: &'static str,
    inputs: &[&Tensor],
    accepted: &[usize],
    expected: &'static str,
) -> Result<()> {
    if accepted.contains(&inputs.len()) {
        Ok(())
    } else {
        Err(Error::Arity {
            op,
            expected,
            got: inputs.len(),
        })
    }
}

/// Checks that a tensor holds floating-point values.
pub(crate) fn float(op: &'static str, tensor: &Tensor) -> Result<()> {
    if tensor.dtype().is_float() {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            op,
            dtype: tensor.dtype(),
        })
    }
}

/// Checks that a tensor holds indices: `u32` or `i32`.
pub(crate) fn indices(op: &'static str, tensor: &Tensor) -> Result<()> {
    match tensor.dtype() {
        DType::U32 | DType::I32 => Ok(()),
        dtype => Err(Error::UnsupportedDType { op, dtype }),
    }
}

/// Checks that every input lives on the device of the first.
pub(crate) fn same_device(op: &'static str, inputs: &[&Tensor]) -> Result<()> {
    if let Some(first) = inputs.first() {
        for input in &inputs[1..] {
            if input.device() != first.device() {
                return Err(Error::DeviceMismatch {
                    op,
                    expected: first.device(),
                    got: input.device(),
                });
            }
        }
    }
    Ok(())
}

/// Checks that a tensor has the given rank.
pub(crate) fn rank(op: &'static str, tensor: &Tensor, rank: usize, what: &str) -> Result<()> {
    if tensor.rank() == rank {
        Ok(())
    } else {
        Err(Error::InvalidArgument {
            op,
            message: format!(
                "{what} must have rank {rank}, got shape {:?}",
                tensor.shape()
            ),
        })
    }
}
