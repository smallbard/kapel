//! Losses.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, indices, rank, same_device};

/// The cross-entropy of each row of logits against its target, fused: the
/// softmax is never formed.
///
/// Inputs: `x [N, V]`, floats, and `targets [N]`, integers. The output is
/// `y [N]` with
///
/// `y[n] = ln Σⱼ e^(x[n, j]) − x[n, t[n]]`,
///
/// and `y[n] = 0` where `t[n]` is [`CrossEntropy::IGNORE`] (all bits set:
/// `u32::MAX`, or −1 as an `i32`): a position that counts in no loss. Any
/// other target outside `0..V` is an error. The output is in `f32`, or
/// `f64` for logits in `f64`, whatever the type of the logits.
///
/// The gradient of `x` is `g[n] · (softmax(x[n])[j] − [j = t[n]])`, zero on
/// the ignored rows, in the type of `x`; the targets receive none. With it,
/// the backward pass of a language model's loss reads the logits once, and
/// writes their gradient in place of a softmax, a log-softmax and a one-hot
/// of the targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CrossEntropy;

impl CrossEntropy {
    /// The target of a row that counts in no loss.
    pub const IGNORE: u32 = u32::MAX;

    /// The column of a target's value, read as an integer of its tensor
    /// (`u32` or `i32`): `None` for [`CrossEntropy::IGNORE`].
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfRange`] for another value outside `0..vocab`.
    pub fn target(value: i64, vocab: usize) -> Result<Option<usize>> {
        if value == -1 || value == i64::from(Self::IGNORE) {
            return Ok(None);
        }
        usize::try_from(value)
            .ok()
            .filter(|&t| t < vocab)
            .map(Some)
            .ok_or(Error::IndexOutOfRange {
                op: Self::NAME,
                index: value,
                size: vocab,
            })
    }

    /// The type of the losses for logits of type `dtype`.
    pub fn output_dtype(dtype: DType) -> DType {
        if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        }
    }
}

impl Operation for CrossEntropy {
    const NAME: &'static str = "cross_entropy";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        arity(op, inputs, &[2], "2")?;
        same_device(op, inputs)?;
        let (x, t) = (inputs[0], inputs[1]);
        float(op, x)?;
        indices(op, t)?;
        rank(op, x, 2, "the logits")?;
        if t.shape() != [x.shape()[0]] {
            return Err(Error::ShapeMismatch {
                op,
                expected: vec![x.shape()[0]],
                got: t.shape().to_vec(),
            });
        }
        Ok(Signature {
            shape: vec![x.shape()[0]],
            dtype: Self::output_dtype(x.dtype()),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}
