//! Updates of parameters in place: the optimizers' steps, fused.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{float, same_device};

/// The name of [`Update::adamw`] in errors.
pub const ADAMW: &str = "adamw";

/// The constants of one AdamW step for one parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWStep {
    /// The learning rate `lr`.
    pub lr: f64,
    /// `β₁`.
    pub beta1: f64,
    /// `β₂`.
    pub beta2: f64,
    /// `ε`.
    pub eps: f64,
    /// The decoupled weight decay `λ`.
    pub weight_decay: f64,
    /// `1 − β₁ᵗ`.
    pub correction1: f64,
    /// `1 − β₂ᵗ`.
    pub correction2: f64,
    /// The factor of the gradient `s` (`1 / accumulated passes`, clipping).
    pub grad_scale: f64,
}

/// Steps of optimizers, written into the parameters and their moments in
/// place.
pub trait Update {
    /// One AdamW step: with `g = s · grad`,
    ///
    /// - `m ← β₁ m + (1 − β₁) g`, `v ← β₂ v + (1 − β₂) g²`;
    /// - `p ← (1 − lr λ) p − (lr / (1 − β₁ᵗ)) · m / (√(v / (1 − β₂ᵗ)) + ε)`,
    ///
    /// computed in `f32`, element by element. `param`, `m` and `v` are `f32`
    /// tensors of one shape, row-major and the only handles on their
    /// storages; `grad` has their shape, in any float type and layout. See
    /// [`check_adamw`].
    ///
    /// # Errors
    ///
    /// Those of [`check_adamw`], and those of the device.
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &AdamWStep,
    ) -> Result<()>;
}

/// Checks a call of [`Update::adamw`].
///
/// # Errors
///
/// [`Error::DeviceMismatch`], [`Error::UnsupportedDType`] for a state that is
/// not `f32` or a gradient that is not a float, [`Error::ShapeMismatch`], and
/// [`Error::InvalidArgument`] for a state that is not row-major from the
/// start of its storage, or not its only handle.
pub fn check_adamw(
    param: &mut Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    grad: &Tensor,
) -> Result<()> {
    let op = ADAMW;
    same_device(op, &[&*param, &*m, &*v, grad])?;
    float(op, grad)?;
    let shape = param.shape().to_vec();
    for t in [&mut *param, &mut *m, &mut *v] {
        if t.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                op,
                dtype: t.dtype(),
            });
        }
        if t.shape() != shape.as_slice() {
            return Err(Error::ShapeMismatch {
                op,
                expected: shape.clone(),
                got: t.shape().to_vec(),
            });
        }
        if !t.is_contiguous() || t.layout().offset() != 0 {
            return Err(Error::InvalidArgument {
                op,
                message: "a parameter or a moment must be row-major from the start of its storage"
                    .into(),
            });
        }
        if t.storage_mut().is_none() {
            return Err(Error::InvalidArgument {
                op,
                message:
                    "a parameter or a moment is shared: updating in place needs its only handle"
                        .into(),
            });
        }
    }
    if grad.shape() != shape.as_slice() {
        return Err(Error::ShapeMismatch {
            op,
            expected: shape,
            got: grad.shape().to_vec(),
        });
    }
    Ok(())
}

impl AdamWStep {
    /// The new `(p, m, v)` of one element, in `f32`, as every backend
    /// computes it.
    pub fn apply(&self, p: f32, m: f32, v: f32, grad: f32) -> (f32, f32, f32) {
        let g = grad * self.grad_scale as f32;
        let m = self.beta1 as f32 * m + (1.0 - self.beta1) as f32 * g;
        let v = self.beta2 as f32 * v + (1.0 - self.beta2) as f32 * (g * g);
        let denominator = (v / self.correction2 as f32).sqrt() + self.eps as f32;
        let p = (1.0 - self.lr * self.weight_decay) as f32 * p
            - (self.lr / self.correction1) as f32 * (m / denominator);
        (p, m, v)
    }
}
