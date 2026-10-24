//! Updates in place.

use kapel_core::{Error, HostStorage, Result, Tensor};
use kapel_ops::{ADAMW, AdamWStep, Update, check_adamw};

use crate::Reference;

/// The `f32` values of a host tensor that can be written.
pub(crate) fn host_f32(t: &mut Tensor) -> Result<&mut [f32]> {
    let storage = t
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| Error::InvalidArgument {
            op: ADAMW,
            message: "the tensor is not in host memory that can be written".into(),
        })?;
    let bytes = storage.bytes_mut();
    let n = bytes.len() / 4;
    // SAFETY: a host storage is 64-byte aligned, and its `n · 4` bytes are
    // valid `f32` values of any bits.
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f32>(), n) })
}

impl Update for Reference {
    /// [`AdamWStep::apply`], element by element.
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &AdamWStep,
    ) -> Result<()> {
        check_adamw(param, m, v, grad)?;
        let g = grad.to_f64_vec()?;
        let n = g.len();
        let (pm, mm, vm) = (host_f32(param)?, host_f32(m)?, host_f32(v)?);
        for i in 0..n {
            let (p, m, v) = step.apply(pm[i], mm[i], vm[i], g[i] as f32);
            (pm[i], mm[i], vm[i]) = (p, m, v);
        }
        Ok(())
    }
}
