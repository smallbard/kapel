//! Updates in place.

use kapel_core::{Error, HostStorage, Result, Tensor};
use kapel_ops::{ADAMW, AdamWStep, BLOCK8, Moments8, Update, check_adamw, check_adamw8bit};

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

/// The bytes of a host tensor that can be written.
pub(crate) fn host_u8(t: &mut Tensor) -> Result<&mut [u8]> {
    t.storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .map(HostStorage::bytes_mut)
        .ok_or_else(|| Error::InvalidArgument {
            op: ADAMW,
            message: "the tensor is not in host memory that can be written".into(),
        })
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

    /// [`AdamWStep::apply_block8`], block after block.
    fn adamw8bit(
        &self,
        param: &mut Tensor,
        moments: &mut Moments8,
        grad: &Tensor,
        maps: &Tensor,
        step: &AdamWStep,
    ) -> Result<()> {
        check_adamw8bit(param, moments, grad, maps)?;
        let g: Vec<f32> = grad.to_f64_vec()?.iter().map(|&x| x as f32).collect();
        let table: Vec<f32> = maps.to_f64_vec()?.iter().map(|&x| x as f32).collect();
        let (signed, unsigned) = table.split_at(256);
        let n = g.len();
        let Moments8 {
            m,
            v,
            m_scale,
            v_scale,
        } = moments;
        let (p, m, v) = (host_f32(param)?, host_u8(m)?, host_u8(v)?);
        let (ms, vs) = (host_f32(m_scale)?, host_f32(v_scale)?);
        for b in 0..n.div_ceil(BLOCK8) {
            let r = b * BLOCK8..(b * BLOCK8 + BLOCK8).min(n);
            step.apply_block8(
                &mut p[r.clone()],
                &mut m[r.clone()],
                &mut v[r.clone()],
                &mut ms[b],
                &mut vs[b],
                &g[r],
                signed,
                unsigned,
            );
        }
        Ok(())
    }
}
