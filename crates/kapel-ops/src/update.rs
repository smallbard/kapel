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

    /// One AdamW step whose moments are kept in 8 bits ([`Moments8`]), as
    /// bitsandbytes' blockwise 8-bit Adam keeps them: for each block of
    /// [`BLOCK8`] elements, the moments are read from their codes
    /// (`m = signed[code] · m_scale`, `v = unsigned[code] · v_scale`), the
    /// step is [`AdamWStep::apply`]'s, and the new moments are written back
    /// as the codes nearest to them over the largest of the block
    /// ([`AdamWStep::apply_block8`]). `maps` holds the 512 values of
    /// [`dynamic_map`], signed then unsigned, on the device. See
    /// [`check_adamw8bit`].
    ///
    /// # Errors
    ///
    /// Those of [`check_adamw8bit`], and those of the device.
    fn adamw8bit(
        &self,
        param: &mut Tensor,
        moments: &mut Moments8,
        grad: &Tensor,
        maps: &Tensor,
        step: &AdamWStep,
    ) -> Result<()>;
}

/// The name of [`Update::adamw8bit`] in errors.
pub const ADAMW8: &str = "adamw8bit";

/// The elements that share one scale in [`Moments8`]: bitsandbytes' block.
pub const BLOCK8: usize = 256;

/// The two moments of AdamW in 8 bits: a code of [`dynamic_map`] per
/// element (`u8`, the parameter's shape), and per block of [`BLOCK8`]
/// elements the largest magnitude of the block (`f32`, `[⌈n / 256⌉]`), by
/// which the codes' values are multiplied. Zeros are moments of zero.
#[derive(Clone, Debug)]
pub struct Moments8 {
    /// The codes of the first moment, in the signed map.
    pub m: Tensor,
    /// The codes of the second moment, in the unsigned map.
    pub v: Tensor,
    /// The scale of each block of `m`.
    pub m_scale: Tensor,
    /// The scale of each block of `v`.
    pub v_scale: Tensor,
}

/// The 256 values a code of 8 bits stands for, sorted: bitsandbytes'
/// dynamic map (`create_dynamic_map`, Dettmers 2016), bit for bit. Signed,
/// a sign, then for each exponent `e` of 0 to 6 the `2ᵉ` means of a
/// subdivision of `[0.1, 1]` scaled by `10^(e − 6)`, with 0 and 1 (and no
/// −1); unsigned, twice as many means of each subdivision, no sign. The
/// values are computed as `torch.linspace` and its products in `f32` give
/// them.
pub fn dynamic_map(signed: bool) -> [f32; 256] {
    const SCALES: [f64; 7] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0];
    let linspace = |n: usize| -> Vec<f32> {
        let (start, end) = (0.1f32, 1.0f32);
        let step = (end - start) / (n - 1) as f32;
        (0..n)
            .map(|i| {
                // Fused, rounded once, as torch's vectorized linspace.
                if i < n / 2 {
                    step.mul_add(i as f32, start)
                } else {
                    (-step).mul_add((n - 1 - i) as f32, end)
                }
            })
            .collect()
    };
    let mut data = Vec::with_capacity(256);
    for (e, &scale) in SCALES.iter().enumerate() {
        let items = if signed {
            (1 << e) + 1
        } else {
            (1 << (e + 1)) + 1
        };
        let scale = scale as f32;
        for pair in linspace(items).windows(2) {
            let mean = (pair[0] + pair[1]) / 2.0;
            data.push(scale * mean);
            if signed {
                data.push(-scale * mean);
            }
        }
    }
    data.push(0.0);
    data.push(1.0);
    data.sort_by(f32::total_cmp);
    let mut map = [0.0; 256];
    map.copy_from_slice(&data);
    map
}

/// The code of `x` (in `[−1, 1]`) in a sorted `map` of 256 values: the
/// value nearest to it, the lower one at a tie (`torch.bucketize` over the
/// midpoints, as bitsandbytes rounds).
pub fn nearest_code(map: &[f32], x: f32) -> u8 {
    let (mut lo, mut hi) = (0usize, 255usize);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (map[mid] + map[mid + 1]) * 0.5 < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// Checks a call of [`Update::adamw8bit`].
///
/// # Errors
///
/// [`Error::DeviceMismatch`]; [`Error::UnsupportedDType`] for a parameter or
/// scales not in `f32`, codes not in `u8`, or a gradient that is not a
/// float; [`Error::ShapeMismatch`] for codes or a gradient not of the
/// parameter's shape, scales not `[⌈n / 256⌉]` or maps not `[512]`; and
/// [`Error::InvalidArgument`] for a state that is not row-major from the
/// start of its storage, or not its only handle.
pub fn check_adamw8bit(
    param: &mut Tensor,
    moments: &mut Moments8,
    grad: &Tensor,
    maps: &Tensor,
) -> Result<()> {
    let op = ADAMW8;
    let Moments8 {
        m,
        v,
        m_scale,
        v_scale,
    } = moments;
    same_device(op, &[&*param, &*m, &*v, &*m_scale, &*v_scale, grad, maps])?;
    float(op, grad)?;
    let shape = param.shape().to_vec();
    let blocks = vec![param.numel().div_ceil(BLOCK8)];
    for (t, dtype, expected) in [
        (&mut *param, DType::F32, &shape),
        (&mut *m, DType::U8, &shape),
        (&mut *v, DType::U8, &shape),
        (&mut *m_scale, DType::F32, &blocks),
        (&mut *v_scale, DType::F32, &blocks),
    ] {
        state(op, t, dtype, expected)?;
    }
    for (t, dtype, expected) in [(grad, None, &shape), (maps, Some(DType::F32), &vec![512])] {
        if let Some(dtype) = dtype
            && t.dtype() != dtype
        {
            return Err(Error::UnsupportedDType {
                op,
                dtype: t.dtype(),
            });
        }
        if t.shape() != expected.as_slice() {
            return Err(Error::ShapeMismatch {
                op,
                expected: expected.clone(),
                got: t.shape().to_vec(),
            });
        }
    }
    Ok(())
}

/// Checks a tensor an update writes in place: its type and shape, row-major
/// from the start of its storage, and its storage's only handle.
fn state(op: &'static str, t: &mut Tensor, dtype: DType, shape: &[usize]) -> Result<()> {
    if t.dtype() != dtype {
        return Err(Error::UnsupportedDType {
            op,
            dtype: t.dtype(),
        });
    }
    if t.shape() != shape {
        return Err(Error::ShapeMismatch {
            op,
            expected: shape.to_vec(),
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
            message: "a parameter or a moment is shared: updating in place needs its only handle"
                .into(),
        });
    }
    Ok(())
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
        state(op, t, DType::F32, &shape)?;
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

    /// One block of [`Update::adamw8bit`], as every backend computes it: at
    /// most [`BLOCK8`] elements of the parameter, their codes and the
    /// block's two scales, updated in place, with the gradient of each
    /// element and the two maps of [`dynamic_map`]. A scale is the largest
    /// magnitude of the block's new moments; each moment over it, kept in
    /// `[−1, 1]`, is written as its [`nearest_code`].
    #[allow(clippy::too_many_arguments)]
    pub fn apply_block8(
        &self,
        p: &mut [f32],
        m: &mut [u8],
        v: &mut [u8],
        m_scale: &mut f32,
        v_scale: &mut f32,
        grad: &[f32],
        signed: &[f32],
        unsigned: &[f32],
    ) {
        let n = p.len();
        let (mut first, mut second) = ([0.0f32; BLOCK8], [0.0f32; BLOCK8]);
        let (mut top_m, mut top_v) = (0.0f32, 0.0f32);
        for i in 0..n {
            let mi = signed[m[i] as usize] * *m_scale;
            let vi = unsigned[v[i] as usize] * *v_scale;
            let (pi, mi, vi) = self.apply(p[i], mi, vi, grad[i]);
            p[i] = pi;
            (first[i], second[i]) = (mi, vi);
            top_m = top_m.max(mi.abs());
            top_v = top_v.max(vi.abs());
        }
        let inverse = |top: f32| if top > 0.0 { 1.0 / top } else { 0.0 };
        let (im, iv) = (inverse(top_m), inverse(top_v));
        for i in 0..n {
            m[i] = nearest_code(signed, (first[i] * im).clamp(-1.0, 1.0));
            v[i] = nearest_code(unsigned, (second[i] * iv).clamp(-1.0, 1.0));
        }
        (*m_scale, *v_scale) = (top_m, top_v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bitsandbytes' maps, as `create_dynamic_map` gives them with
    /// `torch` (the bits of each `f32`).
    const SIGNED: [u32; 256] = [
        0xbf7e3333, 0xbf7a999a, 0xbf770000, 0xbf736666, 0xbf6fcccd, 0xbf6c3333, 0xbf68999a,
        0xbf650000, 0xbf616666, 0xbf5dcccd, 0xbf5a3333, 0xbf56999a, 0xbf530000, 0xbf4f6666,
        0xbf4bcccd, 0xbf483333, 0xbf44999a, 0xbf410000, 0xbf3d6666, 0xbf39cccd, 0xbf363334,
        0xbf32999a, 0xbf2f0000, 0xbf2b6666, 0xbf27cccd, 0xbf243334, 0xbf20999a, 0xbf1d0000,
        0xbf196666, 0xbf15cccd, 0xbf123334, 0xbf0e999a, 0xbf0b0000, 0xbf076666, 0xbf03cccc,
        0xbf003333, 0xbef93332, 0xbef20000, 0xbeeacccc, 0xbee3999a, 0xbedc6666, 0xbed53333,
        0xbece0000, 0xbec6cccc, 0xbebf999a, 0xbeb86666, 0xbeb13333, 0xbeaa0000, 0xbea2cccc,
        0xbe9b999a, 0xbe946666, 0xbe8d3334, 0xbe860000, 0xbe7d9999, 0xbe6f3333, 0xbe60cccd,
        0xbe526666, 0xbe440000, 0xbe35999a, 0xbe273333, 0xbe18cccd, 0xbe0a6666, 0xbdf80000,
        0xbddb3334, 0xbdc9eb85, 0xbdc428f7, 0xbdbe6667, 0xbdb8a3d7, 0xbdb2e148, 0xbdad1eb8,
        0xbda75c2a, 0xbda1999a, 0xbd9bd70a, 0xbd96147b, 0xbd9051eb, 0xbd8a8f5d, 0xbd84cccd,
        0xbd7e147b, 0xbd728f5d, 0xbd670a3d, 0xbd5b851f, 0xbd500000, 0xbd447ae1, 0xbd38f5c3,
        0xbd2d70a3, 0xbd21eb85, 0xbd166667, 0xbd0ae148, 0xbcfeb852, 0xbce7ae15, 0xbcd0a3d7,
        0xbcb9999a, 0xbca28f5d, 0xbc8b851f, 0xbc68f5c3, 0xbc3ae148, 0xbc1f3b64, 0xbc160418,
        0xbc0ccccd, 0xbc039581, 0xbbf4bc6a, 0xbbe24dd3, 0xbbcfdf3b, 0xbbbd70a4, 0xbbab020d,
        0xbb989374, 0xbb8624dd, 0xbb676c8a, 0xbb428f5c, 0xbb1db22d, 0xbaf1a9fc, 0xbaa7ef9d,
        0xba7765ff, 0xba59e83e, 0xba3c6a80, 0xba1eecc1, 0xba016f01, 0xb9c7e283, 0xb98ce705,
        0xb923d70b, 0xb8ba1f4b, 0xb88aefb3, 0xb8378034, 0xb7b24206, 0xb70205ff, 0xb65a1a94,
        0xb513a3b7, 0x00000000, 0x3513a3b7, 0x365a1a94, 0x370205ff, 0x37b24206, 0x38378034,
        0x388aefb3, 0x38ba1f4b, 0x3923d70b, 0x398ce705, 0x39c7e283, 0x3a016f01, 0x3a1eecc1,
        0x3a3c6a80, 0x3a59e83e, 0x3a7765ff, 0x3aa7ef9d, 0x3af1a9fc, 0x3b1db22d, 0x3b428f5c,
        0x3b676c8a, 0x3b8624dd, 0x3b989374, 0x3bab020d, 0x3bbd70a4, 0x3bcfdf3b, 0x3be24dd3,
        0x3bf4bc6a, 0x3c039581, 0x3c0ccccd, 0x3c160418, 0x3c1f3b64, 0x3c3ae148, 0x3c68f5c3,
        0x3c8b851f, 0x3ca28f5d, 0x3cb9999a, 0x3cd0a3d7, 0x3ce7ae15, 0x3cfeb852, 0x3d0ae148,
        0x3d166667, 0x3d21eb85, 0x3d2d70a3, 0x3d38f5c3, 0x3d447ae1, 0x3d500000, 0x3d5b851f,
        0x3d670a3d, 0x3d728f5d, 0x3d7e147b, 0x3d84cccd, 0x3d8a8f5d, 0x3d9051eb, 0x3d96147b,
        0x3d9bd70a, 0x3da1999a, 0x3da75c2a, 0x3dad1eb8, 0x3db2e148, 0x3db8a3d7, 0x3dbe6667,
        0x3dc428f7, 0x3dc9eb85, 0x3ddb3334, 0x3df80000, 0x3e0a6666, 0x3e18cccd, 0x3e273333,
        0x3e35999a, 0x3e440000, 0x3e526666, 0x3e60cccd, 0x3e6f3333, 0x3e7d9999, 0x3e860000,
        0x3e8d3334, 0x3e946666, 0x3e9b999a, 0x3ea2cccc, 0x3eaa0000, 0x3eb13333, 0x3eb86666,
        0x3ebf999a, 0x3ec6cccc, 0x3ece0000, 0x3ed53333, 0x3edc6666, 0x3ee3999a, 0x3eeacccc,
        0x3ef20000, 0x3ef93332, 0x3f003333, 0x3f03cccc, 0x3f076666, 0x3f0b0000, 0x3f0e999a,
        0x3f123334, 0x3f15cccd, 0x3f196666, 0x3f1d0000, 0x3f20999a, 0x3f243334, 0x3f27cccd,
        0x3f2b6666, 0x3f2f0000, 0x3f32999a, 0x3f363334, 0x3f39cccd, 0x3f3d6666, 0x3f410000,
        0x3f44999a, 0x3f483333, 0x3f4bcccd, 0x3f4f6666, 0x3f530000, 0x3f56999a, 0x3f5a3333,
        0x3f5dcccd, 0x3f616666, 0x3f650000, 0x3f68999a, 0x3f6c3333, 0x3f6fcccd, 0x3f736666,
        0x3f770000, 0x3f7a999a, 0x3f7e3333, 0x3f800000,
    ];
    const UNSIGNED: [u32; 256] = [
        0x00000000, 0x34ae7baa, 0x35500998, 0x360e9b38, 0x3692ccf6, 0x36de4c51, 0x3714e5d5,
        0x3783126e, 0x37e171a0, 0x381fe868, 0x384f1800, 0x387e479a, 0x3896bb99, 0x38ae5364,
        0x38c5eb31, 0x3906594b, 0x394154ca, 0x397c5049, 0x399ba5e4, 0x39b923a2, 0x39d6a163,
        0x39f41f22, 0x3a08ce71, 0x3a178d50, 0x3a264c30, 0x3a350b10, 0x3a43c9ef, 0x3a5288d0,
        0x3a6147af, 0x3a70068e, 0x3a7ec56e, 0x3a958106, 0x3aba5e35, 0x3adf3b64, 0x3b020c4a,
        0x3b147ae1, 0x3b26e978, 0x3b395810, 0x3b4bc6a7, 0x3b5e3540, 0x3b70a3d7, 0x3b818937,
        0x3b8ac082, 0x3b93f7cf, 0x3b9d2f1a, 0x3ba66666, 0x3baf9db2, 0x3bb8d4fd, 0x3bc20c4a,
        0x3bcb4395, 0x3bd47ae1, 0x3bddb22e, 0x3be6e978, 0x3bf020c5, 0x3bf9580f, 0x3c0147ae,
        0x3c05e354, 0x3c0a7ef9, 0x3c0f1aa0, 0x3c13b645, 0x3c1851eb, 0x3c1ced92, 0x3c218937,
        0x3c2f5c2a, 0x3c466667, 0x3c5d70a3, 0x3c747ae2, 0x3c85c28f, 0x3c9147af, 0x3c9ccccd,
        0x3ca851eb, 0x3cb3d70b, 0x3cbf5c29, 0x3ccae147, 0x3cd66667, 0x3ce1eb87, 0x3ced70a3,
        0x3cf8f5c3, 0x3d023d70, 0x3d080000, 0x3d0dc28f, 0x3d13851f, 0x3d1947af, 0x3d1f0a3d,
        0x3d24cccd, 0x3d2a8f5c, 0x3d3051eb, 0x3d36147b, 0x3d3bd70a, 0x3d41999a, 0x3d475c28,
        0x3d4d1eb8, 0x3d52e147, 0x3d58a3d7, 0x3d5e6667, 0x3d6428f7, 0x3d69eb87, 0x3d6fae15,
        0x3d7570a3, 0x3d7b3333, 0x3d807ae2, 0x3d835c2a, 0x3d863d71, 0x3d891eb8, 0x3d8c0000,
        0x3d8ee148, 0x3d91c290, 0x3d94a3d7, 0x3d97851f, 0x3d9a6667, 0x3d9d47af, 0x3da028f6,
        0x3da30a3e, 0x3da5eb85, 0x3da8cccd, 0x3dabae15, 0x3dae8f5c, 0x3db170a4, 0x3db451eb,
        0x3db73333, 0x3dba147b, 0x3dbcf5c3, 0x3dbfd70b, 0x3dc2b852, 0x3dc5999a, 0x3dc87ae2,
        0x3dcb5c29, 0x3dd40000, 0x3de26666, 0x3df0cccd, 0x3dff3333, 0x3e06cccc, 0x3e0e0000,
        0x3e153334, 0x3e1c6666, 0x3e23999a, 0x3e2acccc, 0x3e320000, 0x3e393334, 0x3e406666,
        0x3e47999a, 0x3e4ecccc, 0x3e560000, 0x3e5d3334, 0x3e646666, 0x3e6b999a, 0x3e72cccc,
        0x3e7a0000, 0x3e809999, 0x3e843333, 0x3e87cccd, 0x3e8b6666, 0x3e8f0000, 0x3e92999a,
        0x3e963333, 0x3e99cccc, 0x3e9d6666, 0x3ea10000, 0x3ea4999a, 0x3ea83333, 0x3eabcccc,
        0x3eaf6666, 0x3eb30000, 0x3eb6999a, 0x3eba3333, 0x3ebdcccc, 0x3ec16666, 0x3ec50000,
        0x3ec8999a, 0x3ecc3333, 0x3ecfcccc, 0x3ed36666, 0x3ed70000, 0x3eda999a, 0x3ede3333,
        0x3ee1cccc, 0x3ee56666, 0x3ee90000, 0x3eec999a, 0x3ef03333, 0x3ef3cccc, 0x3ef76666,
        0x3efb0000, 0x3efe9999, 0x3f01199a, 0x3f02e666, 0x3f04b333, 0x3f068000, 0x3f084ccc,
        0x3f0a199a, 0x3f0be666, 0x3f0db334, 0x3f0f8000, 0x3f114ccd, 0x3f13199a, 0x3f14e666,
        0x3f16b334, 0x3f188000, 0x3f1a4ccd, 0x3f1c199a, 0x3f1de666, 0x3f1fb334, 0x3f218000,
        0x3f234ccd, 0x3f25199a, 0x3f26e666, 0x3f28b334, 0x3f2a8000, 0x3f2c4ccd, 0x3f2e199a,
        0x3f2fe666, 0x3f31b334, 0x3f338000, 0x3f354ccd, 0x3f37199a, 0x3f38e666, 0x3f3ab334,
        0x3f3c8000, 0x3f3e4ccd, 0x3f40199a, 0x3f41e666, 0x3f43b334, 0x3f458000, 0x3f474ccc,
        0x3f49199a, 0x3f4ae666, 0x3f4cb334, 0x3f4e8000, 0x3f504ccc, 0x3f52199a, 0x3f53e666,
        0x3f55b334, 0x3f578000, 0x3f594ccc, 0x3f5b199a, 0x3f5ce666, 0x3f5eb334, 0x3f608000,
        0x3f624ccc, 0x3f64199a, 0x3f65e666, 0x3f67b334, 0x3f698000, 0x3f6b4ccc, 0x3f6d199a,
        0x3f6ee666, 0x3f70b334, 0x3f728000, 0x3f744ccc, 0x3f76199a, 0x3f77e666, 0x3f79b334,
        0x3f7b8000, 0x3f7d4ccc, 0x3f7f199a, 0x3f800000,
    ];

    #[test]
    fn the_maps_are_those_of_bitsandbytes_bit_for_bit() {
        for (signed, expected) in [(true, &SIGNED), (false, &UNSIGNED)] {
            let map = dynamic_map(signed);
            for (i, (got, want)) in map.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got.to_bits(), *want, "signed {signed}, value {i}: {got}");
            }
        }
    }

    #[test]
    fn a_value_takes_the_nearest_code_the_lower_at_a_tie() {
        let map = dynamic_map(true);
        for (i, &x) in map.iter().enumerate() {
            assert_eq!(nearest_code(&map, x) as usize, i);
        }
        let middle = (map[200] + map[201]) * 0.5;
        assert_eq!(nearest_code(&map, middle), 200);
        assert_eq!(
            nearest_code(&map, f32::from_bits(middle.to_bits() + 1)),
            201
        );
        assert_eq!(nearest_code(&map, -1.0), 0);
        assert_eq!(nearest_code(&map, 1.0), 255);
    }

    #[test]
    fn a_block_of_zero_moments_follows_adamw_and_keeps_its_largest_moments_exactly() {
        let step = AdamWStep {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            correction1: 0.1,
            correction2: 0.05,
            grad_scale: 1.0,
        };
        let (signed, unsigned) = (dynamic_map(true), dynamic_map(false));
        let grad: Vec<f32> = (0..100).map(|i| (i as f32 - 40.0) / 50.0).collect();
        let mut p = vec![0.5f32; 100];
        let (mut m, mut v) = (vec![0u8; 100], vec![0u8; 100]);
        let (mut ms, mut vs) = (0.0, 0.0);
        step.apply_block8(
            &mut p, &mut m, &mut v, &mut ms, &mut vs, &grad, &signed, &unsigned,
        );
        for i in 0..100 {
            let (pe, me, ve) = step.apply(0.5, 0.0, 0.0, grad[i]);
            assert_eq!(p[i], pe);
            let (mq, vq) = (signed[m[i] as usize] * ms, unsigned[v[i] as usize] * vs);
            assert!((mq - me).abs() <= 0.05 * ms, "m {mq} against {me}");
            assert!((vq - ve).abs() <= 0.05 * vs, "v {vq} against {ve}");
        }
        // The largest of each moment (the last element's) is its block's
        // scale, and its code 1.0.
        let (_, m_top, v_top) = step.apply(0.5, 0.0, 0.0, grad[99]);
        assert_eq!((ms, vs), (m_top, v_top));
        assert_eq!(
            (signed[m[99] as usize], unsigned[v[99] as usize]),
            (1.0, 1.0)
        );
    }
}
