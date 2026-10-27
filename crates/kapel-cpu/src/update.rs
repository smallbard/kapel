//! Updates in place.

use kapel_core::{Error, HostStorage, Result, Tensor};
use kapel_ops::{ADAMW, ADAMW8, AdamWStep, BLOCK8, Moments8, Update, check_adamw, check_adamw8bit};

use crate::backend::{CHUNK, Cpu};
use crate::io;
use crate::pool::SharedMut;

/// The `f32` values of a host tensor that can be written.
fn host_f32(t: &mut Tensor) -> Result<&mut [f32]> {
    let storage = t
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| Error::InvalidArgument {
            op: ADAMW,
            message: "the tensor is not in host memory that can be written".into(),
        })?;
    Ok(io::as_f32_mut(storage.bytes_mut()))
}

/// The bytes of a host tensor that can be written.
fn host_u8(t: &mut Tensor) -> Result<&mut [u8]> {
    t.storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .map(HostStorage::bytes_mut)
        .ok_or_else(|| Error::InvalidArgument {
            op: ADAMW8,
            message: "the tensor is not in host memory that can be written".into(),
        })
}

/// A gradient as the updates read it: a contiguous bf16 one in place, each
/// value widened where it is read; any other converted to `f32` first.
enum Gradient<'a> {
    F32(std::borrow::Cow<'a, [f32]>),
    Bf16(&'a [u8]),
}

impl<'a> Gradient<'a> {
    fn new(op: &'static str, grad: &'a Tensor) -> Result<Self> {
        if grad.dtype() == kapel_core::DType::BF16 && grad.is_contiguous() {
            let bytes = io::host_bytes(op, grad)?;
            let start = grad.layout().offset() * 2;
            return Ok(Self::Bf16(&bytes[start..start + grad.numel() * 2]));
        }
        Ok(Self::F32(io::contiguous(op, grad)?))
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Bf16(b) => b.len() / 2,
        }
    }

    /// The values `start..start + len`: borrowed in `f32`, widened into
    /// `local` from bf16.
    fn slice<'s>(&'s self, start: usize, len: usize, local: &'s mut [f32]) -> &'s [f32] {
        match self {
            Self::F32(v) => &v[start..start + len],
            Self::Bf16(b) => {
                for (x, h) in local[..len]
                    .iter_mut()
                    .zip(b[2 * start..2 * (start + len)].as_chunks::<2>().0)
                {
                    *x = f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16);
                }
                &local[..len]
            }
        }
    }

    #[inline]
    fn get(&self, i: usize) -> f32 {
        match self {
            Self::F32(v) => v[i],
            Self::Bf16(b) => {
                f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
            }
        }
    }
}

/// The blocks of [`BLOCK8`] elements each task of `adamw8bit` updates.
const BLOCKS8: usize = CHUNK / BLOCK8;

impl Update for Cpu {
    /// [`AdamWStep::apply`], element by element, pieces of the tensors
    /// shared between threads.
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &AdamWStep,
    ) -> Result<()> {
        check_adamw(param, m, v, grad)?;
        let g = Gradient::new(ADAMW, grad)?;
        let n = g.len();
        let (pm, mm, vm) = (host_f32(param)?, host_f32(m)?, host_f32(v)?);
        let (ps, ms, vs) = (
            SharedMut::new(&mut pm[..n]),
            SharedMut::new(&mut mm[..n]),
            SharedMut::new(&mut vm[..n]),
        );
        self.pool.run(n.div_ceil(CHUNK), |task| {
            let start = task * CHUNK;
            let len = CHUNK.min(n - start);
            // SAFETY: the pieces are disjoint, each updated by one task; the
            // tensors outlive the call to `run`.
            let (p, m, v) = unsafe {
                (
                    ps.slice(start, len),
                    ms.slice(start, len),
                    vs.slice(start, len),
                )
            };
            let mut local = [0.0f32; CHUNK];
            let grad = g.slice(start, len, &mut local);
            for i in 0..len {
                (p[i], m[i], v[i]) = step.apply(p[i], m[i], v[i], grad[i]);
            }
        });
        Ok(())
    }

    /// [`AdamWStep::apply_block8`], runs of blocks shared between threads.
    fn adamw8bit(
        &self,
        param: &mut Tensor,
        moments: &mut Moments8,
        grad: &Tensor,
        maps: &Tensor,
        step: &AdamWStep,
    ) -> Result<()> {
        check_adamw8bit(param, moments, grad, maps)?;
        let g = Gradient::new(ADAMW8, grad)?;
        let table = io::contiguous(ADAMW8, maps)?;
        let (signed, unsigned) = table.split_at(256);
        let n = g.len();
        let blocks = n.div_ceil(BLOCK8);
        let Moments8 {
            m,
            v,
            m_scale,
            v_scale,
        } = moments;
        let (ps, ms, vs) = (
            SharedMut::new(&mut host_f32(param)?[..n]),
            SharedMut::new(&mut host_u8(m)?[..n]),
            SharedMut::new(&mut host_u8(v)?[..n]),
        );
        let (mss, vss) = (
            SharedMut::new(&mut host_f32(m_scale)?[..blocks]),
            SharedMut::new(&mut host_f32(v_scale)?[..blocks]),
        );
        self.pool.run(blocks.div_ceil(BLOCKS8), |task| {
            let first = task * BLOCKS8;
            let count = BLOCKS8.min(blocks - first);
            let start = first * BLOCK8;
            let len = (count * BLOCK8).min(n - start);
            // SAFETY: the runs of blocks are disjoint, each updated by one
            // task; the tensors outlive the call to `run`.
            let (p, m, v, mscale, vscale) = unsafe {
                (
                    ps.slice(start, len),
                    ms.slice(start, len),
                    vs.slice(start, len),
                    mss.slice(first, count),
                    vss.slice(first, count),
                )
            };
            let mut block = [0.0f32; BLOCK8];
            for b in 0..count {
                let r = b * BLOCK8..(b * BLOCK8 + BLOCK8).min(len);
                for (k, x) in block[..r.len()].iter_mut().enumerate() {
                    *x = g.get(start + r.start + k);
                }
                step.apply_block8(
                    &mut p[r.clone()],
                    &mut m[r.clone()],
                    &mut v[r.clone()],
                    &mut mscale[b],
                    &mut vscale[b],
                    &block[..r.len()],
                    signed,
                    unsigned,
                );
            }
        });
        Ok(())
    }
}
