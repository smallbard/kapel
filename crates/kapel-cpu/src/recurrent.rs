//! Linear attention's states, in `f32`: the rows of a pass one after the
//! other, the channels (convolution) or the heads (delta rule) shared between
//! threads, each by one task whatever the number of threads.

use kapel_core::{DType, Error, HostStorage, Result, Tensor};
use kapel_ops::{CAUSAL_CONV, DELTA_RULE, Recurrent, check_conv, check_delta};

use crate::backend::Cpu;
use crate::io;
use crate::pool::SharedMut;

/// The channels of a task of the convolution.
const CHANNELS: usize = 256;

/// The bytes of a host tensor that can be written.
fn host_bytes_mut<'a>(op: &'static str, t: &'a mut Tensor) -> Result<&'a mut [u8]> {
    t.storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .map(HostStorage::bytes_mut)
        .ok_or_else(|| Error::InvalidArgument {
            op,
            message: "the state is not in host memory that can be written".into(),
        })
}

/// `x` once stored in `dtype` (`f32`, bf16 or f16), read back.
#[inline]
fn stored(dtype: DType, x: f32) -> f32 {
    match dtype {
        DType::BF16 => f32::from_bits(u32::from(io::to_bf16(x)) << 16),
        DType::F16 => kapel_core::F16::from_f64(f64::from(x)).to_f64() as f32,
        _ => x,
    }
}

impl Recurrent for Cpu {
    fn causal_conv(
        &self,
        state: &mut Tensor,
        x: &Tensor,
        weight: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor> {
        let (shape, dtype) = check_conv(state, x, weight, slots)?;
        io::supported(CAUSAL_CONV, state.dtype())?;
        let n = state.shape()[0];
        let slots = io::indices(CAUSAL_CONV, slots, n)?;
        let (c, taps) = (weight.shape()[0], weight.shape()[1]);
        let kept = taps - 1;
        let xs = io::contiguous(CAUSAL_CONV, x)?;
        let w = io::contiguous(CAUSAL_CONV, weight)?;
        let kind = state.dtype();
        let mut table = io::contiguous(CAUSAL_CONV, state)?.into_owned();
        let rows = slots.len();
        let mut out = vec![0.0f32; rows * c];
        {
            let (ts, os) = (SharedMut::new(&mut table), SharedMut::new(&mut out));
            let (xs, w, slots) = (&xs, &w, &slots);
            self.pool.run(c.div_ceil(CHANNELS), |task| {
                let first = task * CHANNELS;
                let width = CHANNELS.min(c - first);
                for (r, &s) in slots.iter().enumerate() {
                    for ch in first..first + width {
                        let mut sum = 0.0f32;
                        for j in 0..kept {
                            // SAFETY: each task reads and writes its own
                            // channels of the table, which outlives `run`.
                            let held = unsafe { *ts.ptr((s * kept + j) * c + ch) };
                            sum += w[ch * taps + j] * held;
                        }
                        let input = xs[r * c + ch];
                        sum += w[ch * taps + kept] * input;
                        // SAFETY: element (r, ch) is this task's alone.
                        unsafe { *os.ptr(r * c + ch) = sum };
                        for j in 0..kept {
                            let next = if j + 1 < kept {
                                // SAFETY: as above.
                                unsafe { *ts.ptr((s * kept + j + 1) * c + ch) }
                            } else {
                                stored(kind, input)
                            };
                            // SAFETY: as above.
                            unsafe { *ts.ptr((s * kept + j) * c + ch) = next };
                        }
                    }
                }
            });
        }
        let bytes = host_bytes_mut(CAUSAL_CONV, state)?;
        match kind {
            DType::F32 => io::as_f32_mut(bytes)[..table.len()].copy_from_slice(&table),
            DType::BF16 => {
                for (o, &v) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(&table) {
                    *o = io::to_bf16(v).to_le_bytes();
                }
            }
            _ => {
                for (o, &v) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(&table) {
                    *o = kapel_core::F16::from_f64(f64::from(v)).0.to_le_bytes();
                }
            }
        }
        io::write(&self.buffers, &out, &shape, dtype)
    }

    fn delta_rule(
        &self,
        state: &mut Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        slots: &Tensor,
        layout: kapel_ops::HeadLayout,
    ) -> Result<Tensor> {
        let (shape, dtype) = check_delta(state, q, k, v, g, beta, slots)?;
        let n = state.shape()[0];
        let slots = io::indices(DELTA_RULE, slots, n)?;
        let (hk, dk) = (q.shape()[1], q.shape()[2]);
        let (h, dv) = (v.shape()[1], v.shape()[2]);
        let (q, k, v, g, beta) = (
            io::contiguous(DELTA_RULE, q)?,
            io::contiguous(DELTA_RULE, k)?,
            io::contiguous(DELTA_RULE, v)?,
            io::contiguous(DELTA_RULE, g)?,
            io::contiguous(DELTA_RULE, beta)?,
        );
        let table = io::as_f32_mut(host_bytes_mut(DELTA_RULE, state)?);
        let rows = slots.len();
        let mut out = vec![0.0f32; rows * h * dv];
        {
            let (ts, os) = (SharedMut::new(table), SharedMut::new(&mut out));
            let (q, k, v, g, beta, slots) = (&q, &k, &v, &g, &beta, &slots);
            self.pool.run(h, |head| {
                let kh = layout.key(head, h, hk);
                let mut u = vec![0.0f32; dv];
                for (r, &s) in slots.iter().enumerate() {
                    let qr = &q[(r * hk + kh) * dk..][..dk];
                    let kr = &k[(r * hk + kh) * dk..][..dk];
                    let vr = &v[(r * h + head) * dv..][..dv];
                    let decay = g[r * h + head].exp();
                    let b = beta[r * h + head];
                    // SAFETY: the matrix of (slot, head) is this task's alone:
                    // the tasks are the heads. The table outlives `run`.
                    let m = unsafe { ts.slice((s * h + head) * dk * dv, dk * dv) };
                    for x in m.iter_mut() {
                        *x *= decay;
                    }
                    u.fill(0.0);
                    for (i, &ki) in kr.iter().enumerate() {
                        for (uj, &sij) in u.iter_mut().zip(&m[i * dv..][..dv]) {
                            *uj += sij * ki;
                        }
                    }
                    for (uj, &vj) in u.iter_mut().zip(vr) {
                        *uj = (vj - *uj) * b;
                    }
                    for (i, &ki) in kr.iter().enumerate() {
                        for (sij, &dj) in m[i * dv..][..dv].iter_mut().zip(&u) {
                            *sij += ki * dj;
                        }
                    }
                    // SAFETY: row r of head `head` is this task's alone.
                    let o = unsafe { os.slice((r * h + head) * dv, dv) };
                    o.fill(0.0);
                    for (i, &qi) in qr.iter().enumerate() {
                        for (oj, &sij) in o.iter_mut().zip(&m[i * dv..][..dv]) {
                            *oj += sij * qi;
                        }
                    }
                }
            });
        }
        io::write(&self.buffers, &out, &shape, dtype)
    }
}
