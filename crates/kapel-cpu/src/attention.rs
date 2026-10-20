//! Scaled dot-product attention, one task per batch element and key head.
//!
//! Each task handles the query heads that share one key and value head, one
//! after the other: their gradients of `k` and `v` add up in a fixed order, and
//! no two tasks write the same memory. The scores of a head are computed with
//! the matrix product, the softmax row by row.

use kapel_core::{Result, Tensor};
use kapel_ops::{Attention, Operation, Run};

use crate::backend::Cpu;
use crate::gemm::{self, Matrix};
use crate::io::{self, Strided};
use crate::math;
use crate::pool::SharedMut;

struct Dims {
    b: usize,
    hq: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    dv: usize,
    group: usize,
}

impl Dims {
    fn of(q: &Tensor, k: &Tensor, v: &Tensor) -> Self {
        let (qs, ks) = (q.shape(), k.shape());
        Self {
            b: qs[0],
            hq: qs[1],
            hkv: ks[1],
            tq: qs[2],
            tk: ks[2],
            d: qs[3],
            dv: v.shape()[3],
            group: qs[1] / ks[1],
        }
    }
}

/// The attention weights `P [tq, tk]` of query head `h` of batch element `b`.
/// Called from a task of `pool`, the product runs on the calling thread.
#[allow(clippy::too_many_arguments)]
fn weights(
    op: &Attention,
    pool: &crate::pool::Pool,
    isa: crate::Isa,
    dims: &Dims,
    q: &[f32],
    k: &[f32],
    mask: Option<&Strided<'_>>,
    b: usize,
    h: usize,
    p: &mut [f32],
    mut slope: Option<&mut [f32]>,
) {
    let Dims { tq, tk, d, .. } = *dims;
    let kv = h / dims.group;
    let qh = &q[(b * dims.hq + h) * tq * d..][..tq * d];
    let kh = &k[(b * dims.hkv + kv) * tk * d..][..tk * d];
    gemm::gemm(
        pool,
        isa,
        Matrix::new(qh, tq, d),
        Matrix::new(kh, tk, d).t(),
        p,
        false,
    );
    let scale = op.scale_for(d) as f32;
    for i in 0..tq {
        let row = &mut p[i * tk..(i + 1) * tk];
        let position = tk - tq + i;
        for (j, s) in row.iter_mut().enumerate() {
            let visible =
                (!op.causal || j <= position) && op.window.is_none_or(|w| position < j + w);
            *s = if visible {
                let m = mask.map_or(0.0, |m| {
                    let st = m.layout.strides();
                    m.data[m.layout.offset() + b * st[0] + h * st[1] + i * st[2] + j * st[3]]
                });
                let raw = scale * *s;
                let capped = match op.softcap {
                    Some(cap) => {
                        let t = (raw / cap as f32).tanh();
                        if let Some(slope) = slope.as_deref_mut() {
                            slope[i * tk + j] = 1.0 - t * t;
                        }
                        cap as f32 * t
                    }
                    None => raw,
                };
                capped + m
            } else {
                f32::NEG_INFINITY
            };
        }
        let max = math::max(isa, row);
        if max == f32::NEG_INFINITY {
            row.fill(0.0);
            continue;
        }
        let total = math::exp_shifted_sum(isa, row, max);
        math::scale(isa, row, 1.0 / total);
    }
}

/// The mask, broadcast to `[B, Hq, Tq, Tk]` without copying.
fn mask<'a>(inputs: &[&'a Tensor], dims: &Dims) -> Result<Option<Strided<'a>>> {
    let Some(mask) = inputs.get(3) else {
        return Ok(None);
    };
    let mut strided = io::strided(Attention::NAME, mask)?;
    strided.layout = strided
        .layout
        .broadcast_to(&[dims.b, dims.hq, dims.tq, dims.tk])?;
    Ok(Some(strided))
}

impl Run<Attention> for Cpu {
    fn forward(&self, op: &Attention, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let dims = Dims::of(inputs[0], inputs[1], inputs[2]);
        let q = io::contiguous(Attention::NAME, inputs[0])?;
        let k = io::contiguous(Attention::NAME, inputs[1])?;
        let v = io::contiguous(Attention::NAME, inputs[2])?;
        let mask = mask(inputs, &dims)?;
        let Dims {
            b,
            hq,
            hkv,
            tq,
            tk,
            dv,
            group,
            ..
        } = dims;
        let mut out = vec![0.0; b * hq * tq * dv];
        let shared = SharedMut::new(&mut out);
        let isa = self.isa;
        self.pool.run(b * hkv, |task| {
            let (bi, kv) = (task / hkv, task % hkv);
            let mut p = vec![0.0; tq * tk];
            let vh = &v[(bi * hkv + kv) * tk * dv..][..tk * dv];
            for h in kv * group..(kv + 1) * group {
                weights(
                    op,
                    &self.pool,
                    isa,
                    &dims,
                    &q,
                    &k,
                    mask.as_ref(),
                    bi,
                    h,
                    &mut p,
                    None,
                );
                // SAFETY: each query head's output belongs to one task.
                let y = unsafe { shared.slice((bi * hq + h) * tq * dv, tq * dv) };
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::new(&p, tq, tk),
                    Matrix::new(vh, tk, dv),
                    y,
                    false,
                );
            }
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    /// With `P` the weights and `g` the output gradient of a head:
    /// `∂v += Pᵀ·g`, `∂P = g·vᵀ`, `∂S = P ⊙ (∂P − rowsum(∂P ⊙ P))`,
    /// `∂q = scale·∂S·k`, `∂k += scale·∂Sᵀ·q`.
    fn backward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let dims = Dims::of(inputs[0], inputs[1], inputs[2]);
        let q = io::contiguous(Attention::NAME, inputs[0])?;
        let k = io::contiguous(Attention::NAME, inputs[1])?;
        let v = io::contiguous(Attention::NAME, inputs[2])?;
        let g = io::contiguous(Attention::NAME, grad)?;
        let mask = mask(inputs, &dims)?;
        let Dims {
            b,
            hq,
            hkv,
            tq,
            tk,
            d,
            dv,
            group,
        } = dims;
        let scale = op.scale_for(d) as f32;
        let (mut gq, mut gk, mut gv) = (vec![0.0; q.len()], vec![0.0; k.len()], vec![0.0; v.len()]);
        let (sq, sk, sv) = (
            SharedMut::new(&mut gq),
            SharedMut::new(&mut gk),
            SharedMut::new(&mut gv),
        );
        let isa = self.isa;
        self.pool.run(b * hkv, |task| {
            let (bi, kv) = (task / hkv, task % hkv);
            let (mut p, mut dp) = (vec![0.0; tq * tk], vec![0.0; tq * tk]);
            // The slope of the softcap at each score: 1 without one.
            let mut slope = vec![1.0; tq * tk];
            let kh = &k[(bi * hkv + kv) * tk * d..][..tk * d];
            let vh = &v[(bi * hkv + kv) * tk * dv..][..tk * dv];
            // SAFETY: the key and value head of this task, and the gradients of
            // its query heads, are written by this task alone.
            let (gkh, gvh) = unsafe {
                (
                    sk.slice((bi * hkv + kv) * tk * d, tk * d),
                    sv.slice((bi * hkv + kv) * tk * dv, tk * dv),
                )
            };
            for (n, h) in (kv * group..(kv + 1) * group).enumerate() {
                let accumulate = n > 0;
                weights(
                    op,
                    &self.pool,
                    isa,
                    &dims,
                    &q,
                    &k,
                    mask.as_ref(),
                    bi,
                    h,
                    &mut p,
                    Some(&mut slope),
                );
                let row = (bi * hq + h) * tq;
                let (gh, qh) = (&g[row * dv..][..tq * dv], &q[row * d..][..tq * d]);
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::new(&p, tq, tk).t(),
                    Matrix::new(gh, tq, dv),
                    gvh,
                    accumulate,
                );
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::new(gh, tq, dv),
                    Matrix::new(vh, tk, dv).t(),
                    &mut dp,
                    false,
                );
                for i in 0..tq {
                    let (pr, dr) = (&p[i * tk..(i + 1) * tk], &mut dp[i * tk..(i + 1) * tk]);
                    let dot = math::dot(isa, pr, dr);
                    let sr = &slope[i * tk..(i + 1) * tk];
                    for ((ds, &pv), &sl) in dr.iter_mut().zip(pr).zip(sr) {
                        *ds = scale * pv * (*ds - dot) * sl;
                    }
                }
                // SAFETY: as above.
                let gqh = unsafe { sq.slice(row * d, tq * d) };
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::new(&dp, tq, tk),
                    Matrix::new(kh, tk, d),
                    gqh,
                    false,
                );
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::new(&dp, tq, tk).t(),
                    Matrix::new(qh, tq, d),
                    gkh,
                    accumulate,
                );
            }
        });
        let dtype = signature.dtype;
        let mut grads = vec![
            Some(io::write(&self.buffers, &gq, inputs[0].shape(), dtype)?),
            Some(io::write(&self.buffers, &gk, inputs[1].shape(), dtype)?),
            Some(io::write(&self.buffers, &gv, inputs[2].shape(), dtype)?),
        ];
        if inputs.len() == 4 {
            grads.push(None);
        }
        Ok(grads)
    }
}
