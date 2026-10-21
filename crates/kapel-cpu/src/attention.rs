//! Scaled dot-product attention.
//!
//! The work is split over the batch, the heads and blocks of queries, in a way
//! fixed by the shapes (see `forward` and `backward`). The scores of a block
//! are computed with the matrix product, the softmax row by row.

use std::ops::Range;

use kapel_core::{Result, Tensor};
use kapel_ops::{Attention, Operation, Run};

use crate::backend::{CHUNK, Cpu};
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

/// Rows of queries per task in the forward pass: enough to make a matrix
/// product worth it, few enough to spread one head over several threads.
const FORWARD_ROWS: usize = 128;
/// The backward pass splits the queries of a head in at least this many
/// rows...
const BACKWARD_ROWS: usize = 64;
/// ... into enough blocks to make this many tasks over the batch and the key
/// heads (a constant: the split depends only on the shapes)...
const BACKWARD_TASKS: usize = 32;
/// ... as long as the partial gradients of `k` and `v` of the extra blocks fit
/// in this many elements (256 MiB).
const BACKWARD_BUDGET: usize = 1 << 26;

/// The attention weights `P [rows, tk]` of the queries `rows` of query head
/// `h` of batch element `b`. Called from a task of `pool`, the product runs on
/// the calling thread.
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
    rows: Range<usize>,
    p: &mut [f32],
    mut slope: Option<&mut [f32]>,
) {
    let Dims { tq, tk, d, .. } = *dims;
    let kv = h / dims.group;
    let first = (b * dims.hq + h) * tq + rows.start;
    let qh = &q[first * d..][..rows.len() * d];
    let kh = &k[(b * dims.hkv + kv) * tk * d..][..tk * d];
    gemm::gemm(
        pool,
        isa,
        Matrix::contiguous(qh, rows.len(), d),
        Matrix::contiguous(kh, tk, d).t(),
        p,
        false,
    );
    let scale = op.scale_for(d) as f32;
    for (r, i) in rows.enumerate() {
        let row = &mut p[r * tk..(r + 1) * tk];
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
                            slope[r * tk + j] = 1.0 - t * t;
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
            // No visible key, or all masked with −∞: zeros, and no gradient.
            // `max` skips NaN, which must not be hidden: it propagates.
            let fill = if row.iter().any(|s| s.is_nan()) {
                f32::NAN
            } else {
                0.0
            };
            row.fill(fill);
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

/// Splits `len` rows into `blocks` blocks of equal size, the last one
/// shorter: the block size, and the number of non-empty blocks.
fn split(len: usize, blocks: usize) -> (usize, usize) {
    let size = len.div_ceil(blocks.max(1)).max(1);
    (size, len.div_ceil(size))
}

impl Run<Attention> for Cpu {
    /// One task per batch element, key head and block of `FORWARD_ROWS`
    /// queries: every output row is computed the same way, whatever the
    /// number of threads.
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
        let (size, blocks) = split(tq, tq.div_ceil(FORWARD_ROWS));
        let mut out = vec![0.0; b * hq * tq * dv];
        let shared = SharedMut::new(&mut out);
        let isa = self.isa;
        self.pool.run(b * hkv * blocks, |task| {
            let (head, block) = (task / blocks, task % blocks);
            let (bi, kv) = (head / hkv, head % hkv);
            let rows = block * size..tq.min((block + 1) * size);
            let mut p = vec![0.0; rows.len() * tk];
            let vh = &v[head * tk * dv..][..tk * dv];
            // The query heads that share this key head, one after the other:
            // the key and value head stay in the caches.
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
                    rows.clone(),
                    &mut p,
                    None,
                );
                let first = (bi * hq + h) * tq + rows.start;
                // SAFETY: these output rows belong to this task alone, and
                // `out` outlives the call to `run`.
                let y = unsafe { shared.slice(first * dv, rows.len() * dv) };
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::contiguous(&p, rows.len(), tk),
                    Matrix::contiguous(vh, tk, dv),
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
    ///
    /// One task per batch element, key head and block of queries; a task
    /// handles the query heads of its key head one after the other. The blocks
    /// depend only on the shapes. The first block writes the gradients of `k`
    /// and `v`, the others write partial gradients, added afterwards in block
    /// order: the same bits whatever the number of threads.
    fn backward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
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
        let heads = b * hkv;
        let (kv_len, vv_len) = (k.len(), v.len());
        let wanted = BACKWARD_TASKS
            .div_ceil(heads.max(1))
            .min(tq.div_ceil(BACKWARD_ROWS))
            .min(1 + BACKWARD_BUDGET / (kv_len + vv_len).max(1));
        let (size, blocks) = split(tq, wanted);
        let extra = blocks.saturating_sub(1);
        let (mut gq, mut gk, mut gv) = (vec![0.0; q.len()], vec![0.0; kv_len], vec![0.0; vv_len]);
        let (mut partial_k, mut partial_v) = (vec![0.0; extra * kv_len], vec![0.0; extra * vv_len]);
        let (sq, sk, sv) = (
            SharedMut::new(&mut gq),
            SharedMut::new(&mut gk),
            SharedMut::new(&mut gv),
        );
        let (pk, pv) = (
            SharedMut::new(&mut partial_k),
            SharedMut::new(&mut partial_v),
        );
        let isa = self.isa;
        self.pool.run(heads * blocks, |task| {
            let (head, block) = (task / blocks, task % blocks);
            let (bi, kv) = (head / hkv, head % hkv);
            let rows = block * size..tq.min((block + 1) * size);
            let r = rows.len();
            let (mut p, mut dp) = (vec![0.0; r * tk], vec![0.0; r * tk]);
            // The slope of the softcap at each score: 1 without one.
            let mut slope = vec![1.0; r * tk];
            let kh = &k[head * tk * d..][..tk * d];
            let vh = &v[head * tk * dv..][..tk * dv];
            // SAFETY: the gradients of this key and value head, in the first
            // block's buffers or in this block's partial ones, and the
            // gradients of these query rows, are written by this task alone;
            // all of them outlive the call to `run`.
            let (gkh, gvh) = unsafe {
                if block == 0 {
                    (
                        sk.slice(head * tk * d, tk * d),
                        sv.slice(head * tk * dv, tk * dv),
                    )
                } else {
                    (
                        pk.slice((block - 1) * kv_len + head * tk * d, tk * d),
                        pv.slice((block - 1) * vv_len + head * tk * dv, tk * dv),
                    )
                }
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
                    rows.clone(),
                    &mut p,
                    Some(&mut slope),
                );
                let row = (bi * hq + h) * tq + rows.start;
                let (gh, qh) = (&g[row * dv..][..r * dv], &q[row * d..][..r * d]);
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::contiguous(&p, r, tk).t(),
                    Matrix::contiguous(gh, r, dv),
                    gvh,
                    accumulate,
                );
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::contiguous(gh, r, dv),
                    Matrix::contiguous(vh, tk, dv).t(),
                    &mut dp,
                    false,
                );
                for i in 0..r {
                    let (pr, dr) = (&p[i * tk..(i + 1) * tk], &mut dp[i * tk..(i + 1) * tk]);
                    let dot = math::dot(isa, pr, dr);
                    let sr = &slope[i * tk..(i + 1) * tk];
                    for ((ds, &pv), &sl) in dr.iter_mut().zip(pr).zip(sr) {
                        *ds = scale * pv * (*ds - dot) * sl;
                    }
                }
                // SAFETY: as above.
                let gqh = unsafe { sq.slice(row * d, r * d) };
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::contiguous(&dp, r, tk),
                    Matrix::contiguous(kh, tk, d),
                    gqh,
                    false,
                );
                gemm::gemm(
                    &self.pool,
                    isa,
                    Matrix::contiguous(&dp, r, tk).t(),
                    Matrix::contiguous(qh, r, d),
                    gkh,
                    accumulate,
                );
            }
        });
        // The partial gradients of the other blocks, in block order.
        for (total, partial) in [(&mut gk, &partial_k), (&mut gv, &partial_v)] {
            if partial.is_empty() {
                continue;
            }
            let len = total.len();
            self.fill(total, CHUNK, |start, part| {
                for block in partial.chunks_exact(len) {
                    for (y, &x) in part.iter_mut().zip(&block[start..]) {
                        *y += x;
                    }
                }
            });
        }
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
