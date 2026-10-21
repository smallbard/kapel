//! Attention as cutile-rs tile kernels, in the manner of FlashAttention 2.
//!
//! The forward kernel walks the keys tile by tile with an online softmax, and
//! never forms the `Tq × Tk` weights; it also returns, for each query, the
//! logarithm of its softmax's denominator (`lse`). The backward pass recomputes
//! the weights from it: a small kernel first computes `δ = rowsum(g ⊙ y)` for
//! each query, then one kernel computes the gradient of the queries, and one
//! those of the keys and values. Each output is written by one tile block,
//! which adds its terms in a fixed order: no atomics, the same bits on every
//! run.
//!
//! Tensors are read in place, `[B, H, T, D]` through their strides. A kernel
//! that needs a transposed tile loads it, then transposes it: reading it through
//! a view with swapped strides is about three times slower. Head widths that
//! are not a power of two are padded with zeros by the loads.

use std::sync::Arc;

use cutile::cuda_async::device_operation::DeviceOp;
use cutile::half::{bf16, f16};
use cutile::tensor::{IntoPartition, Tensor as TileTensor};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use kapel_core::{DType, Error, Layout, Result, Tensor};
use kapel_ops::{Attention, Operation, Transfer};

use crate::device::{Cuda, CudaStorage};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// The forward pass for the queries of tile `pid.1` of head `pid.0`
    /// (`b · heads + h`): the output, and `lse = max + ln Σ e^(s − max)` of
    /// each query (`-1e30` for a query that sees no key).
    ///
    /// Key `j` is seen by query `i` if both exist, `j ≤ p(i) + ahead` (`ahead`
    /// is 0 for causal attention, `tk` otherwise) and `p(i) − j < window` (more
    /// than `tk` without a window), with `p(i) = tk − tq + i`. Only the tiles
    /// on an edge of the visible region compute it: their hidden scores become
    /// `HIDDEN`, below the running maximum's start, so their weights vanish.
    /// Scores of −∞ (from the mask) vanish by themselves.
    #[cutile::entry()]
    fn attention_forward<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const D: i32,
        const DV: i32,
        const MASKED: i32,
        const CAPPED: i32,
    >(
        out: &mut Tensor<E, { [1, BM, DV] }>,
        lse: &mut Tensor<f32, { [1, BM, 1] }>,
        q: &Tensor<E, { [-1, -1, -1, -1] }>,
        k: &Tensor<E, { [-1, -1, -1, -1] }>,
        v: &Tensor<E, { [-1, -1, -1, -1] }>,
        mask: &Tensor<f32, { [-1, -1, -1, -1] }>,
        scale: f32,
        softcap: f32,
        heads: i32,
        group: i32,
        tq: i32,
        tk: i32,
        ahead: i32,
        window: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b: i32 = pid.0 / heads;
        let h: i32 = pid.0 % heads;
        let qi: i32 = pid.1;
        let kv: i32 = h / group;
        let tile_q: Tile<E, { [BM, D] }> = q
            .partition(shape![1, 1, BM, D])
            .load([b, h, qi, 0i32])
            .reshape(shape![BM, D]);
        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, DV]);
        let mask_part = mask.partition(shape![1, 1, BM, BN]);
        let zeros: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let scale_tile: Tile<f32, { [BM, BN] }> = broadcast_scalar(scale, shape![BM, BN]);
        let log2e: f32 = 1.442_695_f32;
        let log2e_tile: Tile<f32, { [BM, BN] }> = broadcast_scalar(log2e, shape![BM, BN]);
        let log2e_col: Tile<f32, { [BM, 1] }> = broadcast_scalar(log2e, shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(-1.0e30f32, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, DV] }> = constant(0.0f32, shape![BM, DV]);
        // The key tiles that some query of this tile may see, and among them
        // those that every query sees.
        let first: i32 = tk - tq + qi * BM;
        let last: i32 = min(tk, first + BM + ahead);
        let end: i32 = ceil_div(last, BN);
        let lowest: i32 = first - window + 1i32;
        let start: i32 = max(0i32, lowest) / BN;
        let full_lo: i32 = ceil_div(max(0i32, first + BM - window), BN);
        let full_hi: i32 = min(first + ahead + 1i32, tk) / BN;
        let queries_end: i32 = (qi + 1i32) * BM;
        for jt in start..end {
            let tile_k: Tile<E, { [BN, D] }> =
                k_part.load([b, kv, jt, 0i32]).reshape(shape![BN, D]);
            let raw: Tile<f32, { [BM, BN] }> = mma(tile_q, tile_k.transpose(), zeros);
            let mut s: Tile<f32, { [BM, BN] }> = raw * scale_tile;
            if CAPPED == 1i32 {
                let cap: Tile<f32, { [BM, BN] }> = broadcast_scalar(softcap, shape![BM, BN]);
                s = cap * tanh(true_div(s, cap));
            }
            if MASKED == 1i32 {
                let m: Tile<f32, { [BM, BN] }> =
                    mask_part.load([b, h, qi, jt]).reshape(shape![BM, BN]);
                s = s + m;
            }
            if jt < full_lo || jt >= full_hi || queries_end > tq {
                let rows: Tile<i32, { [BM] }> = iota(shape![BM]);
                let p: Tile<i32, { [BM, 1] }> =
                    rows.reshape(shape![BM, 1]) + broadcast_scalar(first, shape![BM, 1]);
                let p: Tile<i32, { [BM, BN] }> = p.broadcast(shape![BM, BN]);
                let cols: Tile<i32, { [BN] }> = iota(shape![BN]);
                let j: Tile<i32, { [1, BN] }> =
                    cols.reshape(shape![1, BN]) + broadcast_scalar(jt * BN, shape![1, BN]);
                let j: Tile<i32, { [BM, BN] }> = j.broadcast(shape![BM, BN]);
                let end_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(tk, shape![BM, BN]);
                let ahead_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(ahead, shape![BM, BN]);
                let window_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(window, shape![BM, BN]);
                let seen: Tile<bool, { [BM, BN] }> = lt_tile(j, end_tile)
                    & lt_tile(p, end_tile)
                    & le_tile(j, p + ahead_tile)
                    & lt_tile(p - j, window_tile);
                let hidden: Tile<f32, { [BM, BN] }> = constant(-2.0e30f32, shape![BM, BN]);
                s = select(seen, s, hidden);
            }
            let row_max: Tile<f32, { [BM] }> = reduce_max(s, 1i32);
            let m_new: Tile<f32, { [BM, 1] }> = max_tile(m_i, row_max.reshape(shape![BM, 1]));
            let shift: Tile<f32, { [BM, BN] }> = (m_new * log2e_col).broadcast(shape![BM, BN]);
            let w: Tile<f32, { [BM, BN] }> = exp2(s * log2e_tile - shift, ftz::Enabled);
            let alpha: Tile<f32, { [BM, 1] }> = exp2((m_i - m_new) * log2e_col, ftz::Enabled);
            let row_sum: Tile<f32, { [BM] }> = reduce_sum(w, 1i32);
            l_i = l_i * alpha + row_sum.reshape(shape![BM, 1]);
            let tile_v: Tile<E, { [BN, DV] }> =
                v_part.load([b, kv, jt, 0i32]).reshape(shape![BN, DV]);
            let we: Tile<E, { [BM, BN] }> = convert_tile(w);
            acc = mma(we, tile_v, acc * alpha.broadcast(shape![BM, DV]));
            m_i = m_new;
        }
        let ones: Tile<f32, { [BM, 1] }> = constant(1.0f32, shape![BM, 1]);
        let none: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let some: Tile<bool, { [BM, 1] }> = gt_tile(l_i, none);
        let l_safe: Tile<f32, { [BM, 1] }> = select(some, l_i, ones);
        let y: Tile<f32, { [BM, DV] }> = true_div(acc, l_safe.broadcast(shape![BM, DV]));
        let y: Tile<E, { [BM, DV] }> = convert_tile(y);
        out.store(y.reshape(shape![1, BM, DV]));
        let total: Tile<f32, { [BM, 1] }> = m_i + log(l_safe);
        lse.store(total.reshape(shape![1, BM, 1]));
    }

    /// `δ = rowsum(g ⊙ y)` for the queries of tile `pid.1` of head `pid.0`.
    #[cutile::entry()]
    fn attention_delta<E: ElementType, const BM: i32, const DV: i32>(
        delta: &mut Tensor<f32, { [1, BM, 1] }>,
        y: &Tensor<E, { [-1, -1, -1, -1] }>,
        g: &Tensor<E, { [-1, -1, -1, -1] }>,
        heads: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b: i32 = pid.0 / heads;
        let h: i32 = pid.0 % heads;
        let tile_y: Tile<E, { [BM, DV] }> = y
            .partition(shape![1, 1, BM, DV])
            .load([b, h, pid.1, 0i32])
            .reshape(shape![BM, DV]);
        let tile_g: Tile<E, { [BM, DV] }> = g
            .partition(shape![1, 1, BM, DV])
            .load([b, h, pid.1, 0i32])
            .reshape(shape![BM, DV]);
        let yf: Tile<f32, { [BM, DV] }> = convert_tile(tile_y);
        let gf: Tile<f32, { [BM, DV] }> = convert_tile(tile_g);
        let total: Tile<f32, { [BM] }> = reduce_sum(yf * gf, 1i32);
        delta.store(total.reshape(shape![1, BM, 1]));
    }

    /// The gradient of the queries of tile `pid.1` of head `pid.0`: with the
    /// weights `P` recomputed from `lse`, `∂P = g · vᵀ`,
    /// `∂s = scale · P ⊙ (∂P − δ) ⊙ slope`, the slope being that of the softcap
    /// (1 without one), and `∂q = ∂s · k`. Visibility as in the forward pass.
    #[cutile::entry()]
    fn attention_grad_q<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const D: i32,
        const DV: i32,
        const MASKED: i32,
        const CAPPED: i32,
    >(
        dq: &mut Tensor<E, { [1, BM, D] }>,
        q: &Tensor<E, { [-1, -1, -1, -1] }>,
        k: &Tensor<E, { [-1, -1, -1, -1] }>,
        v: &Tensor<E, { [-1, -1, -1, -1] }>,
        mask: &Tensor<f32, { [-1, -1, -1, -1] }>,
        g: &Tensor<E, { [-1, -1, -1, -1] }>,
        lse: &Tensor<f32, { [-1, -1, -1] }>,
        delta: &Tensor<f32, { [-1, -1, -1] }>,
        scale: f32,
        softcap: f32,
        heads: i32,
        group: i32,
        tq: i32,
        tk: i32,
        ahead: i32,
        window: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b: i32 = pid.0 / heads;
        let h: i32 = pid.0 % heads;
        let qi: i32 = pid.1;
        let kv: i32 = h / group;
        let tile_q: Tile<E, { [BM, D] }> = q
            .partition(shape![1, 1, BM, D])
            .load([b, h, qi, 0i32])
            .reshape(shape![BM, D]);
        let tile_g: Tile<E, { [BM, DV] }> = g
            .partition(shape![1, 1, BM, DV])
            .load([b, h, qi, 0i32])
            .reshape(shape![BM, DV]);
        let log2e: f32 = 1.442_695_f32;
        let tile_lse: Tile<f32, { [BM, 1] }> = lse
            .partition(shape![1, 1, BM])
            .load([pid.0, 0i32, qi])
            .reshape(shape![BM, 1]);
        let lse2: Tile<f32, { [BM, BN] }> =
            (tile_lse * broadcast_scalar(log2e, shape![BM, 1])).broadcast(shape![BM, BN]);
        let tile_delta: Tile<f32, { [BM, 1] }> = delta
            .partition(shape![1, 1, BM])
            .load([pid.0, 0i32, qi])
            .reshape(shape![BM, 1]);
        let tile_delta: Tile<f32, { [BM, BN] }> = tile_delta.broadcast(shape![BM, BN]);
        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, DV]);
        let mask_part = mask.partition(shape![1, 1, BM, BN]);
        let zeros: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let scale_tile: Tile<f32, { [BM, BN] }> = broadcast_scalar(scale, shape![BM, BN]);
        let log2e_tile: Tile<f32, { [BM, BN] }> = broadcast_scalar(log2e, shape![BM, BN]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        let first: i32 = tk - tq + qi * BM;
        let last: i32 = min(tk, first + BM + ahead);
        let end: i32 = ceil_div(last, BN);
        let lowest: i32 = first - window + 1i32;
        let start: i32 = max(0i32, lowest) / BN;
        let full_lo: i32 = ceil_div(max(0i32, first + BM - window), BN);
        let full_hi: i32 = min(first + ahead + 1i32, tk) / BN;
        let queries_end: i32 = (qi + 1i32) * BM;
        for jt in start..end {
            let tile_k: Tile<E, { [BN, D] }> =
                k_part.load([b, kv, jt, 0i32]).reshape(shape![BN, D]);
            let raw: Tile<f32, { [BM, BN] }> = mma(tile_q, tile_k.transpose(), zeros);
            let mut s: Tile<f32, { [BM, BN] }> = raw * scale_tile;
            let mut ds_scale: Tile<f32, { [BM, BN] }> = scale_tile;
            if CAPPED == 1i32 {
                let cap: Tile<f32, { [BM, BN] }> = broadcast_scalar(softcap, shape![BM, BN]);
                let t: Tile<f32, { [BM, BN] }> = tanh(true_div(s, cap));
                s = cap * t;
                let ones: Tile<f32, { [BM, BN] }> = constant(1.0f32, shape![BM, BN]);
                ds_scale = scale_tile * (ones - t * t);
            }
            if MASKED == 1i32 {
                let m: Tile<f32, { [BM, BN] }> =
                    mask_part.load([b, h, qi, jt]).reshape(shape![BM, BN]);
                s = s + m;
            }
            if jt < full_lo || jt >= full_hi || queries_end > tq {
                let rows: Tile<i32, { [BM] }> = iota(shape![BM]);
                let p: Tile<i32, { [BM, 1] }> =
                    rows.reshape(shape![BM, 1]) + broadcast_scalar(first, shape![BM, 1]);
                let p: Tile<i32, { [BM, BN] }> = p.broadcast(shape![BM, BN]);
                let cols: Tile<i32, { [BN] }> = iota(shape![BN]);
                let j: Tile<i32, { [1, BN] }> =
                    cols.reshape(shape![1, BN]) + broadcast_scalar(jt * BN, shape![1, BN]);
                let j: Tile<i32, { [BM, BN] }> = j.broadcast(shape![BM, BN]);
                let end_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(tk, shape![BM, BN]);
                let ahead_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(ahead, shape![BM, BN]);
                let window_tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(window, shape![BM, BN]);
                let seen: Tile<bool, { [BM, BN] }> = lt_tile(j, end_tile)
                    & lt_tile(p, end_tile)
                    & le_tile(j, p + ahead_tile)
                    & lt_tile(p - j, window_tile);
                let hidden: Tile<f32, { [BM, BN] }> = constant(-2.0e30f32, shape![BM, BN]);
                s = select(seen, s, hidden);
            }
            let w: Tile<f32, { [BM, BN] }> = exp2(s * log2e_tile - lse2, ftz::Enabled);
            let tile_v: Tile<E, { [BN, DV] }> =
                v_part.load([b, kv, jt, 0i32]).reshape(shape![BN, DV]);
            let dw: Tile<f32, { [BM, BN] }> = mma(tile_g, tile_v.transpose(), zeros);
            let ds: Tile<f32, { [BM, BN] }> = w * (dw - tile_delta) * ds_scale;
            let ds: Tile<E, { [BM, BN] }> = convert_tile(ds);
            acc = mma(ds, tile_k, acc);
        }
        let out: Tile<E, { [BM, D] }> = convert_tile(acc);
        dq.store(out.reshape(shape![1, BM, D]));
    }

    /// The gradients of the keys and values of tile `pid.1` of key head
    /// `pid.0` (`b · kv_heads + kv`), from the transposed scores `k · qᵀ` of
    /// each query head of its group and each query tile, in order:
    /// `∂v = Σ Pᵀ · g` and `∂k = Σ ∂sᵀ · q`.
    #[cutile::entry()]
    fn attention_grad_kv<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const D: i32,
        const DV: i32,
        const MASKED: i32,
        const CAPPED: i32,
    >(
        dk: &mut Tensor<E, { [1, BN, D] }>,
        dv: &mut Tensor<E, { [1, BN, DV] }>,
        q: &Tensor<E, { [-1, -1, -1, -1] }>,
        k: &Tensor<E, { [-1, -1, -1, -1] }>,
        v: &Tensor<E, { [-1, -1, -1, -1] }>,
        mask: &Tensor<f32, { [-1, -1, -1, -1] }>,
        g: &Tensor<E, { [-1, -1, -1, -1] }>,
        lse: &Tensor<f32, { [-1, -1, -1] }>,
        delta: &Tensor<f32, { [-1, -1, -1] }>,
        scale: f32,
        softcap: f32,
        heads: i32,
        group: i32,
        kv_heads: i32,
        tq: i32,
        tk: i32,
        ahead: i32,
        window: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b: i32 = pid.0 / kv_heads;
        let kv: i32 = pid.0 % kv_heads;
        let kj: i32 = pid.1;
        let tile_k: Tile<E, { [BN, D] }> = k
            .partition(shape![1, 1, BN, D])
            .load([b, kv, kj, 0i32])
            .reshape(shape![BN, D]);
        let tile_v: Tile<E, { [BN, DV] }> = v
            .partition(shape![1, 1, BN, DV])
            .load([b, kv, kj, 0i32])
            .reshape(shape![BN, DV]);
        let q_part = q.partition(shape![1, 1, BM, D]);
        let g_part = g.partition(shape![1, 1, BM, DV]);
        let lse_part = lse.partition(shape![1, 1, BM]);
        let delta_part = delta.partition(shape![1, 1, BM]);
        let mask_part = mask.partition(shape![1, 1, BM, BN]);
        let zeros: Tile<f32, { [BN, BM] }> = constant(0.0f32, shape![BN, BM]);
        let scale_tile: Tile<f32, { [BN, BM] }> = broadcast_scalar(scale, shape![BN, BM]);
        let log2e: f32 = 1.442_695_f32;
        let log2e_tile: Tile<f32, { [BN, BM] }> = broadcast_scalar(log2e, shape![BN, BM]);
        let log2e_row: Tile<f32, { [1, BM] }> = broadcast_scalar(log2e, shape![1, BM]);
        let mut acc_k: Tile<f32, { [BN, D] }> = constant(0.0f32, shape![BN, D]);
        let mut acc_v: Tile<f32, { [BN, DV] }> = constant(0.0f32, shape![BN, DV]);
        // The query tiles that may see some key of this tile: query `i` sees
        // key `j` only if `j − ahead ≤ p(i)` and `p(i) < j + window`; and
        // among them those whose queries all see all its keys.
        let shift: i32 = tk - tq;
        let lowest: i32 = kj * BN - ahead - shift;
        let start: i32 = max(0i32, lowest) / BM;
        let highest: i32 = kj * BN + BN - 1i32 + window - shift;
        let last: i32 = min(tq, highest);
        let end: i32 = ceil_div(max(0i32, last), BM);
        let full_lo: i32 = ceil_div(max(0i32, lowest + BN - 1i32), BM);
        let widest: i32 = max(-1i32, window - shift - BM + kj * BN);
        let full_hi: i32 = min((widest + BM) / BM, tq / BM);
        let keys_end: i32 = (kj + 1i32) * BN;
        for gh in 0i32..group {
            let h: i32 = kv * group + gh;
            let bh: i32 = b * heads + h;
            for qi in start..end {
                let tile_q: Tile<E, { [BM, D] }> =
                    q_part.load([b, h, qi, 0i32]).reshape(shape![BM, D]);
                // Rows are keys, columns queries.
                let raw: Tile<f32, { [BN, BM] }> = mma(tile_k, tile_q.transpose(), zeros);
                let mut s: Tile<f32, { [BN, BM] }> = raw * scale_tile;
                let mut ds_scale: Tile<f32, { [BN, BM] }> = scale_tile;
                if CAPPED == 1i32 {
                    let cap: Tile<f32, { [BN, BM] }> = broadcast_scalar(softcap, shape![BN, BM]);
                    let t: Tile<f32, { [BN, BM] }> = tanh(true_div(s, cap));
                    s = cap * t;
                    let ones: Tile<f32, { [BN, BM] }> = constant(1.0f32, shape![BN, BM]);
                    ds_scale = scale_tile * (ones - t * t);
                }
                if MASKED == 1i32 {
                    let m: Tile<f32, { [BM, BN] }> =
                        mask_part.load([b, h, qi, kj]).reshape(shape![BM, BN]);
                    s = s + m.transpose();
                }
                if qi < full_lo || qi >= full_hi || keys_end > tk {
                    let rows: Tile<i32, { [BN] }> = iota(shape![BN]);
                    let j: Tile<i32, { [BN, 1] }> =
                        rows.reshape(shape![BN, 1]) + broadcast_scalar(kj * BN, shape![BN, 1]);
                    let j: Tile<i32, { [BN, BM] }> = j.broadcast(shape![BN, BM]);
                    let cols: Tile<i32, { [BM] }> = iota(shape![BM]);
                    let p: Tile<i32, { [1, BM] }> = cols.reshape(shape![1, BM])
                        + broadcast_scalar(shift + qi * BM, shape![1, BM]);
                    let p: Tile<i32, { [BN, BM] }> = p.broadcast(shape![BN, BM]);
                    let end_tile: Tile<i32, { [BN, BM] }> = broadcast_scalar(tk, shape![BN, BM]);
                    let ahead_tile: Tile<i32, { [BN, BM] }> =
                        broadcast_scalar(ahead, shape![BN, BM]);
                    let window_tile: Tile<i32, { [BN, BM] }> =
                        broadcast_scalar(window, shape![BN, BM]);
                    let seen: Tile<bool, { [BN, BM] }> = lt_tile(j, end_tile)
                        & lt_tile(p, end_tile)
                        & le_tile(j, p + ahead_tile)
                        & lt_tile(p - j, window_tile);
                    let hidden: Tile<f32, { [BN, BM] }> = constant(-2.0e30f32, shape![BN, BM]);
                    s = select(seen, s, hidden);
                }
                let tile_lse: Tile<f32, { [1, BM] }> =
                    lse_part.load([bh, 0i32, qi]).reshape(shape![1, BM]);
                let lse2: Tile<f32, { [BN, BM] }> =
                    (tile_lse * log2e_row).broadcast(shape![BN, BM]);
                let w: Tile<f32, { [BN, BM] }> = exp2(s * log2e_tile - lse2, ftz::Enabled);
                let tile_g: Tile<E, { [BM, DV] }> =
                    g_part.load([b, h, qi, 0i32]).reshape(shape![BM, DV]);
                let we: Tile<E, { [BN, BM] }> = convert_tile(w);
                acc_v = mma(we, tile_g, acc_v);
                let dw: Tile<f32, { [BN, BM] }> = mma(tile_v, tile_g.transpose(), zeros);
                let tile_delta: Tile<f32, { [1, BM] }> =
                    delta_part.load([bh, 0i32, qi]).reshape(shape![1, BM]);
                let ds: Tile<f32, { [BN, BM] }> =
                    w * (dw - tile_delta.broadcast(shape![BN, BM])) * ds_scale;
                let ds: Tile<E, { [BN, BM] }> = convert_tile(ds);
                acc_k = mma(ds, tile_q, acc_k);
            }
        }
        let out_k: Tile<E, { [BN, D] }> = convert_tile(acc_k);
        dk.store(out_k.reshape(shape![1, BN, D]));
        let out_v: Tile<E, { [BN, DV] }> = convert_tile(acc_v);
        dv.store(out_v.reshape(shape![1, BN, DV]));
    }
}

/// A `[B, H, T, D]` tensor of this backend, read in place, or any other
/// strided view of its memory.
#[derive(Clone)]
struct Heads {
    pointer: u64,
    shape: Vec<i32>,
    strides: Vec<i32>,
}

/// How one kernel runs: queries and keys per tile, and the compiler's hints.
#[derive(Clone, Copy)]
struct Launch {
    tiles: [usize; 2],
    occupancy: Option<i32>,
    warps: Option<i32>,
}

impl Launch {
    const fn new(queries: usize, keys: usize) -> Self {
        Self {
            tiles: [queries, keys],
            occupancy: None,
            warps: None,
        }
    }

    /// Asks for `blocks` tile blocks per multiprocessor.
    const fn occupancy(mut self, blocks: i32) -> Self {
        self.occupancy = Some(blocks);
        self
    }

    /// Asks for `warps` warps per tile block.
    const fn warps(mut self, warps: i32) -> Self {
        self.warps = Some(warps);
        self
    }

    fn options(&self) -> CompileOptions {
        let mut options = CompileOptions::new();
        if let Some(blocks) = self.occupancy {
            options = options.occupancy(blocks);
        }
        if let Some(warps) = self.warps {
            options = options.num_worker_warps_per_cta(warps);
        }
        options
    }
}

/// How each kernel runs, and the head widths padded to a power of two (the
/// loads read zeros past the true width).
struct Tiles {
    forward: Launch,
    grad_q: Launch,
    grad_kv: Launch,
    d: usize,
    dv: usize,
}

impl Tiles {
    /// The best of a sweep on an RTX 3090, causal, over 1 to 4 thousand
    /// positions. Neighbouring choices can be up to 6 times slower: the
    /// compiler's results do not follow a rule.
    fn for_(dtype: DType, d: usize, dv: usize) -> Self {
        let pad = |n: usize| n.next_power_of_two().max(16);
        let (d, dv) = (pad(d), pad(dv));
        let l = Launch::new;
        let (forward, grad_q, grad_kv) = match (dtype, d.max(dv)) {
            (DType::F32, _) => (l(32, 32), l(32, 32), l(32, 32)),
            (_, ..=64) => (l(128, 64), l(64, 32), l(64, 32)),
            (_, ..=128) => (
                l(128, 32).occupancy(2),
                l(64, 32).warps(8),
                l(32, 32).warps(8),
            ),
            _ => (
                l(64, 32).warps(8),
                l(32, 32).occupancy(2),
                l(64, 64).warps(8),
            ),
        };
        Self {
            forward,
            grad_q,
            grad_kv,
            d,
            dv,
        }
    }
}

/// The queries per tile of the kernel of `δ`.
const DELTA_ROWS: usize = 64;

fn int(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Overflow {
        op: Attention::NAME,
    })
}

fn ints(values: &[usize]) -> Result<Vec<i32>> {
    values.iter().map(|&v| int(v)).collect()
}

fn device_error(e: impl std::fmt::Display) -> Error {
    Error::Device {
        op: Attention::NAME,
        message: e.to_string(),
    }
}

/// An attention call, ready for the kernels: its sizes, its constants, and
/// its inputs read in place (the mask in `f32`, broadcast, or a placeholder).
struct Prepared {
    dtype: DType,
    name: &'static str,
    sizes: [usize; 7],
    scale: f32,
    softcap: f32,
    ahead: i32,
    window: i32,
    masked: bool,
    tiles: Tiles,
    q: Heads,
    k: Heads,
    v: Heads,
    mask: Heads,
    // Keeps the mask's storage alive when it is a converted copy.
    _mask: Tensor,
}

impl Prepared {
    fn generics(&self, [bm, bn]: [usize; 2]) -> Vec<String> {
        let t = &self.tiles;
        [
            bm,
            bn,
            t.d,
            t.dv,
            usize::from(self.masked),
            usize::from(self.softcap > 0.0),
        ]
        .iter()
        .fold(vec![self.name.to_string()], |mut g, v| {
            g.push(v.to_string());
            g
        })
    }
}

/// A row `[A, 1, C]` of per-query values seen as the column `[A, C, 1]` that
/// a kernel writes; both views share the same memory.
fn column(row: &Heads) -> Heads {
    Heads {
        pointer: row.pointer,
        shape: vec![row.shape[0], row.shape[2], 1],
        strides: vec![row.strides[0], 1, 1],
    }
}

/// Borrows device memory as a tile tensor.
///
/// # Safety
///
/// The memory belongs to a storage of the backend that the caller keeps alive
/// until the kernel that uses it is queued: the kernel runs on the backend's
/// stream, before any later use of the memory. A tensor borrowed as an output
/// is a fresh storage that nothing else reads or writes meanwhile.
unsafe fn borrow<T: cutile::DType>(ordinal: usize, h: &Heads) -> TileTensor<T> {
    // SAFETY: the caller's obligations, above.
    unsafe {
        TileTensor::<T>::borrow_raw_parts(h.pointer, ordinal, h.shape.clone(), h.strides.clone())
    }
}

impl Cuda {
    /// A tensor of this backend seen through `layout`.
    fn heads(&self, tensor: &Tensor, layout: &Layout) -> Result<Heads> {
        let storage = self.storage(Attention::NAME, tensor)?;
        let size = tensor.dtype().block_bytes();
        Ok(Heads {
            pointer: storage.pointer() + (layout.offset() * size) as u64,
            shape: ints(layout.shape())?,
            strides: ints(layout.strides())?,
        })
    }

    /// A contiguous `[A, B, C]` in `storage`.
    fn dense(storage: &CudaStorage, [a, b, c]: [usize; 3]) -> Result<Heads> {
        Ok(Heads {
            pointer: storage.pointer(),
            shape: ints(&[a, b, c])?,
            strides: ints(&[b * c, c, 1])?,
        })
    }

    fn prepare(&self, op: &Attention, inputs: &[&Tensor]) -> Result<Prepared> {
        let (q, k, v) = (inputs[0], inputs[1], inputs[2]);
        let dtype = q.dtype();
        let name = match dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            dtype => {
                return Err(Error::UnsupportedDType {
                    op: Attention::NAME,
                    dtype,
                });
            }
        };
        let [b, hq, tq, d] = [q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]];
        let (hkv, tk, dv) = (k.shape()[1], k.shape()[2], v.shape()[3]);
        let mask = match inputs.get(3) {
            Some(m) if m.dtype() == DType::F32 => (*m).clone(),
            Some(m) => {
                let values: Vec<f32> = self
                    .download(m)?
                    .to_f64_vec()?
                    .into_iter()
                    .map(|v| v as f32)
                    .collect();
                self.upload_f32(&values, m.shape())?
            }
            None => self.upload_f32(&[0.0], &[1, 1, 1, 1])?,
        };
        let masked = inputs.len() == 4;
        let mask_layout = if masked {
            mask.layout().broadcast_to(&[b, hq, tq, tk])?
        } else {
            mask.layout().clone()
        };
        Ok(Prepared {
            dtype,
            name,
            sizes: [b, hq, hkv, tq, tk, d, dv],
            scale: op.scale_for(d) as f32,
            softcap: op.softcap.map_or(0.0, |c| c as f32),
            ahead: if op.causal { 0 } else { int(tk)? },
            // A window wider than the keys hides nothing.
            window: int(op.window.map_or(tk + 1, |w| w.min(tk + 1)))?,
            masked,
            tiles: Tiles::for_(dtype, d, dv),
            q: self.heads(q, q.layout())?,
            k: self.heads(k, k.layout())?,
            v: self.heads(v, v.layout())?,
            mask: self.heads(&mask, &mask_layout)?,
            _mask: mask,
        })
    }

    /// The attention of `inputs` (checked by `infer`): the output
    /// `[B, Hq, Tq, Dv]` and the `lse` of each query, `[B · Hq, Tq, 1]`.
    pub(crate) fn flash_forward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
    ) -> Result<(Tensor, Tensor)> {
        let a = self.prepare(op, inputs)?;
        let [b, hq, _, tq, _, _, dv] = a.sizes;
        let y = self.alloc(b * hq * tq * dv * a.dtype.block_bytes(), false)?;
        let lse = self.alloc(b * hq * tq * 4, false)?;
        if b * hq * tq * dv > 0 {
            let outputs = [
                Self::dense(&y, [b * hq, tq, dv])?,
                Self::dense(&lse, [b * hq, tq, 1])?,
            ];
            match a.dtype {
                DType::F32 => self.forward_kernel::<f32>(&a, outputs),
                DType::BF16 => self.forward_kernel::<bf16>(&a, outputs),
                _ => self.forward_kernel::<f16>(&a, outputs),
            }?;
        }
        let y = Tensor::from_storage(Arc::new(y), Layout::contiguous(&[b, hq, tq, dv])?, a.dtype)?;
        let lse = Tensor::from_storage(
            Arc::new(lse),
            Layout::contiguous(&[b * hq, tq, 1])?,
            DType::F32,
        )?;
        Ok((y, lse))
    }

    fn forward_kernel<T: cutile::DType>(&self, a: &Prepared, [y, lse]: [Heads; 2]) -> Result<()> {
        let o = self.ordinal() as usize;
        let [_, hq, hkv, tq, tk, _, _] = a.sizes;
        let t = &a.tiles;
        // SAFETY: (`borrow`) the inputs are storages that `a` and the caller
        // keep alive; `y` and `lse` are fresh outputs.
        let launch = unsafe {
            kernels::attention_forward(
                borrow::<T>(o, &y).partition([1, t.forward.tiles[0], t.dv]),
                borrow::<f32>(o, &lse).partition([1, t.forward.tiles[0], 1]),
                Arc::new(borrow::<T>(o, &a.q)),
                Arc::new(borrow::<T>(o, &a.k)),
                Arc::new(borrow::<T>(o, &a.v)),
                Arc::new(borrow::<f32>(o, &a.mask)),
                a.scale,
                a.softcap,
                int(hq)?,
                int(hq / hkv)?,
                int(tq)?,
                int(tk)?,
                a.ahead,
                a.window,
            )
        }
        .generics(a.generics(t.forward.tiles))
        .compile_options(t.forward.options());
        // SAFETY: as above; the views returned are dropped without freeing.
        unsafe { launch.async_on(self.tile_stream()) }
            .map(drop)
            .map_err(device_error)
    }

    /// The gradients of `q`, `k` and `v`, given that of the output.
    pub(crate) fn flash_backward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<[Tensor; 3]> {
        let (y, lse) = self.flash_forward(op, inputs)?;
        let a = self.prepare(op, inputs)?;
        if grad.dtype() != a.dtype {
            return Err(Error::DTypeMismatch {
                op: Attention::NAME,
                expected: a.dtype,
                got: grad.dtype(),
            });
        }
        let [b, hq, hkv, tq, tk, d, dv] = a.sizes;
        let size = a.dtype.block_bytes();
        // Without queries, keys or widths, every gradient is zero.
        let run = b * hq * tq * tk * d * dv > 0;
        let gq = self.alloc(b * hq * tq * d * size, !run)?;
        let gk = self.alloc(b * hkv * tk * d * size, !run)?;
        let gv = self.alloc(b * hkv * tk * dv * size, !run)?;
        let delta = self.alloc(b * hq * tq * 4, false)?;
        if run {
            let read = [
                self.heads(&y, y.layout())?,
                self.heads(grad, grad.layout())?,
                Self::dense(self.storage(Attention::NAME, &lse)?, [b * hq, 1, tq])?,
                Self::dense(&delta, [b * hq, 1, tq])?,
            ];
            let outputs = [
                Self::dense(&gq, [b * hq, tq, d])?,
                Self::dense(&gk, [b * hkv, tk, d])?,
                Self::dense(&gv, [b * hkv, tk, dv])?,
            ];
            match a.dtype {
                DType::F32 => self.backward_kernels::<f32>(&a, &read, &outputs),
                DType::BF16 => self.backward_kernels::<bf16>(&a, &read, &outputs),
                _ => self.backward_kernels::<f16>(&a, &read, &outputs),
            }?;
        }
        let tensor = |s: CudaStorage, shape: &[usize]| {
            Tensor::from_storage(Arc::new(s), Layout::contiguous(shape)?, a.dtype)
        };
        Ok([
            tensor(gq, &[b, hq, tq, d])?,
            tensor(gk, &[b, hkv, tk, d])?,
            tensor(gv, &[b, hkv, tk, dv])?,
        ])
    }

    fn backward_kernels<T: cutile::DType>(
        &self,
        a: &Prepared,
        [y, g, lse, delta]: &[Heads; 4],
        [gq, gk, gv]: &[Heads; 3],
    ) -> Result<()> {
        let o = self.ordinal() as usize;
        let [_, hq, hkv, tq, tk, _, _] = a.sizes;
        let t = &a.tiles;
        let stream = self.tile_stream();
        // SAFETY: (`borrow`) the inputs are storages that `a` and the caller
        // keep alive; `delta`, `gq`, `gk` and `gv` are fresh outputs, and the
        // kernels that read `delta` are queued after the one that writes it.
        let deltas = unsafe {
            kernels::attention_delta(
                borrow::<f32>(o, &column(delta)).partition([1, DELTA_ROWS, 1]),
                Arc::new(borrow::<T>(o, y)),
                Arc::new(borrow::<T>(o, g)),
                int(hq)?,
            )
        }
        .generics(vec![
            a.name.to_string(),
            DELTA_ROWS.to_string(),
            t.dv.to_string(),
        ]);
        // SAFETY: as above; the views returned are dropped without freeing.
        unsafe { deltas.async_on(stream) }
            .map(drop)
            .map_err(device_error)?;
        // SAFETY: (`borrow`) as above.
        let grad_q = unsafe {
            kernels::attention_grad_q(
                borrow::<T>(o, gq).partition([1, t.grad_q.tiles[0], t.d]),
                Arc::new(borrow::<T>(o, &a.q)),
                Arc::new(borrow::<T>(o, &a.k)),
                Arc::new(borrow::<T>(o, &a.v)),
                Arc::new(borrow::<f32>(o, &a.mask)),
                Arc::new(borrow::<T>(o, g)),
                Arc::new(borrow::<f32>(o, lse)),
                Arc::new(borrow::<f32>(o, delta)),
                a.scale,
                a.softcap,
                int(hq)?,
                int(hq / hkv)?,
                int(tq)?,
                int(tk)?,
                a.ahead,
                a.window,
            )
        }
        .generics(a.generics(t.grad_q.tiles))
        .compile_options(t.grad_q.options());
        // SAFETY: as above.
        unsafe { grad_q.async_on(stream) }
            .map(drop)
            .map_err(device_error)?;
        // SAFETY: (`borrow`) as above.
        let grad_kv = unsafe {
            kernels::attention_grad_kv(
                borrow::<T>(o, gk).partition([1, t.grad_kv.tiles[1], t.d]),
                borrow::<T>(o, gv).partition([1, t.grad_kv.tiles[1], t.dv]),
                Arc::new(borrow::<T>(o, &a.q)),
                Arc::new(borrow::<T>(o, &a.k)),
                Arc::new(borrow::<T>(o, &a.v)),
                Arc::new(borrow::<f32>(o, &a.mask)),
                Arc::new(borrow::<T>(o, g)),
                Arc::new(borrow::<f32>(o, lse)),
                Arc::new(borrow::<f32>(o, delta)),
                a.scale,
                a.softcap,
                int(hq)?,
                int(hq / hkv)?,
                int(hkv)?,
                int(tq)?,
                int(tk)?,
                a.ahead,
                a.window,
            )
        }
        .generics(a.generics(t.grad_kv.tiles))
        .compile_options(t.grad_kv.options());
        // SAFETY: as above.
        unsafe { grad_kv.async_on(stream) }
            .map(drop)
            .map_err(device_error)
    }
}
