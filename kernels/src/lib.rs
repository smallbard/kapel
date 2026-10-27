//! Kapel's CUDA kernels.
//!
//! Each kernel follows the simplest scheme that gives the same bits on every
//! run: one thread per output element (or per line), which reads what it needs
//! and writes its own output; normalizations take one warp per row, whose lanes
//! add their partial sums by a fixed butterfly. No two threads write the same
//! place, and no sum depends on scheduling: there are no atomics.
//!
//! Matrix products and attention are not here: they are tile kernels written
//! with cutile-rs, in the backend itself.
//!
//! The arguments passed by value are defined in `kapel-cuda-abi`. A slice
//! argument becomes two kernel parameters: its device pointer, then its length.
//!
//! Positions are `u32`: the backend refuses any launch with more than
//! `u32::MAX` threads or a slice longer than `u32::MAX` elements. Reads go
//! through bounds-checked indexing. Writes go through `DisjointSlice::get_mut`
//! at the thread's own index, or, where a thread writes a whole line or row,
//! through its raw pointer once the line's last position has been checked
//! against the slice's length.

// Kernels index explicitly: the indexing is the memory access each thread makes.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use cuda_device::{DisjointSlice, SharedArray, device, kernel, launch_bounds, thread, warp};
use kapel_cuda_abi::{PAGED_CHUNK, PAGED_HEADS, PAGED_WIDTH, Paged, View, binary, float, half, quant, unary};

/// The exact error function, as `kapel_ops::special::erf` computes it: its
/// Taylor series below 2.5, the continued fraction of `erfc` above.
fn erf(x: f64) -> f64 {
    let a = if x < 0.0 { -x } else { x };
    let value = if a < 2.5 {
        let x2 = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0f64);
        loop {
            n += 1.0;
            term *= -x2 / n;
            let next = term / (2.0 * n + 1.0);
            sum += next;
            let size = if next < 0.0 { -next } else { next };
            if size <= 1e-17 * sum {
                break;
            }
        }
        2.0 / core::f64::consts::PI.sqrt() * sum
    } else {
        let tiny = 1e-300;
        let mut f = a;
        let (mut c, mut d) = (a, 0.0f64);
        let mut k = 1;
        while k < 200 {
            let coefficient = k as f64 / 2.0;
            d = a + coefficient * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = a + coefficient / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
            k += 1;
        }
        1.0 - (-a * a).exp() / core::f64::consts::PI.sqrt() / f
    };
    if x < 0.0 { -value } else { value }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `f(x)` for a unary code.
fn unary_value(op: u32, x: f32) -> f32 {
    match op {
        unary::NEG => -x,
        unary::EXP => x.exp(),
        unary::LOG => x.ln(),
        unary::SQRT => x.sqrt(),
        unary::RSQRT => 1.0 / x.sqrt(),
        unary::SQUARE => x * x,
        unary::ABS => x.abs(),
        unary::SIGMOID => sigmoid(x),
        unary::TANH => x.tanh(),
        unary::RELU => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        unary::SILU => x * sigmoid(x),
        unary::GELU_TANH => {
            let c = 0.797_884_6f32;
            0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
        }
        unary::GELU_ERF => {
            let x = x as f64;
            (x * 0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2))) as f32
        }
        _ => f32::NAN,
    }
}

/// `f'(x)` for a unary code.
fn unary_slope(op: u32, x: f32) -> f32 {
    match op {
        unary::NEG => -1.0,
        unary::EXP => x.exp(),
        unary::LOG => 1.0 / x,
        unary::SQRT => 0.5 / x.sqrt(),
        unary::RSQRT => -0.5 / (x * x.sqrt()),
        unary::SQUARE => 2.0 * x,
        unary::ABS => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        unary::SIGMOID => {
            let s = sigmoid(x);
            s * (1.0 - s)
        }
        unary::TANH => {
            let t = x.tanh();
            1.0 - t * t
        }
        unary::RELU => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        unary::SILU => {
            let s = sigmoid(x);
            s + x * s * (1.0 - s)
        }
        unary::GELU_TANH => {
            let c = 0.797_884_6f32;
            let t = (c * (x + 0.044_715 * x * x * x)).tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044_715 * x * x)
        }
        unary::GELU_ERF => {
            let x = x as f64;
            let phi = (-0.5 * x * x).exp() / (2.0 * core::f64::consts::PI).sqrt();
            (0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2)) + x * phi) as f32
        }
        _ => f32::NAN,
    }
}

/// `f(a, b)` and its two partial derivatives, for a binary code.
fn binary_value(op: u32, a: f32, b: f32) -> (f32, f32, f32) {
    match op {
        binary::ADD => (a + b, 1.0, 1.0),
        binary::SUB => (a - b, 1.0, -1.0),
        binary::MUL => (a * b, b, a),
        binary::DIV => (a / b, 1.0 / b, -a / (b * b)),
        binary::MAX => {
            if a >= b {
                (a, 1.0, 0.0)
            } else {
                (b, 0.0, 1.0)
            }
        }
        binary::MIN => {
            if a <= b {
                (a, 1.0, 0.0)
            } else {
                (b, 0.0, 1.0)
            }
        }
        _ => (f32::NAN, 0.0, 0.0),
    }
}

/// `out[i] = x[view.at(i)]`: a strided view made contiguous, for elements of
/// 1, 2, 4 and 8 bytes, copied bit for bit.
#[kernel]
#[launch_bounds(256)]
pub fn copy_view_8(x: &[u8], view: View, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = x[view.at(i as u32) as usize];
    }
}

/// See `copy_view_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_view_16(x: &[u16], view: View, mut out: DisjointSlice<u16>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = x[view.at(i as u32) as usize];
    }
}

/// See `copy_view_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_view_32(x: &[u32], view: View, mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = x[view.at(i as u32) as usize];
    }
}

/// See `copy_view_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_view_64(x: &[u64], view: View, mut out: DisjointSlice<u64>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = x[view.at(i as u32) as usize];
    }
}

/// `out[to.at(i)] = x[from.at(i)]` for every position `i` of the views' shared
/// shape: a tensor copied, bit for bit, into a part of another (elements of 1,
/// 2, 4 or 8 bytes).
///
/// The backend passes a destination view whose positions are distinct (a
/// slice of a row-major tensor), so that no two threads write the same place.
#[kernel]
#[launch_bounds(256)]
pub fn copy_into_8(x: &[u8], from: View, mut out: DisjointSlice<u8>, to: View, count: u32) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    if i < count {
        let p = to.at(i) as usize;
        if p < out.len() {
            // SAFETY: `p` is in bounds (checked above), and no other thread
            // writes it: the positions of `to` are distinct (see above).
            unsafe { *out.as_mut_ptr().add(p) = x[from.at(i) as usize] };
        }
    }
}

/// See `copy_into_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_into_16(x: &[u16], from: View, mut out: DisjointSlice<u16>, to: View, count: u32) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    if i < count {
        let p = to.at(i) as usize;
        if p < out.len() {
            // SAFETY: `p` is in bounds (checked above), and no other thread
            // writes it: the positions of `to` are distinct (see above).
            unsafe { *out.as_mut_ptr().add(p) = x[from.at(i) as usize] };
        }
    }
}

/// See `copy_into_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_into_32(x: &[u32], from: View, mut out: DisjointSlice<u32>, to: View, count: u32) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    if i < count {
        let p = to.at(i) as usize;
        if p < out.len() {
            // SAFETY: `p` is in bounds (checked above), and no other thread
            // writes it: the positions of `to` are distinct (see above).
            unsafe { *out.as_mut_ptr().add(p) = x[from.at(i) as usize] };
        }
    }
}

/// See `copy_into_8`.
#[kernel]
#[launch_bounds(256)]
pub fn copy_into_64(x: &[u64], from: View, mut out: DisjointSlice<u64>, to: View, count: u32) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    if i < count {
        let p = to.at(i) as usize;
        if p < out.len() {
            // SAFETY: `p` is in bounds (checked above), and no other thread
            // writes it: the positions of `to` are distinct (see above).
            unsafe { *out.as_mut_ptr().add(p) = x[from.at(i) as usize] };
        }
    }
}

/// The value of 16-bit float `bits`: `bf16` (`kind` = `half::BF16`) is the top
/// half of an `f32`; `f16` (`half::F16`) widens exactly, subnormals included.
fn widen(kind: u32, bits: u16) -> f32 {
    let bits = bits as u32;
    if kind == half::BF16 {
        return f32::from_bits(bits << 16);
    }
    let sign = (bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    let magnitude = if exponent == 0x1f {
        // Infinities, and NaNs with their payload.
        0x7f80_0000 | (mantissa << 13)
    } else if exponent != 0 {
        ((exponent + 112) << 23) | (mantissa << 13)
    } else if mantissa == 0 {
        0
    } else {
        // A subnormal: `mantissa · 2⁻²⁴`, exact in `f32`.
        let value = mantissa as f32 * (1.0 / 16_777_216.0);
        value.to_bits()
    };
    f32::from_bits(sign | magnitude)
}

/// The 16-bit float nearest to `x`, ties to even, as the host's conversion
/// gives it: overflow to infinity, every NaN to the positive quiet NaN.
fn narrow(kind: u32, x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let a = bits & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return if kind == half::BF16 { 0x7fc0 } else { 0x7e00 };
    }
    if kind == half::BF16 {
        if a == 0x7f80_0000 {
            return (sign | 0x7f80) as u16;
        }
        // Adding half a unit, less one unless the kept part is odd, rounds to
        // nearest even; a carry into the exponent is the right result, up to
        // infinity.
        let rounded = a + 0x7fff + ((a >> 16) & 1);
        return (sign | (rounded >> 16)) as u16;
    }
    // `f16`. From 65520 on (`2¹⁶ − 2⁴`, the midpoint above the largest
    // finite value, which rounds to even, up), infinity.
    if a >= 0x477f_f000 {
        return (sign | 0x7c00) as u16;
    }
    let exponent = (a >> 23) as i32 - 127;
    if exponent < -14 {
        // Subnormal (or zero): in units of `2⁻²⁴`, rounded by the hardware's
        // own ties-to-even, exactly: `|x| · 2²⁴` is exact, and below 2¹⁰.
        let units = f32::from_bits(a) * 16_777_216.0;
        let r = round_ties_even(units);
        return (sign | r as u32) as u16;
    }
    let shifted = a - ((127 - 15) << 23);
    let rounded = shifted + 0xfff + ((shifted >> 13) & 1);
    (sign | (rounded >> 13)) as u16
}

/// `x` rounded to the nearest integer, ties to even, for `0 ≤ x < 2²³`.
fn round_ties_even(x: f32) -> f32 {
    // Adding 2²³ leaves no fractional bits; the addition rounds to even.
    (x + 8_388_608.0) - 8_388_608.0
}

/// `out[i] = x[view.at(i)]` widened to `f32` from a 16-bit float `kind`.
#[kernel]
#[launch_bounds(256)]
pub fn widen_16(kind: u32, x: &[u16], view: View, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = widen(kind, x[view.at(i as u32) as usize]);
    }
}

/// `out[i] = x[i]` rounded to a 16-bit float `kind`.
#[kernel]
#[launch_bounds(256)]
pub fn narrow_16(kind: u32, x: &[f32], mut out: DisjointSlice<u16>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = narrow(kind, x[i]);
    }
}

/// `out[i] = f(x[i])`.
#[kernel]
#[launch_bounds(256)]
pub fn unary_forward(op: u32, x: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = unary_value(op, x[i]);
    }
}

/// `out[i] = g[i] · f'(x[i])`.
#[kernel]
#[launch_bounds(256)]
pub fn unary_backward(op: u32, kx: u32, x: &[u8], vx: View, kg: u32, g: &[u8], vg: View, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < elements(ko, out.len()) {
        let value = read(kx, x, vx.at(i as u32) as usize);
        let grad = read(kg, g, vg.at(i as u32) as usize);
        // SAFETY: `i` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), i, grad * unary_slope(op, value)) };
    }
}

/// `out[i] = scale · x[i] + bias`.
#[kernel]
#[launch_bounds(256)]
pub fn affine(scale: f32, bias: f32, x: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = scale * x[i] + bias;
    }
}

/// `out[i] = f(a, b)`, both read through views broadcast to the output.
#[kernel]
#[launch_bounds(256)]
pub fn binary_forward(op: u32, a: &[f32], va: View, b: &[f32], vb: View, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    if let Some(y) = out.get_mut(idx) {
        *y = binary_value(op, a[va.at(i) as usize], b[vb.at(i) as usize]).0;
    }
}

/// The gradient of one side (0 for `a`, 1 for `b`) at the output's size:
/// `out[i] = g[i] · ∂f/∂side`.
#[kernel]
#[launch_bounds(256)]
pub fn binary_backward(
    op: u32,
    side: u32,
    a: &[f32],
    va: View,
    b: &[f32],
    vb: View,
    g: &[f32],
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let (_, da, db) = binary_value(op, a[va.at(i as u32) as usize], b[vb.at(i as u32) as usize]);
        *y = g[i] * if side == 0 { da } else { db };
    }
}

/// `out[o] = scale · Σᵣ x[bases[o] + inner[r]]`, the terms added in order.
#[kernel]
#[launch_bounds(256)]
pub fn reduce_sum(kx: u32, x: &[u8], bases: &[u32], inner: &[u32], scale: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let o = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let base = bases[o] as usize;
        let mut total = 0.0f32;
        for r in 0..inner.len() {
            total += read(kx, x, base + inner[r] as usize);
        }
        *y = scale * total;
    }
}

/// Sums over the rows of a row-major `[rows, cols]` matrix of float type `kx`:
/// `out[s · cols + o]` is the sum of column `o` over the rows `s · seg` to
/// `(s + 1) · seg`, in order (all of them with `seg ≥ rows`). Adjacent
/// threads take adjacent columns: the reads are coalesced.
#[kernel]
#[launch_bounds(256)]
pub fn reduce_sum_columns(kx: u32, x: &[u8], rows: u32, cols: u32, seg: u32, scale: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let j = idx.get();
    let (rows, cols, seg) = (rows as usize, cols as usize, seg as usize);
    if cols > 0 && seg > 0 && rows * cols <= elements(kx, x.len()) {
        if let Some(y) = out.get_mut(idx) {
            let (s, o) = (j / cols, j % cols);
            let end = ((s + 1) * seg).min(rows);
            let mut total = 0.0f32;
            let mut r = s * seg;
            while r < end {
                total += read(kx, x, r * cols + o);
                r += 1;
            }
            *y = scale * total;
        }
    }
}

/// Sums of consecutive terms: result `o` of `out` is `scale ·` the sum of
/// the `len` values of `x` from `o · len`, in order. With `seg < len`, each
/// result is cut into `parts = ⌈len / seg⌉` segments of `seg` terms, and
/// `out[o · parts + s]` is the sum of segment `s`: a stage of a long sum.
#[kernel]
#[launch_bounds(256)]
pub fn reduce_sum_rows(kx: u32, x: &[u8], len: u32, seg: u32, scale: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let j = idx.get();
    let (len, seg) = (len as usize, seg as usize);
    if len > 0 && seg > 0 {
        if let Some(y) = out.get_mut(idx) {
            let parts = len.div_ceil(seg);
            let (o, s) = (j / parts, j % parts);
            let start = o * len + s * seg;
            let end = (start + seg).min((o + 1) * len).min(elements(kx, x.len()));
            let mut total = 0.0f32;
            for i in start..end {
                total += read(kx, x, i);
            }
            *y = scale * total;
        }
    }
}

/// The first stage of a long sum: `out[s · O + o] = Σ x[bases[o] + inner[r]]`
/// over the `seg` terms `r` of segment `s`, in order, for the `O` results
/// `o` of `bases`. Adjacent threads take adjacent results, whose terms are
/// often adjacent too (the sums over the rows of a matrix): the reads are
/// coalesced. `reduce_sum` then adds the segments of each result, in order.
#[kernel]
#[launch_bounds(256)]
pub fn reduce_sum_segments(kx: u32, x: &[u8], bases: &[u32], inner: &[u32], seg: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let j = idx.get();
    let results = bases.len();
    if results > 0 {
        if let Some(y) = out.get_mut(idx) {
            let (s, o) = (j / results, j % results);
            let base = bases[o] as usize;
            let start = s * seg as usize;
            let end = (start + seg as usize).min(inner.len());
            let mut total = 0.0f32;
            for r in start..end {
                total += read(kx, x, base + inner[r] as usize);
            }
            *y = total;
        }
    }
}

/// One AdamW step in place (`kapel_ops::Update::adamw`), one thread per
/// element: `g = grad_scale · grad`, then the moments, then the parameter,
/// as `AdamWStep::apply` computes them on the host.
#[kernel]
#[launch_bounds(256)]
pub fn adamw(
    mut p: DisjointSlice<f32>,
    mut m: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    kg: u32,
    g: &[u8],
    vg: View,
    beta1: f32,
    beta2: f32,
    keep: f32,
    step: f32,
    correction2: f32,
    eps: f32,
    grad_scale: f32,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < p.len() && i < m.len() && i < v.len() {
        let grad = read(kg, g, vg.at(i as u32) as usize) * grad_scale;
        // SAFETY: `i` is below the three lengths (checked above), and it is
        // this thread's own element.
        unsafe {
            let pm = p.as_mut_ptr().add(i);
            let mm = m.as_mut_ptr().add(i);
            let vm = v.as_mut_ptr().add(i);
            let first = beta1 * *mm + (1.0 - beta1) * grad;
            let second = beta2 * *vm + (1.0 - beta2) * (grad * grad);
            let denominator = (second / correction2).sqrt() + eps;
            *pm = keep * *pm - step * (first / denominator);
            *mm = first;
            *vm = second;
        }
    }
}

/// The code of `x` in the sorted map of 256 values `maps[base..base + 256]`:
/// the number of midpoints between neighbouring values below `x`, so the
/// nearest value, the lower at a tie (`kapel_ops::nearest_code`).
#[inline(always)]
fn nearest_code(maps: &[f32], base: usize, x: f32) -> u8 {
    let (mut lo, mut hi) = (0usize, 255usize);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (maps[base + mid] + maps[base + mid + 1]) * 0.5 < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// `x` within `[−1, 1]`.
#[inline(always)]
fn unit(x: f32) -> f32 {
    if x < -1.0 {
        -1.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// One AdamW step whose moments are kept in 8 bits
/// (`kapel_ops::Update::adamw8bit`), one block of 256 threads per block of
/// 256 elements, one thread per element: the moments read from their codes
/// (`maps[0..256]` signed, `maps[256..512]` unsigned) times their block's
/// scales, the step of `adamw`, then the block's largest magnitudes by
/// `block_max`, the new scales, and each moment written as its nearest
/// code, as `AdamWStep::apply_block8` computes them on the host.
#[kernel]
#[launch_bounds(256)]
pub fn adamw8bit(
    mut p: DisjointSlice<f32>,
    mut m: DisjointSlice<u8>,
    mut v: DisjointSlice<u8>,
    mut m_scale: DisjointSlice<f32>,
    mut v_scale: DisjointSlice<f32>,
    kg: u32,
    g: &[u8],
    vg: View,
    maps: &[f32],
    beta1: f32,
    beta2: f32,
    keep: f32,
    step: f32,
    correction2: f32,
    eps: f32,
    grad_scale: f32,
) {
    let block = thread::blockIdx_x() as usize;
    let i = block * 256 + thread::threadIdx_x() as usize;
    let here = i < p.len() && i < m.len() && i < v.len() && block < m_scale.len() && block < v_scale.len() && maps.len() == 512;
    let (mut first, mut second) = (0.0f32, 0.0f32);
    if here {
        let grad = read(kg, g, vg.at(i as u32) as usize) * grad_scale;
        // SAFETY: `i` and `block` are below the lengths (checked above);
        // element `i` is this thread's own, and the block's scales are only
        // read here, before the barriers of `block_max`.
        unsafe {
            let pm = p.as_mut_ptr().add(i);
            let mm = m.as_mut_ptr().add(i);
            let vm = v.as_mut_ptr().add(i);
            let old_m = maps[*mm as usize] * *m_scale.as_mut_ptr().add(block);
            let old_v = maps[256 + *vm as usize] * *v_scale.as_mut_ptr().add(block);
            first = beta1 * old_m + (1.0 - beta1) * grad;
            second = beta2 * old_v + (1.0 - beta2) * (grad * grad);
            let denominator = (second / correction2).sqrt() + eps;
            *pm = keep * *pm - step * (first / denominator);
        }
    }
    // Every thread of the block, in range or not, takes part.
    let top_m = block_max(first.abs());
    let top_v = block_max(second.abs());
    if here {
        let im = if top_m > 0.0 { 1.0 / top_m } else { 0.0 };
        let iv = if top_v > 0.0 { 1.0 / top_v } else { 0.0 };
        // SAFETY: as above; the block's scales are written by its first
        // thread only, after every thread has read them (the barriers of
        // `block_max`).
        unsafe {
            *m.as_mut_ptr().add(i) = nearest_code(maps, 0, unit(first * im));
            *v.as_mut_ptr().add(i) = nearest_code(maps, 256, unit(second * iv));
            if thread::threadIdx_x() == 0 {
                *m_scale.as_mut_ptr().add(block) = top_m;
                *v_scale.as_mut_ptr().add(block) = top_v;
            }
        }
    }
}

/// `out[i] = scale · g[owner[i]]`: each element receives its result's gradient.
#[kernel]
#[launch_bounds(256)]
pub fn spread(g: &[f32], owner: &[u32], scale: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = scale * g[owner[i] as usize];
    }
}

/// For each line along a dimension, its maximum and the position of the first
/// maximum. Line `l` starts at `starts[l]`, its `len` elements `step` apart.
#[kernel]
#[launch_bounds(256)]
pub fn line_max(x: &[f32], starts: &[u32], step: u32, len: u32, mut value: DisjointSlice<f32>, mut at: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < starts.len() && l < value.len() && l < at.len() {
        let start = starts[l];
        let mut best = start;
        for i in 1..len {
            let p = start + i * step;
            if x[p as usize] > x[best as usize] {
                best = p;
            }
        }
        // SAFETY: `l` is below both lengths (checked above), and it is this
        // thread's own index: no other thread writes slot `l`.
        unsafe {
            *value.as_mut_ptr().add(l) = x[best as usize];
            *at.as_mut_ptr().add(l) = best;
        }
    }
}

/// `out[at[l]] = g[l]` for every line; `out` starts at zero.
///
/// The positions in `at` must be distinct: the backend passes those that
/// `line_max` found, each inside its own line, and lines do not overlap.
#[kernel]
#[launch_bounds(256)]
pub fn place(g: &[f32], at: &[u32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < at.len() {
        let p = at[l] as usize;
        if p < out.len() {
            // SAFETY: `p` is in bounds (checked above), and no other thread
            // writes it: the positions in `at` are distinct (see above).
            unsafe { *out.as_mut_ptr().add(p) = g[l] };
        }
    }
}

/// Whether the line of `len` elements `step` apart from `start` lies within a
/// slice of `bound` elements, its positions computed in `u32` without
/// wrapping.
fn line_fits(start: u32, step: u32, len: u32, bound: usize) -> bool {
    if len == 0 {
        return true;
    }
    let last = start as usize + (len as usize - 1) * step as usize;
    last < bound && last <= u32::MAX as usize
}

/// Softmax (`log` = 0) or log-softmax (`log` = 1) of every line, one thread per
/// line, after subtracting the line's maximum.
#[kernel]
#[launch_bounds(256)]
pub fn softmax_lines(log: u32, x: &[f32], starts: &[u32], step: u32, len: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < starts.len() && line_fits(starts[l], step, len, out.len()) {
        let start = starts[l];
        let mut max = f32::NEG_INFINITY;
        for i in 0..len {
            let v = x[(start + i * step) as usize];
            if v > max {
                max = v;
            }
        }
        let mut total = 0.0f32;
        for i in 0..len {
            total += (x[(start + i * step) as usize] - max).exp();
        }
        let log_total = total.ln();
        for i in 0..len {
            let p = (start + i * step) as usize;
            let y = if log == 1 { x[p] - max - log_total } else { (x[p] - max).exp() / total };
            // SAFETY: `p` is on the line, which lies within `out` (checked
            // above). The lines of the backend's `starts` do not overlap, and
            // each is written by its own thread alone.
            unsafe { *out.as_mut_ptr().add(p) = y };
        }
    }
}

/// The backward pass of `softmax_lines`, from the input `x`:
/// softmax `∂x = y ⊙ (g − Σ g·y)`, log-softmax `∂x = g − softmax(x) · Σ g`.
#[kernel]
#[launch_bounds(256)]
pub fn softmax_lines_backward(
    log: u32,
    x: &[f32],
    g: &[f32],
    starts: &[u32],
    step: u32,
    len: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < starts.len() && line_fits(starts[l], step, len, out.len()) {
        let start = starts[l];
        let mut max = f32::NEG_INFINITY;
        for i in 0..len {
            let v = x[(start + i * step) as usize];
            if v > max {
                max = v;
            }
        }
        let mut total = 0.0f32;
        for i in 0..len {
            total += (x[(start + i * step) as usize] - max).exp();
        }
        let mut dot = 0.0f32;
        let mut g_total = 0.0f32;
        for i in 0..len {
            let p = (start + i * step) as usize;
            let y = (x[p] - max).exp() / total;
            dot += g[p] * y;
            g_total += g[p];
        }
        for i in 0..len {
            let p = (start + i * step) as usize;
            let y = (x[p] - max).exp() / total;
            let value = if log == 1 { g[p] - y * g_total } else { y * (g[p] - dot) };
            // SAFETY: as in `softmax_lines`: in bounds (checked above), and
            // the line is this thread's alone.
            unsafe { *out.as_mut_ptr().add(p) = value };
        }
    }
}

/// The sum of `v` over the 32 lanes of the warp, the same bits on every lane:
/// a butterfly whose every step adds two lanes' values, in either order.
fn warp_sum(v: f32) -> f32 {
    warp::reduce_sum_f32(v)
}

/// LayerNorm (`rms` = 0) or RMSNorm (`rms` = 1) of every row of `width`:
/// `y = x̂ · (offset + w) + b`, one warp per row. Lane `l` reads and writes
/// columns `l, l + 32, …`; the sums over the row add each lane's partial sum,
/// then the lanes by `warp_sum`.
#[kernel]
#[launch_bounds(256)]
pub fn norm_rows(rms: u32, x: &[f32], w: &[f32], b: &[f32], offset: f32, eps: f32, width: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let row = idx.get() / 32;
    let lane = idx.get() % 32;
    let width = width as usize;
    let start = row * width;
    // The same for every lane of the warp: whole warps go on or stop.
    if width > 0 && start + width <= x.len() && start + width <= out.len() {
        let n = width as f32;
        let mut mean = 0.0f32;
        if rms == 0 {
            let mut part = 0.0f32;
            let mut c = lane;
            while c < width {
                part += x[start + c];
                c += 32;
            }
            mean = warp_sum(part) / n;
        }
        let mut part = 0.0f32;
        let mut c = lane;
        while c < width {
            let v = x[start + c] - mean;
            part += v * v;
            c += 32;
        }
        let inv = 1.0 / (warp_sum(part) / n + eps).sqrt();
        let mut c = lane;
        while c < width {
            let hat = (x[start + c] - mean) * inv;
            // SAFETY: `start + c < start + width ≤ out.len()` (checked
            // above); row `row` belongs to this warp, column `c` to this lane.
            unsafe { *out.as_mut_ptr().add(start + c) = hat * (offset + w[c]) + b[c] };
            c += 32;
        }
    }
}

/// The backward pass of `norm_rows`, one warp per row as it: the gradient of
/// `x`, and for every row its contributions `g·x̂` and `g` to the gradients of
/// the weight and the bias, summed over rows afterwards by `reduce_sum_segments` and `reduce_sum`.
#[kernel]
#[launch_bounds(256)]
pub fn norm_rows_backward(
    rms: u32,
    kx: u32,
    x: &[u8],
    w: &[f32],
    kg: u32,
    g: &[u8],
    offset: f32,
    eps: f32,
    width: u32,
    ko: u32,
    mut gx: DisjointSlice<u8>,
    mut gw_rows: DisjointSlice<f32>,
    mut gb_rows: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let row = idx.get() / 32;
    let lane = idx.get() % 32;
    let width = width as usize;
    let start = row * width;
    let end = start + width;
    if width > 0 && end <= elements(kx, x.len()) && end <= elements(kg, g.len()) && end <= elements(ko, gx.len()) && end <= gw_rows.len() && end <= gb_rows.len() {
        let n = width as f32;
        let mut mean = 0.0f32;
        if rms == 0 {
            let mut part = 0.0f32;
            let mut c = lane;
            while c < width {
                part += read(kx, x, start + c);
                c += 32;
            }
            mean = warp_sum(part) / n;
        }
        let mut part = 0.0f32;
        let mut c = lane;
        while c < width {
            let v = read(kx, x, start + c) - mean;
            part += v * v;
            c += 32;
        }
        let inv = 1.0 / (warp_sum(part) / n + eps).sqrt();
        let (mut part_gh, mut part_ghh) = (0.0f32, 0.0f32);
        let mut c = lane;
        while c < width {
            let hat = (read(kx, x, start + c) - mean) * inv;
            let gh = read(kg, g, start + c) * (offset + w[c]);
            part_gh += gh;
            part_ghh += gh * hat;
            c += 32;
        }
        let mut mean_gh = warp_sum(part_gh) / n;
        let mean_ghh = warp_sum(part_ghh) / n;
        if rms == 1 {
            mean_gh = 0.0;
        }
        let mut c = lane;
        while c < width {
            let hat = (read(kx, x, start + c) - mean) * inv;
            let gh = read(kg, g, start + c) * (offset + w[c]);
            // SAFETY: `start + c < end`, within the three outputs (checked
            // above); row `row` belongs to this warp, column `c` to this lane.
            unsafe {
                write(ko, gx.as_mut_ptr(), start + c, inv * (gh - mean_gh - hat * mean_ghh));
                *gw_rows.as_mut_ptr().add(start + c) = read(kg, g, start + c) * hat;
                *gb_rows.as_mut_ptr().add(start + c) = read(kg, g, start + c);
            }
            c += 32;
        }
    }
}

/// The target of a cross-entropy row that counts in no loss: all bits set
/// (`kapel_ops::CrossEntropy::IGNORE`, or −1 as an `i32`).
const IGNORE: u32 = u32::MAX;

/// The maximum of `v` over the 256 threads of the block, the same on every
/// thread. Every thread of the block must call it, the same number of times.
#[inline(always)]
fn block_max(v: f32) -> f32 {
    static mut MAXIMA: SharedArray<f32, 8> = SharedArray::UNINIT;
    let warp_top = warp_max(v);
    let t = thread::threadIdx_x() as usize;
    thread::sync_threads();
    if t % 32 == 0 {
        // SAFETY: as in `block_sum`: each warp's lane 0 writes its own slot,
        // between two barriers.
        unsafe { MAXIMA[t / 32] = warp_top };
    }
    thread::sync_threads();
    let mut top = f32::NEG_INFINITY;
    let mut i = 0;
    while i < 8 {
        // SAFETY: written before the barrier above.
        let m = unsafe { MAXIMA[i] };
        if m > top {
            top = m;
        }
        i += 1;
    }
    top
}

/// The logarithm of `Σⱼ e^(x[j])` over the `vocab` values of `x` (float type
/// `kx`) from `start`, computed by the block of 256 threads: each keeps a
/// running maximum and its sum over columns `t, t + 256, …`, in one
/// reading, then the threads are combined. The same on every thread.
#[inline(always)]
fn row_log_sum_exp(kx: u32, x: &[u8], start: usize, vocab: usize, t: usize) -> f32 {
    let (mut m, mut s) = (f32::NEG_INFINITY, 0.0f32);
    let mut c = t;
    while c < vocab {
        let v = read(kx, x, start + c);
        if v > m {
            s = s * (m - v).exp() + 1.0;
            m = v;
        } else {
            s += (v - m).exp();
        }
        c += 256;
    }
    let top = block_max(m);
    let part = if m == f32::NEG_INFINITY { 0.0 } else { s * (m - top).exp() };
    top + block_sum(part).ln()
}

/// The cross-entropy of every row of logits (`kapel_ops::CrossEntropy`), one
/// block of 256 threads per row: `out[n] = ln Σⱼ e^(x[n, j]) − x[n, t[n]]`, 0
/// for a target `IGNORE`. Row `n` holds the `vocab` values of `x` from
/// `offset + n · stride`. The targets are checked by the backend.
#[kernel]
#[launch_bounds(256)]
pub fn cross_entropy_rows(kx: u32, x: &[u8], offset: u32, stride: u32, targets: &[u32], vocab: u32, mut out: DisjointSlice<f32>) {
    let row = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    // The same for every thread of the block: whole blocks go on or stop,
    // so that every thread reaches the barriers.
    if row < targets.len() && row < out.len() {
        let vocab = vocab as usize;
        let start = offset as usize + row * stride as usize;
        let lse = row_log_sum_exp(kx, x, start, vocab, t);
        let target = targets[row];
        if t == 0 {
            let y = if target == IGNORE { 0.0 } else { lse - read(kx, x, start + target as usize) };
            // SAFETY: `row < out.len()` (checked above); thread 0 of the
            // row's block alone writes it.
            unsafe { *out.as_mut_ptr().add(row) = y };
        }
    }
}

/// The backward pass of `cross_entropy_rows`, one block per row:
/// `out[n · width + j] = g[n] · (softmax(x[n])[j] − [j = t[n]])` in float
/// type `ko` for `j < vocab`, and 0 for `vocab ≤ j < width` (the padding of
/// a row) and on the rows whose target is `IGNORE`.
#[kernel]
#[launch_bounds(256)]
pub fn cross_entropy_rows_backward(
    kx: u32,
    x: &[u8],
    offset: u32,
    stride: u32,
    targets: &[u32],
    g: &[f32],
    vocab: u32,
    width: u32,
    ko: u32,
    mut out: DisjointSlice<u8>,
) {
    let row = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    let (vocab, width) = (vocab as usize, width as usize);
    if row < targets.len() && row < g.len() && (row + 1) * width <= elements(ko, out.len()) {
        let target = targets[row];
        let start = offset as usize + row * stride as usize;
        // The same for the whole block: every thread reaches the barriers.
        let lse = if target == IGNORE { 0.0 } else { row_log_sum_exp(kx, x, start, vocab, t) };
        let scale = g[row];
        let mut c = t;
        while c < width {
            let value = if target == IGNORE || c >= vocab {
                0.0
            } else {
                let p = (read(kx, x, start + c) - lse).exp();
                scale * (p - if c == target as usize { 1.0 } else { 0.0 })
            };
            // SAFETY: `row · width + c` is below the output's elements
            // (checked above); the row is this block's, the column this
            // thread's.
            unsafe { write(ko, out.as_mut_ptr(), row * width + c, value) };
            c += 256;
        }
    }
}

/// The angles of rotary embeddings, in `f64`: for position entry `e` and
/// frequency `j`, `out[2(e·half + j)..][..2] = (s·cos θ, s·sin θ)`, with
/// `θ = direction · p[e] · ω[j]`. `constants` holds the bits of `s`, then of
/// each `ω`. Positions are `u32`, or `i32` when `signed` is 1.
#[kernel]
#[launch_bounds(256)]
pub fn rope_table(positions: &[u32], signed: u32, constants: &[u64], direction: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let t = idx.get();
    let half = constants.len() - 1;
    if half > 0 && t < positions.len() * half && 2 * t + 1 < out.len() {
        let (e, j) = (t / half, t % half);
        let p = if signed == 1 { positions[e] as i32 as f64 } else { positions[e] as f64 };
        let factor = f64::from_bits(constants[0]);
        let angle = direction as f64 * p * f64::from_bits(constants[1 + j]);
        let (sin, cos) = (angle.sin(), angle.cos());
        // SAFETY: `2t + 1 < out.len()` (checked above), and slots `2t`,
        // `2t + 1` belong to thread `t` alone.
        unsafe {
            *out.as_mut_ptr().add(2 * t) = (factor * cos) as f32;
            *out.as_mut_ptr().add(2 * t + 1) = (factor * sin) as f32;
        }
    }
}

/// Rotary embeddings: each element of `x [vectors, d]` is turned with its pair
/// by the angles of its vector's position entry. `table` holds, per entry and
/// frequency, `(s·cos, s·sin)`; `half_pairs` is 1 for pairs `(j, j + r/2)`, 0 for
/// `(2j, 2j + 1)`. Components from `r` on pass unchanged.
#[kernel]
#[launch_bounds(256)]
pub fn rope(x: &[f32], table: &[f32], entry: &[u32], d: u32, r: u32, half_pairs: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let flat = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let (vector, c) = (flat / d as usize, flat as u32 % d);
        let base = vector * d as usize;
        if c >= r {
            *y = x[flat];
        } else {
            let half = r / 2;
            let (j, first, partner) = if half_pairs == 1 {
                if c < half { (c, true, c + half) } else { (c - half, false, c - half) }
            } else if c % 2 == 0 {
                (c / 2, true, c + 1)
            } else {
                (c / 2, false, c - 1)
            };
            let t = ((entry[vector] * half + j) * 2) as usize;
            let (cos, sin) = (table[t], table[t + 1]);
            let (own, other) = (x[flat], x[base + partner as usize]);
            *y = if first { own * cos - other * sin } else { own * cos + other * sin };
        }
    }
}

/// `out[i] = src[rows[i / width] · width + i % width]`.
#[kernel]
#[launch_bounds(256)]
pub fn gather_rows(src: &[f32], rows: &[u32], width: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let flat = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let width = width as usize;
        *y = src[rows[flat / width] as usize * width + flat % width];
    }
}

/// `gather_rows` for 16-bit elements, copied bit for bit.
#[kernel]
#[launch_bounds(256)]
pub fn gather_rows_16(src: &[u16], rows: &[u32], width: u32, mut out: DisjointSlice<u16>) {
    let idx = thread::index_1d();
    let flat = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let width = width as usize;
        *y = src[rows[flat / width] as usize * width + flat % width];
    }
}

/// `out[r, c] = Σ src[s, c]` over the sources `s` of row `r`, listed in
/// `sources[offsets[r]..offsets[r + 1]]` in their order of appearance.
#[kernel]
#[launch_bounds(256)]
pub fn scatter_rows(src: &[f32], offsets: &[u32], sources: &[u32], width: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let flat = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let width = width as usize;
        let (r, c) = (flat / width, flat % width);
        let mut total = 0.0f32;
        for s in offsets[r]..offsets[r + 1] {
            total += src[sources[s as usize] as usize * width + c];
        }
        *y = total;
    }
}

/// The 16-bit format of float code `kind` (`float::BF16` or `float::F16`).
#[inline(always)]
fn half_of(kind: u32) -> u32 {
    if kind == float::BF16 { half::BF16 } else { half::F16 }
}

/// The value of an 8-bit float: E4M3 without infinities (`e4m3`), else E5M2.
#[inline(always)]
fn widen_8(e4m3: bool, byte: u8) -> f32 {
    let (m, bias) = if e4m3 { (3u32, 7i32) } else { (2u32, 15i32) };
    let bits = byte as u32;
    let negative = bits & 0x80 != 0;
    let exponent = (bits >> m) & ((1 << (7 - m)) - 1);
    let mantissa = bits & ((1 << m) - 1);
    let top = (1u32 << (7 - m)) - 1;
    let magnitude = if e4m3 && exponent == top && mantissa == (1 << m) - 1 {
        f32::NAN
    } else if !e4m3 && exponent == top {
        if mantissa == 0 { f32::INFINITY } else { f32::NAN }
    } else if exponent == 0 {
        // Subnormal: `mantissa · 2^(1 − bias − m)`, exact.
        mantissa as f32 * f32::from_bits(((1 - bias - m as i32 + 127) as u32) << 23)
    } else {
        f32::from_bits((((exponent as i32 - bias + 127) as u32) << 23) | (mantissa << (23 - m)))
    };
    if negative { -magnitude } else { magnitude }
}

/// The 8-bit float nearest to `x`, ties to even, as the host's conversion
/// gives it: beyond the largest finite value, infinity in E5M2 and NaN in
/// E4M3, which has no infinity; every NaN to the positive one.
#[inline(always)]
fn narrow_8(e4m3: bool, x: f32) -> u8 {
    let (m, bias) = if e4m3 { (3i32, 7i32) } else { (2i32, 15i32) };
    let top = (1u32 << (7 - m)) - 1;
    let nan = if e4m3 { 0x7fu32 } else { (top << m) | (1 << (m - 1)) };
    let max_finite = if e4m3 { (top << m) | ((1 << m) - 2) } else { ((top - 1) << m) | ((1 << m) - 1) };
    let overflow = if e4m3 { nan } else { top << m };
    if x.is_nan() {
        return nan as u8;
    }
    let bits = x.to_bits();
    let sign = (bits >> 24) & 0x80;
    let a = f32::from_bits(bits & 0x7fff_ffff);
    if a == f32::INFINITY {
        return (sign | overflow) as u8;
    }
    if a == 0.0 {
        return sign as u8;
    }
    let min_normal = 1 - bias;
    let exponent = ((a.to_bits() >> 23) as i32) - 127;
    if exponent < min_normal {
        // In units of the smallest subnormal: `a · 2^(m − min_normal)` is
        // exact (a power of two times `a`), below `2^m`; rounding up to `2^m`
        // gives the smallest normal, the same integer.
        let units = a * f32::from_bits(((m - min_normal + 127) as u32) << 23);
        return (sign | round_ties_even(units) as u32) as u8;
    }
    let mut significand = round_ties_even(a * f32::from_bits(((m - exponent + 127) as u32) << 23)) as u32;
    let mut exponent = exponent;
    if significand == 1 << (m + 1) {
        significand >>= 1;
        exponent += 1;
    }
    let field = (exponent + bias) as u32;
    let pattern = (field << m) | (significand & ((1 << m) - 1));
    if field > top || pattern > max_finite {
        return (sign | overflow) as u8;
    }
    (sign | pattern) as u8
}

/// Element `i` of `x`, bytes holding floats of type `kind`, widened to `f32`;
/// NaN past the end.
#[inline(always)]
fn read(kind: u32, x: &[u8], i: usize) -> f32 {
    if kind == float::F32 {
        if 4 * i + 4 <= x.len() {
            // SAFETY: in bounds (checked above); the backend passes slices of
            // `f32` at 4-byte aligned addresses.
            unsafe { *x.as_ptr().cast::<f32>().add(i) }
        } else {
            f32::NAN
        }
    } else if kind >= float::F8E4M3 {
        if i < x.len() { widen_8(kind == float::F8E4M3, x[i]) } else { f32::NAN }
    } else if 2 * i + 2 <= x.len() {
        // SAFETY: in bounds (checked above), 2-byte aligned (as above).
        widen(half_of(kind), unsafe { *x.as_ptr().cast::<u16>().add(i) })
    } else {
        f32::NAN
    }
}

/// The number of elements of type `kind` that `bytes` bytes hold.
#[inline(always)]
fn elements(kind: u32, bytes: usize) -> usize {
    if kind == float::F32 {
        bytes / 4
    } else if kind >= float::F8E4M3 {
        bytes
    } else {
        bytes / 2
    }
}

/// Writes `value`, rounded to type `kind`, as element `i` of `out`.
///
/// # Safety
///
/// `i < elements(kind, len)` for the length `len` of the bytes at `out`, which
/// are aligned for `kind`, and no other thread writes element `i`.
#[inline(always)]
unsafe fn write(kind: u32, out: *mut u8, i: usize, value: f32) {
    // SAFETY: the caller's contract.
    unsafe {
        if kind == float::F32 {
            *out.cast::<f32>().add(i) = value;
        } else if kind >= float::F8E4M3 {
            *out.add(i) = narrow_8(kind == float::F8E4M3, value);
        } else {
            *out.cast::<u16>().add(i) = narrow(half_of(kind), value);
        }
    }
}

/// `out[i] = f(x[view.at(i)])`, `x` of float type `kx` and `out` of `ko`,
/// computed in `f32`.
#[kernel]
#[launch_bounds(256)]
pub fn unary_any(op: u32, kx: u32, x: &[u8], vx: View, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < elements(ko, out.len()) {
        let y = unary_value(op, read(kx, x, vx.at(i as u32) as usize));
        // SAFETY: `i` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), i, y) };
    }
}

/// `out[i] = scale · x[view.at(i)] + bias`, as `unary_any`.
#[kernel]
#[launch_bounds(256)]
pub fn affine_any(scale: f32, bias: f32, kx: u32, x: &[u8], vx: View, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < elements(ko, out.len()) {
        let y = scale * read(kx, x, vx.at(i as u32) as usize) + bias;
        // SAFETY: `i` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), i, y) };
    }
}

/// `out[i] = f(a, b)`, each read through its view broadcast to the output and
/// of its own float type, computed in `f32`.
#[kernel]
#[launch_bounds(256)]
pub fn binary_any(op: u32, ka: u32, a: &[u8], va: View, kb: u32, b: &[u8], vb: View, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < elements(ko, out.len()) {
        let p = i as u32;
        let y = binary_value(op, read(ka, a, va.at(p) as usize), read(kb, b, vb.at(p) as usize)).0;
        // SAFETY: `i` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), i, y) };
    }
}

/// The float types of a normalization's operands: `x`, the weight, the bias,
/// the output, one byte each, from the lowest.
#[inline(always)]
fn kinds(packed: u32) -> (u32, u32, u32, u32) {
    (packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff, packed >> 24)
}

/// LayerNorm (`rms` = 0) or RMSNorm (`rms` = 1) of every row of `width`, as
/// `norm_rows`, for operands of any float type (`types`, see `kinds`): one
/// warp per row. Lane `l` reads and writes columns `l, l + 32, …`; the sums
/// add each lane's partial sum, then the lanes by a fixed butterfly.
#[kernel]
#[launch_bounds(256)]
pub fn norm_any(rms: u32, types: u32, x: &[u8], w: &[u8], b: &[u8], offset: f32, eps: f32, width: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let row = idx.get() / 32;
    let lane = idx.get() % 32;
    let (kx, kw, kb, ko) = kinds(types);
    let width = width as usize;
    let start = row * width;
    // The same for every lane of the warp: whole warps go on or stop.
    if width > 0 && start + width <= elements(kx, x.len()) && start + width <= elements(ko, out.len()) {
        let n = width as f32;
        let mut mean = 0.0f32;
        if rms == 0 {
            let mut part = 0.0f32;
            let mut c = lane;
            while c < width {
                part += read(kx, x, start + c);
                c += 32;
            }
            mean = warp_sum(part) / n;
        }
        let mut part = 0.0f32;
        let mut c = lane;
        while c < width {
            let v = read(kx, x, start + c) - mean;
            part += v * v;
            c += 32;
        }
        let inv = 1.0 / (warp_sum(part) / n + eps).sqrt();
        let mut c = lane;
        while c < width {
            let hat = (read(kx, x, start + c) - mean) * inv;
            let y = hat * (offset + read(kw, w, c)) + read(kb, b, c);
            // SAFETY: `start + c < start + width`, within `out` (checked
            // above); row `row` belongs to this warp, column `c` to this lane.
            unsafe { write(ko, out.as_mut_ptr(), start + c, y) };
            c += 32;
        }
    }
}

/// The sum of `v` over the 256 threads of the block, the same on every
/// thread: each warp's sum by a fixed butterfly, then the eight warps' sums in
/// order. Every thread of the block must call it, the same number of times.
#[inline(always)]
fn block_sum(v: f32) -> f32 {
    static mut SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
    let warp_total = warp_sum(v);
    let t = thread::threadIdx_x() as usize;
    thread::sync_threads();
    if t % 32 == 0 {
        // SAFETY: each warp's lane 0 writes its own slot; nothing reads
        // before the barrier below, and the barrier above ends the reads of
        // a previous call.
        unsafe { SUMS[t / 32] = warp_total };
    }
    thread::sync_threads();
    let mut total = 0.0f32;
    let mut i = 0;
    while i < 8 {
        // SAFETY: written before the barrier above, not written again before
        // the next call's first barrier.
        total += unsafe { SUMS[i] };
        i += 1;
    }
    total
}

/// The columns of a row that one thread of `norm_block_any` keeps in
/// registers: rows of up to `32 · 256 = 8192` are read once.
const NORM_PER_THREAD: usize = 32;

/// `read` for a type known at compile time.
#[inline(always)]
fn read_as<const K: u32>(x: &[u8], i: usize) -> f32 {
    read(K, x, i)
}

/// `write` for a type known at compile time.
///
/// # Safety
///
/// As [`write`].
#[inline(always)]
unsafe fn write_as<const K: u32>(out: *mut u8, i: usize, value: f32) {
    // SAFETY: the caller's contract.
    unsafe { write(K, out, i, value) }
}

/// The body of `norm_block_any` for operands of type `K`: one
/// specialization per type, so that the unrolled loops hold the conversions
/// of one type only.
#[device]
fn norm_block<const K: u32>(rms: bool, bias: bool, x: &[u8], w: &[u8], b: &[u8], offset: f32, eps: f32, width: usize, out: *mut u8) {
    let row = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    let start = row * width;
    let n = width as f32;
    let mut values = [0.0f32; NORM_PER_THREAD];
    let mut i = 0;
    #[unroll]
    while i < NORM_PER_THREAD {
        let c = t + 256 * i;
        if c < width {
            values[i] = read_as::<K>(x, start + c);
        }
        i += 1;
    }
    let mut mean = 0.0f32;
    if !rms {
        let mut part = 0.0f32;
        let mut i = 0;
        #[unroll]
        while i < NORM_PER_THREAD {
            part += values[i];
            i += 1;
        }
        mean = block_sum(part) / n;
    }
    let mut part = 0.0f32;
    let mut i = 0;
    #[unroll]
    while i < NORM_PER_THREAD {
        if t + 256 * i < width {
            let v = values[i] - mean;
            part += v * v;
        }
        i += 1;
    }
    let inv = 1.0 / (block_sum(part) / n + eps).sqrt();
    let mut i = 0;
    #[unroll]
    while i < NORM_PER_THREAD {
        let c = t + 256 * i;
        if c < width {
            let hat = (values[i] - mean) * inv;
            let shift = if bias { read_as::<K>(b, c) } else { 0.0 };
            let y = hat * (offset + read_as::<K>(w, c)) + shift;
            // SAFETY: the caller checked `start + width` against the output;
            // row `row` belongs to this block, column `c` to this thread.
            unsafe { write_as::<K>(out, start + c, y) };
        }
        i += 1;
    }
}

/// The normalization of `norm_any` with one block of 256 threads per row:
/// for wide rows, as in decoding, where one warp per row would read the row
/// in too many dependent steps. Thread `t` reads columns `t, t + 256, …`
/// once, keeps them in registers, and writes them normalized; the sums add
/// each thread's partial sum, then the threads by `block_sum`.
///
/// For rows of at most 8192 columns, and `x`, the weight, the bias (bit 1 of
/// `flags`; bit 0 is RMSNorm) and the output of one type, `f32`, `bf16` or
/// `f16`. The backend sends the other cases to `norm_any`.
#[kernel]
#[launch_bounds(256)]
pub fn norm_block_any(flags: u32, kind: u32, x: &[u8], w: &[u8], b: &[u8], offset: f32, eps: f32, width: u32, mut out: DisjointSlice<u8>) {
    let row = thread::blockIdx_x() as usize;
    let width = width as usize;
    let start = row * width;
    let (rms, bias) = (flags & 1 == 1, flags & 2 == 2);
    // The same for every thread of the block: whole blocks go on or stop,
    // so that every thread reaches the barriers of `block_sum`.
    if width > 0 && width <= NORM_PER_THREAD * 256 && start + width <= elements(kind, x.len()) && start + width <= elements(kind, out.len()) && width <= elements(kind, w.len()) && (!bias || width <= elements(kind, b.len())) {
        let o = out.as_mut_ptr();
        match kind {
            float::F32 => norm_block::<{ float::F32 }>(rms, bias, x, w, b, offset, eps, width, o),
            float::BF16 => norm_block::<{ float::BF16 }>(rms, bias, x, w, b, offset, eps, width, o),
            float::F16 => norm_block::<{ float::F16 }>(rms, bias, x, w, b, offset, eps, width, o),
            _ => {}
        }
    }
}

/// Rotary embeddings as `rope`, for `x` of float type `kx` read through its
/// view (the output's shape, `[vectors, d]` in row-major order), and `out` of
/// type `ko`.
#[kernel]
#[launch_bounds(256)]
pub fn rope_any(kx: u32, x: &[u8], vx: View, table: &[f32], entry: &[u32], d: u32, r: u32, half_pairs: u32, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let flat = idx.get();
    if flat < elements(ko, out.len()) {
        let (vector, c) = (flat / d as usize, flat as u32 % d);
        let own = read(kx, x, vx.at(flat as u32) as usize);
        let y = if c >= r {
            own
        } else {
            let half = r / 2;
            let (j, first, partner) = if half_pairs == 1 {
                if c < half { (c, true, c + half) } else { (c - half, false, c - half) }
            } else if c % 2 == 0 {
                (c / 2, true, c + 1)
            } else {
                (c / 2, false, c - 1)
            };
            let t = ((entry[vector] * half + j) * 2) as usize;
            let (cos, sin) = (table[t], table[t + 1]);
            let other = read(kx, x, vx.at((vector * d as usize) as u32 + partner) as usize);
            if first { own * cos - other * sin } else { own * cos + other * sin }
        };
        // SAFETY: `flat` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), flat, y) };
    }
}

/// `table[index[r] · width + c] = rows[view.at(r · width + c)]`, converted
/// from float type `kr` to `kt`: rows written in place into a table, as a
/// cache receives the keys and values of new positions. The rows of `index`
/// are distinct (the backend checks them), so that no two threads write the
/// same place.
#[kernel]
#[launch_bounds(256)]
pub fn write_rows_any(kr: u32, rows: &[u8], vr: View, index: &[u32], width: u32, kt: u32, mut table: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let width = width as usize;
    if width > 0 && i < index.len() * width {
        let (r, c) = (i / width, i % width);
        let at = index[r] as usize * width + c;
        if at < elements(kt, table.len()) {
            let value = read(kr, rows, vr.at(i as u32) as usize);
            // SAFETY: `at` is in bounds (checked above); the rows of `index`
            // are distinct, so no other thread writes `at`.
            unsafe { write(kt, table.as_mut_ptr(), at, value) };
        }
    }
}

/// The maximum of `v` over the 32 lanes of the warp, on every lane.
#[inline(always)]
fn warp_max(v: f32) -> f32 {
    let mut m = v;
    let mut step = 16;
    while step > 0 {
        let other = warp::shuffle_xor_f32(m, step);
        if other > m {
            m = other;
        }
        step /= 2;
    }
    m
}

/// Attention over a cache kept by pages (`kapel_ops::PagedAttention`), for
/// the query rows `which`: one block of 256 threads per query row, key head,
/// group of at most 8 query heads of that key head, and chunk of at most 256
/// positions. With one chunk (`p.chunks` = 1) the block writes its rows of
/// `out`; with several, it writes into `partial`, per query row, head and
/// chunk, its unnormalized sums, the maximum of its scores and the sum of
/// their exponentials, which `paged_combine` adds in chunk order.
///
/// In a block, warp `w` computes the scores of positions `w, w + 8, …` of
/// the chunk, its lanes splitting the key's width; warp `h` then turns head
/// `h`'s scores into weights; thread `t` adds the values of column `t mod
/// Dv` over a share of the positions, and the shares are added in order.
/// Every sum is in an order fixed by the shapes: the same bits on every run.
#[kernel]
#[launch_bounds(256)]
pub fn paged_attention(
    p: Paged,
    q: &[u8],
    keys: &[u8],
    values: &[u8],
    pages: &[u32],
    rows: &[u32],
    which: &[u32],
    mut out: DisjointSlice<u8>,
    mut partial: DisjointSlice<f32>,
) {
    static mut Q: SharedArray<f32, 2048> = SharedArray::UNINIT;
    static mut S: SharedArray<f32, 2048> = SharedArray::UNINIT;
    static mut A: SharedArray<f32, 2048> = SharedArray::UNINIT;
    static mut STATS: SharedArray<f32, 16> = SharedArray::UNINIT;
    static mut LINES: SharedArray<u32, 256> = SharedArray::UNINIT;
    let t = thread::threadIdx_x() as usize;
    let (warp_id, lane) = (t / 32, t % 32);
    let block = thread::blockIdx_x() as usize;
    let (kq, kc, ko) = (p.kinds & 0xff, (p.kinds >> 8) & 0xff, p.kinds >> 16);
    let (hq, hkv, d, dv) = (p.heads as usize, p.kv_heads as usize, p.d as usize, p.dv as usize);
    let group = hq / hkv.max(1);
    let subs = group.div_ceil(PAGED_HEADS as usize);
    let (chunk, chunks) = (p.chunk as usize, p.chunks.max(1) as usize);
    let c = block % chunks;
    let rest = block / chunks;
    let sub = rest % subs.max(1);
    let rest = rest / subs.max(1);
    let g = rest % hkv.max(1);
    let qi = rest / hkv.max(1);
    // Whole blocks go on or stop: every thread reaches the same barriers.
    let shapes_fit = hkv > 0
        && group * hkv == hq
        && d > 0
        && d <= PAGED_WIDTH as usize
        && dv > 0
        && dv <= PAGED_WIDTH as usize
        && chunk > 0
        && chunk <= PAGED_CHUNK as usize
        && p.page > 0
        && qi < which.len();
    if shapes_fit {
        let r = which[qi] as usize;
        let s = rows[2 * r] as usize;
        let count = rows[2 * r + 1] as usize;
        let first = if p.window > 0 && count > p.window as usize { count - p.window as usize } else { 0 };
        let lo = if first > c * chunk { first } else { c * chunk };
        let hi = if count < (c + 1) * chunk { count } else { (c + 1) * chunk };
        let n = if hi > lo { hi - lo } else { 0 };
        let h0 = g * group + sub * PAGED_HEADS as usize;
        let nh = if group - sub * (PAGED_HEADS as usize) < PAGED_HEADS as usize { group - sub * PAGED_HEADS as usize } else { PAGED_HEADS as usize };
        let page = p.page as usize;
        // The lines of the chunk's positions, and the queries, scaled, in
        // shared memory.
        if t < n {
            let j = lo + t;
            let at = pages[s * p.per_sequence as usize + j / page] as usize * page + j % page;
            // SAFETY: `t < n ≤ 256`; each thread writes its own slot, read
            // after the barrier below.
            unsafe { LINES[t] = at as u32 };
        }
        let mut i = t;
        while i < nh * d {
            let (hh, col) = (i / d, i % d);
            let at = r * p.q_row as usize + (h0 + hh) * p.q_head as usize + col;
            // SAFETY: `i < nh · d ≤ 8 · 256`; each thread writes its own
            // slots, read after the barrier below.
            unsafe { Q[i] = p.scale * read(kq, q, at) };
            i += 256;
        }
        thread::sync_threads();
        // The scores of the chunk's positions, a warp per position.
        let mut j = lo + warp_id;
        while j < hi {
            // SAFETY: written before the barrier above.
            let base = (unsafe { LINES[j - lo] } as usize * hkv + g) * d;
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut col = lane;
            while col < d {
                let k = read(kc, keys, base + col);
                // SAFETY: the queries' slots, written before the barrier.
                unsafe {
                    s0 = Q[col].mul_add(k, s0);
                    if nh > 1 { s1 = Q[d + col].mul_add(k, s1); }
                    if nh > 2 { s2 = Q[2 * d + col].mul_add(k, s2); }
                    if nh > 3 { s3 = Q[3 * d + col].mul_add(k, s3); }
                    if nh > 4 { s4 = Q[4 * d + col].mul_add(k, s4); }
                    if nh > 5 { s5 = Q[5 * d + col].mul_add(k, s5); }
                    if nh > 6 { s6 = Q[6 * d + col].mul_add(k, s6); }
                    if nh > 7 { s7 = Q[7 * d + col].mul_add(k, s7); }
                }
                col += 32;
            }
            // Only the heads of the block: `nh` is the same for every lane.
            let cap = |v: f32| if p.softcap > 0.0 { p.softcap * (v / p.softcap).tanh() } else { v };
            let slot = j - lo;
            // SAFETY (every write): slot `hh · 256 + (j − lo)` belongs to
            // this warp's position `j`, written by its lane 0, read after the
            // barrier below.
            let r0 = warp_sum(s0);
            if lane == 0 {
                unsafe { S[slot] = cap(r0) };
            }
            if nh > 1 {
                let r = warp_sum(s1);
                if lane == 0 {
                    unsafe { S[PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 2 {
                let r = warp_sum(s2);
                if lane == 0 {
                    unsafe { S[2 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 3 {
                let r = warp_sum(s3);
                if lane == 0 {
                    unsafe { S[3 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 4 {
                let r = warp_sum(s4);
                if lane == 0 {
                    unsafe { S[4 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 5 {
                let r = warp_sum(s5);
                if lane == 0 {
                    unsafe { S[5 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 6 {
                let r = warp_sum(s6);
                if lane == 0 {
                    unsafe { S[6 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            if nh > 7 {
                let r = warp_sum(s7);
                if lane == 0 {
                    unsafe { S[7 * PAGED_CHUNK as usize + slot] = cap(r) };
                }
            }
            j += 8;
        }
        thread::sync_threads();
        // Weights and statistics, a warp per head.
        if warp_id < nh {
            let row = warp_id * PAGED_CHUNK as usize;
            let mut m = f32::NEG_INFINITY;
            let mut i = lane;
            while i < n {
                // SAFETY: written before the barrier above.
                let v = unsafe { S[row + i] };
                if v > m {
                    m = v;
                }
                i += 32;
            }
            let m = warp_max(m);
            let mut l = 0.0f32;
            let mut i = lane;
            while i < n {
                // SAFETY: each lane rewrites its own positions of this row.
                unsafe {
                    let w = (S[row + i] - m).exp();
                    S[row + i] = w;
                    l += w;
                }
                i += 32;
            }
            let l = warp_sum(l);
            if lane == 0 {
                // SAFETY: slots of this warp's head, read after the barrier.
                unsafe {
                    STATS[warp_id] = m;
                    STATS[8 + warp_id] = l;
                }
            }
        }
        thread::sync_threads();
        // The weighted values: thread `t` adds column `t mod Dv` over the
        // positions `share, share + shares, …` of the chunk.
        let shares = 256 / dv;
        let (col, share) = (t % dv, t / dv);
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut a4, mut a5, mut a6, mut a7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        if share < shares {
            // SAFETY (the reads of `LINES` and `S`): written before the
            // barriers above.
            let value = |i: usize| read(kc, values, (unsafe { LINES[i] } as usize * hkv + g) * dv + col);
            let mut add = |i: usize, v: f32| unsafe {
                let w = |hh: usize| S[hh * PAGED_CHUNK as usize + i];
                a0 = w(0).mul_add(v, a0);
                if nh > 1 { a1 = w(1).mul_add(v, a1); }
                if nh > 2 { a2 = w(2).mul_add(v, a2); }
                if nh > 3 { a3 = w(3).mul_add(v, a3); }
                if nh > 4 { a4 = w(4).mul_add(v, a4); }
                if nh > 5 { a5 = w(5).mul_add(v, a5); }
                if nh > 6 { a6 = w(6).mul_add(v, a6); }
                if nh > 7 { a7 = w(7).mul_add(v, a7); }
            };
            // Four loads in flight, then their sums in order.
            let mut i = share;
            while i + 3 * shares < n {
                let (v0, v1) = (value(i), value(i + shares));
                let (v2, v3) = (value(i + 2 * shares), value(i + 3 * shares));
                add(i, v0);
                add(i + shares, v1);
                add(i + 2 * shares, v2);
                add(i + 3 * shares, v3);
                i += 4 * shares;
            }
            while i < n {
                add(i, value(i));
                i += shares;
            }
            // SAFETY (every write): the slot of this thread's share, head
            // and column, read after the barrier below.
            let slot = |hh: usize| (share * PAGED_HEADS as usize + hh) * dv + col;
            unsafe {
                A[slot(0)] = a0;
                if nh > 1 { A[slot(1)] = a1; }
                if nh > 2 { A[slot(2)] = a2; }
                if nh > 3 { A[slot(3)] = a3; }
                if nh > 4 { A[slot(4)] = a4; }
                if nh > 5 { A[slot(5)] = a5; }
                if nh > 6 { A[slot(6)] = a6; }
                if nh > 7 { A[slot(7)] = a7; }
            }
        }
        thread::sync_threads();
        if share == 0 {
            let mut hh = 0;
            while hh < nh {
                let mut total = 0.0f32;
                let mut k = 0;
                while k < shares {
                    // SAFETY: written before the barrier above.
                    total += unsafe { A[(k * PAGED_HEADS as usize + hh) * dv + col] };
                    k += 1;
                }
                // SAFETY: written before the barrier above the values' sums.
                let (m, l) = unsafe { (STATS[hh], STATS[8 + hh]) };
                let h = h0 + hh;
                if chunks == 1 {
                    let at = (r * hq + h) * dv + col;
                    let y = if l > 0.0 { total / l } else { 0.0 };
                    if at < elements(ko, out.len()) {
                        // SAFETY: in bounds (checked above); row `r`, head
                        // `h`, column `col` belong to this block and thread.
                        unsafe { write(ko, out.as_mut_ptr(), at, y) };
                    }
                } else {
                    let at = ((qi * hq + h) * chunks + c) * (dv + 2);
                    if at + dv + 2 <= partial.len() {
                        // SAFETY: in bounds (checked above); the slots of
                        // query `qi`, head `h` and chunk `c` belong to this
                        // block, column `col` to this thread, the statistics
                        // to the thread of column 0.
                        unsafe {
                            let base = partial.as_mut_ptr().add(at);
                            *base.add(col) = total;
                            if col == 0 {
                                *base.add(dv) = m;
                                *base.add(dv + 1) = l;
                            }
                        }
                    }
                }
                hh += 1;
            }
        }
    }
}

/// The chunks of `paged_attention` added: for query row `which[qi]`, head
/// `h` and column `col` (one thread each), `y = Σ_c e^(m_c − m) · acc_c / Σ_c
/// e^(m_c − m) · l_c`, the chunks in order, `m` their largest maximum; zeros
/// for a query that saw nothing.
#[kernel]
#[launch_bounds(256)]
pub fn paged_combine(p: Paged, which: &[u32], partial: &[f32], mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let (hq, dv, chunks) = (p.heads as usize, p.dv as usize, p.chunks as usize);
    let ko = p.kinds >> 16;
    if dv > 0 && chunks > 0 && i < which.len() * hq * dv {
        let (qi, h, col) = (i / (hq * dv), (i / dv) % hq, i % dv);
        let base = (qi * hq + h) * chunks * (dv + 2);
        let mut m = f32::NEG_INFINITY;
        let mut c = 0;
        while c < chunks {
            let at = base + c * (dv + 2);
            if partial[at + dv + 1] > 0.0 && partial[at + dv] > m {
                m = partial[at + dv];
            }
            c += 1;
        }
        let (mut total, mut l) = (0.0f32, 0.0f32);
        let mut c = 0;
        while c < chunks {
            let at = base + c * (dv + 2);
            let lc = partial[at + dv + 1];
            if lc > 0.0 {
                let f = (partial[at + dv] - m).exp();
                total += f * partial[at + col];
                l += f * lc;
            }
            c += 1;
        }
        let r = which[qi] as usize;
        let at = (r * hq + h) * dv + col;
        if at < elements(ko, out.len()) {
            let y = if l > 0.0 { total / l } else { 0.0 };
            // SAFETY: in bounds (checked above), this thread's own element.
            unsafe { write(ko, out.as_mut_ptr(), at, y) };
        }
    }
}

/// Four `u32` at `p`, in one 128-bit load through the read-only cache.
///
/// # Safety
///
/// `p` must be 16-byte aligned, and the 16 bytes from it readable and not
/// written while the kernel runs.
#[inline(always)]
unsafe fn load_128(p: *const u32) -> (u32, u32, u32, u32) {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: the caller's contract.
    unsafe {
        cuda_device::ptx_asm!(
            "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") p as u64,
        );
    }
    (a, b, c, d)
}

/// Stores four `u32` at `p` in one 128-bit store.
///
/// # Safety
///
/// `p` must be 16-byte aligned, the 16 bytes from it writable, and written
/// by no other thread.
#[inline(always)]
unsafe fn store_128(p: *mut u32, v: (u32, u32, u32, u32)) {
    // SAFETY: the caller's contract.
    unsafe {
        cuda_device::ptx_asm!(
            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
            in("l") p as u64,
            in("r") v.0,
            in("r") v.1,
            in("r") v.2,
            in("r") v.3,
            clobber("memory"),
        );
    }
}

/// The two 16-bit floats of format `kind` (`half`) packed in `pair`, the first
/// in the low half.
#[inline(always)]
fn unpack(kind: u32, pair: u32) -> (f32, f32) {
    if kind == half::BF16 {
        (f32::from_bits(pair << 16), f32::from_bits(pair & 0xffff_0000))
    } else {
        (f16_hw(pair), f16_hw(pair >> 16))
    }
}

/// Two values rounded to the 16-bit format `kind` and packed, the first low.
#[inline(always)]
fn pack(kind: u32, a: f32, b: f32) -> u32 {
    narrow(kind, a) as u32 | (narrow(kind, b) as u32) << 16
}

/// The eight 16-bit floats of format `kind` in a 128-bit piece, widened.
#[inline(always)]
fn unpack_8(kind: u32, p: (u32, u32, u32, u32)) -> [f32; 8] {
    let (a, b) = unpack(kind, p.0);
    let (c, d) = unpack(kind, p.1);
    let (e, f) = unpack(kind, p.2);
    let (g, h) = unpack(kind, p.3);
    [a, b, c, d, e, f, g, h]
}

/// Eight values rounded to the 16-bit format `kind`, as a 128-bit piece.
#[inline(always)]
fn pack_8(kind: u32, v: [f32; 8]) -> (u32, u32, u32, u32) {
    (
        pack(kind, v[0], v[1]),
        pack(kind, v[2], v[3]),
        pack(kind, v[4], v[5]),
        pack(kind, v[6], v[7]),
    )
}

/// `out = x` widened from the 16-bit format `kind` to `f32`, eight values a
/// thread: one 128-bit load, two 128-bit stores. `x` holds whole pieces of
/// eight values, 16-byte aligned, as `u32` pairs; `out` the `f32` bits.
#[kernel]
#[launch_bounds(256)]
pub fn widen_16v(kind: u32, x: &[u32], mut out: DisjointSlice<u32>) {
    let p = thread::index_1d().get();
    if 4 * p + 4 <= x.len() && 8 * p + 8 <= out.len() {
        // SAFETY: piece `p` is within both slices (checked above), which are
        // 16-byte aligned; it is this thread's own.
        unsafe {
            let v = unpack_8(kind, load_128(x.as_ptr().add(4 * p)));
            let o = out.as_mut_ptr().add(8 * p);
            store_128(o, (v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()));
            store_128(o.add(4), (v[4].to_bits(), v[5].to_bits(), v[6].to_bits(), v[7].to_bits()));
        }
    }
}

/// `out = x` rounded from `f32` to the 16-bit format `kind`, eight values a
/// thread; the layout of `widen_16v` the other way.
#[kernel]
#[launch_bounds(256)]
pub fn narrow_16v(kind: u32, x: &[u32], mut out: DisjointSlice<u32>) {
    let p = thread::index_1d().get();
    if 8 * p + 8 <= x.len() && 4 * p + 4 <= out.len() {
        // SAFETY: as in `widen_16v`.
        unsafe {
            let a = load_128(x.as_ptr().add(8 * p));
            let b = load_128(x.as_ptr().add(8 * p + 4));
            let v = [
                f32::from_bits(a.0),
                f32::from_bits(a.1),
                f32::from_bits(a.2),
                f32::from_bits(a.3),
                f32::from_bits(b.0),
                f32::from_bits(b.1),
                f32::from_bits(b.2),
                f32::from_bits(b.3),
            ];
            store_128(out.as_mut_ptr().add(4 * p), pack_8(kind, v));
        }
    }
}

/// `out = f(a, b)` on 16-bit floats of format `kind`, eight values a thread.
/// `b` has the shape of `a`, or with `row` pieces of eight, is a row of `8 ·
/// row` values repeated over the rows of `a` (a bias).
#[kernel]
#[launch_bounds(256)]
pub fn binary_16(op: u32, kind: u32, a: &[u32], b: &[u32], row: u32, mut out: DisjointSlice<u32>) {
    let p = thread::index_1d().get();
    let q = if row == 0 { p } else { p % row as usize };
    if 4 * p + 4 <= a.len() && 4 * q + 4 <= b.len() && 4 * p + 4 <= out.len() {
        // SAFETY: as in `widen_16v`.
        unsafe {
            let x = unpack_8(kind, load_128(a.as_ptr().add(4 * p)));
            let y = unpack_8(kind, load_128(b.as_ptr().add(4 * q)));
            let mut v = [0.0f32; 8];
            let mut i = 0;
            #[unroll]
            while i < 8 {
                v[i] = binary_value(op, x[i], y[i]).0;
                i += 1;
            }
            store_128(out.as_mut_ptr().add(4 * p), pack_8(kind, v));
        }
    }
}

/// `out = f(x)` on 16-bit floats of format `kind`, eight values a thread.
#[kernel]
#[launch_bounds(256)]
pub fn unary_16(op: u32, kind: u32, x: &[u32], mut out: DisjointSlice<u32>) {
    let p = thread::index_1d().get();
    if 4 * p + 4 <= x.len() && 4 * p + 4 <= out.len() {
        // SAFETY: as in `widen_16v`.
        unsafe {
            let v = unpack_8(kind, load_128(x.as_ptr().add(4 * p)));
            let mut y = [0.0f32; 8];
            let mut i = 0;
            #[unroll]
            while i < 8 {
                y[i] = unary_value(op, v[i]);
                i += 1;
            }
            store_128(out.as_mut_ptr().add(4 * p), pack_8(kind, y));
        }
    }
}

/// `out = g · f'(x)` on 16-bit floats of format `kind`, eight values a
/// thread.
#[kernel]
#[launch_bounds(256)]
pub fn unary_backward_16(op: u32, kind: u32, x: &[u32], g: &[u32], mut out: DisjointSlice<u32>) {
    let p = thread::index_1d().get();
    if 4 * p + 4 <= x.len() && 4 * p + 4 <= g.len() && 4 * p + 4 <= out.len() {
        // SAFETY: as in `widen_16v`.
        unsafe {
            let v = unpack_8(kind, load_128(x.as_ptr().add(4 * p)));
            let d = unpack_8(kind, load_128(g.as_ptr().add(4 * p)));
            let mut y = [0.0f32; 8];
            let mut i = 0;
            #[unroll]
            while i < 8 {
                y[i] = d[i] * unary_slope(op, v[i]);
                i += 1;
            }
            store_128(out.as_mut_ptr().add(4 * p), pack_8(kind, y));
        }
    }
}

/// The pieces of 8 columns one thread of `norm_16` keeps: rows of up to
/// `4 · 256 · 8 = 8192` columns.
const NORM_PIECES: usize = 4;

/// The body of `norm_16` for the 16-bit format `K`.
#[device]
fn norm_row_16<const K: u32>(rms: bool, bias: bool, x: &[u32], w: &[u32], b: &[u32], offset: f32, eps: f32, width: usize, out: *mut u32) {
    let row = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    let (pairs, pieces) = (width / 2, width / 8);
    let start = row * pairs;
    let mut v = [0.0f32; 8 * NORM_PIECES];
    let mut i = 0;
    #[unroll]
    while i < NORM_PIECES {
        let piece = t + 256 * i;
        if piece < pieces {
            // SAFETY: piece `piece < pieces` lies within row `row` of `x`
            // (checked by the kernel); rows start at multiples of 16 bytes
            // (`width` is a multiple of 8) of 16-byte aligned slices.
            let p = unsafe { load_128(x.as_ptr().add(start + 4 * piece)) };
            let (a, b) = unpack(K, p.0);
            let (c, d) = unpack(K, p.1);
            let (e, f) = unpack(K, p.2);
            let (g, h) = unpack(K, p.3);
            v[8 * i] = a;
            v[8 * i + 1] = b;
            v[8 * i + 2] = c;
            v[8 * i + 3] = d;
            v[8 * i + 4] = e;
            v[8 * i + 5] = f;
            v[8 * i + 6] = g;
            v[8 * i + 7] = h;
        }
        i += 1;
    }
    let n = width as f32;
    let mut mean = 0.0f32;
    if !rms {
        let mut part = 0.0f32;
        let mut k = 0;
        #[unroll]
        while k < 8 * NORM_PIECES {
            part += v[k];
            k += 1;
        }
        mean = block_sum(part) / n;
    }
    let mut part = 0.0f32;
    let mut m = 0;
    #[unroll]
    while m < 8 * NORM_PIECES {
        if t + 256 * (m / 8) < pieces {
            let centered = v[m] - mean;
            part += centered * centered;
        }
        m += 1;
    }
    let inv = 1.0 / (block_sum(part) / n + eps).sqrt();
    let mut q = 0;
    #[unroll]
    while q < NORM_PIECES {
        let i = q;
        let at = t + 256 * i;
        if at < pieces {
            // SAFETY: as the loads of `x`, within the vectors' `width`.
            let ws = unsafe { load_128(w.as_ptr().add(4 * at)) };
            let bs = if bias { unsafe { load_128(b.as_ptr().add(4 * at)) } } else { (0, 0, 0, 0) };
            let y = |j: usize, wp: u32, bp: u32| {
                let (w0, w1) = unpack(K, wp);
                let (b0, b1) = if bias { unpack(K, bp) } else { (0.0, 0.0) };
                let y0 = (v[8 * i + j] - mean) * inv * (offset + w0) + b0;
                let y1 = (v[8 * i + j + 1] - mean) * inv * (offset + w1) + b1;
                pack(K, y0, y1)
            };
            let packed = (y(0, ws.0, bs.0), y(2, ws.1, bs.1), y(4, ws.2, bs.2), y(6, ws.3, bs.3));
            // SAFETY: piece `at` of row `row` of the output, in bounds
            // (checked by the kernel) and aligned as `x`'s; this thread's
            // alone.
            unsafe { store_128(out.add(start + 4 * at), packed) };
        }
        q += 1;
    }
}

/// LayerNorm (bit 0 of `flags` clear) or RMSNorm (set) of every row of
/// `width`, as `norm_rows`, for `x`, the weight, the bias (bit 1 of `flags`)
/// and the output in one 16-bit format `kind` (`half`): one block of 256
/// threads per row, for wide rows, as in decoding. Thread `t` reads the
/// 16-byte pieces `t, t + 256, …` of the row (at most 4) in one load each,
/// all issued before the sums, keeps them in registers and writes them
/// normalized; the sums add each thread's partial sums, then the threads by
/// `block_sum`. `width` is a multiple of 8, at most 8192; the slices are
/// pairs of values, 16-byte aligned.
#[kernel]
#[launch_bounds(256)]
pub fn norm_16(flags: u32, kind: u32, x: &[u32], w: &[u32], b: &[u32], offset: f32, eps: f32, width: u32, mut out: DisjointSlice<u32>) {
    let row = thread::blockIdx_x() as usize;
    let width = width as usize;
    let (pairs, pieces) = (width / 2, width / 8);
    let start = row * pairs;
    let (rms, bias) = (flags & 1 == 1, flags & 2 == 2);
    // The same for every thread of the block: whole blocks go on or stop,
    // so that every thread reaches the barriers of `block_sum`.
    if width > 0 && width % 8 == 0 && pieces <= NORM_PIECES * 256 && start + pairs <= x.len() && start + pairs <= out.len() && pairs <= w.len() && (!bias || pairs <= b.len()) {
        let o = out.as_mut_ptr();
        if kind == half::BF16 {
            norm_row_16::<{ half::BF16 }>(rms, bias, x, w, b, offset, eps, width, o);
        } else {
            norm_row_16::<{ half::F16 }>(rms, bias, x, w, b, offset, eps, width, o);
        }
    }
}

/// `out[r, j] = f(x[r, j]) · x[r, n + j]` for rows of `2n` of `x`: a gated
/// unit, the gate first, `x` of float type `kx` (row-major) and `out` of `ko`.
#[kernel]
#[launch_bounds(256)]
pub fn glu_any(op: u32, kx: u32, x: &[u8], n: u32, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let n = n as usize;
    if n > 0 && i < elements(ko, out.len()) {
        let (r, j) = (i / n, i % n);
        let gate = read(kx, x, 2 * n * r + j);
        let value = read(kx, x, 2 * n * r + n + j);
        // SAFETY: `i` is in bounds (checked above) and this thread's own.
        unsafe { write(ko, out.as_mut_ptr(), i, unary_value(op, gate) * value) };
    }
}

/// The gradient of `glu_any` in `f32`, one thread per element of `x`:
/// `∂gate = g · value · f'(gate)`, `∂value = g · f(gate)`.
#[kernel]
#[launch_bounds(256)]
pub fn glu_backward(op: u32, x: &[f32], g: &[f32], n: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let n = n as usize;
    if let Some(y) = out.get_mut(idx) {
        if n > 0 {
            let (r, c) = (i / (2 * n), i % (2 * n));
            let j = if c < n { c } else { c - n };
            let gate = x[2 * n * r + j];
            let grad = g[n * r + j];
            *y = if c < n { grad * x[2 * n * r + n + j] * unary_slope(op, gate) } else { grad * unary_value(op, gate) };
        }
    }
}

/// The position of the maximum of each row of `width` of `x` (float type
/// `kind`), the first one on a tie, NaN never chosen, 0 for a row of NaN:
/// one block of 256 threads per row. Thread `t` keeps the best of columns
/// `t, t + 256, …`; the threads' candidates are then merged in a fixed
/// order, the larger value winning, the smaller position on a tie.
#[kernel]
#[launch_bounds(256)]
pub fn argmax_rows(kind: u32, x: &[u8], width: u32, mut out: DisjointSlice<u32>) {
    static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut AT: SharedArray<u32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    let width = width as usize;
    let start = row * width;
    // `u32::MAX` marks a thread that saw no number.
    if width > 0 && row < out.len() && start + width <= elements(kind, x.len()) {
        let (mut best, mut at) = (f32::NEG_INFINITY, u32::MAX);
        let mut c = t;
        while c < width {
            let v = read(kind, x, start + c);
            if !v.is_nan() && (at == u32::MAX || v > best) {
                best = v;
                at = c as u32;
            }
            c += 256;
        }
        // SAFETY: each thread writes its own slot, read after the barrier.
        unsafe {
            VALUES[t] = best;
            AT[t] = at;
        }
        thread::sync_threads();
        if t == 0 {
            let (mut best, mut at) = (f32::NEG_INFINITY, u32::MAX);
            let mut i = 0;
            while i < 256 {
                // SAFETY: written before the barrier above.
                let (v, a) = unsafe { (VALUES[i], AT[i]) };
                if a != u32::MAX && (at == u32::MAX || v > best || (v == best && a < at)) {
                    best = v;
                    at = a;
                }
                i += 1;
            }
            // SAFETY: `row < out.len()` (checked above); slot `row` belongs
            // to this block, and only its thread 0 writes.
            unsafe { *out.as_mut_ptr().add(row) = if at == u32::MAX { 0 } else { at } };
        }
    }
}

/// The cache kinds of `paged_attention_16` beyond `half`'s two: 8-bit floats,
/// decoded through a table of their 256 values in shared memory (the device
/// has no conversion for them).
const CACHE_E4M3: u32 = 2;
/// See [`CACHE_E4M3`].
const CACHE_E5M2: u32 = 3;

/// Eight bytes at `p`, in one 64-bit load through the read-only cache.
///
/// # Safety
///
/// `p` must be 8-byte aligned, and the 8 bytes from it readable and not
/// written while the kernel runs.
#[inline(always)]
unsafe fn load_64(p: *const u32) -> (u32, u32) {
    let (a, b): (u32, u32);
    // SAFETY: the caller's contract.
    unsafe {
        cuda_device::ptx_asm!(
            "ld.global.nc.v2.u32 {%0, %1}, [%2];",
            out("=r") a,
            out("=r") b,
            in("l") p as u64,
        );
    }
    (a, b)
}

/// The query heads of one key head that a block of `paged_attention_16` takes.
const PAGED_16_HEADS: usize = 4;

/// The partial values of a block of `paged_attention_16`, per warp, head and
/// column: heads of at most 128.
const PAGED_16_WIDTH: usize = 128;

/// The sum of `v` over the `L` lanes of each group of `L` consecutive lanes,
/// on each of them, by a fixed butterfly.
#[inline(always)]
fn group_sum<const L: usize>(v: f32) -> f32 {
    let mut v = v;
    let mut step = (L / 2) as u32;
    while step > 0 {
        v += warp::shuffle_xor_f32(v, step);
        step /= 2;
    }
    v
}

/// The body of `paged_attention_16` for caches of the 16-bit format `K` and
/// `L = D / 8` lanes per head (8 or 16): lane `p = lane mod L` of a group of
/// `L` lanes reads columns `8p..8p + 8` of a key or value in one 128-bit
/// load, and a warp reads `32 / L` positions at a time.
#[device]
fn paged_16<const K: u32, const L: usize>(
    p: Paged,
    q: &[u8],
    keys: &[u32],
    values: &[u32],
    pages: &[u32],
    rows: &[u32],
    which: &[u32],
    out: *mut u8,
    out_len: usize,
    partial: *mut f32,
    partial_len: usize,
) {
    static mut QS: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut S: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut A: SharedArray<f32, 4096> = SharedArray::UNINIT;
    static mut STATS: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut LINES: SharedArray<u32, 256> = SharedArray::UNINIT;
    static mut LUT: SharedArray<f32, 256> = SharedArray::UNINIT;
    let t = thread::threadIdx_x() as usize;
    let (warp_id, lane) = (t / 32, t % 32);
    let block = thread::blockIdx_x() as usize;
    let (kq, ko) = (p.kinds & 0xff, p.kinds >> 16);
    let eight = K == CACHE_E4M3 || K == CACHE_E5M2;
    let (hq, hkv, d) = (p.heads as usize, p.kv_heads as usize, p.d as usize);
    let group = hq / hkv;
    let subs = group.div_ceil(PAGED_16_HEADS);
    let (chunk, chunks) = (p.chunk as usize, p.chunks as usize);
    let c = block % chunks;
    let rest = block / chunks;
    let sub = rest % subs;
    let rest = rest / subs;
    let g = rest % hkv;
    let qi = rest / hkv;
    if qi >= which.len() {
        return;
    }
    let r = which[qi] as usize;
    let s = rows[2 * r] as usize;
    let count = rows[2 * r + 1] as usize;
    let first = if p.window > 0 && count > p.window as usize { count - p.window as usize } else { 0 };
    let lo = if first > c * chunk { first } else { c * chunk };
    let hi = if count < (c + 1) * chunk { count } else { (c + 1) * chunk };
    let n = if hi > lo { hi - lo } else { 0 };
    let h0 = g * group + sub * PAGED_16_HEADS;
    let nh = if group - sub * PAGED_16_HEADS < PAGED_16_HEADS { group - sub * PAGED_16_HEADS } else { PAGED_16_HEADS };
    if n == 0 {
        // A chunk past the query's positions (the table of pages may hold
        // more): nothing to add, only its statistics, which say so. The
        // same for the whole block: no barrier is skipped by part of it.
        if chunks > 1 {
            let mut slot = t;
            while slot < nh * d {
                let (hh, col) = (slot / d, slot % d);
                let at = ((qi * hq + h0 + hh) * chunks + c) * (d + 2);
                if at + d + 2 <= partial_len {
                    // SAFETY: as the writes of `partial` below.
                    unsafe {
                        *partial.add(at + col) = 0.0;
                        if col == 0 {
                            *partial.add(at + d) = f32::NEG_INFINITY;
                            *partial.add(at + d + 1) = 0.0;
                        }
                    }
                }
                slot += 256;
            }
        } else if count == 0 {
            let mut slot = t;
            while slot < nh * d {
                let (hh, col) = (slot / d, slot % d);
                let at = (r * hq + h0 + hh) * d + col;
                if at < elements(ko, out_len) {
                    // SAFETY: as the writes of `out` below.
                    unsafe { write(ko, out, at, 0.0) };
                }
                slot += 256;
            }
        }
        return;
    }
    let page = p.page as usize;
    if t < n {
        let j = lo + t;
        let at = pages[s * p.per_sequence as usize + j / page] as usize * page + j % page;
        // SAFETY: `t < n ≤ 256`; this thread's slot, read after the barrier.
        unsafe { LINES[t] = at as u32 };
    }
    let mut i = t;
    while i < nh * d {
        let (hh, col) = (i / d, i % d);
        let at = r * p.q_row as usize + (h0 + hh) * p.q_head as usize + col;
        // SAFETY: `i < nh · d ≤ 4 · 128`; this thread's slot, read after the
        // barrier below.
        unsafe { QS[i] = p.scale * read(kq, q, at) };
        i += 256;
    }
    if eight {
        // SAFETY: `t < 256`; this thread's slot, read after the barrier.
        unsafe { LUT[t] = widen_8(K == CACHE_E4M3, t as u8) };
    }
    thread::sync_threads();
    // The eight values of the cache in the words `w` of a load: 16-bit
    // floats, or bytes looked up in the table.
    let values_of = |w: (u32, u32, u32, u32)| -> [f32; 8] {
        if eight {
            // SAFETY: the table, written before the barrier above.
            let v = |x: u32, i: u32| unsafe { LUT[((x >> (8 * i)) & 0xff) as usize] };
            [v(w.0, 0), v(w.0, 1), v(w.0, 2), v(w.0, 3), v(w.1, 0), v(w.1, 1), v(w.1, 2), v(w.1, 3)]
        } else {
            let (a, b) = unpack(K, w.0);
            let (c, d) = unpack(K, w.1);
            let (e, f) = unpack(K, w.2);
            let (g, h) = unpack(K, w.3);
            [a, b, c, d, e, f, g, h]
        }
    };
    let part = lane % L;
    // The eight values of row `line` of a table at this lane's columns: 16
    // bytes in 16-bit floats, 8 in bytes.
    let load_at = |table: &[u32], line: u32| -> (u32, u32, u32, u32) {
        let row = line as usize * hkv + g;
        // SAFETY: the bytes of row `line`, head `g`, columns `8p..8p + 8` lie
        // within the cache (the backend checked the lines) and are aligned
        // (rows of `D` values, `D` a multiple of 8).
        if eight {
            let (a, b) = unsafe { load_64(table.as_ptr().add(row * (d / 4) + 2 * part)) };
            (a, b, 0, 0)
        } else {
            unsafe { load_128(table.as_ptr().add(row * (d / 2) + 4 * part)) }
        }
    };
    // This lane's columns of the queries, in registers.
    let mut qr = [[0.0f32; 8]; PAGED_16_HEADS];
    let mut hh = 0;
    #[unroll]
    while hh < PAGED_16_HEADS {
        let mut k = 0;
        #[unroll]
        while k < 8 {
            if hh < nh {
                // SAFETY: written before the barrier above.
                qr[hh][k] = unsafe { QS[hh * d + 8 * part + k] };
            }
            k += 1;
        }
        hh += 1;
    }
    // The scores: each group of `L` lanes one position at a time, two
    // positions in flight per group.
    let per_warp = 32 / L;
    let slot0 = warp_id * per_warp + lane / L;
    let stride = 8 * per_warp;
    // The loop runs the same number of times on every lane of the warp: the
    // shuffles of `group_sum` need them all.
    let mut base = warp_id * per_warp;
    while base < n {
        let i = base + lane / L;
        let i2 = i + stride;
        // SAFETY (both loads): `LINES` written before the barrier above;
        // the 16 bytes at row `line`, head `g`, columns `8p..8p + 8` lie
        // within the cache (the backend checked the lines) and are 16-byte
        // aligned (rows of `D` 16-bit values, `D` a multiple of 8).
        let k1 = if i < n { load_at(keys, unsafe { LINES[i] }) } else { (0, 0, 0, 0) };
        let k2 = if i2 < n { load_at(keys, unsafe { LINES[i2] }) } else { (0, 0, 0, 0) };
        let (v1, v2) = (values_of(k1), values_of(k2));
        let dot = |q: &[f32; 8], v: &[f32; 8]| {
            q[7].mul_add(v[7], q[6].mul_add(v[6], q[5].mul_add(v[5], q[4].mul_add(v[4], q[3].mul_add(v[3], q[2].mul_add(v[2], q[1].mul_add(v[1], q[0] * v[0])))))))
        };
        let mut hh = 0;
        #[unroll]
        while hh < PAGED_16_HEADS {
            if hh < nh {
                let a = group_sum::<L>(dot(&qr[hh], &v1));
                let b = group_sum::<L>(dot(&qr[hh], &v2));
                let cap = |v: f32| if p.softcap > 0.0 { p.softcap * (v / p.softcap).tanh() } else { v };
                if part == 0 {
                    // SAFETY: slot `(hh, i)` belongs to this group's
                    // position `i`, read after the barrier below.
                    if i < n {
                        unsafe { S[hh * 256 + i] = cap(a) };
                    }
                    if i2 < n {
                        unsafe { S[hh * 256 + i2] = cap(b) };
                    }
                }
            }
            hh += 1;
        }
        base += 2 * stride;
    }
    thread::sync_threads();
    // Weights and statistics, a warp per head.
    if warp_id < nh {
        let row = warp_id * 256;
        let mut m = f32::NEG_INFINITY;
        let mut i = lane;
        while i < n {
            // SAFETY: written before the barrier above.
            let v = unsafe { S[row + i] };
            if v > m {
                m = v;
            }
            i += 32;
        }
        let m = warp_max(m);
        let mut l = 0.0f32;
        let mut i = lane;
        while i < n {
            // SAFETY: each lane rewrites its own positions of this row.
            unsafe {
                let w = (S[row + i] - m).exp();
                S[row + i] = w;
                l += w;
            }
            i += 32;
        }
        let l = warp_sum(l);
        if lane == 0 {
            // SAFETY: this warp's head's slots, read after the barrier.
            unsafe {
                STATS[warp_id] = m;
                STATS[4 + warp_id] = l;
            }
        }
    }
    thread::sync_threads();
    // The weighted values: each group of `L` lanes adds its positions into
    // its columns `8p..8p + 8`, for every head.
    let mut acc = [[0.0f32; 8]; PAGED_16_HEADS];
    let mut i = slot0;
    while i < n {
        // SAFETY: `LINES`, written before the barriers above.
        let vs = values_of(load_at(values, unsafe { LINES[i] }));
        let mut hh = 0;
        #[unroll]
        while hh < PAGED_16_HEADS {
            if hh < nh {
                // SAFETY: the weights, written before the barrier above.
                let w = unsafe { S[hh * 256 + i] };
                let mut k = 0;
                #[unroll]
                while k < 8 {
                    acc[hh][k] = w.mul_add(vs[k], acc[hh][k]);
                    k += 1;
                }
            }
            hh += 1;
        }
        i += stride;
    }
    // The groups of a warp added (lanes `p`, `p + L`, …), then the warps in
    // order through shared memory.
    let mut hh = 0;
    #[unroll]
    while hh < PAGED_16_HEADS {
        let mut k = 0;
        #[unroll]
        while k < 8 {
            let mut v = acc[hh][k];
            let mut step = 16;
            while step >= L as u32 {
                v += warp::shuffle_xor_f32(v, step);
                step /= 2;
            }
            if hh < nh && lane < L {
                // SAFETY: slot of this warp, head and column, read after
                // the barrier below.
                unsafe { A[(warp_id * PAGED_16_HEADS + hh) * PAGED_16_WIDTH + 8 * part + k] = v };
            }
            k += 1;
        }
        hh += 1;
    }
    thread::sync_threads();
    // The heads' columns, `nh · d` of them (up to 512), 256 at a time.
    let mut slot = t;
    while slot < nh * d {
        let (hh, col) = (slot / d, slot % d);
        let mut total = 0.0f32;
        let mut w = 0;
        while w < 8 {
            // SAFETY: written before the barrier above.
            total += unsafe { A[(w * PAGED_16_HEADS + hh) * PAGED_16_WIDTH + col] };
            w += 1;
        }
        // SAFETY: written before the barriers above.
        let (m, l) = unsafe { (STATS[hh], STATS[4 + hh]) };
        let h = h0 + hh;
        if chunks == 1 {
            let at = (r * hq + h) * d + col;
            if at < elements(ko, out_len) {
                // SAFETY: in bounds (checked above); row `r`, head `h`,
                // column `col` belong to this block and thread.
                unsafe { write(ko, out, at, if l > 0.0 { total / l } else { 0.0 }) };
            }
        } else {
            let at = ((qi * hq + h) * chunks + c) * (d + 2);
            if at + d + 2 <= partial_len {
                // SAFETY: in bounds (checked above); the slots of query `qi`,
                // head `h` and chunk `c` belong to this block, column `col` to
                // this thread, the statistics to the thread of column 0.
                unsafe {
                    *partial.add(at + col) = total;
                    if col == 0 {
                        *partial.add(at + d) = m;
                        *partial.add(at + d + 1) = l;
                    }
                }
            }
        }
        slot += 256;
    }
}

/// `paged_attention` for a cache of 16-bit floats (`kind` = `half::BF16` or
/// `half::F16`) or 8-bit ones (`CACHE_E4M3`, `CACHE_E5M2`), with heads of 64
/// or 128 for both keys and values: eight values of a key or a value per lane
/// in one load (128 or 64 bits), a group of `D / 8` lanes per position, the
/// queries' columns kept in registers, groups of at most 4 query heads per
/// block. The same results as `paged_attention`, but for the order of the
/// sums, fixed as well.
#[kernel]
#[launch_bounds(256)]
pub fn paged_attention_16(
    p: Paged,
    kind: u32,
    q: &[u8],
    keys: &[u32],
    values: &[u32],
    pages: &[u32],
    rows: &[u32],
    which: &[u32],
    mut out: DisjointSlice<u8>,
    mut partial: DisjointSlice<f32>,
) {
    let fits = p.kv_heads > 0
        && p.heads % p.kv_heads == 0
        && p.d == p.dv
        && (p.d == 64 || p.d == 128)
        && p.chunk > 0
        && p.chunk <= 256
        && p.chunks > 0
        && p.page > 0;
    // The same for every thread of the block: whole blocks go on or stop.
    if fits {
        let (o, ol) = (out.as_mut_ptr(), out.len());
        let (pp, pl) = (partial.as_mut_ptr(), partial.len());
        match (kind, p.d == 128) {
            (half::BF16, true) => paged_16::<{ half::BF16 }, 16>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (half::BF16, false) => paged_16::<{ half::BF16 }, 8>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (half::F16, true) => paged_16::<{ half::F16 }, 16>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (half::F16, false) => paged_16::<{ half::F16 }, 8>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (CACHE_E4M3, true) => paged_16::<CACHE_E4M3, 16>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (CACHE_E4M3, false) => paged_16::<CACHE_E4M3, 8>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (CACHE_E5M2, true) => paged_16::<CACHE_E5M2, 16>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            (CACHE_E5M2, false) => paged_16::<CACHE_E5M2, 8>(p, q, keys, values, pages, rows, which, o, ol, pp, pl),
            _ => {}
        }
    }
}

/// The `f16` at byte `at` of `b`, widened; 0 past the end.
#[inline(always)]
fn f16_at(b: &[u8], at: usize) -> f32 {
    if at + 1 < b.len() { widen(half::F16, b[at] as u16 | (b[at + 1] as u16) << 8) } else { 0.0 }
}

/// The `f32` at byte `at` of `b`; 0 past the end.
#[inline(always)]
fn f32_at(b: &[u8], at: usize) -> f32 {
    if at + 3 < b.len() {
        f32::from_bits(b[at] as u32 | (b[at + 1] as u32) << 8 | (b[at + 2] as u32) << 16 | (b[at + 3] as u32) << 24)
    } else {
        0.0
    }
}

/// The elements and bytes of a block of `format` (`quant`), `group` for
/// Kapel's formats; `(0, 0)` for a code this file does not decode.
#[inline(always)]
fn quant_block(format: u32, group: usize) -> (usize, usize) {
    match format {
        quant::INT8 => (group, 4 + group),
        quant::INT4 => (group, 4 + group / 2),
        quant::Q8_0 => (32, 34),
        quant::Q4_0 => (32, 18),
        quant::Q4_1 => (32, 20),
        quant::Q5_0 => (32, 22),
        quant::Q5_1 => (32, 24),
        quant::Q4K => (256, 144),
        quant::Q5K => (256, 176),
        quant::Q6K => (256, 210),
        _ => (0, 0),
    }
}

/// The six-bit scale and minimum of sub-block `k` (of 32 elements) of a Q4_K
/// or Q5_K block, from its 12 packed bytes at `p`.
#[inline(always)]
fn k_scale(b: &[u8], p: usize, k: usize) -> (f32, f32) {
    let byte = |i: usize| if p + i < b.len() { b[p + i] as u32 } else { 0 };
    let (sc, mn) = if k < 4 {
        (byte(k) & 0x3f, byte(4 + k) & 0x3f)
    } else {
        let i = k - 4;
        ((byte(8 + i) & 0x0f) | ((byte(i) >> 2) & 0x30), (byte(8 + i) >> 4) | ((byte(4 + i) >> 2) & 0x30))
    };
    (sc as f32, mn as f32)
}

/// Elements `e..e + 8` of a row of `format` (`row`, whole blocks), `e` a
/// multiple of 8: `(a, b, q)` with element `e + j` equal to `a · q[j] + b`,
/// as `kapel_core`'s decoding computes it.
#[inline(always)]
fn decode_8(format: u32, group: usize, row: &[u8], e: usize) -> (f32, f32, [f32; 8]) {
    let (len, bytes) = quant_block(format, group);
    let mut q = [0.0f32; 8];
    if len == 0 {
        return (f32::NAN, 0.0, q);
    }
    let (block, off) = (e / len, e % len);
    let base = block * bytes;
    let byte = |at: usize| if at < row.len() { row[at] } else { 0 };
    let (mut a, mut b) = (0.0f32, 0.0f32);
    let mut j = 0;
    match format {
        quant::INT8 => {
            a = f32_at(row, base);
            while j < 8 {
                q[j] = byte(base + 4 + off + j) as i8 as f32;
                j += 1;
            }
        }
        quant::INT4 => {
            a = f32_at(row, base);
            while j < 8 {
                let v = byte(base + 4 + (off + j) / 2);
                q[j] = (if (off + j) % 2 == 0 { v & 0x0f } else { v >> 4 }) as f32 - 8.0;
                j += 1;
            }
        }
        quant::Q8_0 => {
            a = f16_at(row, base);
            while j < 8 {
                q[j] = byte(base + 2 + off + j) as i8 as f32;
                j += 1;
            }
        }
        quant::Q4_0 | quant::Q4_1 => {
            a = f16_at(row, base);
            let codes = if format == quant::Q4_1 {
                b = f16_at(row, base + 2);
                base + 4
            } else {
                base + 2
            };
            while j < 8 {
                let at = off + j;
                let v = if at < 16 { byte(codes + at) & 0x0f } else { byte(codes + at - 16) >> 4 };
                q[j] = if format == quant::Q4_1 { v as f32 } else { v as f32 - 8.0 };
                j += 1;
            }
        }
        quant::Q5_0 | quant::Q5_1 => {
            a = f16_at(row, base);
            let high_at = if format == quant::Q5_1 {
                b = f16_at(row, base + 2);
                base + 4
            } else {
                base + 2
            };
            let high = byte(high_at) as u32 | (byte(high_at + 1) as u32) << 8 | (byte(high_at + 2) as u32) << 16 | (byte(high_at + 3) as u32) << 24;
            let codes = high_at + 4;
            while j < 8 {
                let at = off + j;
                let low = if at < 16 { byte(codes + at) & 0x0f } else { byte(codes + at - 16) >> 4 };
                let v = low as u32 | ((high >> at) & 1) << 4;
                q[j] = if format == quant::Q5_1 { v as f32 } else { v as f32 - 16.0 };
                j += 1;
            }
        }
        quant::Q4K | quant::Q5K => {
            let (d, dmin) = (f16_at(row, base), f16_at(row, base + 2));
            let (c, t, l) = (off / 64, off % 64 / 32, off % 32);
            let (sc, mn) = k_scale(row, base + 4, off / 32);
            a = d * sc;
            b = -(dmin * mn);
            let codes = if format == quant::Q5K { base + 48 } else { base + 16 };
            while j < 8 {
                let mut v = (byte(codes + c * 32 + l + j) >> (4 * t)) & 0x0f;
                if format == quant::Q5K {
                    v |= ((byte(base + 16 + l + j) >> (off / 32)) & 1) << 4;
                }
                q[j] = v as f32;
                j += 1;
            }
        }
        quant::Q6K => {
            let (h, t, m) = (off / 128, off % 128 / 64, off % 64);
            let (s, l) = (off % 128 / 32, off % 32);
            a = f16_at(row, base + 208) * (byte(base + 192 + off / 16) as i8 as f32);
            while j < 8 {
                let low = (byte(base + h * 64 + m + j) >> (4 * t)) & 0x0f;
                let high = (byte(base + 128 + h * 32 + l + j) >> (2 * s)) & 0x03;
                q[j] = (low | high << 4) as f32 - 32.0;
                j += 1;
            }
        }
        _ => {}
    }
    (a, b, q)
}

/// `total + Σ_j (a · q[j] + b) · x[at + j]`, the eight values of `x` of float
/// type `kx`, as `a · Σ q·x + b · Σ x`.
#[inline(always)]
fn dot_quant(kx: u32, x: &[u8], at: usize, a: f32, b: f32, q: &[f32; 8], total: f32) -> f32 {
    let (mut dot, mut sum) = (0.0f32, 0.0f32);
    let mut j = 0;
    while j < 8 {
        let v = read(kx, x, at + j);
        dot = q[j].mul_add(v, dot);
        sum += v;
        j += 1;
    }
    let total = a.mul_add(dot, total);
    b.mul_add(sum, total)
}

/// `out[m, c0 + n] = Σ_k x[m, k] · w[n, k]` for `rows ≤ 8` rows of `x` (float
/// type `kx`) and the weights `w` `[N, K]` of a quantized `format` (`quant`,
/// `group` for Kapel's), its rows a whole number of blocks: the product of a
/// few activations by quantized weights, read in place. `out` has `width`
/// columns, of type `ko`; the weights fill columns `c0..c0 + N`.
///
/// One warp per weight row `n`: lane `l` decodes elements `8l, 8l + 256, …`
/// of the row, the scale of each 8 taken out of the sum (`a · Σ q·x + b ·
/// Σ x`), and the lanes' sums are added by a fixed butterfly.
#[kernel]
#[launch_bounds(256)]
pub fn qgemv(format: u32, group: u32, k: u32, w: &[u8], kx: u32, x: &[u8], rows: u32, c0: u32, width: u32, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let n = idx.get() / 32;
    let lane = idx.get() % 32;
    let (k, rows, group) = (k as usize, rows as usize, group as usize);
    let (len, bytes) = quant_block(format, group);
    let row_bytes = if len > 0 { k / len * bytes } else { 0 };
    let columns = if row_bytes > 0 { w.len() / row_bytes } else { 0 };
    let (c0, width) = (c0 as usize, width as usize);
    // The same for every lane of the warp: whole warps go on or stop.
    if len > 0 && k % len == 0 && len % 8 == 0 && rows >= 1 && rows <= 8 && n < columns && c0 + columns <= width && rows * width <= elements(ko, out.len()) {
        let row = &w[n * row_bytes..(n + 1) * row_bytes];
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut a4, mut a5, mut a6, mut a7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut e = lane * 8;
        while e < k {
            let (a, b, q) = decode_8(format, group, row, e);
            a0 = dot_quant(kx, x, e, a, b, &q, a0);
            if rows > 1 { a1 = dot_quant(kx, x, k + e, a, b, &q, a1); }
            if rows > 2 { a2 = dot_quant(kx, x, 2 * k + e, a, b, &q, a2); }
            if rows > 3 { a3 = dot_quant(kx, x, 3 * k + e, a, b, &q, a3); }
            if rows > 4 { a4 = dot_quant(kx, x, 4 * k + e, a, b, &q, a4); }
            if rows > 5 { a5 = dot_quant(kx, x, 5 * k + e, a, b, &q, a5); }
            if rows > 6 { a6 = dot_quant(kx, x, 6 * k + e, a, b, &q, a6); }
            if rows > 7 { a7 = dot_quant(kx, x, 7 * k + e, a, b, &q, a7); }
            e += 256;
        }
        let sums = [warp_sum(a0), warp_sum(a1), warp_sum(a2), warp_sum(a3), warp_sum(a4), warp_sum(a5), warp_sum(a6), warp_sum(a7)];
        if lane == 0 {
            let o = out.as_mut_ptr();
            let mut m = 0;
            while m < 8 {
                if m < rows {
                    // SAFETY: `m · width + c0 + n < rows · width` (checked
                    // above); the column belongs to this warp, and only its
                    // lane 0 writes.
                    unsafe { write(ko, o, m * width + c0 + n, sums[m]) };
                }
                m += 1;
            }
        }
    }
}

/// `out[r, e] = w[first + r, e]` dequantized to float type `ko`, one thread
/// per 8 elements: rows `first..` of quantized weights `[N, K]` (`format`,
/// `group`) as a float matrix `[R, K]`, for the products with many rows.
#[kernel]
#[launch_bounds(256)]
pub fn dequantize(format: u32, group: u32, k: u32, w: &[u8], first: u32, ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let t = idx.get();
    let (k, group) = (k as usize, group as usize);
    let (len, bytes) = quant_block(format, group);
    if len > 0 && k % len == 0 && len % 8 == 0 && 8 * t < elements(ko, out.len()) {
        let (r, e) = (t / (k / 8), t % (k / 8) * 8);
        let row_bytes = k / len * bytes;
        let start = (first as usize + r) * row_bytes;
        if start + row_bytes <= w.len() {
            let (a, b, q) = decode_8(format, group, &w[start..start + row_bytes], e);
            let mut j = 0;
            while j < 8 {
                // SAFETY: `8t + j < elements(ko, out.len())` (`8t` checked
                // above, the output a whole number of rows of 8s); these 8
                // belong to this thread.
                unsafe { write(ko, out.as_mut_ptr(), r * k + e + j, a.mul_add(q[j], b)) };
                j += 1;
            }
        }
    }
}

/// `out[i, e] = w[index[i], e]` dequantized to float type `ko`, one thread
/// per 8 elements: rows of a quantized table (`format`, `group`), as the
/// gather of embeddings reads them.
#[kernel]
#[launch_bounds(256)]
pub fn dequantize_rows(format: u32, group: u32, k: u32, w: &[u8], index: &[u32], ko: u32, mut out: DisjointSlice<u8>) {
    let idx = thread::index_1d();
    let t = idx.get();
    let (k, group) = (k as usize, group as usize);
    let (len, bytes) = quant_block(format, group);
    if len > 0 && k % len == 0 && len % 8 == 0 && t < index.len() * (k / 8) && 8 * t < elements(ko, out.len()) {
        let (i, e) = (t / (k / 8), t % (k / 8) * 8);
        let row_bytes = k / len * bytes;
        let start = index[i] as usize * row_bytes;
        if start + row_bytes <= w.len() {
            let (a, b, q) = decode_8(format, group, &w[start..start + row_bytes], e);
            let mut j = 0;
            while j < 8 {
                // SAFETY: as in `dequantize`.
                unsafe { write(ko, out.as_mut_ptr(), i * k + e + j, a.mul_add(q[j], b)) };
                j += 1;
            }
        }
    }
}

/// The `u16` at byte `at` of `row`, 2-byte aligned (the caller checks the
/// bounds of the block).
#[inline(always)]
fn u16_at(row: &[u8], at: usize) -> u32 {
    // SAFETY: the caller checked that the block holding `at..at + 2` lies
    // within `row`, and blocks and fields of the fast formats start at even
    // offsets of an aligned buffer.
    unsafe { *row.as_ptr().add(at).cast::<u16>() as u32 }
}

/// The `u32` at byte `at` of `row`, 4-byte aligned (as [`u16_at`]).
#[inline(always)]
fn u32_at(row: &[u8], at: usize) -> u32 {
    // SAFETY: as in `u16_at`, at offsets that are multiples of 4.
    unsafe { *row.as_ptr().add(at).cast::<u32>() }
}

/// The raw words of eight weights `e..e + 8` of a row of format `F` (one of
/// `qgemv_16`'s), read with aligned loads and nothing else: what
/// [`decode_raw`] turns into weights. Inlined, so that the loads of several
/// groups of eight are in flight together.
#[inline(always)]
fn load_raw<const F: u32>(group: usize, row: &[u8], e: usize) -> [u32; 10] {
    let mut r = [0u32; 10];
    if F == quant::INT8 {
        let base = e / group * (4 + group);
        let off = e % group;
        r[0] = u32_at(row, base);
        r[1] = u32_at(row, base + 4 + off);
        r[2] = u32_at(row, base + 8 + off);
    } else if F == quant::INT4 {
        let base = e / group * (4 + group / 2);
        r[0] = u32_at(row, base);
        r[1] = u32_at(row, base + 4 + e % group / 2);
    } else if F == quant::Q8_0 {
        let base = e / 32 * 34;
        let at = base + 2 + e % 32;
        r[0] = u16_at(row, base);
        r[1] = u16_at(row, at);
        r[2] = u16_at(row, at + 2);
        r[3] = u16_at(row, at + 4);
        r[4] = u16_at(row, at + 6);
    } else if F == quant::Q4_0 || F == quant::Q4_1 {
        // A block: the scale (and Q4_1's minimum), then 16 bytes whose low
        // nibbles are elements 0 to 15 and high nibbles 16 to 31.
        let size = if F == quant::Q4_1 { 20 } else { 18 };
        let base = e / 32 * size;
        let at = base + size - 16 + e % 16;
        r[0] = u16_at(row, base);
        r[1] = u16_at(row, at);
        r[2] = u16_at(row, at + 2);
        r[3] = u16_at(row, at + 4);
        r[4] = u16_at(row, at + 6);
        if F == quant::Q4_1 {
            r[5] = u16_at(row, base + 2);
        }
    } else if F == quant::Q4K {
        let base = e / 256 * 144;
        let off = e % 256;
        let at = base + 16 + off / 64 * 32 + off % 32;
        r[0] = u32_at(row, base);
        r[1] = u32_at(row, base + 4);
        r[2] = u32_at(row, base + 8);
        r[3] = u32_at(row, base + 12);
        r[4] = u32_at(row, at);
        r[5] = u32_at(row, at + 4);
    } else if F == quant::Q6K {
        let base = e / 256 * 210;
        let off = e % 256;
        let (h, m, l) = (off / 128, off % 64, off % 32);
        r[0] = u16_at(row, base + 208);
        r[1] = u16_at(row, base + 192 + (off / 16 & !1));
        let low = base + h * 64 + m;
        let high = base + 128 + h * 32 + l;
        r[2] = u16_at(row, low);
        r[3] = u16_at(row, low + 2);
        r[4] = u16_at(row, low + 4);
        r[5] = u16_at(row, low + 6);
        r[6] = u16_at(row, high);
        r[7] = u16_at(row, high + 2);
        r[8] = u16_at(row, high + 4);
        r[9] = u16_at(row, high + 6);
    }
    r
}

/// The `f16` of the low half of `bits`, widened by the hardware's conversion
/// (exact, as `widen`).
#[inline(always)]
fn f16_hw(bits: u32) -> f32 {
    let h = bits as u16;
    let v: f32;
    // SAFETY: a conversion between registers.
    unsafe {
        cuda_device::ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") v, in("h") h, options(register_only));
    }
    v
}

/// `q` as a float, for `0 ≤ q < 2²³`, without the slow conversion
/// instruction: the bits `2²³ + q`, minus `2²³`. Exact.
#[inline(always)]
fn small_float(q: u32) -> f32 {
    f32::from_bits(0x4b00_0000 | q) - 8_388_608.0
}

/// The eight weights of `load_raw`'s words for elements `e..e + 8`, as
/// `kapel_core`'s decoding computes them: `(a, b, q)`, weight `e + j` equal to
/// `a · q[j] + b`. Straight-line code, inlined: no loop, no call.
#[inline(always)]
fn decode_raw<const F: u32>(r: &[u32; 10], e: usize) -> (f32, f32, [f32; 8]) {
    let byte = |w: u32, i: u32| ((w >> (8 * i)) & 0xff) as u8;
    if F == quant::INT8 {
        // A signed byte `b` is `(b ^ 0x80) − 128`, an unsigned one.
        let c = |w: u32, i: u32| small_float(((w >> (8 * i)) & 0xff) ^ 0x80) - 128.0;
        let q = [c(r[1], 0), c(r[1], 1), c(r[1], 2), c(r[1], 3), c(r[2], 0), c(r[2], 1), c(r[2], 2), c(r[2], 3)];
        (f32::from_bits(r[0]), 0.0, q)
    } else if F == quant::INT4 {
        let c = |i: u32| small_float((r[1] >> (4 * i)) & 0x0f) - 8.0;
        (f32::from_bits(r[0]), 0.0, [c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)])
    } else if F == quant::Q8_0 {
        let c = |w: u32, i: u32| small_float(((w >> (8 * i)) & 0xff) ^ 0x80) - 128.0;
        let q = [c(r[1], 0), c(r[1], 1), c(r[2], 0), c(r[2], 1), c(r[3], 0), c(r[3], 1), c(r[4], 0), c(r[4], 1)];
        (f16_hw(r[0]), 0.0, q)
    } else if F == quant::Q4_0 || F == quant::Q4_1 {
        // Elements 16 to 31 of a block are the high nibbles.
        let t = (e % 32 / 16) as u32;
        let offset = if F == quant::Q4_1 { 0.0 } else { 8.0 };
        let c = |w: u32, i: u32| small_float((w >> (8 * i + 4 * t)) & 0x0f) - offset;
        let q = [c(r[1], 0), c(r[1], 1), c(r[2], 0), c(r[2], 1), c(r[3], 0), c(r[3], 1), c(r[4], 0), c(r[4], 1)];
        let min = if F == quant::Q4_1 { f16_hw(r[5]) } else { 0.0 };
        (f16_hw(r[0]), min, q)
    } else if F == quant::Q4K {
        let off = e % 256;
        let (t, k) = ((off % 64 / 32) as u32, off / 32);
        let (d, dmin) = (f16_hw(r[0]), f16_hw(r[0] >> 16));
        let (sc, mn) = if k < 4 {
            let k = k as u32;
            (byte(r[1], k) as u32 & 0x3f, byte(r[2], k) as u32 & 0x3f)
        } else {
            let i = (k - 4) as u32;
            let (s0, m0, md) = (byte(r[1], i) as u32, byte(r[2], i) as u32, byte(r[3], i) as u32);
            ((md & 0x0f) | ((s0 >> 2) & 0x30), (md >> 4) | ((m0 >> 2) & 0x30))
        };
        let c = |w: u32, i: u32| small_float((w >> (8 * i + 4 * t)) & 0x0f);
        let q = [c(r[4], 0), c(r[4], 1), c(r[4], 2), c(r[4], 3), c(r[5], 0), c(r[5], 1), c(r[5], 2), c(r[5], 3)];
        (d * sc as f32, -(dmin * mn as f32), q)
    } else if F == quant::Q6K {
        let off = e % 256;
        let (t, s) = ((off % 128 / 64) as u32, (off % 128 / 32) as u32);
        let scale = (r[1] >> (8 * (off / 16 % 2))) as u8 as i8 as f32;
        let c = |low: u32, high: u32, i: u32| {
            let lo = (low >> (8 * i + 4 * t)) & 0x0f;
            let hi = (high >> (8 * i + 2 * s)) & 0x03;
            small_float(lo | hi << 4) - 32.0
        };
        let q = [
            c(r[2], r[6], 0),
            c(r[2], r[6], 1),
            c(r[3], r[7], 0),
            c(r[3], r[7], 1),
            c(r[4], r[8], 0),
            c(r[4], r[8], 1),
            c(r[5], r[9], 0),
            c(r[5], r[9], 1),
        ];
        (f16_hw(r[0]) * scale, 0.0, q)
    } else {
        (f32::NAN, 0.0, [0.0; 8])
    }
}

/// `acc + a · Σ q·x + b · Σ x` for the eight 16-bit activations of format
/// `KX` packed in `xs`.
#[inline(always)]
fn dot_raw<const KX: u32>(a: f32, b: f32, q: &[f32; 8], xs: (u32, u32, u32, u32), acc: f32) -> f32 {
    let (x0, x1) = unpack(KX, xs.0);
    let (x2, x3) = unpack(KX, xs.1);
    let (x4, x5) = unpack(KX, xs.2);
    let (x6, x7) = unpack(KX, xs.3);
    let dot = q[7].mul_add(x7, q[6].mul_add(x6, q[5].mul_add(x5, q[4].mul_add(x4, q[3].mul_add(x3, q[2].mul_add(x2, q[1].mul_add(x1, q[0] * x0)))))));
    let sum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    b.mul_add(sum, a.mul_add(dot, acc))
}

/// The sums of `qgemv_16` for one weight row of format `F` and activations
/// of the 16-bit format `KX` (`half`): lane `l` takes the weights `8l..8l +
/// 8` of each 256 of the row, two groups in flight at a time. Inlined, its
/// rows of `x` written out: no call, no array indexed at run time.
#[inline(always)]
fn qgemv_row<const F: u32, const KX: u32>(group: usize, k: usize, row: &[u8], x: &[u32], rows: usize, lane: usize) -> [f32; 8] {
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut a4, mut a5, mut a6, mut a7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    // SAFETY (every load of `x`): `e + 8 ≤ k` (`k` a multiple of 8, `e`
    // too), so the 16 bytes lie within row `m < rows` of `x`, checked by the
    // kernel; rows of `x` start at multiples of 16 bytes.
    let xs = |m: usize, e: usize| unsafe { load_128(x.as_ptr().add((m * k + e) / 2)) };
    let mut e = lane * 8;
    while e < k {
        let second = e + 256 < k;
        let r0 = load_raw::<F>(group, row, e);
        let r1 = if second { load_raw::<F>(group, row, e + 256) } else { [0; 10] };
        let (wa, wb, wq) = decode_raw::<F>(&r0, e);
        a0 = dot_raw::<KX>(wa, wb, &wq, xs(0, e), a0);
        if rows > 1 { a1 = dot_raw::<KX>(wa, wb, &wq, xs(1, e), a1); }
        if rows > 2 { a2 = dot_raw::<KX>(wa, wb, &wq, xs(2, e), a2); }
        if rows > 3 { a3 = dot_raw::<KX>(wa, wb, &wq, xs(3, e), a3); }
        if rows > 4 { a4 = dot_raw::<KX>(wa, wb, &wq, xs(4, e), a4); }
        if rows > 5 { a5 = dot_raw::<KX>(wa, wb, &wq, xs(5, e), a5); }
        if rows > 6 { a6 = dot_raw::<KX>(wa, wb, &wq, xs(6, e), a6); }
        if rows > 7 { a7 = dot_raw::<KX>(wa, wb, &wq, xs(7, e), a7); }
        if second {
            let f = e + 256;
            let (wa, wb, wq) = decode_raw::<F>(&r1, f);
            a0 = dot_raw::<KX>(wa, wb, &wq, xs(0, f), a0);
            if rows > 1 { a1 = dot_raw::<KX>(wa, wb, &wq, xs(1, f), a1); }
            if rows > 2 { a2 = dot_raw::<KX>(wa, wb, &wq, xs(2, f), a2); }
            if rows > 3 { a3 = dot_raw::<KX>(wa, wb, &wq, xs(3, f), a3); }
            if rows > 4 { a4 = dot_raw::<KX>(wa, wb, &wq, xs(4, f), a4); }
            if rows > 5 { a5 = dot_raw::<KX>(wa, wb, &wq, xs(5, f), a5); }
            if rows > 6 { a6 = dot_raw::<KX>(wa, wb, &wq, xs(6, f), a6); }
            if rows > 7 { a7 = dot_raw::<KX>(wa, wb, &wq, xs(7, f), a7); }
        }
        e += 512;
    }
    [a0, a1, a2, a3, a4, a5, a6, a7]
}

/// The sum of `qgemv_row` for a single row of `x`: decoding.
#[inline(always)]
fn qgemv_one<const F: u32, const KX: u32>(group: usize, k: usize, row: &[u8], x: &[u32], lane: usize) -> f32 {
    let mut acc = 0.0f32;
    // SAFETY (every load of `x`): as in `qgemv_row`, for row 0.
    let xs = |e: usize| unsafe { load_128(x.as_ptr().add(e / 2)) };
    let mut e = lane * 8;
    while e < k {
        let second = e + 256 < k;
        let r0 = load_raw::<F>(group, row, e);
        let r1 = if second { load_raw::<F>(group, row, e + 256) } else { [0; 10] };
        let x0 = xs(e);
        let x1 = if second { xs(e + 256) } else { (0, 0, 0, 0) };
        let (wa, wb, wq) = decode_raw::<F>(&r0, e);
        acc = dot_raw::<KX>(wa, wb, &wq, x0, acc);
        if second {
            let (wa, wb, wq) = decode_raw::<F>(&r1, e + 256);
            acc = dot_raw::<KX>(wa, wb, &wq, x1, acc);
        }
        e += 512;
    }
    acc
}

/// The checks and the shape of a launch of the `qgemv_*` kernels: the row
/// of weights of this warp, its column, its lane, and the sizes; `None` for
/// a warp past the weights or arguments that do not fit.
#[inline(always)]
fn qgemv_launch<'w>(format: u32, group: usize, k: usize, w: &'w [u8], x: &[u32], rows: usize, c0: usize, width: usize, out_len: usize, n: usize) -> Option<&'w [u8]> {
    let (len, bytes) = quant_block(format, group);
    let row_bytes = if len > 0 { k / len * bytes } else { 0 };
    let columns = if row_bytes > 0 { w.len() / row_bytes } else { 0 };
    let fits = len > 0 && k % len == 0 && len % 8 == 0 && rows >= 1 && rows <= 8 && n < columns && c0 + columns <= width && rows * width <= out_len && rows * k / 2 <= x.len();
    if fits { Some(&w[n * row_bytes..(n + 1) * row_bytes]) } else { None }
}

/// The kernels `qgemv_<format>` (at most 8 rows of activations) and
/// `qgemv_<format>_1` (one row, decoding) of a format of `qgemv_16`'s: one
/// small kernel each, for the instruction cache.
macro_rules! qgemv_kernels {
    ($many:ident, $one:ident, $format:expr) => {
        /// `qgemv` for the format of its name, activations and output in one
        /// 16-bit format (`kind`, of `half`): aligned loads, eight weights
        /// decoded at once, the activations read in 128-bit loads.
        #[kernel]
        #[launch_bounds(256)]
        pub fn $many(group: u32, k: u32, w: &[u8], kind: u32, x: &[u32], rows: u32, c0: u32, width: u32, mut out: DisjointSlice<u16>) {
            let idx = thread::index_1d();
            let (n, lane) = (idx.get() / 32, idx.get() % 32);
            let (group, k, rows, c0, width) = (group as usize, k as usize, rows as usize, c0 as usize, width as usize);
            // The same for every lane of the warp: whole warps go on or stop.
            if let Some(row) = qgemv_launch($format, group, k, w, x, rows, c0, width, out.len(), n) {
                let acc = if kind == half::BF16 {
                    qgemv_row::<{ $format }, { half::BF16 }>(group, k, row, x, rows, lane)
                } else {
                    qgemv_row::<{ $format }, { half::F16 }>(group, k, row, x, rows, lane)
                };
                let sums = [warp_sum(acc[0]), warp_sum(acc[1]), warp_sum(acc[2]), warp_sum(acc[3]), warp_sum(acc[4]), warp_sum(acc[5]), warp_sum(acc[6]), warp_sum(acc[7])];
                if lane == 0 {
                    let o = out.as_mut_ptr();
                    let mut m = 0;
                    while m < 8 {
                        if m < rows {
                            // SAFETY: `m · width + c0 + n < rows · width ≤
                            // out.len()` (checked by `qgemv_launch`); the
                            // column belongs to this warp, and only its lane
                            // 0 writes.
                            unsafe { *o.add(m * width + c0 + n) = narrow(kind, sums[m]) };
                        }
                        m += 1;
                    }
                }
            }
        }

        /// The kernel of the same format for one row of activations.
        #[kernel]
        #[launch_bounds(256)]
        pub fn $one(group: u32, k: u32, w: &[u8], kind: u32, x: &[u32], c0: u32, width: u32, mut out: DisjointSlice<u16>) {
            let idx = thread::index_1d();
            let (n, lane) = (idx.get() / 32, idx.get() % 32);
            let (group, k, c0, width) = (group as usize, k as usize, c0 as usize, width as usize);
            // The same for every lane of the warp: whole warps go on or stop.
            if let Some(row) = qgemv_launch($format, group, k, w, x, 1, c0, width, out.len(), n) {
                let acc = if kind == half::BF16 {
                    qgemv_one::<{ $format }, { half::BF16 }>(group, k, row, x, lane)
                } else {
                    qgemv_one::<{ $format }, { half::F16 }>(group, k, row, x, lane)
                };
                let total = warp_sum(acc);
                if lane == 0 {
                    // SAFETY: `c0 + n < width ≤ out.len()` (checked by
                    // `qgemv_launch`); this warp's column, its lane 0 alone.
                    unsafe { *out.as_mut_ptr().add(c0 + n) = narrow(kind, total) };
                }
            }
        }
    };
}

qgemv_kernels!(qgemv_int8, qgemv_int8_1, quant::INT8);
qgemv_kernels!(qgemv_int4, qgemv_int4_1, quant::INT4);
qgemv_kernels!(qgemv_q8_0, qgemv_q8_0_1, quant::Q8_0);
qgemv_kernels!(qgemv_q4_0, qgemv_q4_0_1, quant::Q4_0);
qgemv_kernels!(qgemv_q4_1, qgemv_q4_1_1, quant::Q4_1);
qgemv_kernels!(qgemv_q4k, qgemv_q4k_1, quant::Q4K);
qgemv_kernels!(qgemv_q6k, qgemv_q6k_1, quant::Q6K);

/// The best of each part of `parts` of each row of `width` of `x` (float type
/// `kind`): its value and its position, as `argmax_rows` finds them, for
/// `argmax_merge` to merge. One block of 256 threads per row and part; a
/// part is `ceil(width / parts)` columns.
#[kernel]
#[launch_bounds(256)]
pub fn argmax_parts(kind: u32, x: &[u8], width: u32, parts: u32, mut values: DisjointSlice<f32>, mut at: DisjointSlice<u32>) {
    static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut AT: SharedArray<u32, 256> = SharedArray::UNINIT;
    let block = thread::blockIdx_x() as usize;
    let t = thread::threadIdx_x() as usize;
    let (width, parts) = (width as usize, parts as usize);
    let (row, part) = (block / parts.max(1), block % parts.max(1));
    let size = width.div_ceil(parts.max(1));
    let (from, to) = (part * size, if (part + 1) * size < width { (part + 1) * size } else { width });
    let start = row * width;
    if parts > 0 && block < values.len() && block < at.len() && start + width <= elements(kind, x.len()) {
        let (mut best, mut best_at) = (f32::NEG_INFINITY, u32::MAX);
        let mut c = from + t;
        while c < to {
            let v = read(kind, x, start + c);
            if !v.is_nan() && (best_at == u32::MAX || v > best) {
                best = v;
                best_at = c as u32;
            }
            c += 256;
        }
        // SAFETY: each thread writes its own slot, read after the barrier.
        unsafe {
            VALUES[t] = best;
            AT[t] = best_at;
        }
        thread::sync_threads();
        if t == 0 {
            let (mut best, mut best_at) = (f32::NEG_INFINITY, u32::MAX);
            let mut i = 0;
            while i < 256 {
                // SAFETY: written before the barrier above.
                let (v, a) = unsafe { (VALUES[i], AT[i]) };
                if a != u32::MAX && (best_at == u32::MAX || v > best || (v == best && a < best_at)) {
                    best = v;
                    best_at = a;
                }
                i += 1;
            }
            // SAFETY: `block` is below both lengths (checked above), and only
            // thread 0 of this block writes its slots.
            unsafe {
                *values.as_mut_ptr().add(block) = best;
                *at.as_mut_ptr().add(block) = best_at;
            }
        }
    }
}

/// The position of each row's maximum from the `parts` bests of
/// `argmax_parts`, merged in order: the larger value, the smaller position
/// on a tie; 0 for a row of NaN. One thread per row.
#[kernel]
#[launch_bounds(256)]
pub fn argmax_merge(parts: u32, values: &[f32], at: &[u32], mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let row = idx.get();
    let parts = parts as usize;
    if let Some(y) = out.get_mut(idx) {
        let (mut best, mut best_at) = (f32::NEG_INFINITY, u32::MAX);
        let mut p = 0;
        while p < parts {
            let (v, a) = (values[row * parts + p], at[row * parts + p]);
            if a != u32::MAX && (best_at == u32::MAX || v > best || (v == best && a < best_at)) {
                best = v;
                best_at = a;
            }
            p += 1;
        }
        *y = if best_at == u32::MAX { 0 } else { best_at };
    }
}
