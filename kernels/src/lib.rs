//! Kapel's CUDA kernels.
//!
//! Each kernel follows the simplest scheme that gives the same bits on every
//! run: one thread per output element (or per line, row, query), which reads
//! what it needs and writes its own output. No two threads write the same place,
//! and no sum depends on scheduling: there are no atomics.
//!
//! Matrix products and attention are not here: they are tile kernels written
//! with cutile-rs, in the backend itself.
//!
//! The arguments passed by value are defined in `kapel-cuda-abi`. A slice
//! argument becomes two kernel parameters: its device pointer, then its length.

// Kernels index explicitly: the indexing is the memory access each thread makes.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};
use kapel_cuda_abi::{View, binary, unary};

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

/// `out[i] = x[view.at(i)]`: a strided view made contiguous.
#[kernel]
#[launch_bounds(256)]
pub fn copy_view(x: &[f32], view: View, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = x[view.at(i as u32) as usize];
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
pub fn unary_backward(op: u32, x: &[f32], g: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(y) = out.get_mut(idx) {
        *y = g[i] * unary_slope(op, x[i]);
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
pub fn reduce_sum(x: &[f32], bases: &[u32], inner: &[u32], scale: f32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let o = idx.get();
    if let Some(y) = out.get_mut(idx) {
        let base = bases[o] as usize;
        let mut total = 0.0f32;
        for r in 0..inner.len() {
            total += x[base + inner[r] as usize];
        }
        *y = scale * total;
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
    if l < starts.len() {
        let start = starts[l];
        let mut best = start;
        for i in 1..len {
            let p = start + i * step;
            if x[p as usize] > x[best as usize] {
                best = p;
            }
        }
        // SAFETY: one thread per line, and each line has its own slot.
        unsafe {
            *value.as_mut_ptr().add(l) = x[best as usize];
            *at.as_mut_ptr().add(l) = best;
        }
    }
}

/// `out[at[l]] = g[l]` for every line; `out` starts at zero.
#[kernel]
#[launch_bounds(256)]
pub fn place(g: &[f32], at: &[u32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < at.len() {
        // SAFETY: the positions of distinct lines are distinct elements.
        unsafe { *out.as_mut_ptr().add(at[l] as usize) = g[l] };
    }
}

/// Softmax (`log` = 0) or log-softmax (`log` = 1) of every line, one thread per
/// line, after subtracting the line's maximum.
#[kernel]
#[launch_bounds(256)]
pub fn softmax_lines(log: u32, x: &[f32], starts: &[u32], step: u32, len: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let l = idx.get();
    if l < starts.len() {
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
            // SAFETY: the elements of a line belong to it alone.
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
    if l < starts.len() {
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
            // SAFETY: as in `softmax_lines`.
            unsafe { *out.as_mut_ptr().add(p) = value };
        }
    }
}

/// LayerNorm (`rms` = 0) or RMSNorm (`rms` = 1) of every row of `width`:
/// `y = x̂ · (offset + w) + b`, one thread per row.
#[kernel]
#[launch_bounds(256)]
pub fn norm_rows(rms: u32, x: &[f32], w: &[f32], b: &[f32], offset: f32, eps: f32, width: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let row = idx.get();
    let width = width as usize;
    if row * width < x.len() {
        let start = row * width;
        let mut mean = 0.0f32;
        if rms == 0 {
            for c in 0..width {
                mean += x[start + c];
            }
            mean /= width as f32;
        }
        let mut squares = 0.0f32;
        for c in 0..width {
            let v = x[start + c] - mean;
            squares += v * v;
        }
        let inv = 1.0 / (squares / width as f32 + eps).sqrt();
        for c in 0..width {
            let hat = (x[start + c] - mean) * inv;
            // SAFETY: each row is written by its own thread.
            unsafe { *out.as_mut_ptr().add(start + c) = hat * (offset + w[c]) + b[c] };
        }
    }
}

/// The backward pass of `norm_rows`: the gradient of `x`, and for every row its
/// contributions `g·x̂` and `g` to the gradients of the weight and the bias,
/// summed over rows afterwards by `column_sum`.
#[kernel]
#[launch_bounds(256)]
pub fn norm_rows_backward(
    rms: u32,
    x: &[f32],
    w: &[f32],
    g: &[f32],
    offset: f32,
    eps: f32,
    width: u32,
    mut gx: DisjointSlice<f32>,
    mut gw_rows: DisjointSlice<f32>,
    mut gb_rows: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let row = idx.get();
    let width = width as usize;
    if row * width < x.len() {
        let start = row * width;
        let n = width as f32;
        let mut mean = 0.0f32;
        if rms == 0 {
            for c in 0..width {
                mean += x[start + c];
            }
            mean /= n;
        }
        let mut squares = 0.0f32;
        for c in 0..width {
            let v = x[start + c] - mean;
            squares += v * v;
        }
        let inv = 1.0 / (squares / n + eps).sqrt();
        let (mut mean_gh, mut mean_ghh) = (0.0f32, 0.0f32);
        for c in 0..width {
            let hat = (x[start + c] - mean) * inv;
            let gh = g[start + c] * (offset + w[c]);
            mean_gh += gh;
            mean_ghh += gh * hat;
        }
        mean_gh /= n;
        mean_ghh /= n;
        if rms == 1 {
            mean_gh = 0.0;
        }
        for c in 0..width {
            let hat = (x[start + c] - mean) * inv;
            let gh = g[start + c] * (offset + w[c]);
            // SAFETY: each row is written by its own thread.
            unsafe {
                *gx.as_mut_ptr().add(start + c) = inv * (gh - mean_gh - hat * mean_ghh);
                *gw_rows.as_mut_ptr().add(start + c) = g[start + c] * hat;
                *gb_rows.as_mut_ptr().add(start + c) = g[start + c];
            }
        }
    }
}

/// `out[c] = Σ_r x[r·width + c]`, the rows added in order.
#[kernel]
#[launch_bounds(256)]
pub fn column_sum(x: &[f32], rows: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let c = idx.get();
    let width = out.len();
    if let Some(y) = out.get_mut(idx) {
        let mut total = 0.0f32;
        for r in 0..rows as usize {
            total += x[r * width + c];
        }
        *y = total;
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
