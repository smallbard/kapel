//! Products by quantized weights with activations in bytes.
//!
//! The kernels of [`crate::quantized`] convert each code of the weights to
//! `f32` before multiplying it: several instructions for sixteen weights.
//! Here each row of `x` is quantized in turn, by blocks of 32 elements, as
//! llama.cpp does: `x ≈ s · q` with `s = max |x| / 127` and integers `q`
//! between −127 and 127, rounded to the nearest (ties to even). A group of weights, codes `c` with a scale `d` and a
//! minimum `m` (`w = d · c − m`), then gives
//!
//! ```text
//! Σ w · x ≈ d · s · Σ c · q  −  m · s · Σ q
//! ```
//!
//! where `Σ c · q` is a sum of products of bytes, exact in 32-bit integers:
//! `vpdpbusd` (AVX-512 VNNI) adds 32 of them in one instruction, without
//! converting anything. `s · Σ q` is computed once per row of `x`, for all
//! the rows of `w`.
//!
//! This is no longer the product by the dequantized weights: the rounding of
//! the activations adds to that of the weights. So the mode is chosen
//! explicitly ([`crate::Cpu::with_byte_activations`]), and its effect is
//! measured on the perplexity.
//!
//! The formats read this way are those of the generation of a usual model
//! (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, and Kapel's int8 and int4 by groups of 32,
//! 64 or 128); the others go through [`crate::quantized`].

use kapel_core::QuantFormat;

use crate::Isa;
use crate::pool::{Pool, SharedMut};
use crate::quantized::{Task, Weights, tasks};

/// Elements per block of the quantized activations.
const BLOCK: usize = 32;

/// The rows of `x` multiplied by the same codes at a time.
const ROWS: usize = 4;

/// How a format is read: its block, its groups (elements per scale), and
/// whether it has minimums.
#[derive(Clone, Copy)]
struct Layout {
    len: usize,
    bytes: usize,
    group: usize,
    minimum: bool,
}

/// The layout of `format`, if it is read with activations in bytes.
fn layout(format: QuantFormat) -> Option<Layout> {
    let l = |len, bytes, group, minimum| {
        Some(Layout {
            len,
            bytes,
            group,
            minimum,
        })
    };
    match format {
        QuantFormat::Q8_0 => l(32, 34, 32, false),
        QuantFormat::Q4_0 => l(32, 18, 32, true),
        QuantFormat::Q4K => l(256, 144, 32, true),
        QuantFormat::Q5K => l(256, 176, 32, true),
        // Q6_K's offset of 32 goes into its codes, signed.
        QuantFormat::Q6K => l(256, 210, 16, false),
        QuantFormat::Int8 { group } if matches!(group, 32 | 64 | 128) => {
            let g = usize::from(group);
            l(g, 4 + g, g, false)
        }
        QuantFormat::Int4 { group } if matches!(group, 32 | 64 | 128) => {
            let g = usize::from(group);
            l(g, 4 + g / 2, g, true)
        }
        _ => None,
    }
}

/// Whether `format` is read with activations in bytes.
pub(crate) fn supported(format: QuantFormat) -> bool {
    layout(format).is_some()
}

/// Whether this processor has AVX-512 VNNI, with the instructions of
/// [`crate::IsaKind::Avx512`].
pub(crate) fn vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::IsaKind::Avx512.available() && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The rows of `x`, quantized by blocks of [`BLOCK`].
struct Activations {
    /// The integers, `[rows, k]`.
    q: Vec<i8>,
    /// The scale of each block, `[rows, k / BLOCK]`.
    scale: Vec<f32>,
    /// For each size of group that has minimums, `s · Σ q` over each group,
    /// `[rows, k / group]`.
    sums: Vec<(usize, Vec<f32>)>,
}

/// `x [rows, k]` in bytes, with the sums of its groups of each of `groups`
/// elements.
fn quantize(isa: Isa, x: &[f32], rows: usize, k: usize, groups: &[usize]) -> Activations {
    let mut q = vec![0i8; rows * k];
    let mut scale = vec![0.0f32; rows * k / BLOCK];
    blocks_to_bytes(isa, x, &mut q, &mut scale);
    let sums = groups
        .iter()
        .map(|&group| {
            let mut sums = vec![0.0f32; rows * k / group];
            for (g, sum) in sums.iter_mut().enumerate() {
                // A group of weights may span several blocks of activations,
                // each with its scale; groups of 16 lie within one.
                let start = g * group;
                *sum = (start..start + group)
                    .step_by(group.min(BLOCK))
                    .map(|at| {
                        let piece = &q[at..at + group.min(BLOCK)];
                        let total: i32 = piece.iter().map(|&v| i32::from(v)).sum();
                        scale[at / BLOCK] * total as f32
                    })
                    .sum();
            }
            (group, sums)
        })
        .collect();
    Activations { q, scale, sums }
}

multiversion! {
    /// `q`, `scale`: the blocks of `x` in bytes, `x ≈ scale · q`, rounded to
    /// the nearest, ties to even.
    fn blocks_to_bytes(x: &[f32], q: &mut [i8], scale: &mut [f32]) -> () = to_bytes_body;
}

#[inline(always)]
fn to_bytes_body<const FMA: bool>(x: &[f32], q: &mut [i8], scale: &mut [f32]) {
    let (blocks, _) = x.as_chunks::<BLOCK>();
    let (codes, _) = q.as_chunks_mut::<BLOCK>();
    for ((x, q), s) in blocks.iter().zip(codes).zip(scale) {
        let mut max = 0.0f32;
        for &v in x {
            max = max.max(v.abs());
        }
        *s = max / 127.0;
        let inverse = if max > 0.0 { 127.0 / max } else { 0.0 };
        for (q, &x) in q.iter_mut().zip(x) {
            *q = (x * inverse).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// `out[r, :] = x[r] · [w₁; w₂; …]ᵀ` with the rows of `x` in bytes, for the
/// rows of the matrices `weights` stacked, all of supported formats (see
/// [`supported`]). `out` is `[rows, Σ nᵢ]`.
///
/// # Panics
///
/// If the sizes do not match, or a format is not supported.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product(
    pool: &Pool,
    isa: Isa,
    vnni: bool,
    weights: &[Weights<'_>],
    k: usize,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    let width: usize = weights.iter().map(|w| w.n).sum();
    let mut groups = Vec::new();
    for w in weights {
        let l = layout(w.format).expect("a supported format");
        assert!(k.is_multiple_of(l.len), "whole blocks");
        assert_eq!(w.bytes.len(), w.n * (k / l.len * l.bytes), "w holds n rows");
        if l.minimum && !groups.contains(&l.group) {
            groups.push(l.group);
        }
    }
    assert_eq!(x.len(), rows * k, "x holds rows × k values");
    assert_eq!(out.len(), rows * width, "out holds rows × n values");
    if width == 0 || rows == 0 {
        return;
    }
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let a = quantize(isa, x, rows, k, &groups);
    let tasks = tasks(pool.threads(), weights);
    let shared = SharedMut::new(out);
    pool.run(tasks.len(), |task| {
        let Task {
            matrix,
            first,
            count,
            column,
        } = tasks[task];
        let w = weights[matrix];
        let l = layout(w.format).expect("checked above");
        let row_bytes = w.bytes.len() / w.n;
        let bytes = &w.bytes[first * row_bytes..(first + count) * row_bytes];
        let sums = a
            .sums
            .iter()
            .find(|(g, _)| *g == l.group)
            .map_or(&[][..], |(_, s)| s);
        let mut part = vec![0.0f32; rows * count];
        for r0 in (0..rows).step_by(ROWS) {
            let taken = ROWS.min(rows - r0);
            let view = View {
                a: &a,
                sums,
                first: r0,
                rows: taken,
                k,
            };
            let part = &mut part[r0 * count..(r0 + taken) * count];
            match () {
                // SAFETY: `vnni` is only set after checking that the
                // processor has AVX-512 VNNI and the instructions of `vnni`.
                #[cfg(target_arch = "x86_64")]
                () if vnni => unsafe { vnni::rows(w.format, l, bytes, &view, part) },
                () => portable(w.format, l, bytes, &view, part),
            }
        }
        for r in 0..rows {
            // SAFETY: the task writes columns `column .. column + count` of
            // each row of `out`, which no other task writes.
            let line = unsafe { shared.slice(r * width + column, count) };
            line.copy_from_slice(&part[r * count..][..count]);
        }
    });
}

/// Rows `first .. first + rows` of quantized activations, with the sums of
/// the groups of one format.
struct View<'a> {
    a: &'a Activations,
    sums: &'a [f32],
    first: usize,
    rows: usize,
    k: usize,
}

impl View<'_> {
    fn q(&self, r: usize) -> &[i8] {
        &self.a.q[(self.first + r) * self.k..][..self.k]
    }

    fn scale(&self, r: usize) -> &[f32] {
        let blocks = self.k / BLOCK;
        &self.a.scale[(self.first + r) * blocks..][..blocks]
    }

    fn sums(&self, r: usize) -> &[f32] {
        let rows = self.a.q.len() / self.k;
        let groups = self.sums.len() / rows;
        self.sums
            .get((self.first + r) * groups..)
            .map_or(&[], |s| &s[..groups])
    }
}

/// The scales and minimums of the groups of a block (up to 16 groups), and
/// its codes: `w = scale · code − minimum`.
struct Decoded {
    scale: [f32; 16],
    minimum: [f32; 16],
    codes: [i16; 256],
}

/// The codes, scales and minimums of one block of `format`.
fn decode(format: QuantFormat, block: &[u8]) -> Decoded {
    use crate::quantized::{half, k_scales, single};
    let mut d = Decoded {
        scale: [0.0; 16],
        minimum: [0.0; 16],
        codes: [0; 256],
    };
    match format {
        QuantFormat::Q8_0 => {
            d.scale[0] = half(block, 0);
            for (c, &b) in d.codes.iter_mut().zip(&block[2..34]) {
                *c = i16::from(b as i8);
            }
        }
        QuantFormat::Q4_0 => {
            d.scale[0] = half(block, 0);
            d.minimum[0] = 8.0 * d.scale[0];
            for (j, &b) in block[2..18].iter().enumerate() {
                d.codes[j] = i16::from(b & 0x0f);
                d.codes[j + 16] = i16::from(b >> 4);
            }
        }
        QuantFormat::Q4K | QuantFormat::Q5K => {
            let (scale, minimum) = k_scales(block);
            d.scale[..8].copy_from_slice(&scale);
            d.minimum[..8].copy_from_slice(&minimum);
            let five = format == QuantFormat::Q5K;
            let codes = if five {
                &block[48..176]
            } else {
                &block[16..144]
            };
            for e in 0..256 {
                let (c, t, l) = (e / 64, e % 64 / 32, e % 32);
                let mut q = (codes[c * 32 + l] >> (4 * t)) & 0x0f;
                if five {
                    q |= ((block[16 + l] >> (e / 32)) & 1) << 4;
                }
                d.codes[e] = i16::from(q);
            }
        }
        QuantFormat::Q6K => {
            let dd = half(block, 208);
            for (i, &k) in block[192..208].iter().enumerate() {
                d.scale[i] = dd * f32::from(k as i8);
            }
            for f in 0..256 {
                let (h, t, m) = (f / 128, f % 128 / 64, f % 64);
                let ql = (block[h * 64 + m] >> (4 * t)) & 0x0f;
                let (s, l) = (f % 128 / 32, f % 32);
                let qh = (block[128 + h * 32 + l] >> (2 * s)) & 0x03;
                d.codes[f] = i16::from(ql | (qh << 4)) - 32;
            }
        }
        QuantFormat::Int8 { group } => {
            d.scale[0] = single(block);
            for (c, &b) in d.codes.iter_mut().zip(&block[4..4 + usize::from(group)]) {
                *c = i16::from(b as i8);
            }
        }
        QuantFormat::Int4 { group } => {
            d.scale[0] = single(block);
            d.minimum[0] = 8.0 * d.scale[0];
            for (i, &b) in block[4..4 + usize::from(group) / 2].iter().enumerate() {
                d.codes[2 * i] = i16::from(b & 0x0f);
                d.codes[2 * i + 1] = i16::from(b >> 4);
            }
        }
        _ => unreachable!("checked by `supported`"),
    }
    d
}

/// The products, one integer sum per group: on any processor.
fn portable(format: QuantFormat, l: Layout, w: &[u8], view: &View<'_>, out: &mut [f32]) {
    let count = out.len() / view.rows;
    let k = view.k;
    for j in 0..count {
        let row = &w[j * (k / l.len) * l.bytes..][..(k / l.len) * l.bytes];
        let mut acc = [0.0f32; ROWS];
        let mut low = [0.0f32; ROWS];
        for (b, block) in row.chunks_exact(l.bytes).enumerate() {
            let d = decode(format, block);
            for r in 0..view.rows {
                let (q, scale, sums) = (view.q(r), view.scale(r), view.sums(r));
                // One sum per block of the activations and group of weights.
                for c in 0..l.len / BLOCK {
                    let at = b * l.len + c * BLOCK;
                    let s = scale[at / BLOCK];
                    for half in 0..BLOCK / l.group.min(BLOCK) {
                        let width = l.group.min(BLOCK);
                        let start = c * BLOCK + half * width;
                        let g = start / l.group;
                        let dot: i32 = (0..width)
                            .map(|i| {
                                i32::from(d.codes[start + i]) * i32::from(q[b * l.len + start + i])
                            })
                            .sum();
                        acc[r] += d.scale[g] * s * dot as f32;
                    }
                }
                if l.minimum {
                    for g in 0..l.len / l.group {
                        low[r] += d.minimum[g] * sums[b * l.len / l.group + g];
                    }
                }
            }
        }
        for r in 0..view.rows {
            out[r * count + j] = acc[r] - low[r];
        }
    }
}

/// The products with `vpdpbusd`, 32 bytes at a time.
#[cfg(target_arch = "x86_64")]
mod vnni {
    use std::arch::x86_64::*;

    use kapel_core::QuantFormat;

    use super::{BLOCK, Layout, View};
    use crate::quantized::{half, k_scales, single};

    /// The scales and minimums of the groups of a block.
    type Scales = ([f32; 16], [f32; 16]);

    /// [`super::portable`], with these instructions.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    pub(super) fn rows(format: QuantFormat, l: Layout, w: &[u8], view: &View<'_>, out: &mut [f32]) {
        match view.rows {
            4 => by_format::<4>(format, l, w, view, out),
            3 => by_format::<3>(format, l, w, view, out),
            2 => by_format::<2>(format, l, w, view, out),
            _ => by_format::<1>(format, l, w, view, out),
        }
    }

    /// Thirty-two bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn load32(b: &[u8]) -> __m256i {
        let b: &[u8; 32] = b[..32].try_into().expect("32 bytes");
        // SAFETY: `b` holds the 32 bytes read.
        unsafe { _mm256_loadu_si256(b.as_ptr().cast()) }
    }

    /// Sixteen bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn load16(b: &[u8]) -> __m128i {
        let b: &[u8; 16] = b[..16].try_into().expect("16 bytes");
        // SAFETY: `b` holds the 16 bytes read.
        unsafe { _mm_loadu_si128(b.as_ptr().cast()) }
    }

    /// `(v >> shift) & mask` on each byte (`shift < 8`).
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn take(v: __m256i, shift: usize, mask: i8) -> __m256i {
        let shifted = _mm256_srl_epi16(v, _mm_cvtsi64_si128(shift as i64));
        _mm256_and_si256(shifted, _mm256_set1_epi8(mask))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn take16(v: __m128i, shift: usize, mask: i8) -> __m128i {
        let shifted = _mm_srl_epi16(v, _mm_cvtsi64_si128(shift as i64));
        _mm_and_si128(shifted, _mm_set1_epi8(mask))
    }

    fn one(scale: f32, minimum: f32) -> Scales {
        let mut s = ([0.0; 16], [0.0; 16]);
        s.0[0] = scale;
        s.1[0] = minimum;
        s
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn by_format<const R: usize>(
        format: QuantFormat,
        l: Layout,
        w: &[u8],
        view: &View<'_>,
        out: &mut [f32],
    ) {
        let k_formats = |b: &[u8]| {
            let (scale, minimum) = k_scales(b);
            let mut s = ([0.0; 16], [0.0; 16]);
            s.0[..8].copy_from_slice(&scale);
            s.1[..8].copy_from_slice(&minimum);
            s
        };
        let kapel_signed = |b: &[u8]| one(single(b), 0.0);
        let kapel_int4 = |b: &[u8]| {
            let s = single(b);
            one(s, 8.0 * s)
        };
        let int8 = |b: &[u8], c: usize| load32(&b[4 + 32 * c..]);
        let int4 = |b: &[u8], c: usize| {
            // Elements 2i and 2i + 1 are the nibbles of byte i.
            let v = load16(&b[4 + 16 * c..]);
            let (low, high) = (take16(v, 0, 0x0f), take16(v, 4, 0x0f));
            _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high))
        };
        match format {
            QuantFormat::Q8_0 => fused::<R, 32, 34, 32, true, false>(
                w,
                view,
                out,
                |b| one(half(b, 0), 0.0),
                |b, _| load32(&b[2..]),
            ),
            QuantFormat::Q4_0 => fused::<R, 32, 18, 32, false, true>(
                w,
                view,
                out,
                |b| {
                    let d = half(b, 0);
                    one(d, 8.0 * d)
                },
                |b, _| {
                    let v = load16(&b[2..]);
                    _mm256_set_m128i(take16(v, 4, 0x0f), take16(v, 0, 0x0f))
                },
            ),
            QuantFormat::Q4K => {
                fused::<R, 256, 144, 32, false, true>(w, view, out, k_formats, |b, c| {
                    take(load32(&b[16 + c / 2 * 32..]), 4 * (c % 2), 0x0f)
                })
            }
            QuantFormat::Q5K => {
                fused::<R, 256, 176, 32, false, true>(w, view, out, k_formats, |b, c| {
                    let low = take(load32(&b[48 + c / 2 * 32..]), 4 * (c % 2), 0x0f);
                    let top = take(load32(&b[16..]), c, 1);
                    _mm256_or_si256(low, _mm256_slli_epi16::<4>(top))
                })
            }
            QuantFormat::Q6K => fused::<R, 256, 210, 16, true, false>(
                w,
                view,
                out,
                |b| {
                    let mut s = ([0.0; 16], [0.0; 16]);
                    let scales = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(load16(&b[192..])));
                    let scales = _mm512_mul_ps(scales, _mm512_set1_ps(half(b, 208)));
                    // SAFETY: `s.0` holds the 16 values written.
                    unsafe { _mm512_storeu_ps(s.0.as_mut_ptr(), scales) };
                    s
                },
                |b, c| {
                    let (h, s) = (c / 4, c % 4);
                    let ql = take(load32(&b[h * 64 + (s % 2) * 32..]), 4 * (s / 2), 0x0f);
                    let qh = take(load32(&b[128 + h * 32..]), 2 * s, 3);
                    let q = _mm256_or_si256(ql, _mm256_slli_epi16::<4>(qh));
                    _mm256_sub_epi8(q, _mm256_set1_epi8(32))
                },
            ),
            QuantFormat::Int8 { group: 32 } => {
                fused::<R, 32, 36, 32, true, false>(w, view, out, kapel_signed, int8);
            }
            QuantFormat::Int8 { group: 64 } => {
                fused::<R, 64, 68, 64, true, false>(w, view, out, kapel_signed, int8);
            }
            QuantFormat::Int8 { group: 128 } => {
                fused::<R, 128, 132, 128, true, false>(w, view, out, kapel_signed, int8);
            }
            QuantFormat::Int4 { group: 32 } => {
                fused::<R, 32, 20, 32, false, true>(w, view, out, kapel_int4, int4);
            }
            QuantFormat::Int4 { group: 64 } => {
                fused::<R, 64, 36, 64, false, true>(w, view, out, kapel_int4, int4);
            }
            QuantFormat::Int4 { group: 128 } => {
                fused::<R, 128, 68, 128, false, true>(w, view, out, kapel_int4, int4);
            }
            _ => super::portable(format, l, w, view, out),
        }
    }

    /// `out[r, j] = x[r] · w[j]` for `R` rows of activations in bytes, with
    /// blocks of `LEN` elements in `BYTES` bytes, a scale (and with `MIN` a
    /// minimum) per `GROUP` elements, and codes that are signed bytes with
    /// `SIGNED`, unsigned otherwise. `codes` reads codes `32 c … 32 c + 31`
    /// of a block.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn fused<
        const R: usize,
        const LEN: usize,
        const BYTES: usize,
        const GROUP: usize,
        const SIGNED: bool,
        const MIN: bool,
    >(
        w: &[u8],
        view: &View<'_>,
        out: &mut [f32],
        scales: impl Fn(&[u8]) -> Scales,
        codes: impl Fn(&[u8], usize) -> __m256i,
    ) {
        debug_assert_eq!(view.rows, R);
        let k = view.k;
        let count = out.len() / R;
        let blocks = k / LEN;
        let rows: [(&[i8], &[f32], &[f32]); R] =
            std::array::from_fn(|r| (view.q(r), view.scale(r), view.sums(r)));
        for j in 0..count {
            let row = &w[j * blocks * BYTES..][..blocks * BYTES];
            let mut acc = [_mm256_setzero_ps(); R];
            let mut low = [0.0f32; R];
            let mut lows = [_mm256_setzero_ps(); R];
            for (b, block) in row.as_chunks::<BYTES>().0.iter().enumerate() {
                let (scale, minimum) = scales(block);
                // The scale of each lane, for each block of the activations:
                // shared by the rows of `x`.
                // At most 8 blocks of 32 in a block of weights.
                let factors: [__m256; 8] = std::array::from_fn(|c| {
                    if c >= LEN / BLOCK {
                        _mm256_setzero_ps()
                    } else if GROUP >= BLOCK {
                        _mm256_set1_ps(scale[c * BLOCK / GROUP])
                    } else {
                        // Two groups of 16: four lanes each.
                        _mm256_set_m128(_mm_set1_ps(scale[2 * c + 1]), _mm_set1_ps(scale[2 * c]))
                    }
                });
                for (c, &factor) in factors.iter().enumerate().take(LEN / BLOCK) {
                    let u = codes(block, c);
                    // Signed codes: `vpdpbusd` multiplies unsigned bytes by
                    // signed ones, so the signs move to the activations.
                    let (u, sign) = if SIGNED {
                        (_mm256_abs_epi8(u), u)
                    } else {
                        (u, _mm256_setzero_si256())
                    };
                    for (r, &(q, s, _)) in rows.iter().enumerate() {
                        let at = b * LEN + c * BLOCK;
                        let mut xq = load32(bytes_of(&q[at..at + BLOCK]));
                        if SIGNED {
                            xq = _mm256_sign_epi8(xq, sign);
                        }
                        let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, xq);
                        let factor = _mm256_mul_ps(factor, _mm256_set1_ps(s[at / BLOCK]));
                        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), factor, acc[r]);
                    }
                }
                if MIN && LEN / GROUP == 8 {
                    // The eight minimums of a K block, in one product.
                    // SAFETY: `minimum` holds the 16 values, 8 of them read.
                    let m = unsafe { _mm256_loadu_ps(minimum.as_ptr()) };
                    for (r, &(_, _, sums)) in rows.iter().enumerate() {
                        let sums = &sums[b * 8..b * 8 + 8];
                        // SAFETY: `sums` holds the 8 values read.
                        let v = unsafe { _mm256_loadu_ps(sums.as_ptr()) };
                        lows[r] = _mm256_fmadd_ps(m, v, lows[r]);
                    }
                } else if MIN {
                    for (r, &(_, _, sums)) in rows.iter().enumerate() {
                        for g in 0..LEN / GROUP {
                            low[r] += minimum[g] * sums[b * (LEN / GROUP) + g];
                        }
                    }
                }
            }
            for r in 0..R {
                out[r * count + j] = reduce(acc[r]) - (low[r] + reduce(lows[r]));
            }
        }
    }

    /// The bytes of signed bytes.
    fn bytes_of(q: &[i8]) -> &[u8] {
        // SAFETY: `i8` and `u8` have the same size and alignment, and every
        // bit pattern is valid for both.
        unsafe { std::slice::from_raw_parts(q.as_ptr().cast(), q.len()) }
    }

    /// The sum of eight lanes, pairwise.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
    #[inline]
    fn reduce(v: __m256) -> f32 {
        let quad = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
        let one = _mm_add_ss(pair, _mm_movehdup_ps(pair));
        _mm_cvtss_f32(one)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapel_core::{DType, F16, dequantize};

    /// Random blocks of `format`, with `f16` scales at their places.
    fn blocks(format: QuantFormat, count: usize, seed: u64) -> Vec<u8> {
        let dtype = DType::Quantized(format);
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let halves: &[usize] = match format {
            QuantFormat::Q8_0 | QuantFormat::Q4_0 => &[0],
            QuantFormat::Q4K | QuantFormat::Q5K => &[0, 2],
            QuantFormat::Q6K => &[208],
            _ => &[],
        };
        let mut out = Vec::new();
        for _ in 0..count {
            let mut block: Vec<u8> = (0..dtype.block_bytes()).map(|_| next()).collect();
            for &at in halves {
                let value = 0.001 + f64::from(next()) / 255.0 * 0.02;
                block[at..at + 2].copy_from_slice(&F16::from_f64(value).0.to_le_bytes());
            }
            if halves.is_empty() {
                block[..4].copy_from_slice(&0.0125f32.to_le_bytes());
            }
            out.extend(block);
        }
        out
    }

    const FORMATS: [QuantFormat; 9] = [
        QuantFormat::Q8_0,
        QuantFormat::Q4_0,
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
        QuantFormat::Int8 { group: 32 },
        QuantFormat::Int8 { group: 128 },
        QuantFormat::Int4 { group: 32 },
        QuantFormat::Int4 { group: 64 },
    ];

    #[test]
    fn products_are_those_of_the_activations_in_bytes() {
        let (n, k, rows) = (13, 1024, 6);
        // Blocks of very different sizes: each has its own scale.
        let x: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 7919) % 201) as f32 / 100.0 - 1.0) * (1 + (i / 32) % 7) as f32)
            .collect();
        let vnni = vnni_available();
        for format in FORMATS {
            let dtype = DType::Quantized(format);
            let w = blocks(format, n * k / dtype.block_len(), 3);
            let dense = dequantize(dtype, &w).unwrap();
            let pool = Pool::new(3);
            let mut results = Vec::new();
            for vnni in [false, vnni] {
                let mut out = vec![0.0f32; rows * n];
                let weights = [Weights {
                    format,
                    bytes: &w,
                    n,
                }];
                product(
                    &pool,
                    crate::Isa::detect(),
                    vnni,
                    &weights,
                    k,
                    &x,
                    rows,
                    &mut out,
                );
                results.push(out);
            }
            // The exact product of the activations in bytes, as they are
            // quantized, by the dequantized weights.
            let a = quantize(crate::Isa::detect(), &x, rows, k, &[]);
            for r in 0..rows {
                let xq: Vec<f64> = (0..k)
                    .map(|i| f64::from(a.scale[(r * k + i) / BLOCK]) * f64::from(a.q[r * k + i]))
                    .collect();
                for j in 0..n {
                    let w = &dense[j * k..(j + 1) * k];
                    let e: f64 = xq.iter().zip(w).map(|(x, w)| x * w).sum();
                    let size: f64 = xq.iter().zip(w).map(|(x, w)| (x * w).abs()).sum();
                    for out in &results {
                        let g = f64::from(out[r * n + j]);
                        assert!(
                            (g - e).abs() <= 1e-5 * size + 1e-6,
                            "{format:?} [{r}, {j}]: {g} instead of {e}"
                        );
                    }
                }
            }
        }
    }
}
