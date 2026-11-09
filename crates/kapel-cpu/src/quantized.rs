//! Products by quantized weights, read in place.
//!
//! A quantized matrix `w [N, K]` stores each of its rows as a run of blocks
//! along `K`: the bytes of a GGUF file, or of Kapel's own formats, as they are.
//! In generation, each weight is used once per token: the product is bound by
//! the reading of the weights, and fewer bytes per weight is the whole gain.
//! So the weights are never expanded in memory. Each thread takes a range of
//! rows of `w`, decodes each block into `f32` in the first-level cache, and
//! multiplies it with the matching part of every row of `x` (up to
//! [`ROWS`] at a time, so that a block is decoded once for all of them).
//!
//! The result is the product by the dequantized weights, up to the rounding of
//! the decoding in `f32` and of the sums. Each output element is a sum over `K`
//! in an order fixed by `K` and the format: the result does not depend on the
//! number of threads, nor on how many rows of `x` are taken together.

use kapel_core::QuantFormat;

use crate::Isa;
use crate::fma;
use crate::pool::{Pool, SharedMut};

/// The rows of `x` multiplied by one decoded block.
pub(crate) const ROWS: usize = 8;

/// The width of a packet of partial sums.
const LANES: usize = 16;

/// The largest block of a fixed-size format (the K formats).
const LARGEST: usize = 256;

/// Rows of `w` per task, at least: enough for a task to stream several
/// kilobytes of weights.
const MIN_TASK_ROWS: usize = 8;

/// Tasks per thread, at least: threads that finish early take more.
const TASKS_PER_THREAD: usize = 4;

/// The weight bytes of a task, at most.
const TASK_BYTES: usize = 128 << 10;

/// Whether the backend multiplies `format` in place.
pub(crate) fn supported(format: QuantFormat) -> bool {
    matches!(
        format,
        QuantFormat::Int8 { .. }
            | QuantFormat::Int4 { .. }
            | QuantFormat::Q8_0
            | QuantFormat::Q4_0
            | QuantFormat::Q4_1
            | QuantFormat::Q5_0
            | QuantFormat::Q5_1
            | QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
            | QuantFormat::Iq4Nl
            | QuantFormat::Iq4Xs
            | QuantFormat::Iq3S
    )
}

/// One matrix of a product: `n` rows of `k` elements in `format`, as bytes.
#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    pub(crate) format: QuantFormat,
    pub(crate) bytes: &'a [u8],
    pub(crate) n: usize,
}

/// A task of a product: rows `first .. first + count` of matrix `matrix`,
/// whose rows start at column `column` of the output.
#[derive(Clone, Copy)]
pub(crate) struct Task {
    pub(crate) matrix: usize,
    pub(crate) first: usize,
    pub(crate) count: usize,
    pub(crate) column: usize,
}

/// The tasks of a product by `weights`, their rows stacked: about the same
/// number of bytes each, so that the threads finish together, at most
/// [`TASK_BYTES`] and at least [`MIN_TASK_ROWS`] rows. Taken in order, small
/// tasks keep the threads reading neighbouring weights: on a Ryzen 7 7700,
/// tasks of 128 KiB give 2 % more than one large task per thread and quarter.
pub(crate) fn tasks(threads: usize, weights: &[Weights<'_>]) -> Vec<Task> {
    let total: usize = weights.iter().map(|w| w.bytes.len()).sum();
    let target = total
        .div_ceil(threads * TASKS_PER_THREAD)
        .clamp(1, TASK_BYTES);
    let mut tasks = Vec::new();
    let mut column = 0;
    for (matrix, w) in weights.iter().enumerate() {
        let row_bytes = w.bytes.len() / w.n.max(1);
        let chunk = target.div_ceil(row_bytes.max(1)).max(MIN_TASK_ROWS);
        for first in (0..w.n).step_by(chunk) {
            tasks.push(Task {
                matrix,
                first,
                count: chunk.min(w.n - first),
                column: column + first,
            });
        }
        column += w.n;
    }
    tasks
}

/// `out[r, :] = x[r] · [w₁; w₂; …]ᵀ`, for `rows` rows of `x [rows, k]` and the
/// rows of the matrices `weights` stacked. `out` is `[rows, Σ nᵢ]`.
///
/// # Panics
///
/// If the sizes of the weights, `x` or `out` do not match, or if `k` is not
/// a whole number of blocks: the backend checks them first.
pub(crate) fn product(
    pool: &Pool,
    isa: Isa,
    weights: &[Weights<'_>],
    k: usize,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    let width: usize = weights.iter().map(|w| w.n).sum();
    for w in weights {
        let dtype = kapel_core::DType::Quantized(w.format);
        let row_bytes = dtype.storage_bytes(k).expect("whole blocks");
        assert_eq!(w.bytes.len(), w.n * row_bytes, "w holds n rows");
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
        let row_bytes = w.bytes.len() / w.n;
        let bytes = &w.bytes[first * row_bytes..(first + count) * row_bytes];
        let mut part = vec![0.0f32; rows * count];
        for (group, x) in x.chunks(ROWS * k).enumerate() {
            let taken = x.len() / k;
            let part = &mut part[group * ROWS * count..][..taken * count];
            match isa.kind() {
                // SAFETY: an `Isa` of kind `Avx512` proves the instructions
                // `wide::rows` is compiled with.
                #[cfg(target_arch = "x86_64")]
                crate::IsaKind::Avx512 => unsafe {
                    wide::rows(w.format, bytes, k, x, taken, part);
                },
                _ => rows_of(isa, w.format, bytes, k, x, taken, part),
            }
        }
        for r in 0..rows {
            // SAFETY: the task writes columns `column .. column + count` of
            // each row of `out`, which no other task writes; `out` outlives
            // the call to `run`.
            let line = unsafe { shared.slice(r * width + column, count) };
            line.copy_from_slice(&part[r * count..][..count]);
        }
    });
}

multiversion! {
    /// `out[r, j] = x[r] · w[j]` for the rows `w[j]` given by `w` and the
    /// `rows ≤ ROWS` rows of `x`.
    fn rows_of(
        format: QuantFormat,
        w: &[u8],
        k: usize,
        x: &[f32],
        rows: usize,
        out: &mut [f32],
    ) -> () = rows_body;
}

#[inline(always)]
fn rows_body<const FMA: bool>(
    format: QuantFormat,
    w: &[u8],
    k: usize,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    debug_assert!(rows <= ROWS);
    let count = out.len() / rows.max(1);
    // The rows by groups of 8, 4, 2 and 1, whose sums stay in registers.
    let mut first = 0;
    while first < rows {
        let x = &x[first * k..];
        let at = first * count;
        first += match rows - first {
            8.. => by_format::<FMA, 8>(format, (w, k, x, &mut out[at..at + 8 * count])),
            4.. => by_format::<FMA, 4>(format, (w, k, x, &mut out[at..at + 4 * count])),
            2.. => by_format::<FMA, 2>(format, (w, k, x, &mut out[at..at + 2 * count])),
            _ => by_format::<FMA, 1>(format, (w, k, x, &mut out[at..at + count])),
        };
    }
}

/// The weights, `k`, the rows of `x`, and the output `[R, count]`.
pub(crate) type Args<'a> = (&'a [u8], usize, &'a [f32], &'a mut [f32]);

/// [`products`] for `R` rows of `x`, with the decoder of `format`. The sizes
/// of the fixed formats are constants, for the compiler to unroll the loops.
/// Returns `R`.
#[inline(always)]
pub(crate) fn by_format<const FMA: bool, const R: usize>(
    format: QuantFormat,
    args: Args<'_>,
) -> usize {
    let dtype = kapel_core::DType::Quantized(format);
    let group = (dtype.block_len(), dtype.block_bytes());
    match format {
        // Kapel's usual group sizes as constants too.
        QuantFormat::Int8 { group: 32 } => products::<FMA, R>(args, (32, 36), int8),
        QuantFormat::Int8 { group: 64 } => products::<FMA, R>(args, (64, 68), int8),
        QuantFormat::Int8 { group: 128 } => products::<FMA, R>(args, (128, 132), int8),
        QuantFormat::Int8 { .. } => products::<FMA, R>(args, group, int8),
        QuantFormat::Int4 { group: 32 } => products::<FMA, R>(args, (32, 20), int4),
        QuantFormat::Int4 { group: 64 } => products::<FMA, R>(args, (64, 36), int4),
        QuantFormat::Int4 { group: 128 } => products::<FMA, R>(args, (128, 68), int4),
        QuantFormat::Int4 { .. } => products::<FMA, R>(args, group, int4),
        QuantFormat::Q8_0 => products::<FMA, R>(args, (32, 34), q8_0),
        QuantFormat::Q4_0 => products::<FMA, R>(args, (32, 18), q4_0),
        QuantFormat::Q4_1 => products::<FMA, R>(args, (32, 20), q4_1),
        QuantFormat::Q5_0 => products::<FMA, R>(args, (32, 22), q5_0),
        QuantFormat::Q5_1 => products::<FMA, R>(args, (32, 24), q5_1),
        QuantFormat::Q2K => products::<FMA, R>(args, (256, 84), q2_k),
        QuantFormat::Q3K => products::<FMA, R>(args, (256, 110), q3_k),
        QuantFormat::Q4K => products::<FMA, R>(args, (256, 144), q4_k),
        QuantFormat::Q5K => products::<FMA, R>(args, (256, 176), q5_k),
        QuantFormat::Q6K => products::<FMA, R>(args, (256, 210), q6_k),
        QuantFormat::Iq4Nl => products::<FMA, R>(args, (32, 18), iq4_nl),
        QuantFormat::Iq4Xs => products::<FMA, R>(args, (256, 136), iq4_xs),
        QuantFormat::Iq3S => products::<FMA, R>(args, (256, 110), iq3_s),
        _ => unreachable!("checked by `supported`"),
    }
    R
}

/// `out[r, j] = x[r] · w[j]` for `R` rows of `x` and the rows of `w`, in
/// blocks of `len` elements stored in `bytes` bytes that `decode` turns into
/// `f32`. Each row of `x` keeps two packets of partial sums, which the even
/// and odd packets of a block feed: two chains of additions instead of one.
#[inline(always)]
fn products<const FMA: bool, const R: usize>(
    (w, k, x, out): Args<'_>,
    (len, bytes): (usize, usize),
    decode: impl Fn(&[u8], &mut [f32]),
) {
    let count = out.len() / R;
    let row_bytes = k / len * bytes;
    // A fixed-size buffer for the fixed formats; Kapel's groups may be longer.
    let mut fixed = [0.0f32; LARGEST];
    let mut long = Vec::new();
    let block: &mut [f32] = if len <= LARGEST {
        &mut fixed[..len]
    } else {
        long.resize(len, 0.0);
        &mut long
    };
    for j in 0..count {
        let mut acc = [[[0.0f32; LANES]; 2]; R];
        let row = &w[j * row_bytes..][..row_bytes];
        for (b, code) in row.chunks_exact(bytes).enumerate() {
            decode(code, block);
            let (pairs, rest) = block.as_chunks::<{ 2 * LANES }>();
            for (r, [even, odd]) in acc.iter_mut().enumerate() {
                let x = &x[r * k + b * len..][..len];
                let (inputs, tail) = x.as_chunks::<{ 2 * LANES }>();
                for (w, x) in pairs.iter().zip(inputs) {
                    for i in 0..LANES {
                        even[i] = fma::<FMA>(w[i], x[i], even[i]);
                        odd[i] = fma::<FMA>(w[LANES + i], x[LANES + i], odd[i]);
                    }
                }
                for (i, (&w, &x)) in rest.iter().zip(tail).enumerate() {
                    even[i % LANES] = fma::<FMA>(w, x, even[i % LANES]);
                }
            }
        }
        for (r, [even, odd]) in acc.iter().enumerate() {
            let mut sums = [0.0f32; LANES];
            for i in 0..LANES {
                sums[i] = even[i] + odd[i];
            }
            out[r * count + j] = reduce(sums);
        }
    }
}

/// The sum of the partial sums, pairwise.
#[inline(always)]
fn reduce(mut sums: [f32; LANES]) -> f32 {
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            sums[i] += sums[i + width];
        }
    }
    sums[0]
}

/// `Σ x[i]`, in packets, then pairwise.
#[inline(always)]
fn reduce_sum(x: &[f32]) -> f32 {
    let mut sums = [0.0f32; LANES];
    for (i, &x) in x.iter().enumerate() {
        sums[i % LANES] += x;
    }
    reduce(sums)
}

/// The values of whole blocks of `format`, `bytes`, into `out`.
pub(crate) fn decode_row(isa: Isa, format: QuantFormat, bytes: &[u8], out: &mut [f32]) {
    decode_rows(isa, format, bytes, out);
}

multiversion! {
    /// [`decode_row`], compiled for each instruction set.
    fn decode_rows(format: QuantFormat, bytes: &[u8], out: &mut [f32]) -> () = decode_body;
}

#[inline(always)]
fn decode_body<const FMA: bool>(format: QuantFormat, bytes: &[u8], out: &mut [f32]) {
    let dtype = kapel_core::DType::Quantized(format);
    let (len, size) = (dtype.block_len(), dtype.block_bytes());
    // One loop per format, so that each decoder is inlined and vectorized.
    #[inline(always)]
    fn each(
        bytes: &[u8],
        out: &mut [f32],
        (len, size): (usize, usize),
        decode: impl Fn(&[u8], &mut [f32]),
    ) {
        for (block, out) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(len)) {
            decode(block, out);
        }
    }
    let sizes = (len, size);
    match format {
        QuantFormat::Int8 { .. } => each(bytes, out, sizes, int8),
        QuantFormat::Int4 { .. } => each(bytes, out, sizes, int4),
        QuantFormat::Q8_0 => each(bytes, out, sizes, q8_0),
        QuantFormat::Q4_0 => each(bytes, out, sizes, q4_0),
        QuantFormat::Q4_1 => each(bytes, out, sizes, q4_1),
        QuantFormat::Q5_0 => each(bytes, out, sizes, q5_0),
        QuantFormat::Q5_1 => each(bytes, out, sizes, q5_1),
        QuantFormat::Q2K => each(bytes, out, sizes, q2_k),
        QuantFormat::Q3K => each(bytes, out, sizes, q3_k),
        QuantFormat::Q4K => each(bytes, out, sizes, q4_k),
        QuantFormat::Q5K => each(bytes, out, sizes, q5_k),
        QuantFormat::Q6K => each(bytes, out, sizes, q6_k),
        QuantFormat::Iq4Nl => each(bytes, out, sizes, iq4_nl),
        QuantFormat::Iq4Xs => each(bytes, out, sizes, iq4_xs),
        QuantFormat::Iq3S => each(bytes, out, sizes, iq3_s),
        _ => unreachable!("checked by `supported`"),
    }
}

// --- Decoding, one block at a time ---
//
// Each function writes the values of one block, in the formulas of
// `kapel_core::dequantize`, in `f32`.

/// The `f16` at byte `at`, exactly, without branches on normal and
/// subnormal values: the exponent and mantissa, moved to their places in an
/// `f32`, are rebiased by a product with `2¹¹²`, exact for both.
#[inline(always)]
pub(crate) fn half(block: &[u8], at: usize) -> f32 {
    let bits = u32::from(u16::from_le_bytes([block[at], block[at + 1]]));
    let magnitude = bits & 0x7fff;
    let value = if magnitude >= 0x7c00 {
        // Infinities and NaNs keep their payload.
        f32::from_bits(0x7f80_0000 | (magnitude & 0x3ff) << 13)
    } else {
        f32::from_bits(magnitude << 13) * f32::from_bits(0x7780_0000)
    };
    f32::from_bits(value.to_bits() | (bits & 0x8000) << 16)
}

#[inline(always)]
pub(crate) fn single(block: &[u8]) -> f32 {
    f32::from_le_bytes([block[0], block[1], block[2], block[3]])
}

#[inline(always)]
fn int8(block: &[u8], out: &mut [f32]) {
    let scale = single(block);
    for (y, &q) in out.iter_mut().zip(&block[4..]) {
        *y = scale * f32::from(q as i8);
    }
}

#[inline(always)]
fn int4(block: &[u8], out: &mut [f32]) {
    let scale = single(block);
    for (pair, &byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(&block[4..]) {
        pair[0] = scale * (f32::from(byte & 0x0f) - 8.0);
        pair[1] = scale * (f32::from(byte >> 4) - 8.0);
    }
}

/// The 32 values of IQ4_NL: `scale × value[code]`, the values of the
/// non-linear table. Products of a scale of `f16` and values of 7 bits:
/// exact in `f32`, as the reference's.
#[inline(always)]
fn iq4_nl(block: &[u8], out: &mut [f32]) {
    let scale = half(block, 0);
    let codes: &[u8; 16] = block[2..18].try_into().expect("an IQ4_NL block");
    let (low, high) = out.split_at_mut(16);
    for ((l, h), &byte) in low.iter_mut().zip(high.iter_mut()).zip(codes) {
        *l = scale * f32::from(kapel_core::iq::IQ4NL_VALUES[usize::from(byte & 0x0f)]);
        *h = scale * f32::from(kapel_core::iq::IQ4NL_VALUES[usize::from(byte >> 4)]);
    }
}

/// The 256 values of IQ4_XS: IQ4_NL's codes by groups of 32, each group's
/// scale `d × (s − 32)` of a 6-bit `s`.
#[inline(always)]
fn iq4_xs(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let high = u16::from_le_bytes([block[2], block[3]]);
    for g in 0..8 {
        let s = i32::from((block[4 + g / 2] >> (4 * (g % 2))) & 0x0f)
            | i32::from((high >> (2 * g)) & 3) << 4;
        let scale = d * (s - 32) as f32;
        let codes: &[u8; 16] = block[8 + g * 16..][..16].try_into().expect("16 bytes");
        let (low, top) = out[g * 32..][..32].split_at_mut(16);
        for ((l, h), &byte) in low.iter_mut().zip(top.iter_mut()).zip(codes) {
            *l = scale * f32::from(kapel_core::iq::IQ4NL_VALUES[usize::from(byte & 0x0f)]);
            *h = scale * f32::from(kapel_core::iq::IQ4NL_VALUES[usize::from(byte >> 4)]);
        }
    }
}

/// The 256 values of IQ3_S: for each group of 32, its scale `d × (1 + 2s)`,
/// and for each 4 of its values a 9-bit index into the grid of magnitudes,
/// each value with its sign.
#[inline(always)]
fn iq3_s(block: &[u8], out: &mut [f32]) {
    let d = half(block, 0);
    let (codes, high, signs, scales) = (
        &block[2..66],
        &block[66..74],
        &block[74..106],
        &block[106..110],
    );
    for g in 0..8 {
        let s = (scales[g / 2] >> (4 * (g % 2))) & 0x0f;
        let db = d * f32::from(1 + 2 * s);
        for l in 0..4 {
            let sign = signs[g * 4 + l];
            for half_group in 0..2 {
                let code = usize::from(codes[g * 8 + 2 * l + half_group]);
                let index = code | ((usize::from(high[g]) << (8 - 2 * l - half_group)) & 256);
                let grid = kapel_core::iq::IQ3S_GRID[index];
                for j in 0..4 {
                    let bit = 4 * half_group + j;
                    let magnitude = db * ((grid >> (8 * j)) & 0xff) as f32;
                    out[g * 32 + l * 8 + bit] = if (sign >> bit) & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    };
                }
            }
        }
    }
}

#[inline(always)]
fn q8_0(block: &[u8], out: &mut [f32]) {
    let scale = half(block, 0);
    let codes: &[u8; 32] = block[2..34].try_into().expect("a Q8_0 block");
    for (y, &q) in out.iter_mut().zip(codes) {
        *y = scale * f32::from(q as i8);
    }
}

/// The 32 values of Q4_0 or Q4_1: `scale × code + offset`.
#[inline(always)]
fn nibbles(codes: &[u8; 16], scale: f32, offset: f32, out: &mut [f32]) {
    let (low, high) = out.split_at_mut(16);
    for ((l, h), &byte) in low.iter_mut().zip(high.iter_mut()).zip(codes) {
        *l = scale * f32::from(byte & 0x0f) + offset;
        *h = scale * f32::from(byte >> 4) + offset;
    }
}

#[inline(always)]
fn q4_0(block: &[u8], out: &mut [f32]) {
    let scale = half(block, 0);
    let codes: &[u8; 16] = block[2..18].try_into().expect("a Q4_0 block");
    let (low, high) = out.split_at_mut(16);
    for ((l, h), &byte) in low.iter_mut().zip(high.iter_mut()).zip(codes) {
        *l = scale * (f32::from(byte & 0x0f) - 8.0);
        *h = scale * (f32::from(byte >> 4) - 8.0);
    }
}

#[inline(always)]
fn q4_1(block: &[u8], out: &mut [f32]) {
    let (scale, minimum) = (half(block, 0), half(block, 2));
    nibbles(
        block[4..20].try_into().expect("a Q4_1 block"),
        scale,
        minimum,
        out,
    );
}

/// The 32 five-bit codes of Q5_0 or Q5_1: four low bits in `codes` as in
/// Q4_0, the fifth in bit `j` of `high`.
#[inline(always)]
fn five_bits(high: &[u8], codes: &[u8], out: &mut [u8; 32]) {
    let high = u32::from_le_bytes([high[0], high[1], high[2], high[3]]);
    for j in 0..16 {
        out[j] = codes[j] & 0x0f;
        out[j + 16] = codes[j] >> 4;
    }
    for (j, q) in out.iter_mut().enumerate() {
        *q |= (((high >> j) & 1) as u8) << 4;
    }
}

#[inline(always)]
fn q5_0(block: &[u8], out: &mut [f32]) {
    let scale = half(block, 0);
    let mut q = [0u8; 32];
    five_bits(&block[2..6], &block[6..22], &mut q);
    for (y, &q) in out.iter_mut().zip(&q) {
        *y = scale * (f32::from(q) - 16.0);
    }
}

#[inline(always)]
fn q5_1(block: &[u8], out: &mut [f32]) {
    let (scale, minimum) = (half(block, 0), half(block, 2));
    let mut q = [0u8; 32];
    five_bits(&block[4..8], &block[8..24], &mut q);
    for (y, &q) in out.iter_mut().zip(&q) {
        *y = scale * f32::from(q) + minimum;
    }
}

#[inline(always)]
fn q2_k(block: &[u8], out: &mut [f32]) {
    let (scales, codes) = (&block[..16], &block[16..80]);
    let (d, dmin) = (half(block, 80), half(block, 82));
    for h in 0..2 {
        for s in 0..4 {
            for half16 in 0..2 {
                let k = scales[h * 8 + s * 2 + half16];
                let (scale, minimum) = (d * f32::from(k & 0x0f), dmin * f32::from(k >> 4));
                let at = h * 128 + s * 32 + half16 * 16;
                let codes = &codes[h * 32 + half16 * 16..][..16];
                for (y, &c) in out[at..at + 16].iter_mut().zip(codes) {
                    *y = scale * f32::from((c >> (2 * s)) & 3) - minimum;
                }
            }
        }
    }
}

#[inline(always)]
fn q3_k(block: &[u8], out: &mut [f32]) {
    let (mask, codes, packed) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half(block, 108);
    for i in 0..16 {
        let low = (packed[i % 8] >> (4 * (i / 8))) & 0x0f;
        let high = (packed[8 + i % 4] >> (2 * (i / 4))) & 0x03;
        let scale = d * f32::from(((low | (high << 4)) as i8) - 32);
        // Elements 16i … 16i + 15: h = i / 8, s = i % 8 / 2.
        let (h, s, part) = (i / 8, i % 8 / 2, i % 2);
        for l in part * 16..part * 16 + 16 {
            let low = i32::from((codes[h * 32 + l] >> (2 * s)) & 3);
            let high = i32::from(((mask[l] >> (h * 4 + s)) & 1) ^ 1);
            out[16 * i + l - part * 16] = scale * (low - 4 * high) as f32;
        }
    }
}

/// The eight scales and minimums of a Q4_K or Q5_K block, times `d` and
/// `dmin`.
#[inline(always)]
pub(crate) fn k_scales(block: &[u8]) -> ([f32; 8], [f32; 8]) {
    let (d, dmin) = (half(block, 0), half(block, 2));
    let packed = &block[4..16];
    let (sc, mn, md) = (&packed[..4], &packed[4..8], &packed[8..12]);
    let mut scale = [0.0; 8];
    let mut minimum = [0.0; 8];
    for i in 0..4 {
        scale[i] = d * f32::from(sc[i] & 0x3f);
        minimum[i] = dmin * f32::from(mn[i] & 0x3f);
        scale[4 + i] = d * f32::from((md[i] & 0x0f) | ((sc[i] >> 2) & 0x30));
        minimum[4 + i] = dmin * f32::from((md[i] >> 4) | ((mn[i] >> 2) & 0x30));
    }
    (scale, minimum)
}

#[inline(always)]
fn q4_k(block: &[u8], out: &mut [f32]) {
    let (scale, minimum) = k_scales(block);
    let codes = &block[16..144];
    for c in 0..4 {
        let codes: &[u8; 32] = codes[c * 32..][..32].try_into().expect("32 bytes");
        let (low, high) = out[c * 64..][..64].split_at_mut(32);
        let (s0, m0, s1, m1) = (
            scale[2 * c],
            minimum[2 * c],
            scale[2 * c + 1],
            minimum[2 * c + 1],
        );
        for ((l, h), &byte) in low.iter_mut().zip(high.iter_mut()).zip(codes) {
            *l = s0 * f32::from(byte & 0x0f) - m0;
            *h = s1 * f32::from(byte >> 4) - m1;
        }
    }
}

#[inline(always)]
fn q5_k(block: &[u8], out: &mut [f32]) {
    let (scale, minimum) = k_scales(block);
    let high: &[u8; 32] = block[16..48].try_into().expect("32 bytes");
    let codes = &block[48..176];
    for c in 0..4 {
        let codes: &[u8; 32] = codes[c * 32..][..32].try_into().expect("32 bytes");
        let (low, top) = out[c * 64..][..64].split_at_mut(32);
        let (s0, m0, s1, m1) = (
            scale[2 * c],
            minimum[2 * c],
            scale[2 * c + 1],
            minimum[2 * c + 1],
        );
        for (((l, t), &byte), &h) in low.iter_mut().zip(top.iter_mut()).zip(codes).zip(high) {
            let q0 = (byte & 0x0f) | (((h >> (2 * c)) & 1) << 4);
            let q1 = (byte >> 4) | (((h >> (2 * c + 1)) & 1) << 4);
            *l = s0 * f32::from(q0) - m0;
            *t = s1 * f32::from(q1) - m1;
        }
    }
}

#[inline(always)]
fn q6_k(block: &[u8], out: &mut [f32]) {
    let (low, high, scales) = (&block[..128], &block[128..192], &block[192..208]);
    let d = half(block, 208);
    for h in 0..2 {
        let low = &low[h * 64..][..64];
        let high: &[u8; 32] = high[h * 32..][..32].try_into().expect("32 bytes");
        for s in 0..4 {
            let low: &[u8; 32] = low[(s % 2) * 32..][..32].try_into().expect("32 bytes");
            let at = h * 128 + s * 32;
            let (first, second) = out[at..at + 32].split_at_mut(16);
            let scale0 = d * f32::from(scales[at / 16] as i8);
            let scale1 = d * f32::from(scales[at / 16 + 1] as i8);
            for (l, y) in first.iter_mut().chain(second.iter_mut()).enumerate() {
                let ql = (low[l] >> (4 * (s / 2))) & 0x0f;
                let qh = (high[l] >> (2 * s)) & 0x03;
                let q = f32::from(ql | (qh << 4)) - 32.0;
                *y = if l < 16 { scale0 } else { scale1 } * q;
            }
        }
    }
}

/// The AVX-512 products: each block is decoded in registers, sixteen codes
/// per vector, and multiplied there with the rows of `x`.
///
/// The scales and minimums are taken out of the sums: within a group of
/// elements that share a scale `s` and a minimum `m`,
///
/// ```text
/// Σ (s · q − m) · x = s · Σ q · x − m · Σ x
/// ```
///
/// so the codes `q` are multiplied as they are, `s` once per group, and the
/// sums `Σ x` of each group are taken once per row of `x`, for all the rows
/// of `w`. The compiler does not find these by itself: the codes are spread
/// over bytes, nibbles and bit planes that it reads one at a time.
#[cfg(target_arch = "x86_64")]
mod wide {
    use std::arch::x86_64::*;

    use kapel_core::QuantFormat;

    use super::{Args, half, k_scales, single};

    /// The scales and minimums of the groups of a block (up to 16 groups).
    type Scales = ([f32; 16], [f32; 16]);

    /// [`super::rows_of`] on a processor with AVX-512.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    pub(super) fn rows(
        format: QuantFormat,
        w: &[u8],
        k: usize,
        x: &[f32],
        rows: usize,
        out: &mut [f32],
    ) {
        let count = out.len() / rows.max(1);
        let mut first = 0;
        while first < rows {
            let x = &x[first * k..];
            let at = first * count;
            first += match rows - first {
                8.. => by_format::<8>(format, (w, k, x, &mut out[at..at + 8 * count])),
                4.. => by_format::<4>(format, (w, k, x, &mut out[at..at + 4 * count])),
                2.. => by_format::<2>(format, (w, k, x, &mut out[at..at + 2 * count])),
                _ => by_format::<1>(format, (w, k, x, &mut out[at..at + count])),
            };
        }
    }

    /// The scale of a block of one group.
    fn one(scale: f32) -> Scales {
        let mut scales = ([0.0; 16], [0.0; 16]);
        scales.0[0] = scale;
        scales
    }

    /// The scale and minimum of a block of one group.
    fn one_with(scale: f32, minimum: f32) -> Scales {
        let mut scales = one(scale);
        scales.1[0] = minimum;
        scales
    }

    /// [`fused`] for `R` rows, with the layout of `format`. Returns `R`.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn by_format<const R: usize>(format: QuantFormat, args: Args<'_>) -> usize {
        // The codes are taken from their bytes in 128-bit registers, sixteen
        // at a time, then widened to sixteen integers and converted: one
        // instruction on 512 bits for each step on 128, which Zen 4 runs at
        // half the rate.
        let int8 = |b: &[u8], s: usize| signed(bytes(&b[4 + 16 * s..]));
        let int4 = |b: &[u8], s: usize| {
            // Elements 2i and 2i + 1 are the nibbles of byte i.
            let v = bytes(&b[4 + s / 2 * 16..]);
            let (low, high) = (take(v, 0, 0x0f), take(v, 4, 0x0f));
            let q = if s.is_multiple_of(2) {
                _mm_unpacklo_epi8(low, high)
            } else {
                _mm_unpackhi_epi8(low, high)
            };
            signed(_mm_sub_epi8(q, _mm_set1_epi8(8)))
        };
        let kapel = |b: &[u8]| one(single(b));
        match format {
            QuantFormat::Int8 { group: 32 } => fused::<R, 32, 36, 32, false>(args, kapel, int8),
            QuantFormat::Int8 { group: 64 } => fused::<R, 64, 68, 64, false>(args, kapel, int8),
            QuantFormat::Int8 { group: 128 } => {
                fused::<R, 128, 132, 128, false>(args, kapel, int8);
            }
            QuantFormat::Int4 { group: 32 } => fused::<R, 32, 20, 32, false>(args, kapel, int4),
            QuantFormat::Int4 { group: 64 } => fused::<R, 64, 36, 64, false>(args, kapel, int4),
            QuantFormat::Int4 { group: 128 } => {
                fused::<R, 128, 68, 128, false>(args, kapel, int4);
            }
            QuantFormat::Q8_0 => fused::<R, 32, 34, 32, false>(
                args,
                |b| one(half(b, 0)),
                |b, s| signed(bytes(&b[2 + 16 * s..])),
            ),
            QuantFormat::Q4_0 => fused::<R, 32, 18, 32, false>(
                args,
                |b| one(half(b, 0)),
                |b, s| {
                    signed(_mm_sub_epi8(
                        take(bytes(&b[2..]), 4 * s, 0x0f),
                        _mm_set1_epi8(8),
                    ))
                },
            ),
            QuantFormat::Q4_1 => fused::<R, 32, 20, 32, true>(
                args,
                |b| one_with(half(b, 0), -half(b, 2)),
                |b, s| unsigned(take(bytes(&b[4..]), 4 * s, 0x0f)),
            ),
            QuantFormat::Q5_0 => fused::<R, 32, 22, 32, false>(
                args,
                |b| one(half(b, 0)),
                |b, s| {
                    let q = five_bits(&b[2..6], &b[6..22], s);
                    signed(_mm_sub_epi8(q, _mm_set1_epi8(16)))
                },
            ),
            QuantFormat::Q5_1 => fused::<R, 32, 24, 32, true>(
                args,
                |b| one_with(half(b, 0), -half(b, 2)),
                |b, s| unsigned(five_bits(&b[4..8], &b[8..24], s)),
            ),
            QuantFormat::Q2K => fused::<R, 256, 84, 16, true>(
                args,
                |b| {
                    let (d, dmin) = (half(b, 80), half(b, 82));
                    let mut scales = ([0.0; 16], [0.0; 16]);
                    for (i, &k) in b[..16].iter().enumerate() {
                        scales.0[i] = d * f32::from(k & 0x0f);
                        scales.1[i] = dmin * f32::from(k >> 4);
                    }
                    scales
                },
                |b, s| {
                    let (h, shift, part) = (s / 8, s % 8 / 2, s % 2);
                    unsigned(take(bytes(&b[16 + h * 32 + part * 16..]), 2 * shift, 3))
                },
            ),
            QuantFormat::Q3K => fused::<R, 256, 110, 16, false>(
                args,
                |b| {
                    let (packed, d) = (&b[96..108], half(b, 108));
                    let mut scales = ([0.0; 16], [0.0; 16]);
                    for (i, scale) in scales.0.iter_mut().enumerate() {
                        let low = (packed[i % 8] >> (4 * (i / 8))) & 0x0f;
                        let high = (packed[8 + i % 4] >> (2 * (i / 4))) & 0x03;
                        *scale = d * f32::from(((low | (high << 4)) as i8) - 32);
                    }
                    scales
                },
                |b, s| {
                    let (h, shift, part) = (s / 8, s % 8 / 2, s % 2);
                    let low = take(bytes(&b[32 + h * 32 + part * 16..]), 2 * shift, 3);
                    let mask = take(bytes(&b[part * 16..]), h * 4 + shift, 1);
                    // The high bit subtracts 4 when it is clear.
                    let clear = _mm_xor_si128(mask, _mm_set1_epi8(1));
                    signed(_mm_sub_epi8(low, _mm_slli_epi16::<2>(clear)))
                },
            ),
            QuantFormat::Q4K => fused::<R, 256, 144, 32, true>(
                args,
                |b| {
                    let (scale, minimum) = k_scales(b);
                    let mut scales = ([0.0; 16], [0.0; 16]);
                    scales.0[..8].copy_from_slice(&scale);
                    scales.1[..8].copy_from_slice(&minimum);
                    scales
                },
                |b, s| {
                    // Segment s: group s / 2, whose codes are the low (even
                    // groups) or high (odd) nibbles of 32 bytes.
                    let (c, high, part) = (s / 4, s % 4 / 2, s % 2);
                    unsigned(take(bytes(&b[16 + c * 32 + part * 16..]), 4 * high, 0x0f))
                },
            ),
            QuantFormat::Q5K => fused::<R, 256, 176, 32, true>(
                args,
                |b| {
                    let (scale, minimum) = k_scales(b);
                    let mut scales = ([0.0; 16], [0.0; 16]);
                    scales.0[..8].copy_from_slice(&scale);
                    scales.1[..8].copy_from_slice(&minimum);
                    scales
                },
                |b, s| {
                    let (c, high, part) = (s / 4, s % 4 / 2, s % 2);
                    let low = take(bytes(&b[48 + c * 32 + part * 16..]), 4 * high, 0x0f);
                    let top = take(bytes(&b[16 + part * 16..]), 2 * c + high, 1);
                    unsigned(_mm_or_si128(low, _mm_slli_epi16::<4>(top)))
                },
            ),
            QuantFormat::Q6K => fused::<R, 256, 210, 16, false>(
                args,
                |b| {
                    let d = _mm512_set1_ps(half(b, 208));
                    let mut scales = ([0.0; 16], [0.0; 16]);
                    let values = _mm512_mul_ps(signed(bytes(&b[192..208])), d);
                    // SAFETY: `scales.0` holds the 16 values written.
                    unsafe { _mm512_storeu_ps(scales.0.as_mut_ptr(), values) };
                    scales
                },
                |b, s| {
                    let (h, shift, part) = (s / 8, s % 8 / 2, s % 2);
                    let low = bytes(&b[h * 64 + (shift % 2) * 32 + part * 16..]);
                    let ql = take(low, 4 * (shift / 2), 0x0f);
                    let qh = take(bytes(&b[128 + h * 32 + part * 16..]), 2 * shift, 3);
                    let q = _mm_or_si128(ql, _mm_slli_epi16::<4>(qh));
                    signed(_mm_sub_epi8(q, _mm_set1_epi8(32)))
                },
            ),
            // Other group sizes: the portable decoders, with these instructions.
            _ => {
                super::by_format::<true, R>(format, args);
            }
        }
        R
    }

    /// `out[r, j] = x[r] · w[j]` for `R` rows of `x`, with blocks of `LEN`
    /// elements in `BYTES` bytes, a scale (and with `MIN` a minimum,
    /// subtracted) per `GROUP` elements. `scales` reads them from a block;
    /// `codes` reads its codes `16 s … 16 s + 15` as integers in floats.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn fused<
        const R: usize,
        const LEN: usize,
        const BYTES: usize,
        const GROUP: usize,
        const MIN: bool,
    >(
        (w, k, x, out): Args<'_>,
        scales: impl Fn(&[u8]) -> Scales,
        codes: impl Fn(&[u8], usize) -> __m512,
    ) {
        let count = out.len() / R;
        let (blocks, groups) = (k / LEN, k / GROUP);
        // Σ x over each group of each row, for the minimums.
        let mut sums = Vec::new();
        if MIN {
            sums.resize(R * groups, 0.0f32);
            for r in 0..R {
                for (g, sum) in sums[r * groups..][..groups].iter_mut().enumerate() {
                    *sum = super::reduce_sum(&x[r * k + g * GROUP..][..GROUP]);
                }
            }
        }
        let x = &x[..R * k];
        assert!(k.is_multiple_of(LEN) && w.len() >= count * blocks * BYTES);
        let base = x.as_ptr();
        for j in 0..count {
            let row = &w[j * blocks * BYTES..][..blocks * BYTES];
            let mut acc = [_mm512_setzero_ps(); R];
            let mut low = [0.0f32; R];
            for (b, block) in row.as_chunks::<BYTES>().0.iter().enumerate() {
                let (scale, minimum) = scales(block);
                for g in 0..LEN / GROUP {
                    let mut t = [_mm512_setzero_ps(); R];
                    for s in 0..GROUP / 16 {
                        let q = codes(block, g * GROUP / 16 + s);
                        for (r, t) in t.iter_mut().enumerate() {
                            let at = r * k + b * LEN + g * GROUP + s * 16;
                            // SAFETY: `at + 16 ≤ R · k`, the length of `x`:
                            // `r < R`, and the block `b` of a row of `w`
                            // covers elements `b · LEN … b · LEN + LEN − 1 < k`
                            // of the row of `x`.
                            let v = unsafe { _mm512_loadu_ps(base.add(at)) };
                            *t = _mm512_fmadd_ps(q, v, *t);
                        }
                    }
                    let scale = _mm512_set1_ps(scale[g]);
                    for (acc, &t) in acc.iter_mut().zip(&t) {
                        *acc = _mm512_fmadd_ps(t, scale, *acc);
                    }
                    if MIN {
                        for (r, low) in low.iter_mut().enumerate() {
                            *low += minimum[g] * sums[r * groups + b * (LEN / GROUP) + g];
                        }
                    }
                }
            }
            for r in 0..R {
                out[r * count + j] = _mm512_reduce_add_ps(acc[r]) - low[r];
            }
        }
    }

    /// Sixteen bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn bytes(b: &[u8]) -> __m128i {
        let b: &[u8; 16] = b[..16].try_into().expect("16 bytes");
        // SAFETY: `b` holds the 16 bytes read.
        unsafe { _mm_loadu_si128(b.as_ptr().cast()) }
    }

    /// `(v >> shift) & mask` on each byte (`shift < 8`: the bits that the
    /// 16-bit shift brings from the next byte are masked off).
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn take(v: __m128i, shift: usize, mask: i8) -> __m128i {
        let shifted = _mm_srl_epi16(v, _mm_cvtsi64_si128(shift as i64));
        _mm_and_si128(shifted, _mm_set1_epi8(mask))
    }

    /// Sixteen unsigned bytes as floats.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn unsigned(v: __m128i) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v))
    }

    /// Sixteen signed bytes as floats.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn signed(v: __m128i) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v))
    }

    /// Codes `16 s … 16 s + 15` of Q5_0 or Q5_1, as bytes: four low bits in
    /// `codes` as in Q4_0, the fifth in bit `j` of `high`.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[inline]
    fn five_bits(high: &[u8], codes: &[u8], s: usize) -> __m128i {
        let high = u32::from_le_bytes([high[0], high[1], high[2], high[3]]);
        let low = take(bytes(codes), 4 * s, 0x0f);
        let fifth = (high >> (16 * s)) as u16;
        _mm_mask_blend_epi8(fifth, low, _mm_or_si128(low, _mm_set1_epi8(16)))
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
            QuantFormat::Q8_0
            | QuantFormat::Q4_0
            | QuantFormat::Q5_0
            | QuantFormat::Iq4Nl
            | QuantFormat::Iq4Xs
            | QuantFormat::Iq3S => &[0],
            QuantFormat::Q4_1 | QuantFormat::Q5_1 | QuantFormat::Q4K | QuantFormat::Q5K => &[0, 2],
            QuantFormat::Q2K => &[80, 82],
            QuantFormat::Q3K => &[108],
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
            if matches!(format, QuantFormat::Int8 { .. } | QuantFormat::Int4 { .. }) {
                block[..4].copy_from_slice(&0.0125f32.to_le_bytes());
            }
            out.extend(block);
        }
        out
    }

    const FORMATS: [QuantFormat; 17] = [
        QuantFormat::Int8 { group: 32 },
        QuantFormat::Int8 { group: 40 },
        QuantFormat::Int4 { group: 64 },
        QuantFormat::Int4 { group: 40 },
        QuantFormat::Q8_0,
        QuantFormat::Q4_0,
        QuantFormat::Q4_1,
        QuantFormat::Q5_0,
        QuantFormat::Q5_1,
        QuantFormat::Q2K,
        QuantFormat::Q3K,
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
        QuantFormat::Iq4Nl,
        QuantFormat::Iq4Xs,
        QuantFormat::Iq3S,
    ];

    #[test]
    fn halves_decode_exactly() {
        for bits in 0..=u16::MAX {
            let expected = F16(bits).to_f64() as f32;
            let got = half(&bits.to_le_bytes(), 0);
            assert!(
                got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
                "{bits:#06x}: {got} instead of {expected}"
            );
        }
    }

    #[test]
    fn blocks_decode_as_the_reference() {
        for format in FORMATS {
            let dtype = DType::Quantized(format);
            let bytes = blocks(format, 3, 7);
            let expected = dequantize(dtype, &bytes).unwrap();
            let len = dtype.block_len();
            let mut got = vec![0.0f32; expected.len()];
            for (code, out) in bytes
                .chunks_exact(dtype.block_bytes())
                .zip(got.chunks_mut(len))
            {
                portable(format, code, out);
            }
            close(format, &got, &expected);
        }
    }

    fn close(format: QuantFormat, got: &[f32], expected: &[f64]) {
        for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
            let tolerance = 1e-6 * e.abs().max(1e-3);
            assert!(
                (f64::from(g) - e).abs() <= tolerance,
                "{format:?} element {i}: {g} instead of {e}"
            );
        }
    }

    fn portable(format: QuantFormat, code: &[u8], out: &mut [f32]) {
        {
            {
                match format {
                    QuantFormat::Int8 { .. } => int8(code, out),
                    QuantFormat::Int4 { .. } => int4(code, out),
                    QuantFormat::Q8_0 => q8_0(code, out),
                    QuantFormat::Q4_0 => q4_0(code, out),
                    QuantFormat::Q4_1 => q4_1(code, out),
                    QuantFormat::Q5_0 => q5_0(code, out),
                    QuantFormat::Q5_1 => q5_1(code, out),
                    QuantFormat::Q2K => q2_k(code, out),
                    QuantFormat::Q3K => q3_k(code, out),
                    QuantFormat::Q4K => q4_k(code, out),
                    QuantFormat::Q5K => q5_k(code, out),
                    QuantFormat::Q6K => q6_k(code, out),
                    QuantFormat::Iq4Nl => iq4_nl(code, out),
                    QuantFormat::Iq4Xs => iq4_xs(code, out),
                    QuantFormat::Iq3S => iq3_s(code, out),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn products_match_the_dequantized_weights_on_any_threads() {
        let (n, k, rows) = (37, 1280, 11);
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0)
            .collect();
        for format in FORMATS {
            let dtype = DType::Quantized(format);
            let w = blocks(format, n * k / dtype.block_len(), 11);
            let dense = dequantize(dtype, &w).unwrap();
            let mut first = None;
            for threads in [1, 3] {
                for isa in Isa::all_available() {
                    let pool = Pool::new(threads);
                    let mut out = vec![0.0f32; rows * n];
                    let weights = [Weights {
                        format,
                        bytes: &w,
                        n,
                    }];
                    product(&pool, isa, &weights, k, &x, rows, &mut out);
                    for r in 0..rows {
                        for j in 0..n {
                            let e: f64 = (0..k)
                                .map(|i| f64::from(x[r * k + i]) * dense[j * k + i])
                                .sum();
                            let g = f64::from(out[r * n + j]);
                            assert!(
                                (g - e).abs() <= 1e-4 * (1.0 + e.abs()),
                                "{format:?} [{r}, {j}]: {g} instead of {e}"
                            );
                        }
                    }
                    if isa.kind() == Isa::detect().kind() {
                        match &first {
                            None => first = Some(out),
                            Some(first) => assert_eq!(first, &out, "{format:?}: same bits"),
                        }
                    }
                }
            }
        }
    }
}
