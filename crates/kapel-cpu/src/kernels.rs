//! The micro-kernels of the matrix product: AVX-512, AVX2 and scalar.
//!
//! A micro-kernel computes a whole small `MR × NR` block of `C`, along an
//! entire slice of `kc` indices of the shared dimension. The block stays in
//! vector registers from start to finish: at each index of the slice, the
//! kernel loads a row of `NR` elements of `B`, broadcasts the `MR` elements of
//! the column of `A` one by one, and issues `MR × NR / width` FMAs. It only
//! touches `C` once, at the end.
//!
//! The panels of `A` and `B` arrive packed (see `gemm.rs`): for each index `p`
//! of the slice, the `MR` elements of `A`, then the `NR` elements of `B`, are
//! contiguous. Ragged edges were padded with zeros during packing: the kernel
//! always computes a full block, and only writes the useful part.

/// What the matrix product requires from a micro-kernel.
pub trait MicroKernel: Sync {
    /// Rows of the block of `C`.
    const MR: usize;
    /// Columns of the block of `C`.
    const NR: usize;

    /// `C[..rows, ..cols] (+)= A · B` over a slice of `kc` indices.
    ///
    /// `a` holds `kc × MR` elements, `b` holds `kc × NR`. `c` points to element
    /// `(0, 0)` of the block, in a matrix of `ldc` columns. With `accumulate`,
    /// the product is added to `C`; otherwise it replaces it.
    ///
    /// # Safety
    ///
    /// The pointers must cover everything that is read and written, and the
    /// processor must support the kernel's instructions.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    );
}

/// Writes a block computed in a local buffer to `C`, keeping only `rows × cols`
/// of its elements.
///
/// # Safety
///
/// `c` must cover `rows` rows of `cols` elements, `ldc` elements apart.
#[inline(always)]
unsafe fn write_back<const MR: usize, const NR: usize>(
    block: &[[f32; NR]; MR],
    c: *mut f32,
    ldc: usize,
    rows: usize,
    cols: usize,
    accumulate: bool,
) {
    for (i, line) in block.iter().enumerate().take(rows) {
        for (j, &value) in line.iter().enumerate().take(cols) {
            // SAFETY: `i < rows` and `j < cols`, which the caller guarantees
            // are covered by `c`.
            unsafe {
                let target = c.add(i * ldc + j);
                *target = if accumulate { *target + value } else { value };
            }
        }
    }
}

// --- Scalar ---

/// The fallback without explicit vector instructions: a `4 × 8` block, which
/// the compiler vectorizes as best it can.
pub struct Scalar;

impl MicroKernel for Scalar {
    const MR: usize = 4;
    const NR: usize = 8;

    unsafe fn run(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut block = [[0.0f32; 8]; 4];
        for p in 0..kc {
            // SAFETY: the caller guarantees that `a` holds `kc × 4` elements
            // and `b` holds `kc × 8`.
            let (a, b) = unsafe {
                (
                    std::slice::from_raw_parts(a.add(p * 4), 4),
                    std::slice::from_raw_parts(b.add(p * 8), 8),
                )
            };
            for i in 0..4 {
                for j in 0..8 {
                    block[i][j] += a[i] * b[j];
                }
            }
        }
        // SAFETY: the caller guarantees that `c` covers the `rows × cols` block.
        unsafe { write_back(&block, c, ldc, rows, cols, accumulate) };
    }
}

// --- AVX2 ---

#[cfg(target_arch = "x86_64")]
pub use x86::{Avx2, Avx512, bf16_14x32};

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{MicroKernel, write_back};

    /// AVX2 and FMA: sixteen 256-bit registers. A `6 × 16` block, that is
    /// twelve accumulators, two registers for the row of `B` and one for the
    /// broadcast element of `A`.
    pub struct Avx2;

    impl MicroKernel for Avx2 {
        const MR: usize = 6;
        const NR: usize = 16;

        unsafe fn run(
            kc: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
            rows: usize,
            cols: usize,
            accumulate: bool,
        ) {
            // SAFETY: the caller upholds the contract of `MicroKernel::run`,
            // including AVX2 and FMA support.
            unsafe { avx2_6x16(kc, a, b, c, ldc, rows, cols, accumulate) }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn avx2_6x16(
        kc: usize,
        mut a: *const f32,
        mut b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm256_setzero_ps(); 2]; 6];
        for _ in 0..kc {
            // SAFETY: the packed panels hold `kc × 6` elements of `A` and
            // `kc × 16` of `B`.
            unsafe {
                let b0 = _mm256_loadu_ps(b);
                let b1 = _mm256_loadu_ps(b.add(8));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai = _mm256_broadcast_ss(&*a.add(i));
                    line[0] = _mm256_fmadd_ps(ai, b0, line[0]);
                    line[1] = _mm256_fmadd_ps(ai, b1, line[1]);
                }
                a = a.add(6);
                b = b.add(16);
            }
        }
        if rows == 6 && cols == 16 {
            for (i, line) in acc.iter().enumerate() {
                // SAFETY: the block is full, so `c` covers 6 rows of 16
                // elements.
                unsafe {
                    let target = c.add(i * ldc);
                    let (mut low, mut high) = (line[0], line[1]);
                    if accumulate {
                        low = _mm256_add_ps(low, _mm256_loadu_ps(target));
                        high = _mm256_add_ps(high, _mm256_loadu_ps(target.add(8)));
                    }
                    _mm256_storeu_ps(target, low);
                    _mm256_storeu_ps(target.add(8), high);
                }
            }
        } else {
            let mut block = [[0.0f32; 16]; 6];
            for (line, values) in block.iter_mut().zip(&acc) {
                // SAFETY: each `line` of the local block holds 16 elements.
                unsafe {
                    _mm256_storeu_ps(line.as_mut_ptr(), values[0]);
                    _mm256_storeu_ps(line.as_mut_ptr().add(8), values[1]);
                }
            }
            // SAFETY: the caller guarantees that `c` covers the `rows × cols`
            // block.
            unsafe { write_back(&block, c, ldc, rows, cols, accumulate) };
        }
    }

    // --- AVX-512 ---

    /// AVX-512: thirty-two 512-bit registers. A `14 × 32` block, that is
    /// twenty-eight accumulators, two registers for the row of `B` and one
    /// for the broadcast element of `A`.
    ///
    /// Zen 4 executes a 512-bit instruction in two halves: this kernel does
    /// not compute faster per cycle than the AVX2 one, but it loads half as
    /// much of `A` and `B` per FMA, and decodes half as many instructions.
    pub struct Avx512;

    impl MicroKernel for Avx512 {
        const MR: usize = 14;
        const NR: usize = 32;

        unsafe fn run(
            kc: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
            rows: usize,
            cols: usize,
            accumulate: bool,
        ) {
            // SAFETY: the caller upholds the contract of `MicroKernel::run`,
            // including AVX-512 support.
            unsafe { avx512_14x32(kc, a, b, c, ldc, rows, cols, accumulate) }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn avx512_14x32(
        kc: usize,
        mut a: *const f32,
        mut b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 14];
        for _ in 0..kc {
            // SAFETY: the packed panels hold `kc × 14` elements of `A` and
            // `kc × 32` of `B`.
            unsafe {
                let b0 = _mm512_loadu_ps(b);
                let b1 = _mm512_loadu_ps(b.add(16));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai = _mm512_set1_ps(*a.add(i));
                    line[0] = _mm512_fmadd_ps(ai, b0, line[0]);
                    line[1] = _mm512_fmadd_ps(ai, b1, line[1]);
                }
                a = a.add(14);
                b = b.add(32);
            }
        }
        // SAFETY: the caller guarantees that `c` covers the `rows × cols`
        // block.
        unsafe { store_14x32(&acc, c, ldc, rows, cols, accumulate) };
    }

    /// Writes the `14 × 32` block of accumulators to `C`, keeping only
    /// `rows × cols` of its elements: masked loads and stores, which neither
    /// read nor write the columns beyond `cols`.
    ///
    /// # Safety
    ///
    /// AVX-512F; `c` covers `rows ≤ 14` rows of `cols ≤ 32` elements, `ldc`
    /// elements apart.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_14x32(
        acc: &[[__m512; 2]; 14],
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        // The useful columns of each half, as a mask.
        let low_mask: __mmask16 = if cols >= 16 {
            0xffff
        } else {
            ((1u32 << cols) - 1) as u16
        };
        let high_mask: __mmask16 = if cols >= 32 {
            0xffff
        } else if cols > 16 {
            ((1u32 << (cols - 16)) - 1) as u16
        } else {
            0
        };
        for (i, line) in acc.iter().enumerate().take(rows) {
            // SAFETY: `i < rows`, and the masks keep loads and stores within
            // the `cols` columns covered by `c`.
            unsafe {
                let target = c.add(i * ldc);
                let (mut low, mut high) = (line[0], line[1]);
                if accumulate {
                    low = _mm512_add_ps(low, _mm512_maskz_loadu_ps(low_mask, target));
                    high = _mm512_add_ps(
                        high,
                        _mm512_maskz_loadu_ps(high_mask, target.wrapping_add(16)),
                    );
                }
                _mm512_mask_storeu_ps(target, low_mask, low);
                _mm512_mask_storeu_ps(target.wrapping_add(16), high_mask, high);
            }
        }
    }

    // --- bf16 ---

    /// The mixed-precision micro-kernel: `vdpbf16ps` multiplies two pairs of
    /// `bf16` per lane and adds both products to an `f32`, twice the
    /// operations of an `f32` FMA per instruction. The block of `C` is the
    /// same, `14 × 32`, in `f32`. `a` and `b` arrive packed by pairs of shared
    /// indices: for each pair, the `MR` pairs of `A` (four bytes each), then
    /// the `NR` pairs of `B`.
    ///
    /// # Safety
    ///
    /// As for [`MicroKernel::run`], with `pairs` the number of pairs.
    #[target_feature(enable = "avx512f,avx512bf16")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bf16_14x32(
        pairs: usize,
        mut a: *const u32,
        mut b: *const u32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 14];
        for _ in 0..pairs {
            // SAFETY: the packed panels hold `pairs × 14` pairs of `A` and
            // `pairs × 32` of `B`; `__m512bh` has the size of `__m512i`.
            unsafe {
                let b0: __m512bh = std::mem::transmute(_mm512_loadu_si512(b as *const _));
                let b1: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(16) as *const _));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai: __m512bh = std::mem::transmute(_mm512_set1_epi32(*a.add(i) as i32));
                    line[0] = _mm512_dpbf16_ps(line[0], ai, b0);
                    line[1] = _mm512_dpbf16_ps(line[1], ai, b1);
                }
                a = a.add(14);
                b = b.add(32);
            }
        }
        // SAFETY: the caller guarantees that `c` covers the `rows × cols`
        // block; AVX512-BF16 implies AVX-512F.
        unsafe { store_14x32(&acc, c, ldc, rows, cols, accumulate) };
    }
}
