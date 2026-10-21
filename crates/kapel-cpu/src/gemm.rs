//! Blocked matrix product, in the style of Goto and BLIS.
//!
//! The triple-loop product rereads all of `B` for each row of `A`. As soon as
//! `B` no longer fits in the caches, every FMA waits for memory. The blocked
//! product splits the computation so that each cache level keeps what it
//! rereads:
//!
//! ```text
//! for each slice of nc columns of B              (jc)
//!   for each slice of kc shared indices          (pc)  B[pc, jc] packed, in L3
//!     for each block of mc rows of A             (ic)  A[ic, pc] packed, in L2
//!       for each panel of NR columns             (jr)  one panel of B, in L1
//!         for each panel of MR rows              (ir)
//!           micro-kernel: C[MR × NR] += A[MR × kc] · B[kc × NR]
//! ```
//!
//! Packing means copying a block of `A` or `B` in the exact order in which the
//! micro-kernel will read it, padded with zeros at the edges. This is also
//! where transposed matrices are handled: packing reads any strides, and the
//! micro-kernel only ever sees contiguous panels.
//!
//! The threads share the row blocks and, when there are not enough of them,
//! groups of column panels. This sharing does not change how an element of
//! `C` is computed: its sum always proceeds `kc` slice after `kc` slice, in the
//! same order. So the result does not depend on the number of threads.

use std::cell::RefCell;
use std::thread::LocalKey;

use kapel_core::{Error, Result};

use crate::kernels::{MicroKernel, Scalar};
use crate::pool::{Pool, SharedMut};
use crate::{Isa, IsaKind};

/// A matrix read with arbitrary strides: element `(i, j)` is
/// `data[i × row_stride + j × col_stride]`. Transposing a matrix just swaps
/// the strides; a stride of zero repeats a row or a column (broadcasting).
///
/// A `Matrix` always fits in its data: the kernels read it through raw
/// pointers, so the constructors check that its last element,
/// `(rows − 1) × row_stride + (cols − 1) × col_stride`, lies within `data`,
/// and the fields are private so that this cannot be undone.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a> Matrix<'a> {
    /// A matrix stored row by row.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if `data` holds fewer than `rows × cols`
    /// elements.
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Result<Self> {
        Self::strided(data, rows, cols, cols, 1)
    }

    /// A matrix with any strides, possibly zero.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an element lies beyond `data` (or its
    /// position overflows), unless the matrix is empty.
    pub fn strided(
        data: &'a [f32],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self> {
        if rows > 0 && cols > 0 {
            let last = (rows - 1)
                .checked_mul(row_stride)
                .zip((cols - 1).checked_mul(col_stride))
                .and_then(|(r, c)| r.checked_add(c));
            if !last.is_some_and(|last| last < data.len()) {
                return Err(Error::InvalidArgument {
                    op: "Matrix",
                    message: format!(
                        "a {rows} × {cols} matrix with strides ({row_stride}, {col_stride}) \
                         reads beyond its {} elements",
                        data.len()
                    ),
                });
            }
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// A matrix stored row by row, in a buffer the crate sized itself.
    ///
    /// # Panics
    ///
    /// If `data` is too short: an internal invariant is broken.
    pub(crate) fn contiguous(data: &'a [f32], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols).expect("a buffer sized for its matrix")
    }

    /// The transpose, without copying anything.
    pub fn t(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Distance in the data between two consecutive rows.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Distance in the data between two consecutive columns.
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// Columns `begin..end`. The result still fits in its data: it starts
    /// `begin` columns further and ends no later.
    fn columns(self, begin: usize, end: usize) -> Self {
        debug_assert!(begin < end && end <= self.cols);
        Self {
            data: &self.data[begin * self.col_stride..],
            cols: end - begin,
            ..self
        }
    }

    /// Rows `begin..end`, as [`Matrix::columns`].
    fn row_range(self, begin: usize, end: usize) -> Self {
        self.t().columns(begin, end).t()
    }

    #[inline(always)]
    fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.row_stride + j * self.col_stride]
    }
}

/// The block sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blocks {
    /// The slice of the shared dimension. This is the only size that changes
    /// the result: it fixes the order of the additions.
    pub kc: usize,
    /// The largest block of rows of `A`.
    pub mc: usize,
    /// The widest slice of columns of `B`.
    pub nc: usize,
    /// How the threads share the product.
    pub split: Split,
    /// Up to this many rows, the product is computed without packing: each
    /// row of `B` is read once and used for every row of `A`.
    pub few_rows: usize,
}

/// How the product is shared between the threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Each thread takes blocks of rows of `A` and `C`, and reads the same
    /// packed slice of `B`, shared in L3. Columns are only shared when there
    /// are not enough rows.
    Rows,
    /// Each thread takes a group of columns of `B` and `C`, which stays in its
    /// L2, and packs each block of `A` itself.
    Columns,
}

impl Blocks {
    /// The sizes chosen for Zen 4: a `kc × NR` panel of `B` in half of the
    /// 32 KiB L1, an `mc × kc` block of `A` in half of the 1 MiB L2, and the
    /// `kc × nc` slice of `B` in a share of L3.
    pub const ZEN4: Blocks = Blocks {
        kc: 512,
        mc: 392,
        nc: 4096,
        split: Split::Rows,
        few_rows: FEW_ROWS,
    };
}

/// The target number of tasks per thread: threads take new tasks as they
/// finish previous ones, and having more tasks than threads makes them finish
/// together.
const TASKS_PER_THREAD: usize = 3;

/// The boundary between the two paths when `B` is stored row by row, as
/// measured on Zen 4: up to 256 rows, the block of `C` stays in registers and
/// each row of `B` is read in one sweep.
pub const FEW_ROWS: usize = 256;
/// The same boundary when `B` is transposed, where each element is a dot
/// product: up to 16 rows.
pub const FEW_ROWS_TRANSPOSED: usize = 16;

thread_local! {
    /// The packed block of `A`, private to each thread.
    static PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// The packed slice of `B`, private to the thread that starts the product.
    static PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Checks the shapes of a product. These checks, with the invariant of
/// [`Matrix`], are what make every access of the kernels in bounds.
fn check_shapes(a: &Matrix, b: &Matrix, c: &[f32]) {
    assert_eq!(
        a.cols, b.rows,
        "gemm: A has {} columns but B has {} rows",
        a.cols, b.rows
    );
    let size = a.rows.checked_mul(b.cols);
    assert_eq!(
        size,
        Some(c.len()),
        "gemm: C must hold {} × {} elements",
        a.rows,
        b.cols
    );
}

/// `C = A · B`, or `C += A · B` with `accumulate`. `C` is stored row by row,
/// `a.rows() × b.cols()`.
///
/// # Panics
///
/// If `a.cols() != b.rows()`, or if `c` does not hold `a.rows() × b.cols()`
/// elements. This is a low-level function: the backend checks the shapes of
/// its tensors, with typed errors, before calling it.
pub fn gemm(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    gemm_with_blocks(pool, isa, a, b, c, accumulate, Blocks::ZEN4);
}

/// Like [`gemm`], with given block sizes: for benchmarking them. Sizes of
/// zero are taken as one.
///
/// # Panics
///
/// As [`gemm`].
pub fn gemm_with_blocks(
    pool: &Pool,
    isa: Isa,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    check_shapes(&a, &b, c);
    let blocks = Blocks {
        kc: blocks.kc.max(1),
        mc: blocks.mc.max(1),
        nc: blocks.nc.max(1),
        ..blocks
    };
    if c.is_empty() {
        return;
    }
    if a.cols == 0 {
        if !accumulate {
            c.fill(0.0);
        }
        return;
    }
    let limit = if b.col_stride == 1 {
        blocks.few_rows
    } else {
        blocks.few_rows.min(FEW_ROWS_TRANSPOSED)
    };
    if a.rows <= limit {
        few_rows(pool, isa, a, b, c, accumulate);
        return;
    }
    match isa.kind() {
        // SAFETY: an `Isa` of kind `Avx512` proves that the processor has
        // AVX-512F, the instructions of this micro-kernel.
        #[cfg(target_arch = "x86_64")]
        IsaKind::Avx512 => unsafe {
            blocked::<F32<crate::kernels::Avx512>>(pool, a, b, c, accumulate, blocks)
        },
        // SAFETY: an `Isa` of kind `Avx2` proves AVX2 and FMA.
        #[cfg(target_arch = "x86_64")]
        IsaKind::Avx2 => unsafe {
            blocked::<F32<crate::kernels::Avx2>>(pool, a, b, c, accumulate, blocks)
        },
        // SAFETY: the scalar kernel needs no particular instructions.
        _ => unsafe { blocked::<F32<Scalar>>(pool, a, b, c, accumulate, blocks) },
    }
}

/// How the blocked product packs its operands and multiplies the panels: in
/// `f32` with a [`MicroKernel`], or by pairs of `bf16`.
trait Packing {
    /// A packed element.
    type Packed: Copy + Default + Send + Sync + 'static;
    /// Rows of the block of `C`.
    const MR: usize;
    /// Columns of the block of `C`.
    const NR: usize;

    /// The packed length of `depth` shared indices.
    fn packed(depth: usize) -> usize;

    /// Packs `rows ≤ MR` rows of `A` from `row`, over `depth` shared indices
    /// from `pc`, padded with zeros, into `out` (`MR × packed(depth)`).
    fn pack_a(
        a: Matrix,
        row: usize,
        rows: usize,
        pc: usize,
        depth: usize,
        out: &mut [Self::Packed],
    );

    /// Packs `cols ≤ NR` columns of `B` from `col`, over `depth` shared
    /// indices from `pc`, padded with zeros, into `out` (`NR × packed(depth)`).
    fn pack_b(
        b: Matrix,
        pc: usize,
        depth: usize,
        col: usize,
        cols: usize,
        out: &mut [Self::Packed],
    );

    /// `C[..rows, ..cols] (+)= A · B` over `packed` packed indices.
    ///
    /// # Safety
    ///
    /// As [`MicroKernel::run`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        packed: usize,
        a: *const Self::Packed,
        b: *const Self::Packed,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    );

    /// The packed block of `A`, private to each thread.
    fn scratch_a() -> &'static LocalKey<RefCell<Vec<Self::Packed>>>;
    /// The packed slice of `B`, private to the thread that starts the product.
    fn scratch_b() -> &'static LocalKey<RefCell<Vec<Self::Packed>>>;
}

/// The `f32` product with micro-kernel `K`.
struct F32<K>(std::marker::PhantomData<K>);

impl<K: MicroKernel> Packing for F32<K> {
    type Packed = f32;
    const MR: usize = K::MR;
    const NR: usize = K::NR;

    #[inline(always)]
    fn packed(depth: usize) -> usize {
        depth
    }

    fn pack_a(a: Matrix, row: usize, rows: usize, pc: usize, depth: usize, out: &mut [f32]) {
        pack_a(a, row, rows, pc, depth, K::MR, out);
    }

    fn pack_b(b: Matrix, pc: usize, depth: usize, col: usize, cols: usize, out: &mut [f32]) {
        pack_b(b, pc, depth, col, cols, K::NR, out);
    }

    #[inline(always)]
    unsafe fn run(
        packed: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        // SAFETY: the caller upholds the contract of `MicroKernel::run`.
        unsafe { K::run(packed, a, b, c, ldc, rows, cols, accumulate) }
    }

    fn scratch_a() -> &'static LocalKey<RefCell<Vec<f32>>> {
        &PACKED_A
    }

    fn scratch_b() -> &'static LocalKey<RefCell<Vec<f32>>> {
        &PACKED_B
    }
}

/// The blocked product (see the module documentation), for a given packing
/// and micro-kernel.
///
/// # Safety
///
/// The processor must support the instructions of `P::run`. The shapes must
/// have been checked by `check_shapes`, and `C` must not be empty.
unsafe fn blocked<P: Packing>(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let (mr, nr) = (P::MR, P::NR);
    let kc = blocks.kc.max(1);
    let nc = blocks.nc.max(1).div_ceil(nr) * nr;
    let threads = pool.threads();

    // Row blocks: at most `mc` rows, and enough of them for at least
    // `TASKS_PER_THREAD` tasks per thread, so that the threads finish together
    // even when the tasks do not divide evenly. When there are not enough
    // rows, the columns are shared too.
    let wanted = TASKS_PER_THREAD * threads;
    let row_panels = m.div_ceil(mr);
    let row_blocks = match blocks.split {
        Split::Rows => m.div_ceil(blocks.mc.max(mr)).max(wanted.min(row_panels)),
        Split::Columns => m.div_ceil(blocks.mc.max(mr)),
    };
    let mc = row_panels.div_ceil(row_blocks) * mr;
    let row_blocks = m.div_ceil(mc);

    let out = SharedMut::new(c);
    P::scratch_b().with(|packed_b| {
        let mut packed_b = packed_b.borrow_mut();
        for jc in (0..n).step_by(nc) {
            let width = nc.min(n - jc);
            let panels = width.div_ceil(nr);
            // Groups of at least two panels, when the rows are not enough to
            // keep the threads busy.
            let col_groups = match blocks.split {
                Split::Rows => wanted.div_ceil(row_blocks),
                Split::Columns => wanted,
            };
            let col_groups = col_groups.min(panels.div_ceil(2)).max(1);
            let per_group = panels.div_ceil(col_groups);
            let col_groups = panels.div_ceil(per_group);
            for pc in (0..k).step_by(kc) {
                let depth = kc.min(k - pc);
                let packed = P::packed(depth);
                let accumulate = accumulate || pc > 0;

                // The slice of B, packed in panels of NR columns.
                let size = panels * nr * packed;
                if packed_b.len() < size {
                    packed_b.resize(size, P::Packed::default());
                }
                let shared_b = SharedMut::new(&mut packed_b[..size]);
                pool.run(panels, |panel| {
                    // SAFETY: each task packs its own panel of `packed_b`.
                    let target = unsafe { shared_b.slice(panel * nr * packed, nr * packed) };
                    let col = jc + panel * nr;
                    P::pack_b(b, pc, depth, col, nr.min(width - panel * nr), target);
                });
                let packed_b = &packed_b[..size];

                pool.run(row_blocks * col_groups, |task| {
                    let (row_block, group) = (task / col_groups, task % col_groups);
                    let ic = row_block * mc;
                    let height = mc.min(m - ic);
                    let first = group * per_group;
                    let last = panels.min(first + per_group);
                    P::scratch_a().with(|packed_a| {
                        let mut packed_a = packed_a.borrow_mut();
                        let row_panels = height.div_ceil(mr);
                        let size = row_panels * mr * packed;
                        if packed_a.len() < size {
                            packed_a.resize(size, P::Packed::default());
                        }
                        for panel in 0..row_panels {
                            let rows = mr.min(height - panel * mr);
                            let target = &mut packed_a[panel * mr * packed..][..mr * packed];
                            P::pack_a(a, ic + panel * mr, rows, pc, depth, target);
                        }
                        for jr in first..last {
                            let col = jc + jr * nr;
                            let cols = nr.min(n - col);
                            let panel_b = packed_b[jr * nr * packed..][..nr * packed].as_ptr();
                            for ir in 0..row_panels {
                                let row = ic + ir * mr;
                                let rows = mr.min(m - row);
                                let panel_a = packed_a[ir * mr * packed..][..mr * packed].as_ptr();
                                // SAFETY: the caller guarantees the
                                // instructions of `P::run`. The packed panels
                                // hold `mr × packed` and `nr × packed`
                                // elements. `C` holds `m × n` elements (checked
                                // by `check_shapes`), so the `rows × cols`
                                // block at `(row, col)` lies within it; each
                                // task writes its own rows and columns.
                                unsafe {
                                    P::run(
                                        packed,
                                        panel_a,
                                        panel_b,
                                        out.ptr(row * n + col),
                                        n,
                                        rows,
                                        cols,
                                        accumulate,
                                    );
                                }
                            }
                        }
                    });
                });
            }
        }
    });
}

// --- Mixed precision ---

/// Whether the processor can multiply `bf16` values (`vdpbf16ps`).
pub fn bf16_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        IsaKind::Avx512.available() && is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Like [`gemm`], with the operands of a large product rounded to `bf16` and
/// their products accumulated in `f32`: mixed precision. Products with few
/// rows, which are memory-bound, stay in `f32`, as on the GPU, and so does
/// everything on a processor without `vdpbf16ps`.
///
/// # Panics
///
/// As [`gemm`].
pub fn gemm_mixed(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    check_shapes(&a, &b, c);
    #[cfg(target_arch = "x86_64")]
    {
        let limit = if b.col_stride == 1 {
            FEW_ROWS
        } else {
            FEW_ROWS_TRANSPOSED
        };
        let large = a.rows > limit && a.cols > 0 && !c.is_empty();
        if large && isa.kind() == IsaKind::Avx512 && bf16_available() {
            // SAFETY: `bf16_available` checked AVX-512F and AVX512-BF16, the
            // instructions of the kernel; the shapes are checked and `C` is
            // not empty.
            unsafe { blocked::<Bf16>(pool, a, b, c, accumulate, Blocks::ZEN4) };
            return;
        }
    }
    gemm(pool, isa, a, b, c, accumulate);
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// The packed `bf16` pairs of a block of `A`, private to each thread.
    static PAIRS_A: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// The packed `bf16` pairs of a slice of `B`, private to the thread that
    /// starts the product.
    static PAIRS_B: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// The mixed-precision product: the `kc` slice is counted in indices, and
/// packed by pairs of `bf16` in a `u32`, the lower index in the low half.
#[cfg(target_arch = "x86_64")]
struct Bf16;

#[cfg(target_arch = "x86_64")]
impl Bf16 {
    /// The pair of `bf16` of elements `(p, p + 1)` read by `at`; zero beyond
    /// `depth`.
    #[inline(always)]
    fn pair(at: impl Fn(usize) -> f32, p: usize, depth: usize) -> u32 {
        let low = u32::from(crate::io::to_bf16(at(p)));
        let high = if p + 1 < depth {
            u32::from(crate::io::to_bf16(at(p + 1)))
        } else {
            0
        };
        low | (high << 16)
    }
}

#[cfg(target_arch = "x86_64")]
impl Packing for Bf16 {
    type Packed = u32;
    const MR: usize = 14;
    const NR: usize = 32;

    #[inline(always)]
    fn packed(depth: usize) -> usize {
        depth.div_ceil(2)
    }

    fn pack_a(a: Matrix, row: usize, rows: usize, pc: usize, depth: usize, out: &mut [u32]) {
        out.fill(0);
        for p in 0..depth.div_ceil(2) {
            for i in 0..rows {
                out[p * Self::MR + i] = Self::pair(|q| a.at(row + i, pc + q), 2 * p, depth);
            }
        }
    }

    fn pack_b(b: Matrix, pc: usize, depth: usize, col: usize, cols: usize, out: &mut [u32]) {
        out.fill(0);
        for p in 0..depth.div_ceil(2) {
            for j in 0..cols {
                out[p * Self::NR + j] = Self::pair(|q| b.at(pc + q, col + j), 2 * p, depth);
            }
        }
    }

    #[inline(always)]
    unsafe fn run(
        packed: usize,
        a: *const u32,
        b: *const u32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        // SAFETY: the caller upholds the contract, AVX512-BF16 included.
        unsafe { crate::kernels::bf16_14x32(packed, a, b, c, ldc, rows, cols, accumulate) }
    }

    fn scratch_a() -> &'static LocalKey<RefCell<Vec<u32>>> {
        &PAIRS_A
    }

    fn scratch_b() -> &'static LocalKey<RefCell<Vec<u32>>> {
        &PAIRS_B
    }
}

/// Packs `rows` rows of `A`, starting at `row`, over `depth` shared indices
/// starting at `pc`: for each index, the `mr` elements of the column, padded
/// with zeros.
fn pack_a(a: Matrix, row: usize, rows: usize, pc: usize, depth: usize, mr: usize, out: &mut [f32]) {
    if rows < mr {
        out.fill(0.0);
    }
    if a.col_stride == 1 {
        for i in 0..rows {
            let line = &a.data[(row + i) * a.row_stride + pc..][..depth];
            for (p, &value) in line.iter().enumerate() {
                out[p * mr + i] = value;
            }
        }
    } else {
        for p in 0..depth {
            for i in 0..rows {
                out[p * mr + i] = a.at(row + i, pc + p);
            }
        }
    }
}

/// Packs `cols` columns of `B`, starting at `col`, over `depth` shared indices
/// starting at `pc`: for each index, the `nr` elements of the row, padded
/// with zeros.
fn pack_b(b: Matrix, pc: usize, depth: usize, col: usize, cols: usize, nr: usize, out: &mut [f32]) {
    if cols < nr {
        out.fill(0.0);
    }
    if b.col_stride == 1 {
        for p in 0..depth {
            let line = &b.data[(pc + p) * b.row_stride + col..][..cols];
            out[p * nr..p * nr + cols].copy_from_slice(line);
        }
    } else {
        for j in 0..cols {
            for p in 0..depth {
                out[p * nr + j] = b.at(pc + p, col + j);
            }
        }
    }
}

// --- Few rows ---

/// Columns come in groups of this size: one thread's task. A single row keeps
/// all of them in its accumulators; several rows take them in blocks of
/// `FEW_ROWS_BLOCK`, so that their accumulators fit in registers.
const FEW_ROWS_COLUMNS: usize = 256;
/// The column block for several rows (see `FEW_ROWS_COLUMNS`).
const FEW_ROWS_BLOCK: usize = 64;
/// The target number of tasks: enough to keep every thread busy several
/// times. It does not depend on the number of threads, so that the result
/// does not either.
const FEW_ROWS_TASKS: usize = 32;
/// A slice of the shared dimension has at least this length.
const FEW_ROWS_DEPTH: usize = 128;
/// Rows of `Bᵀ` come in groups of this size, for the product with a
/// transpose.
const FEW_ROWS_DOTS: usize = 64;

/// The product when `A` has only a few rows, as during generation. There is
/// nothing to reuse in the caches: each weight is used once per row. Packing
/// `B` would cost more than the product itself.
fn few_rows(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    if b.col_stride == 1 {
        // Groups of columns, plus slices of the shared dimension when there
        // are not enough columns to make `FEW_ROWS_TASKS` tasks: each slice
        // writes its partial sum, and the sums are then added in slice order.
        // This split depends only on the shapes.
        let width = if m == 1 && n >= FEW_ROWS_COLUMNS {
            FEW_ROWS_COLUMNS
        } else {
            FEW_ROWS_BLOCK
        };
        let groups = n.div_ceil(width);
        let slices = FEW_ROWS_TASKS
            .div_ceil(groups)
            .min(k.div_ceil(FEW_ROWS_DEPTH))
            .max(1);
        let depth = k.div_ceil(slices);
        let slices = k.div_ceil(depth);
        let mut partial = if slices > 1 {
            vec![0.0f32; (slices - 1) * m * n]
        } else {
            Vec::new()
        };
        let out = SharedMut::new(c);
        let rest = SharedMut::new(&mut partial);
        pool.run(groups * slices, |task| {
            let (slice, group) = (task / groups, task % groups);
            let first = group * width;
            let end = n.min(first + width);
            let (begin, stop) = (slice * depth, k.min((slice + 1) * depth));
            let (a, b) = (a.columns(begin, stop), b.row_range(begin, stop));
            // The first slice writes into C, the others into separate buffers.
            let (target, accumulate) = if slice == 0 {
                (out, accumulate)
            } else {
                (
                    // SAFETY: each slice after the first owns its own `m × n`
                    // part of `partial`.
                    unsafe { SharedMut::from_ptr(rest.ptr((slice - 1) * m * n)) },
                    false,
                )
            };
            if width == FEW_ROWS_COLUMNS {
                let block = Block {
                    a,
                    row: 0,
                    b,
                    start: first,
                    width: end - first,
                    out: target,
                    n,
                    accumulate,
                };
                direct::<FEW_ROWS_COLUMNS>(isa, block, 1);
                return;
            }
            for row in (0..m).step_by(4) {
                let block = Block {
                    a,
                    row,
                    b,
                    start: first,
                    width: end - first,
                    out: target,
                    n,
                    accumulate,
                };
                direct::<FEW_ROWS_BLOCK>(isa, block, m - row);
            }
        });
        for part in partial.chunks(m * n) {
            for (value, &x) in c.iter_mut().zip(part) {
                *value += x;
            }
        }
    } else if b.row_stride == 1 {
        // `B` is a transpose: each element is the dot product of two
        // contiguous rows.
        let out = SharedMut::new(c);
        let groups = n.div_ceil(FEW_ROWS_DOTS);
        pool.run(groups, |group| {
            let start = group * FEW_ROWS_DOTS;
            let width = FEW_ROWS_DOTS.min(n - start);
            for i in 0..m {
                // SAFETY: each task writes its own columns of C.
                let target = unsafe { out.slice(i * n + start, width) };
                for (offset, value) in target.iter_mut().enumerate() {
                    let column = &b.data[(start + offset) * b.col_stride..][..b.rows];
                    let dot = dot_with_row(isa, a, i, column);
                    *value = if accumulate { *value + dot } else { dot };
                }
            }
        });
    } else {
        // SAFETY: the scalar kernel needs no particular instructions; the
        // caller checked the shapes, and `C` is not empty.
        unsafe { blocked::<F32<Scalar>>(pool, a, b, c, accumulate, Blocks::ZEN4) };
    }
}

/// A block of `C` for the product without packing: a few rows starting at
/// `row`, `width` columns starting at `start`.
#[derive(Clone, Copy)]
struct Block<'a> {
    a: Matrix<'a>,
    row: usize,
    b: Matrix<'a>,
    start: usize,
    width: usize,
    out: SharedMut<f32>,
    n: usize,
    accumulate: bool,
}

/// `C[R rows, W columns] (+)= A · B`, without packing: the block of `C` stays
/// in registers (sixteen AVX-512 registers for four rows of 64 columns, or one
/// row of 256), and each row of `B` serves the `R` rows of `A` at once.
#[inline(always)]
fn direct_body<const R: usize, const W: usize, const FMA: bool>(block: Block) {
    let Block {
        a,
        row,
        b,
        start,
        width,
        out,
        n,
        accumulate,
    } = block;
    let mut acc = [[0.0f32; W]; R];
    let mut padded = [0.0f32; W];
    for p in 0..a.cols {
        let line = &b.data[p * b.row_stride + start..][..width];
        let line: &[f32; W] = if width == W {
            line.try_into().unwrap()
        } else {
            padded[..width].copy_from_slice(line);
            &padded
        };
        for (r, sums) in acc.iter_mut().enumerate() {
            let x = a.at(row + r, p);
            for j in 0..W {
                sums[j] = if FMA {
                    x.mul_add(line[j], sums[j])
                } else {
                    sums[j] + x * line[j]
                };
            }
        }
    }
    for (r, sums) in acc.iter().enumerate() {
        // SAFETY: each task writes its own columns.
        let target = unsafe { out.slice((row + r) * n + start, width) };
        for (value, &sum) in target.iter_mut().zip(sums.iter()) {
            *value = if accumulate { *value + sum } else { sum };
        }
    }
}

/// `C[rows, W columns] (+)= A · B` for a block of `C` with `rows` rows (four at
/// most are used) and `block.width ≤ W` columns.
fn direct<const W: usize>(isa: Isa, block: Block, rows: usize) {
    assert!(
        block.width <= W
            && block.start + block.width <= block.b.cols
            && block.row + rows.min(4) <= block.a.rows,
        "a block within C"
    );
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2<const R: usize, const W: usize>(block: Block) {
        direct_body::<R, W, true>(block)
    }

    fn with<const R: usize, const W: usize>(isa: Isa, block: Block) {
        match isa.kind() {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: an `Isa` of kind `Avx512` proves AVX-512F. `W == 64` is
            // four 16-lane registers, and `direct` passes at most `W` columns
            // and `R` rows that exist (see `direct_avx512`).
            IsaKind::Avx512 if W == 64 => unsafe { x86::direct_avx512::<R, 4>(block) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above; `W == 256` is sixteen 16-lane registers.
            IsaKind::Avx512 if W == 256 && R == 1 => unsafe { x86::direct_avx512::<1, 16>(block) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: an `Isa` of either kind proves AVX2 and FMA;
            // `direct_body` only makes checked accesses.
            IsaKind::Avx2 | IsaKind::Avx512 => unsafe { avx2::<R, W>(block) },
            _ => direct_body::<R, W, false>(block),
        }
    }

    match rows {
        1 => with::<1, W>(isa, block),
        2 => with::<2, W>(isa, block),
        3 => with::<3, W>(isa, block),
        _ => with::<4, W>(isa, block),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Block;

    /// The product without packing in AVX-512: `R` rows of `16 × V` columns,
    /// that is `R × V` accumulators, which stay in registers from start to
    /// finish. Columns beyond `width` are masked out.
    ///
    /// # Safety
    ///
    /// The processor has AVX-512F; `row + R ≤ a.rows`, `width ≤ 16 × V`,
    /// `start + width ≤ b.cols`, `a.cols == b.rows`; `out` points to a
    /// row-major `C` of `n` columns with at least `row + R` rows and
    /// `n ≥ start + width`, whose block this task alone writes.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn direct_avx512<const R: usize, const V: usize>(block: Block) {
        let Block {
            a,
            row,
            b,
            start,
            width,
            out,
            n,
            accumulate,
        } = block;
        let masks: [__mmask16; V] = std::array::from_fn(|v| {
            let lanes = width.saturating_sub(16 * v).min(16);
            ((1u32 << lanes) - 1) as u16
        });
        let mut acc = [[_mm512_setzero_ps(); V]; R];
        let a_data = a.data.as_ptr();
        let mut line = b.data.as_ptr().wrapping_add(start);
        for p in 0..a.cols {
            // SAFETY: `row + r < a.rows` and `p < a.cols`, so the element of
            // `A` lies within `a.data` (invariant of `Matrix`). `line + 16v +
            // lane` is element `(p, start + 16v + lane)` of `B`, which exists
            // for every unmasked lane since `start + width ≤ b.cols` and
            // `p < b.rows`; masked-out lanes are not read.
            unsafe {
                let x: [__m512; R] = std::array::from_fn(|r| {
                    _mm512_set1_ps(*a_data.add((row + r) * a.row_stride + p * a.col_stride))
                });
                for (v, &mask) in masks.iter().enumerate() {
                    let w = _mm512_maskz_loadu_ps(mask, line.wrapping_add(16 * v));
                    for r in 0..R {
                        acc[r][v] = _mm512_fmadd_ps(x[r], w, acc[r][v]);
                    }
                }
            }
            line = line.wrapping_add(b.row_stride);
        }
        for (r, sums) in acc.iter().enumerate() {
            // SAFETY: row `row + r` of `C` exists and its columns
            // `start..start + width` lie within its `n` columns; this task
            // alone writes them, and masked-out columns are neither read nor
            // written.
            unsafe {
                let target = out.ptr((row + r) * n + start);
                for (v, &mask) in masks.iter().enumerate() {
                    let at = target.wrapping_add(16 * v);
                    let mut value = sums[v];
                    if accumulate {
                        value = _mm512_add_ps(value, _mm512_maskz_loadu_ps(mask, at));
                    }
                    _mm512_mask_storeu_ps(at, mask, value);
                }
            }
        }
    }
}

/// The dot product of row `i` of `A` with a contiguous row.
fn dot_with_row(isa: Isa, a: Matrix, i: usize, row: &[f32]) -> f32 {
    if a.col_stride == 1 {
        let line = &a.data[i * a.row_stride..][..a.cols];
        crate::math::dot(isa, line, row)
    } else {
        (0..a.cols).map(|p| a.at(i, p) * row[p]).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: Matrix, b: Matrix) -> Vec<f64> {
        let mut c = vec![0.0; a.rows * b.cols];
        for i in 0..a.rows {
            for j in 0..b.cols {
                c[i * b.cols + j] = (0..a.cols)
                    .map(|p| a.at(i, p) as f64 * b.at(p, j) as f64)
                    .sum();
            }
        }
        c
    }

    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// A row-major `rows × cols` matrix, or the transpose of a `cols × rows`
    /// one.
    fn matrix(data: &[f32], rows: usize, cols: usize, transpose: bool) -> Matrix<'_> {
        if transpose {
            Matrix::new(data, cols, rows).unwrap().t()
        } else {
            Matrix::new(data, rows, cols).unwrap()
        }
    }

    fn blocks(m: usize, k: usize, n: usize) -> Blocks {
        Blocks {
            kc: 48,
            mc: 40,
            nc: 64,
            split: if (m + k + n).is_multiple_of(2) {
                Split::Rows
            } else {
                Split::Columns
            },
            few_rows: FEW_ROWS,
        }
    }

    /// Checks `A · B`, then `A · B` accumulated, against the definition.
    fn check_matrices(isa: Isa, a: Matrix, b: Matrix, blocks: Blocks, what: &str) {
        let pool = Pool::new(4);
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let expected = reference(a, b);
        let mut c = vec![f32::NAN; m * n];
        gemm_with_blocks(&pool, isa, a, b, &mut c, false, blocks);
        for (index, (&got, &want)) in c.iter().zip(&expected).enumerate() {
            assert!(
                (got as f64 - want).abs() < 1e-4 * (k as f64).sqrt(),
                "{isa:?} {m}×{k}×{n} {what}, element {index}: {got} instead of {want}",
            );
        }
        // Accumulating adds the same product a second time.
        gemm_with_blocks(&pool, isa, a, b, &mut c, true, blocks);
        for (&got, &want) in c.iter().zip(&expected) {
            assert!((got as f64 - 2.0 * want).abs() < 2e-4 * (k as f64).sqrt());
        }
    }

    fn check(isa: Isa, m: usize, k: usize, n: usize, transpose_a: bool, transpose_b: bool) {
        let data_a = values(m * k, 1);
        let data_b = values(k * n, 2);
        let a = matrix(&data_a, m, k, transpose_a);
        let b = matrix(&data_b, k, n, transpose_b);
        let what = format!("(tA {transpose_a}, tB {transpose_b})");
        check_matrices(isa, a, b, blocks(m, k, n), &what);
    }

    fn isas() -> Vec<Isa> {
        Isa::all_available().collect()
    }

    #[test]
    fn blocked_product_matches_the_definition() {
        for isa in isas() {
            // Edges everywhere: no dimension is a multiple of MR, NR, kc, mc or nc.
            for &(m, k, n) in &[(5, 7, 3), (33, 100, 70), (130, 49, 150), (29, 200, 33)] {
                for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                    check(isa, m, k, n, ta, tb);
                }
            }
        }
    }

    #[test]
    fn few_rows_match_the_definition() {
        for isa in isas() {
            for &(m, k, n) in &[
                (1, 512, 300),
                (3, 70, 513),
                (4, 5, 1),
                (13, 100, 70),
                (16, 33, 300),
            ] {
                for (ta, tb) in [(false, false), (false, true), (true, true)] {
                    check(isa, m, k, n, ta, tb);
                }
            }
        }
    }

    #[test]
    fn any_strides_match_the_definition() {
        for isa in isas() {
            // Rows of 1, 3 and 300 cover the products with few rows, by rows
            // of `B` and by dot products, and the blocked one.
            for &(m, k, n) in &[(1, 70, 40), (3, 70, 300), (300, 70, 40)] {
                let data = values(4 * (m + n + 3) * (k + 3), 9);
                // Padded rows, and every second column.
                let a = Matrix::strided(&data, m, k, 2 * k + 3, 2).unwrap();
                let b = Matrix::strided(&data[5..], k, n, 1, k + 7).unwrap();
                check_matrices(isa, a, b, blocks(m, k, n), "padded");
                let b = Matrix::strided(&data, k, n, 2 * n + 1, 2).unwrap();
                check_matrices(isa, a, b, blocks(m, k, n), "every second column");
                // A stride of zero repeats a row or a column: broadcasting.
                let a = Matrix::strided(&data, m, k, 0, 1).unwrap();
                let b = Matrix::strided(&data, k, n, 1, 0).unwrap();
                check_matrices(isa, a, b, blocks(m, k, n), "broadcast A, B");
                let a = Matrix::strided(&data, m, k, 1, 0).unwrap();
                let b = Matrix::strided(&data, k, n, 0, 1).unwrap();
                check_matrices(isa, a, b, blocks(m, k, n), "broadcast Aᵀ, Bᵀ");
                let b = Matrix::strided(&data, k, n, 0, 0).unwrap();
                check_matrices(isa, a, b, blocks(m, k, n), "a single value");
            }
        }
    }

    #[test]
    fn matrices_must_fit_in_their_data() {
        let data = [0.0f32; 12];
        assert!(Matrix::new(&data, 3, 4).is_ok());
        assert!(Matrix::new(&data, 4, 4).is_err());
        assert!(Matrix::strided(&data, 2, 2, 10, 1).is_ok());
        assert!(Matrix::strided(&data, 2, 2, 11, 1).is_err());
        assert!(Matrix::strided(&data, 1 << 40, 3, 0, 1).is_ok());
        assert!(Matrix::strided(&data, usize::MAX, 2, usize::MAX, 1).is_err());
        assert!(Matrix::strided(&data, 2, 2, usize::MAX / 2 + 1, 1).is_err());
        // An empty matrix reads nothing.
        assert!(Matrix::strided(&[], 0, 5, 7, 1).is_ok());
        assert!(Matrix::strided(&[], 5, 0, 7, 1).is_ok());
    }

    #[test]
    #[should_panic(expected = "C must hold")]
    fn a_short_output_is_refused() {
        let data = [1.0f32; 4];
        let a = Matrix::new(&data, 2, 2).unwrap();
        gemm(&Pool::new(1), Isa::detect(), a, a, &mut [0.0; 3], false);
    }

    #[test]
    #[should_panic(expected = "C must hold")]
    fn an_overflowing_output_is_refused() {
        let data = [1.0f32; 4];
        let a = Matrix::strided(&data, 1 << 33, 2, 0, 1).unwrap();
        let b = Matrix::strided(&data, 2, 1 << 31, 1, 0).unwrap();
        gemm(&Pool::new(1), Isa::detect(), a, b, &mut [0.0; 0], false);
    }

    #[test]
    fn zero_block_sizes_are_taken_as_one() {
        let data = values(40 * 30, 3);
        let a = Matrix::new(&data, 40, 30).unwrap();
        let b = Matrix::new(&data[..30 * 20], 30, 20).unwrap();
        let blocks = Blocks {
            kc: 0,
            mc: 0,
            nc: 0,
            split: Split::Rows,
            few_rows: 0,
        };
        for isa in isas() {
            check_matrices(isa, a, b, blocks, "with blocks of zero");
        }
    }

    /// `NaN` and infinities propagate: a `NaN` in row `i` of `A` makes row `i`
    /// of `C` `NaN`, a `+∞` in column `j` of `B` makes column `j` `+∞` where
    /// `A` is positive.
    fn check_special_values(isa: Isa, mixed: bool) {
        let pool = Pool::new(3);
        for &(m, k, n) in &[(3, 40, 70), (20, 40, 70), (300, 600, 70)] {
            for (ta, tb) in [(false, false), (true, true), (false, true)] {
                let positive = |len, seed| -> Vec<f32> {
                    values(len, seed).iter().map(|v| 1.0 + 0.5 * v).collect()
                };
                let (mut data_a, mut data_b) = (positive(m * k, 1), positive(k * n, 2));
                let (row, col) = (m / 2, n - 1);
                // Element `(row, 7)` of A and `(k - 1, col)` of B, whichever
                // the storage.
                data_a[if ta { 7 * m + row } else { row * k + 7 }] = f32::NAN;
                data_b[if tb {
                    col * k + k - 1
                } else {
                    (k - 1) * n + col
                }] = f32::INFINITY;
                let (a, b) = (matrix(&data_a, m, k, ta), matrix(&data_b, k, n, tb));
                let mut c = vec![0.0; m * n];
                if mixed {
                    gemm_mixed(&pool, isa, a, b, &mut c, false);
                } else {
                    gemm(&pool, isa, a, b, &mut c, false);
                }
                for i in 0..m {
                    for j in 0..n {
                        let value = c[i * n + j];
                        let what = format!("{isa:?} {m}×{k}×{n} ({ta}, {tb}) at ({i}, {j})");
                        if i == row {
                            assert!(value.is_nan(), "{what}: {value}");
                        } else if j == col {
                            assert_eq!(value, f32::INFINITY, "{what}");
                        } else {
                            assert!(value.is_finite(), "{what}: {value}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nan_and_infinities_propagate() {
        for isa in isas() {
            check_special_values(isa, false);
            check_special_values(isa, true);
        }
    }

    #[test]
    fn mixed_products_are_exact_on_small_integers_and_close_otherwise() {
        if !bf16_available() {
            return;
        }
        let pool = Pool::new(4);
        let isa = Isa::new(IsaKind::Avx512).unwrap();
        for &(m, k, n, ta, tb) in &[
            (300, 70, 150, false, false),
            (290, 33, 65, true, false),
            (400, 129, 70, false, true),
        ] {
            // Integers between -8 and 8 are exact in bf16, and so are their
            // sums in f32: the result must match the f32 one.
            let small = |len: usize, seed: u64| -> Vec<f32> {
                values(len, seed)
                    .iter()
                    .map(|v| (v * 8.0).round())
                    .collect()
            };
            for (data_a, data_b, exact) in [
                (small(m * k, 5), small(k * n, 6), true),
                (values(m * k, 7), values(k * n, 8), false),
            ] {
                let a = matrix(&data_a, m, k, ta);
                let b = matrix(&data_b, k, n, tb);
                let mut c = vec![0.0; m * n];
                gemm_mixed(&pool, isa, a, b, &mut c, false);
                let expected = reference(a, b);
                for (i, (&got, &want)) in c.iter().zip(&expected).enumerate() {
                    if exact {
                        assert_eq!(got as f64, want, "{m}×{k}×{n}, element {i}");
                    } else {
                        // Each operand is rounded to within 2⁻⁹: the error is
                        // bounded by 2⁻⁸ × Σ |a| × |b| ≤ 2⁻⁸ × k.
                        assert!(
                            (got as f64 - want).abs() <= k as f64 / 256.0,
                            "{m}×{k}×{n}: {got} versus {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_number_of_threads() {
        // Blocked, with few rows by rows of `B`, with few rows by dot products.
        for (m, k, n, tb) in [
            (300, 700, 200, false),
            (7, 700, 900, false),
            (5, 300, 200, true),
        ] {
            let data_a = values(m * k, 3);
            let data_b = values(k * n, 4);
            let (a, b) = (matrix(&data_a, m, k, false), matrix(&data_b, k, n, tb));
            let run = |threads: usize, isa: Isa, mixed: bool| {
                let mut c = vec![0.0; m * n];
                let pool = Pool::new(threads);
                if mixed {
                    gemm_mixed(&pool, isa, a, b, &mut c, false);
                } else {
                    gemm(&pool, isa, a, b, &mut c, false);
                }
                c
            };
            for isa in isas() {
                for mixed in [false, true] {
                    let one = run(1, isa, mixed);
                    assert_eq!(one, run(2, isa, mixed), "{isa:?} {m}×{k}×{n}");
                    assert_eq!(one, run(7, isa, mixed), "{isa:?} {m}×{k}×{n}");
                }
            }
        }
    }
}
