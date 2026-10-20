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

use crate::Isa;
use crate::kernels::{MicroKernel, Scalar};
use crate::pool::{Pool, SharedMut};

/// A matrix read with arbitrary strides: element `(i, j)` is
/// `data[i × row_stride + j × col_stride]`. Transposing a matrix just swaps
/// the strides.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    /// The underlying elements.
    pub data: &'a [f32],
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Distance in `data` between two consecutive rows.
    pub row_stride: usize,
    /// Distance in `data` between two consecutive columns.
    pub col_stride: usize,
}

impl<'a> Matrix<'a> {
    /// A matrix stored row by row.
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        }
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

/// `C = A · B`, or `C += A · B` with `accumulate`. `C` is stored row by row,
/// `a.rows × b.cols`.
pub fn gemm(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    gemm_with_blocks(pool, isa, a, b, c, accumulate, Blocks::ZEN4);
}

/// Like [`gemm`], with given block sizes: for benchmarking them.
pub fn gemm_with_blocks(
    pool: &Pool,
    isa: Isa,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    assert_eq!(a.cols, b.rows, "gemm: incompatible shapes");
    assert_eq!(c.len(), a.rows * b.cols, "gemm: unexpected output size");
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
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => blocked::<crate::kernels::Avx512>(pool, a, b, c, accumulate, blocks),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => blocked::<crate::kernels::Avx2>(pool, a, b, c, accumulate, blocks),
        _ => blocked::<Scalar>(pool, a, b, c, accumulate, blocks),
    }
}

/// The blocked product, for a given micro-kernel.
fn blocked<K: MicroKernel>(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let (mr, nr) = (K::MR, K::NR);
    let kc = blocks.kc;
    let nc = blocks.nc.div_ceil(nr) * nr;
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
    PACKED_B.with(|packed_b| {
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
                let accumulate = accumulate || pc > 0;

                // The slice of B, packed in panels of NR columns.
                let size = panels * nr * depth;
                if packed_b.len() < size {
                    packed_b.resize(size, 0.0);
                }
                let shared_b = SharedMut::new(&mut packed_b[..size]);
                pool.run(panels, |panel| {
                    // SAFETY: each task packs its own panel of `packed_b`.
                    let target = unsafe { shared_b.slice(panel * nr * depth, nr * depth) };
                    pack_b(
                        b,
                        pc,
                        depth,
                        jc + panel * nr,
                        nr.min(width - panel * nr),
                        nr,
                        target,
                    );
                });
                let packed_b = &packed_b[..size];

                pool.run(row_blocks * col_groups, |task| {
                    let (row_block, group) = (task / col_groups, task % col_groups);
                    let ic = row_block * mc;
                    let height = mc.min(m - ic);
                    let first = group * per_group;
                    let last = panels.min(first + per_group);
                    PACKED_A.with(|packed_a| {
                        let mut packed_a = packed_a.borrow_mut();
                        let row_panels = height.div_ceil(mr);
                        let size = row_panels * mr * depth;
                        if packed_a.len() < size {
                            packed_a.resize(size, 0.0);
                        }
                        for panel in 0..row_panels {
                            let rows = mr.min(height - panel * mr);
                            pack_a(
                                a,
                                ic + panel * mr,
                                rows,
                                pc,
                                depth,
                                mr,
                                &mut packed_a[panel * mr * depth..][..mr * depth],
                            );
                        }
                        for jr in first..last {
                            let col = jc + jr * nr;
                            let cols = nr.min(n - col);
                            let panel_b = packed_b[jr * nr * depth..].as_ptr();
                            for ir in 0..row_panels {
                                let row = ic + ir * mr;
                                let rows = mr.min(m - row);
                                // SAFETY: each task writes its own rows and
                                // columns of C, and the packed panels cover
                                // `depth` indices.
                                unsafe {
                                    let target = out.ptr(row * n + col);
                                    K::run(
                                        depth,
                                        packed_a[ir * mr * depth..].as_ptr(),
                                        panel_b,
                                        target,
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
        Isa::Avx512.available() && is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// `f32` rounded to the nearest `bf16`, ties to even: keeps the 16 most
/// significant bits.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn to_bf16(x: f32) -> u32 {
    let bits = x.to_bits();
    (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
}

/// Like [`gemm`], with the operands of a large product rounded to `bf16` and
/// their products accumulated in `f32`: mixed precision. Products with few
/// rows, which are memory-bound, stay in `f32`, as on the GPU.
pub fn gemm_mixed(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    let limit = if b.col_stride == 1 {
        FEW_ROWS
    } else {
        FEW_ROWS_TRANSPOSED
    };
    if a.rows <= limit || a.cols == 0 || c.is_empty() || isa != Isa::Avx512 || !bf16_available() {
        gemm(pool, isa, a, b, c, accumulate);
        return;
    }
    assert_eq!(a.cols, b.rows, "gemm: incompatible shapes");
    assert_eq!(c.len(), a.rows * b.cols, "gemm: unexpected output size");
    blocked_bf16(pool, a, b, c, accumulate, Blocks::ZEN4);
}

thread_local! {
    /// The packed `bf16` pairs of a block of `A`, private to each thread.
    static PAIRS_A: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// The packed `bf16` pairs of a slice of `B`, private to the thread that
    /// starts the product.
    static PAIRS_B: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// The blocked product of [`blocked`], with pairs of `bf16`: the `kc` slice
/// is counted in indices, and packed by pairs.
#[cfg(target_arch = "x86_64")]
fn blocked_bf16(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    const MR: usize = 14;
    const NR: usize = 32;
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let kc = blocks.kc;
    let nc = blocks.nc.div_ceil(NR) * NR;
    let threads = pool.threads();
    let wanted = TASKS_PER_THREAD * threads;
    let row_panels = m.div_ceil(MR);
    let row_blocks = m.div_ceil(blocks.mc.max(MR)).max(wanted.min(row_panels));
    let mc = row_panels.div_ceil(row_blocks) * MR;
    let row_blocks = m.div_ceil(mc);
    let out = SharedMut::new(c);
    PAIRS_B.with(|packed_b| {
        let mut packed_b = packed_b.borrow_mut();
        for jc in (0..n).step_by(nc) {
            let width = nc.min(n - jc);
            let panels = width.div_ceil(NR);
            let col_groups = wanted.div_ceil(row_blocks).min(panels.div_ceil(2)).max(1);
            let per_group = panels.div_ceil(col_groups);
            let col_groups = panels.div_ceil(per_group);
            for pc in (0..k).step_by(kc) {
                let depth = kc.min(k - pc);
                let pairs = depth.div_ceil(2);
                let accumulate = accumulate || pc > 0;
                let size = panels * NR * pairs;
                if packed_b.len() < size {
                    packed_b.resize(size, 0);
                }
                let shared_b = SharedMut::new(&mut packed_b[..size]);
                pool.run(panels, |panel| {
                    // SAFETY: each task packs its own panel of `packed_b`.
                    let target = unsafe { shared_b.slice(panel * NR * pairs, NR * pairs) };
                    let (col, cols) = (jc + panel * NR, NR.min(width - panel * NR));
                    target.fill(0);
                    for p in 0..pairs {
                        for j in 0..cols {
                            let low = to_bf16(b.at(pc + 2 * p, col + j));
                            let high = if 2 * p + 1 < depth {
                                to_bf16(b.at(pc + 2 * p + 1, col + j))
                            } else {
                                0
                            };
                            target[p * NR + j] = low | (high << 16);
                        }
                    }
                });
                let packed_b = &packed_b[..size];
                pool.run(row_blocks * col_groups, |task| {
                    let (row_block, group) = (task / col_groups, task % col_groups);
                    let ic = row_block * mc;
                    let height = mc.min(m - ic);
                    let first = group * per_group;
                    let last = panels.min(first + per_group);
                    PAIRS_A.with(|packed_a| {
                        let mut packed_a = packed_a.borrow_mut();
                        let panels_a = height.div_ceil(MR);
                        let size = panels_a * MR * pairs;
                        if packed_a.len() < size {
                            packed_a.resize(size, 0);
                        }
                        for panel in 0..panels_a {
                            let target = &mut packed_a[panel * MR * pairs..][..MR * pairs];
                            target.fill(0);
                            let rows = MR.min(height - panel * MR);
                            for p in 0..pairs {
                                for i in 0..rows {
                                    let row = ic + panel * MR + i;
                                    let low = to_bf16(a.at(row, pc + 2 * p));
                                    let high = if 2 * p + 1 < depth {
                                        to_bf16(a.at(row, pc + 2 * p + 1))
                                    } else {
                                        0
                                    };
                                    target[p * MR + i] = low | (high << 16);
                                }
                            }
                        }
                        for jr in first..last {
                            let col = jc + jr * NR;
                            let cols = NR.min(n - col);
                            let panel_b = packed_b[jr * NR * pairs..].as_ptr();
                            for ir in 0..panels_a {
                                let row = ic + ir * MR;
                                let rows = MR.min(m - row);
                                // SAFETY: `bf16_available` checked the
                                // instructions; each task writes its own rows
                                // and columns of C.
                                unsafe {
                                    crate::kernels::bf16_14x32(
                                        pairs,
                                        packed_a[ir * MR * pairs..].as_ptr(),
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

#[cfg(not(target_arch = "x86_64"))]
fn blocked_bf16(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    blocked::<Scalar>(pool, a, b, c, accumulate, blocks)
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
            let a = Matrix {
                data: &a.data[begin * a.col_stride..],
                cols: stop - begin,
                ..a
            };
            let b = Matrix {
                data: &b.data[begin * b.row_stride..],
                rows: stop - begin,
                ..b
            };
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
        blocked::<Scalar>(pool, a, b, c, accumulate, Blocks::ZEN4);
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

fn direct<const W: usize>(isa: Isa, block: Block, rows: usize) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2<const R: usize, const W: usize>(block: Block) {
        direct_body::<R, W, true>(block)
    }

    fn with<const R: usize, const W: usize>(isa: Isa, block: Block) {
        match isa {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `Isa::Avx512` is only chosen when the processor has
            // these instructions; `W == 64` is four 16-lane registers.
            Isa::Avx512 if W == 64 => unsafe { x86::direct_avx512::<R, 4>(block) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above; `W == 256` is sixteen 16-lane registers.
            Isa::Avx512 if W == 256 && R == 1 => unsafe { x86::direct_avx512::<1, 16>(block) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: both ISAs imply AVX2 and FMA support.
            Isa::Avx2 | Isa::Avx512 => unsafe { avx2::<R, W>(block) },
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
            // SAFETY: the caller guarantees that the rows, columns and slice
            // exist; masked-out columns are not read.
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
            // SAFETY: each task writes its own columns of C; masked-out
            // columns are neither read nor written.
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

    fn check(isa: Isa, m: usize, k: usize, n: usize, transpose_a: bool, transpose_b: bool) {
        let pool = Pool::new(4);
        let data_a = values(m * k, 1);
        let data_b = values(k * n, 2);
        let a = if transpose_a {
            Matrix::new(&data_a, k, m).t()
        } else {
            Matrix::new(&data_a, m, k)
        };
        let b = if transpose_b {
            Matrix::new(&data_b, n, k).t()
        } else {
            Matrix::new(&data_b, k, n)
        };
        let expected = reference(a, b);
        let mut c = vec![f32::NAN; m * n];
        let blocks = Blocks {
            kc: 48,
            mc: 40,
            nc: 64,
            split: if (m + k + n).is_multiple_of(2) {
                Split::Rows
            } else {
                Split::Columns
            },
            few_rows: FEW_ROWS,
        };
        gemm_with_blocks(&pool, isa, a, b, &mut c, false, blocks);
        for (index, (&got, &want)) in c.iter().zip(&expected).enumerate() {
            assert!(
                (got as f64 - want).abs() < 1e-4 * (k as f64).sqrt(),
                "{isa:?} {m}×{k}×{n} (tA {transpose_a}, tB {transpose_b}), element {index}: {got} instead of {want}",
            );
        }
        // Accumulating adds the same product a second time.
        gemm_with_blocks(&pool, isa, a, b, &mut c, true, blocks);
        for (&got, &want) in c.iter().zip(&expected) {
            assert!((got as f64 - 2.0 * want).abs() < 2e-4 * (k as f64).sqrt());
        }
    }

    fn isas() -> Vec<Isa> {
        [Isa::Scalar, Isa::Avx2, Isa::Avx512]
            .into_iter()
            .filter(|isa| isa.available())
            .collect()
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
    fn mixed_products_are_exact_on_small_integers_and_close_otherwise() {
        if !bf16_available() {
            return;
        }
        let pool = Pool::new(4);
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
                let a = if ta {
                    Matrix::new(&data_a, k, m).t()
                } else {
                    Matrix::new(&data_a, m, k)
                };
                let b = if tb {
                    Matrix::new(&data_b, n, k).t()
                } else {
                    Matrix::new(&data_b, k, n)
                };
                let mut c = vec![0.0; m * n];
                gemm_mixed(&pool, Isa::Avx512, a, b, &mut c, false);
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
        let (m, k, n) = (300, 700, 200);
        let data_a = values(m * k, 3);
        let data_b = values(k * n, 4);
        let (a, b) = (Matrix::new(&data_a, m, k), Matrix::new(&data_b, k, n));
        let run = |threads: usize, isa: Isa| {
            let mut c = vec![0.0; m * n];
            gemm(&Pool::new(threads), isa, a, b, &mut c, false);
            c
        };
        for isa in isas() {
            let one = run(1, isa);
            assert_eq!(one, run(2, isa));
            assert_eq!(one, run(7, isa));
        }
    }
}
