//! The dense operations, as cutile-rs tile kernels.
//!
//! A tile kernel is compiled by `tileiras` at its first launch for a given
//! specialization (element type, tile sizes, alignments), then kept in memory
//! and in a cache on disk (`~/.cache/cutile/kernels`). Tile kernels read the
//! backend's buffers in place, through their strides, and run on its stream.

use std::sync::{Arc, Once};

use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::{Device as Gpu, Stream};
use cutile::half::{bf16, f16};
use cutile::tensor::{IntoPartition, Tensor as TileTensor};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use kapel_core::{DType, Error, Layout, Result, Tensor};
use kapel_ops::{Matmul, Operation};

use crate::device::{Args, Cuda, CudaStorage, float_code, small};

/// The most rows of a product [`Cuda::small_product`] computes.
pub(crate) const SMALL_ROWS: usize = 8;

/// The most columns of a product [`Cuda::small_product`] computes: the
/// small projections of a decoder (a router, the gates of linear attention),
/// not its large ones, which the tiles compute well.
pub(crate) const SMALL_COLUMNS: usize = 512;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// `z[t] = bias + x[t] · y[t]` for each matrix `t` of a batch, the bias
    /// `[1, N]` added to every row, with an `f32` accumulator. Each tile block computes one `BM × BN` tile of the output,
    /// adding the products along `K` in order. Loads past the edges read
    /// zeros, and stores past them are dropped, so no size needs to divide
    /// the tile.
    #[cutile::entry()]
    fn batched_matmul<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<E, { [1, BM, BN] }>,
        x: &Tensor<E, { [-1, -1, -1] }>,
        y: &Tensor<E, { [-1, -1, -1] }>,
        bias: &Tensor<E, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, BM, BK]);
        let part_y = y.partition(shape![1, BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        // The accumulator starts from the bias, a row repeated (zeros
        // without one).
        let row: Tile<E, { [1, BN] }> = bias.partition(shape![1, BN]).load([0i32, pid.2]);
        let row: Tile<f32, { [1, BN] }> = convert_tile(row);
        let mut acc: Tile<f32, { [BM, BN] }> = row.broadcast(shape![BM, BN]);
        for i in 0i32..num_tiles(&part_x, 2) {
            let tx: Tile<E, { [BM, BK] }> = part_x.load([pid.0, pid.1, i]).reshape(shape![BM, BK]);
            let ty: Tile<E, { [BK, BN] }> = part_y.load([pid.0, i, pid.2]).reshape(shape![BK, BN]);
            acc = mma(tx, ty, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out.reshape(shape![1, BM, BN]));
    }

    /// `z[t] = x[t] · y[t]ᵀ`, `y[t]` given as `[N, K]`: the weights of a
    /// linear layer as stored. Each tile of `y` is loaded along `K`, its
    /// contiguous dimension, then transposed in registers: reading it through
    /// swapped strides is several times slower.
    #[cutile::entry()]
    fn batched_matmul_nt<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<E, { [1, BM, BN] }>,
        x: &Tensor<E, { [-1, -1, -1] }>,
        y: &Tensor<E, { [-1, -1, -1] }>,
        bias: &Tensor<E, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, BM, BK]);
        let part_y = y.partition(shape![1, BN, BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        // The accumulator starts from the bias, a row repeated (zeros
        // without one).
        let row: Tile<E, { [1, BN] }> = bias.partition(shape![1, BN]).load([0i32, pid.2]);
        let row: Tile<f32, { [1, BN] }> = convert_tile(row);
        let mut acc: Tile<f32, { [BM, BN] }> = row.broadcast(shape![BM, BN]);
        for i in 0i32..num_tiles(&part_x, 2) {
            let tx: Tile<E, { [BM, BK] }> = part_x.load([pid.0, pid.1, i]).reshape(shape![BM, BK]);
            let ty: Tile<E, { [BN, BK] }> = part_y.load([pid.0, pid.2, i]).reshape(shape![BN, BK]);
            acc = mma(tx, ty.transpose(), acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out.reshape(shape![1, BM, BN]));
    }

    /// `z[t] = x[t]ᵀ · y[t]`, `x[t]` given as `[K, M]`: the gradient of a
    /// weight, `xᵀ · g`. Each tile of `x` is loaded along `M`, its contiguous
    /// dimension, then transposed in registers.
    #[cutile::entry()]
    fn batched_matmul_tn<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<E, { [1, BM, BN] }>,
        x: &Tensor<E, { [-1, -1, -1] }>,
        y: &Tensor<E, { [-1, -1, -1] }>,
        bias: &Tensor<E, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, BK, BM]);
        let part_y = y.partition(shape![1, BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        // The accumulator starts from the bias, a row repeated (zeros
        // without one).
        let row: Tile<E, { [1, BN] }> = bias.partition(shape![1, BN]).load([0i32, pid.2]);
        let row: Tile<f32, { [1, BN] }> = convert_tile(row);
        let mut acc: Tile<f32, { [BM, BN] }> = row.broadcast(shape![BM, BN]);
        for i in 0i32..num_tiles(&part_x, 1) {
            let tx: Tile<E, { [BK, BM] }> = part_x.load([pid.0, i, pid.1]).reshape(shape![BK, BM]);
            let ty: Tile<E, { [BK, BN] }> = part_y.load([pid.0, i, pid.2]).reshape(shape![BK, BN]);
            acc = mma(tx.transpose(), ty, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out.reshape(shape![1, BM, BN]));
    }
}

/// The tile runtime's handles on the backend's device and stream.
pub(crate) struct Tiles {
    // The stream is borrowed from the backend, which outlives it.
    stream: Arc<Stream>,
    _gpu: Arc<Gpu>,
}

impl Tiles {
    /// Handles on device `ordinal` and on the backend's stream `cu_stream`,
    /// and the cache on disk, opened once per process.
    pub(crate) fn new(ordinal: u32, cu_stream: *mut std::ffi::c_void) -> Result<Self> {
        static CACHE: Once = Once::new();
        CACHE.call_once(|| {
            // Without a cache directory, kernels are compiled at each start.
            if let Ok(store) = cutile::jit_cache::FileSystemJitStore::default_location() {
                cutile::jit_cache::enable(Arc::new(store));
            }
        });
        let gpu = Gpu::new(ordinal as usize).map_err(|e| Error::Device {
            op: "Cuda::new",
            message: e.to_string(),
        })?;
        // SAFETY: `cu_stream` is the backend's stream, in the primary context
        // of the same device; the backend keeps it alive longer than `Tiles`,
        // and a borrowed stream is never destroyed through this handle.
        let stream = unsafe { Stream::borrow_raw(cu_stream, &gpu) };
        Ok(Self { stream, _gpu: gpu })
    }
}

/// The second operand of a product: as it is (`Nn`), or transposed (`Nt`).
#[derive(Clone, Copy)]
enum Kernel {
    Nn,
    Nt,
    Tn,
}

/// A batch of matrices in device memory: `[batch, rows, columns]` with any
/// strides, the batch stride possibly 0 (broadcast).
struct Matrices {
    pointer: u64,
    shape: [usize; 3],
    strides: [usize; 3],
}

/// The one stride that walks the batch dimensions `shape` in row-major order,
/// if there is one: dimensions of size 1 do not count, and each must step
/// over the whole of the next.
fn batch_stride(shape: &[usize], strides: &[usize]) -> Option<usize> {
    let dims: Vec<(usize, usize)> = shape
        .iter()
        .zip(strides)
        .filter(|(n, _)| **n > 1)
        .map(|(&n, &s)| (n, s))
        .collect();
    for pair in dims.windows(2) {
        let ((_, outer), (n, inner)) = (pair[0], pair[1]);
        if outer != inner * n {
            return None;
        }
    }
    Some(dims.last().map_or(0, |&(_, s)| s))
}

/// The tile sizes for an `m × n` output of `dtype`, and the tile blocks asked
/// for per multiprocessor: the best of sweeps on an RTX 3090, whose neighbours
/// can be 3 times slower. Larger tiles reuse more of what they load, but leave
/// multiprocessors idle on small outputs.
///
/// With few rows, as in decoding, the product reads its weights once and is
/// bound by memory: thin tiles of 16 rows, long along `K`, keep every
/// multiprocessor reading. One row times the weights of Llama 3.2 3B reads
/// them at 650 to 880 GB/s, against 100 with the tiles of large products.
/// With many rows, as in a prefill, it is bound by the tensor cores: tiles of
/// `64 × 128 × 32`, two per multiprocessor, reach 60 to 70 Top/s on the
/// shapes of Llama 3.2 3B from 1024 rows on.
fn tiles_for(dtype: DType, m: usize, n: usize) -> ([usize; 3], Option<i32>) {
    match dtype {
        DType::F32 => ([64, 64, 32], None),
        _ if m <= 16 && n <= 1024 => ([16, 32, 256], Some(4)),
        _ if m <= 16 => ([16, 64, 128], None),
        _ if m <= 32 => ([32, 64, 128], Some(2)),
        _ if m <= 256 => ([64, 64, 64], Some(2)),
        _ => ([64, 128, 32], Some(2)),
    }
}

fn i32s(op: &'static str, values: &[usize]) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&v| i32::try_from(v).map_err(|_| Error::Overflow { op }))
        .collect()
}

impl Cuda {
    /// A tensor `[..., rows, columns]` as a batch of `batch` matrices read in
    /// place, its batch dimensions broadcast to `batch`. When they cannot be
    /// walked with one stride, the matrices are first copied (`f32` only).
    fn matrices(
        &self,
        op: &'static str,
        x: &Tensor,
        batch: &[usize],
    ) -> Result<(Matrices, Tensor)> {
        let r = x.rank();
        let mut full = batch.to_vec();
        full.extend_from_slice(&x.shape()[r - 2..]);
        let layout = x.layout().broadcast_to(&full)?;
        let lead = batch.len();
        let (x, layout) = match batch_stride(batch, &layout.strides()[..lead]) {
            Some(_) => (x.clone(), layout),
            None => {
                let copy = self.materialize(op, x, &layout)?;
                let layout = copy.layout().clone();
                (copy, layout)
            }
        };
        let stride = match batch_stride(batch, &layout.strides()[..lead]) {
            // With one matrix, the batch stride is never followed: one past
            // the matrix keeps the alignments the compiler sees.
            _ if batch.iter().product::<usize>() == 1 => full[lead] * full[lead + 1],
            Some(stride) => stride,
            None => unreachable!("copied above"),
        };
        let size = x.dtype().block_bytes();
        let storage: &CudaStorage = self.storage(op, &x)?;
        let s = layout.strides();
        let matrices = Matrices {
            pointer: storage.pointer() + (layout.offset() * size) as u64,
            shape: [batch.iter().product(), full[lead], full[lead + 1]],
            strides: [stride, s[lead], s[lead + 1]],
        };
        Ok((matrices, x))
    }

    /// The batched product of two tensors of this backend, of shape `shape`
    /// (from `infer`): `f32`, `bf16` or `f16`, accumulated in `f32`.
    pub(crate) fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        shape: &[usize],
    ) -> Result<Tensor> {
        let op = Matmul::NAME;
        let dtype = a.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::UnsupportedDType { op, dtype });
        }
        let rank = shape.len();
        let (m, n, k) = (shape[rank - 2], shape[rank - 1], a.shape()[a.rank() - 1]);
        if let Some(y) = self.small_product(a, b, bias, shape)? {
            return Ok(y);
        }
        let layout = Layout::contiguous(shape)?;
        let numel = layout.numel();
        let storage = self.alloc(numel * dtype.block_bytes(), k == 0)?;
        if numel > 0 && k > 0 {
            let batch = &shape[..rank - 2];
            // `a` read as the transpose of row-major `[K, M]` matrices (`xᵀ ·
            // g`, the gradient of a weight): loaded along `M`. Through its
            // swapped strides, it was three to four times slower.
            let ra = a.rank();
            let a_transposed = a.layout().strides()[ra - 2] == 1 && m > 1 && k > 1;
            let (x, _keep_a) = if a_transposed {
                self.matrices(op, &a.transpose(ra - 2, ra - 1)?, batch)?
            } else {
                self.matrices(op, a, batch)?
            };
            // `b` read as the transpose of row-major `[N, K]` matrices (the
            // weights of a linear layer): loaded along `K`.
            let rb = b.rank();
            // With `a` transposed too, `b` is read through its strides:
            // there is no kernel for both.
            let transposed = b.layout().strides()[rb - 2] == 1 && n > 1 && !a_transposed;
            let (y, _keep_b) = if transposed {
                self.matrices(op, &b.transpose(rb - 2, rb - 1)?, batch)?
            } else {
                self.matrices(op, b, batch)?
            };
            let z = Matrices {
                pointer: storage.pointer(),
                shape: [numel / (m * n), m, n],
                strides: [m * n, n, 1],
            };
            // The bias as one row `[1, N]`, row-major; zeros without one.
            let zeros;
            let bias = match bias {
                Some(b) if b.is_contiguous() => b.clone(),
                Some(b) => self.contiguous_any(op, b)?,
                None => {
                    zeros = self.alloc(n * dtype.block_bytes(), true)?;
                    Tensor::from_storage(Arc::new(zeros), Layout::contiguous(&[n])?, dtype)?
                }
            };
            let storage = self.storage(op, &bias)?;
            let row = Matrices {
                pointer: storage.pointer() + (bias.layout().offset() * dtype.block_bytes()) as u64,
                shape: [1, 1, n],
                strides: [n, n, 1],
            };
            let launch = tiles_for(dtype, m, n);
            let kernel = match (a_transposed, transposed) {
                (false, false) => Kernel::Nn,
                (false, true) => Kernel::Nt,
                (true, _) => Kernel::Tn,
            };
            match dtype {
                DType::F32 => {
                    self.batched_matmul::<f32>("f32", kernel, &z, &x, &y, &row, launch)?
                }
                DType::BF16 => {
                    self.batched_matmul::<bf16>("bf16", kernel, &z, &x, &y, &row, launch)?
                }
                _ => self.batched_matmul::<f16>("f16", kernel, &z, &x, &y, &row, launch)?,
            }
        }
        Tensor::from_storage(Arc::new(storage), layout, dtype)
    }

    /// `a · b` by `gemv_float` when it is a small product of decoding: at
    /// most [`SMALL_ROWS`] rows of `a` by the transpose of a row-major weight
    /// `[N, K]` with at most [`SMALL_COLUMNS`] rows, without a bias, which
    /// the tiles compute with a block or two; `None` otherwise. Each output
    /// is one warp's, whatever the number of rows: the bits of a row do not
    /// depend on the others.
    fn small_product(
        &self,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        shape: &[usize],
    ) -> Result<Option<Tensor>> {
        let op = Matmul::NAME;
        let (ra, rank) = (a.rank(), shape.len());
        let (n, k) = (shape[rank - 1], a.shape()[ra - 1]);
        let rows: usize = shape[..rank - 1].iter().product();
        let weight_like = b.rank() == 2 && b.layout().strides() == [1, k] && b.shape() == [k, n];
        if bias.is_some()
            || !weight_like
            || rows == 0
            || rows > SMALL_ROWS
            || n == 0
            || n > SMALL_COLUMNS
            || k == 0
            || a.shape()[..ra - 1].iter().product::<usize>() != rows
        {
            return Ok(None);
        }
        let dtype = a.dtype();
        let x = if a.is_contiguous() && a.layout().offset() == 0 {
            a.clone()
        } else {
            self.materialize(op, a, a.layout())?
        };
        let xa = self.float_arg(op, &x, x.layout())?;
        // The weights row-major, from their first element.
        let w = b.transpose(0, 1)?;
        if !w.is_contiguous() {
            return Ok(None);
        }
        let wa = self.float_arg(op, &w, w.layout())?;
        let w_size = w.dtype().block_bytes();
        let w_start = wa.base + (w.layout().offset() * w_size) as u64;
        let w_bytes = n * k * w_size;
        let y = self.output_as(shape, dtype, false, |out, count| {
            self.launch(
                "gemv_float",
                rows * n * 256,
                Args::new()
                    .u32(wa.code)
                    .slice(w_start, w_bytes)
                    .u32(xa.code)
                    .slice(xa.base, xa.bytes)
                    .u32(small(op, rows)?)
                    .u32(small(op, k)?)
                    .u32(small(op, n)?)
                    .u32(float_code(op, dtype)?)
                    .slice(out, dtype.storage_bytes(count)?),
            )
        })?;
        Ok(Some(y))
    }

    /// The backend's stream, as the tile runtime sees it.
    pub(crate) fn tile_stream(&self) -> &Arc<Stream> {
        &self.tiles().stream
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_matmul<T: cutile::DType>(
        &self,
        name: &str,
        kernel: Kernel,
        z: &Matrices,
        x: &Matrices,
        y: &Matrices,
        bias: &Matrices,
        (tiles, occupancy): ([usize; 3], Option<i32>),
    ) -> Result<()> {
        let op = Matmul::NAME;
        let ordinal = self.ordinal() as usize;
        let tensor = |m: &Matrices| -> Result<TileTensor<T>> {
            let (shape, strides) = (i32s(op, &m.shape)?, i32s(op, &m.strides)?);
            // SAFETY: the pointer, shape and strides describe memory of a
            // storage of this backend that the caller keeps alive; the kernel
            // is queued on the backend's stream, so the storage, given back to
            // the pool afterwards, is only reused by work queued after it.
            // Only `z`, a fresh output, is written.
            Ok(unsafe { TileTensor::<T>::borrow_raw_parts(m.pointer, ordinal, shape, strides) })
        };
        let [bm, bn, bk] = tiles;
        let out = tensor(z)?.partition([1, bm, bn]);
        let generics = vec![
            name.to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
        ];
        let options = match occupancy {
            Some(blocks) => CompileOptions::new().occupancy(blocks),
            None => CompileOptions::new(),
        };
        let (x, y) = (Arc::new(tensor(x)?), Arc::new(tensor(y)?));
        let row = |m: &Matrices| -> Result<TileTensor<T>> {
            let (shape, strides) = (i32s(op, &m.shape[1..])?, i32s(op, &m.strides[1..])?);
            // SAFETY: as for `tensor`: a row of a storage the caller keeps
            // alive, only read.
            Ok(unsafe { TileTensor::<T>::borrow_raw_parts(m.pointer, ordinal, shape, strides) })
        };
        let bias = Arc::new(row(bias)?);
        let error = |e: cutile::prelude::DeviceError| Error::Device {
            op,
            message: e.to_string(),
        };
        match kernel {
            // SAFETY: as above; the outputs returned are views of borrowed
            // memory, dropped without freeing it.
            Kernel::Nn => unsafe {
                kernels::batched_matmul(out, x, y, bias)
                    .generics(generics)
                    .compile_options(options)
                    .async_on(self.tile_stream())
            }
            .map(drop)
            .map_err(error),
            // SAFETY: as for `Kernel::Nn`.
            Kernel::Nt => unsafe {
                kernels::batched_matmul_nt(out, x, y, bias)
                    .generics(generics)
                    .compile_options(options)
                    .async_on(self.tile_stream())
            }
            .map(drop)
            .map_err(error),
            // SAFETY: as for `Kernel::Nn`.
            Kernel::Tn => unsafe {
                kernels::batched_matmul_tn(out, x, y, bias)
                    .generics(generics)
                    .compile_options(options)
                    .async_on(self.tile_stream())
            }
            .map(drop)
            .map_err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_dimensions_collapse_when_they_can() {
        assert_eq!(batch_stride(&[2, 3], &[60, 20]), Some(20));
        assert_eq!(batch_stride(&[2, 3], &[0, 0]), Some(0));
        assert_eq!(batch_stride(&[2, 1, 3], &[60, 7, 20]), Some(20));
        assert_eq!(batch_stride(&[], &[]), Some(0));
        assert_eq!(batch_stride(&[2, 3], &[20, 0]), None);
    }
}
