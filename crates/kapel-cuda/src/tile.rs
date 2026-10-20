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
use cutile::tile_kernel::TileKernel;
use kapel_core::{DType, Error, Layout, Result, Tensor};
use kapel_ops::{Matmul, Operation};

use crate::device::{Cuda, CudaStorage};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// `z[t] = x[t] · y[t]` for each matrix `t` of a batch, with an `f32`
    /// accumulator. Each tile block computes one `BM × BN` tile of the output,
    /// adding the products along `K` in order. Loads past the edges read
    /// zeros, and stores past them are dropped, so no size needs to divide
    /// the tile.
    #[cutile::entry()]
    fn batched_matmul<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<E, { [1, BM, BN] }>,
        x: &Tensor<E, { [-1, -1, -1] }>,
        y: &Tensor<E, { [-1, -1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, BM, BK]);
        let part_y = y.partition(shape![1, BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..num_tiles(&part_x, 2) {
            let tx: Tile<E, { [BM, BK] }> = part_x.load([pid.0, pid.1, i]).reshape(shape![BM, BK]);
            let ty: Tile<E, { [BK, BN] }> = part_y.load([pid.0, i, pid.2]).reshape(shape![BK, BN]);
            acc = mma(tx, ty, acc);
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

/// The tile sizes for an `m × n` output of `dtype`. Larger tiles reuse more of
/// what they load, but leave blocks idle on small outputs.
fn tiles_for(dtype: DType, m: usize) -> [usize; 3] {
    match dtype {
        DType::F32 => [64, 64, 32],
        _ if m >= 128 => [128, 64, 64],
        _ => [64, 64, 64],
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
    pub(crate) fn matmul(&self, a: &Tensor, b: &Tensor, shape: &[usize]) -> Result<Tensor> {
        let op = Matmul::NAME;
        let dtype = a.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::UnsupportedDType { op, dtype });
        }
        let rank = shape.len();
        let (m, n, k) = (shape[rank - 2], shape[rank - 1], a.shape()[a.rank() - 1]);
        let layout = Layout::contiguous(shape)?;
        let numel = layout.numel();
        let storage = self.alloc(numel * dtype.block_bytes(), k == 0)?;
        if numel > 0 && k > 0 {
            let batch = &shape[..rank - 2];
            let (x, _keep_a) = self.matrices(op, a, batch)?;
            let (y, _keep_b) = self.matrices(op, b, batch)?;
            let z = Matrices {
                pointer: storage.pointer(),
                shape: [numel / (m * n), m, n],
                strides: [m * n, n, 1],
            };
            let tiles = tiles_for(dtype, m);
            match dtype {
                DType::F32 => self.batched_matmul::<f32>("f32", &z, &x, &y, tiles)?,
                DType::BF16 => self.batched_matmul::<bf16>("bf16", &z, &x, &y, tiles)?,
                _ => self.batched_matmul::<f16>("f16", &z, &x, &y, tiles)?,
            }
        }
        Tensor::from_storage(Arc::new(storage), layout, dtype)
    }

    /// The backend's stream, as the tile runtime sees it.
    pub(crate) fn tile_stream(&self) -> &Arc<Stream> {
        &self.tiles().stream
    }

    fn batched_matmul<T: cutile::DType>(
        &self,
        name: &str,
        z: &Matrices,
        x: &Matrices,
        y: &Matrices,
        tiles: [usize; 3],
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
        let launch = kernels::batched_matmul(out, Arc::new(tensor(x)?), Arc::new(tensor(y)?))
            .generics(generics);
        // SAFETY: as above; the outputs returned are views of borrowed memory,
        // dropped without freeing it.
        unsafe { launch.async_on(self.tile_stream()) }
            .map(drop)
            .map_err(|e| Error::Device {
                op,
                message: e.to_string(),
            })
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
