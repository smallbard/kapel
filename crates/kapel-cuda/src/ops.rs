//! The catalogue of operations on the device.
//!
//! Products and attention compute in their inputs' type (see `tile.rs` and
//! `attention.rs`). The other operations compute in `f32`: inputs in `bf16`
//! or `f16` are widened to `f32` on the device, and results rounded back to
//! their type ([`InF32`]). Copies and row gathers move the bits of any type.
//!
//! The host prepares what depends only on shapes (the offsets of reductions,
//! the lines of a softmax, the rows of a scatter, the angles of rotary
//! embeddings) and the kernels do the arithmetic. Quantized weights, and
//! conversions to or from types other than `f32`, `bf16` and `f16`, go
//! through the host for now.

use std::sync::Arc;

use kapel_core::{DType, Error, HostStorage, Layout, Result, Tensor};
use kapel_cuda_abi::{binary, unary};
use kapel_ops::{
    Affine, ArgMax, Attention, Binary, BinaryKind, Cast, Concat, Contiguous, CrossEntropy,
    GatherRows, Glu, LayerNorm, LogSoftmax, Matmul, Max, Mean, Operation, QuantizedMatmul, RmsNorm,
    Rope, RopeLayout, Run, ScatterAddRows, Softmax, Sum, Transfer, Unary, UnaryKind,
};

use crate::device::{
    Args, Cuda, CudaStorage, Flat, FloatArg, Table, float_code, small, smalls, view_of,
};

/// The code of `f32` for the kernels that read any float type.
const F32_CODE: u32 = kapel_cuda_abi::float::F32;

/// The terms a thread adds in one stage of a long sum.
const SEGMENT: usize = 128;

fn unary_code(op: &'static str, kind: UnaryKind) -> Result<u32> {
    Ok(match kind {
        UnaryKind::Neg => unary::NEG,
        UnaryKind::Exp => unary::EXP,
        UnaryKind::Log => unary::LOG,
        UnaryKind::Sqrt => unary::SQRT,
        UnaryKind::Rsqrt => unary::RSQRT,
        UnaryKind::Square => unary::SQUARE,
        UnaryKind::Abs => unary::ABS,
        UnaryKind::Sigmoid => unary::SIGMOID,
        UnaryKind::Tanh => unary::TANH,
        UnaryKind::Relu => unary::RELU,
        UnaryKind::Silu => unary::SILU,
        UnaryKind::GeluTanh => unary::GELU_TANH,
        UnaryKind::GeluErf => unary::GELU_ERF,
        UnaryKind::Softplus => unary::SOFTPLUS,
        kind => {
            return Err(Error::InvalidArgument {
                op,
                message: format!("{kind:?} is not implemented on the device"),
            });
        }
    })
}

fn binary_code(op: &'static str, kind: BinaryKind) -> Result<u32> {
    Ok(match kind {
        BinaryKind::Add => binary::ADD,
        BinaryKind::Sub => binary::SUB,
        BinaryKind::Mul => binary::MUL,
        BinaryKind::Div => binary::DIV,
        BinaryKind::Max => binary::MAX,
        BinaryKind::Min => binary::MIN,
        kind => {
            return Err(Error::InvalidArgument {
                op,
                message: format!("{kind:?} is not implemented on the device"),
            });
        }
    })
}

/// Row-major strides.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Every combination of the coordinates of `dims`, as offsets in `shape`.
fn offsets_of(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let mut offsets = vec![0];
    for &d in dims {
        let (size, stride) = (shape[d], s[d]);
        offsets = offsets
            .iter()
            .flat_map(|&o| (0..size).map(move |i| o + i * stride))
            .collect();
    }
    offsets
}

/// For reducing `dims` of `shape`: the offset of the first element of each
/// result, in row-major order, and the offsets of a result's elements from it.
fn reduction(shape: &[usize], dims: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let kept: Vec<usize> = (0..shape.len()).filter(|d| !dims.contains(d)).collect();
    (offsets_of(shape, &kept), offsets_of(shape, dims))
}

/// The first element of each line of `shape` along `dim`; the elements of a
/// line are `shape[dim + 1..].product()` apart.
fn line_starts(shape: &[usize], dim: usize) -> Vec<usize> {
    let len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    (0..outer * inner)
        .map(|l| (l / inner) * len * inner + l % inner)
        .collect()
}

impl Cuda {
    /// `scale ·` the sum of `x` (row-major, of `shape`) over `dims`, as a
    /// tensor of `out_shape`, which has one element per result.
    fn reduce(
        &self,
        op: &'static str,
        x: &Flat,
        shape: &[usize],
        dims: &[usize],
        scale: f32,
        out_shape: &[usize],
    ) -> Result<Tensor> {
        self.reduce_raw(
            op,
            F32_CODE,
            x.ptr,
            x.len * 4,
            shape,
            dims,
            scale,
            out_shape,
        )
    }

    /// [`Cuda::reduce`] of the `bytes` bytes at `ptr`, values of float type
    /// `kx` (a code of `kapel_cuda_abi::float`).
    ///
    /// A result of more than [`SEGMENT`] terms is summed in stages, for speed
    /// and precision: segments of `SEGMENT` terms each, all at once
    /// (`reduce_sum_segments`), then the segments of each result, the same
    /// way, until a stage has `SEGMENT` terms or fewer (`reduce_sum`). The
    /// order of the additions is fixed: the same bits on every run.
    #[allow(clippy::too_many_arguments)]
    fn reduce_raw(
        &self,
        op: &'static str,
        kx: u32,
        ptr: u64,
        len: usize,
        shape: &[usize],
        dims: &[usize],
        scale: f32,
        out_shape: &[usize],
    ) -> Result<Tensor> {
        // The last dimensions of a row-major tensor: each result's terms
        // follow each other, and no table of offsets is needed (that of a
        // whole tensor would be as large as the tensor).
        let rank = shape.len();
        let suffix = !dims.is_empty() && {
            let mut sorted = dims.to_vec();
            sorted.sort_unstable();
            sorted
                .iter()
                .enumerate()
                .all(|(i, &d)| d == rank - sorted.len() + i)
        };
        if suffix {
            let terms: usize = dims.iter().map(|&d| shape[d]).product();
            let results = shape.iter().product::<usize>() / terms.max(1);
            return self.reduce_rows(op, kx, ptr, len, results, terms, scale, out_shape);
        }
        // The first dimensions: sums of the columns of a matrix (the
        // gradient of a bias), without a table either.
        let prefix = !dims.is_empty() && {
            let mut sorted = dims.to_vec();
            sorted.sort_unstable();
            sorted.iter().enumerate().all(|(i, &d)| d == i)
        };
        if prefix && !suffix {
            let rows: usize = dims.iter().map(|&d| shape[d]).product();
            let cols = shape.iter().product::<usize>() / rows.max(1);
            return self.reduce_columns(op, kx, ptr, len, rows, cols, scale, out_shape);
        }
        let (b, i) = self.reduction_tables(op, shape, dims)?;
        let (results, terms) = (b.len, i.len);
        if terms <= SEGMENT || results == 0 {
            return self.output(out_shape, false, |out, n| {
                self.launch(
                    "reduce_sum",
                    n,
                    Args::new()
                        .u32(kx)
                        .slice(ptr, len)
                        .table(&b)
                        .table(&i)
                        .f32(scale)
                        .slice(out, n),
                )
            });
        }
        let segments = terms.div_ceil(SEGMENT);
        let parts = self.output(&[segments, results], false, |out, n| {
            self.launch(
                "reduce_sum_segments",
                n,
                Args::new()
                    .u32(kx)
                    .slice(ptr, len)
                    .table(&b)
                    .table(&i)
                    .u32(small(op, SEGMENT)?)
                    .slice(out, n),
            )
        })?;
        // The segments of each result, summed the same way: as many stages
        // as it takes to come down to `SEGMENT` terms.
        let parts = self.flat(op, &parts)?;
        self.reduce_raw(
            op,
            F32_CODE,
            parts.ptr,
            parts.len * 4,
            &[segments, results],
            &[0],
            scale,
            out_shape,
        )
    }

    /// `scale ·` the sums of the columns of a row-major `[rows, cols]` matrix
    /// at `ptr`, in stages of [`SEGMENT`] rows beyond that.
    #[allow(clippy::too_many_arguments)]
    fn reduce_columns(
        &self,
        op: &'static str,
        kx: u32,
        ptr: u64,
        len: usize,
        rows: usize,
        cols: usize,
        scale: f32,
        out_shape: &[usize],
    ) -> Result<Tensor> {
        if rows == 0 || cols == 0 {
            return self.output(out_shape, true, |_, _| Ok(()));
        }
        let launch = |shape: &[usize], seg: usize, scale: f32| {
            self.output(shape, false, |out, n| {
                self.launch(
                    "reduce_sum_columns",
                    n,
                    Args::new()
                        .u32(kx)
                        .slice(ptr, len)
                        .u32(small(op, rows)?)
                        .u32(small(op, cols)?)
                        .u32(small(op, seg)?)
                        .f32(scale)
                        .slice(out, n),
                )
            })
        };
        if rows <= SEGMENT {
            return launch(out_shape, rows, scale);
        }
        let parts = rows.div_ceil(SEGMENT);
        let partial = launch(&[parts, cols], SEGMENT, 1.0)?;
        let partial = self.flat(op, &partial)?;
        self.reduce_columns(
            op,
            F32_CODE,
            partial.ptr,
            partial.len * 4,
            parts,
            cols,
            scale,
            out_shape,
        )
    }

    /// `scale ·` the sums of `results` runs of `terms` consecutive values at
    /// `ptr`, in stages of [`SEGMENT`] terms beyond that.
    #[allow(clippy::too_many_arguments)]
    fn reduce_rows(
        &self,
        op: &'static str,
        kx: u32,
        ptr: u64,
        len: usize,
        results: usize,
        terms: usize,
        scale: f32,
        out_shape: &[usize],
    ) -> Result<Tensor> {
        if terms == 0 {
            return self.output(out_shape, true, |_, _| Ok(()));
        }
        let launch = |shape: &[usize], seg: usize, scale: f32| {
            self.output(shape, false, |out, n| {
                self.launch(
                    "reduce_sum_rows",
                    n,
                    Args::new()
                        .u32(kx)
                        .slice(ptr, len)
                        .u32(small(op, terms)?)
                        .u32(small(op, seg)?)
                        .f32(scale)
                        .slice(out, n),
                )
            })
        };
        if terms <= SEGMENT {
            return launch(out_shape, terms, scale);
        }
        let parts = terms.div_ceil(SEGMENT);
        let partial = launch(&[results, parts], SEGMENT, 1.0)?;
        let partial = self.flat(op, &partial)?;
        self.reduce_rows(
            op,
            F32_CODE,
            partial.ptr,
            partial.len * 4,
            results,
            parts,
            scale,
            out_shape,
        )
    }

    /// The key of a table computed from `shape` and `dims`.
    fn key(shape: &[usize], dims: &[usize]) -> Vec<usize> {
        let mut key = shape.to_vec();
        key.push(usize::MAX);
        key.extend_from_slice(dims);
        key
    }

    /// The tables of `reduction(shape, dims)`, on the device.
    fn reduction_tables(
        &self,
        op: &'static str,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<(Arc<Table>, Arc<Table>)> {
        let key = Self::key(shape, dims);
        let bases = self.table("reduction bases", &key, || {
            smalls(op, &reduction(shape, dims).0)
        })?;
        let inner = self.table("reduction offsets", &key, || {
            smalls(op, &offsets_of(shape, dims))
        })?;
        Ok((bases, inner))
    }

    /// The gradient `scale · g` of each result given to every element it
    /// reduced: the backward pass of a sum over `dims` of a tensor of `shape`.
    fn spread(
        &self,
        op: &'static str,
        g: &Tensor,
        shape: &[usize],
        dims: &[usize],
        scale: f32,
    ) -> Result<Tensor> {
        let g = self.flat(op, g)?;
        let owner = self.table("owners", &Self::key(shape, dims), || {
            let (bases, inner) = reduction(shape, dims);
            let mut owner = vec![0u32; shape.iter().product()];
            for (r, &base) in bases.iter().enumerate() {
                for &o in &inner {
                    owner[base + o] = small(op, r)?;
                }
            }
            Ok(owner)
        })?;
        self.output(shape, false, |out, n| {
            self.launch(
                "spread",
                n,
                Args::new().flat(&g).table(&owner).f32(scale).slice(out, n),
            )
        })
    }

    /// A gradient computed at a broadcast shape, summed back to `target`: over
    /// the leading dimensions `target` lacks, and those where it has size 1.
    fn sum_to(&self, op: &'static str, full: &Tensor, target: &[usize]) -> Result<Tensor> {
        let shape = full.shape().to_vec();
        let lead = shape.len() - target.len();
        let dims: Vec<usize> = (0..shape.len())
            .filter(|&i| i < lead || (target[i - lead] == 1 && shape[i] != 1))
            .collect();
        if dims.is_empty() {
            return self.contiguous(op, full)?.reshape(target);
        }
        let x = self.flat(op, full)?;
        self.reduce(op, &x, &shape, &dims, 1.0, target)
    }

    /// Index values (`u32` or `i32`), read back to the host and each checked
    /// against `size`.
    pub(crate) fn indices(
        &self,
        op: &'static str,
        tensor: &Tensor,
        size: usize,
    ) -> Result<Vec<usize>> {
        let host = match self.storage(op, tensor)?.host() {
            // Uploaded from the host, and kept there: no need to wait.
            Some(bytes) => Tensor::from_storage(
                Arc::new(HostStorage::from_bytes(bytes)),
                tensor.layout().clone(),
                tensor.dtype(),
            )?,
            None => self.download(tensor)?,
        };
        host.to_f64_vec()?
            .into_iter()
            .map(|v| {
                let index = v as i64;
                usize::try_from(index)
                    .ok()
                    .filter(|&i| i < size)
                    .ok_or(Error::IndexOutOfRange { op, index, size })
            })
            .collect()
    }

    /// The rows of a gather on the device, as a pointer to `u32` values: those
    /// of `index` in place when it holds them in row-major order (`u32`, or
    /// `i32` checked non-negative: the same bits), otherwise `rows`
    /// uploaded. The storage returned keeps them alive.
    pub(crate) fn row_list(
        &self,
        op: &'static str,
        index: &Tensor,
        rows: &[usize],
    ) -> Result<(u64, Arc<dyn kapel_core::Storage>)> {
        if index.is_contiguous() && matches!(index.dtype(), DType::U32 | DType::I32) {
            let storage = self.storage(op, index)?;
            let pointer = storage.pointer() + index.layout().offset() as u64 * 4;
            return Ok((pointer, Arc::clone(index.storage())));
        }
        let uploaded = self.upload_u32(&smalls(op, rows)?)?;
        Ok((uploaded.pointer(), Arc::new(uploaded)))
    }

    /// A tensor's exact values, read back to the host.
    fn values(&self, tensor: &Tensor) -> Result<Vec<f64>> {
        self.download(tensor)?.to_f64_vec()
    }

    /// `values` encoded as `dtype`, uploaded as a tensor of `shape`.
    fn encoded(&self, values: &[f64], shape: &[usize], dtype: DType) -> Result<Tensor> {
        let size = dtype.block_bytes();
        let mut storage = HostStorage::new(values.len() * size);
        for (&value, out) in values
            .iter()
            .zip(storage.bytes_mut().chunks_exact_mut(size))
        {
            dtype.encode(value, out)?;
        }
        let host = Tensor::from_storage(Arc::new(storage), Layout::contiguous(shape)?, dtype)?;
        self.upload(&host)
    }

    /// The weights of a quantized product, dequantized and uploaded as `f32`.
    fn dequantized(&self, w: &Tensor) -> Result<Tensor> {
        self.stacked(&[w])
    }

    /// Weights `[Nᵢ, K]`, dequantized, their rows stacked, and uploaded as
    /// `f32`.
    fn stacked(&self, weights: &[&Tensor]) -> Result<Tensor> {
        let k = weights[0].shape()[1];
        let mut values = Vec::new();
        for w in weights {
            values.extend(self.values(w)?.into_iter().map(|v| v as f32));
        }
        self.upload_f32(&values, &[values.len() / k.max(1), k])
    }

    /// The rows of a scatter, grouped by target: `offsets[r]..offsets[r + 1]`
    /// in `sources` are the sources of row `r`, in their order of appearance.
    fn scatter(
        &self,
        op: &'static str,
        src: &Tensor,
        rows: &[usize],
        count: usize,
        shape: &[usize],
    ) -> Result<Tensor> {
        let width: usize = shape[1..].iter().product();
        let mut grouped: Vec<Vec<usize>> = vec![Vec::new(); count];
        for (i, &r) in rows.iter().enumerate() {
            grouped[r].push(i);
        }
        let mut offsets = vec![0u32];
        let mut sources = Vec::with_capacity(rows.len());
        for group in &grouped {
            sources.extend(smalls(op, group)?);
            offsets.push(small(op, sources.len())?);
        }
        let src = self.flat(op, src)?;
        let (o, s) = (self.upload_u32(&offsets)?, self.upload_u32(&sources)?);
        self.output(shape, false, |out, n| {
            self.launch(
                "scatter_rows",
                n,
                Args::new()
                    .flat(&src)
                    .list(&o, offsets.len())
                    .list(&s, sources.len())
                    .u32(small(op, width)?)
                    .slice(out, n),
            )
        })
    }

    /// `out[i] = src[rows[i]]`, rows of the given width; `rows` are the
    /// values of `index`, checked.
    fn gather(
        &self,
        op: &'static str,
        src: &Tensor,
        index: &Tensor,
        rows: &[usize],
        width: usize,
        shape: &[usize],
    ) -> Result<Tensor> {
        let src = self.flat(op, src)?;
        let (r, _keep) = self.row_list(op, index, rows)?;
        self.output(shape, false, |out, n| {
            self.launch(
                "gather_rows",
                n,
                Args::new()
                    .flat(&src)
                    .slice(r, rows.len())
                    .u32(small(op, width)?)
                    .slice(out, n),
            )
        })
    }
}

/// An operation computed in `f32`, for inputs in `f32`; [`Run`] widens the
/// others first.
trait InF32<O> {
    fn forward_f32(&self, op: &O, inputs: &[&Tensor]) -> Result<Tensor>;

    fn backward_f32(
        &self,
        op: &O,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>>;

    /// The backward pass from inputs and a gradient of any float type
    /// directly, computing in `f32` in registers, when the operation has
    /// one: `None` to widen them first. Called after `infer_backward`.
    fn backward_native(
        &self,
        _op: &O,
        _inputs: &[&Tensor],
        _grad: &Tensor,
    ) -> Option<Result<Vec<Option<Tensor>>>> {
        None
    }
}

impl Cuda {
    /// Each input in `f32` if it is in `bf16` or `f16`; the others as they are.
    fn widened_all(&self, op: &'static str, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        inputs.iter().map(|x| self.widened(op, x)).collect()
    }
}

/// [`Run`] for operations computed in `f32`: inputs widened, the output
/// rounded to the type of its signature, and each gradient to its input's.
/// The operations listed after `native:` compute their forward pass from
/// inputs of any float type directly ([`Native`]), without widening them
/// first: one kernel instead of three.
macro_rules! in_f32 {
    ($($op:ty),* $(,)?; native: $($native:ty),* $(,)?) => {
        $(
        impl Run<$op> for Cuda {
            fn forward(&self, op: &$op, inputs: &[&Tensor]) -> Result<Tensor> {
                let name = <$op>::NAME;
                let dtype = op.infer(inputs)?.dtype;
                let wide = self.widened_all(name, inputs)?;
                let wide: Vec<&Tensor> = wide.iter().collect();
                let y = InF32::<$op>::forward_f32(self, op, &wide)?;
                self.narrowed(name, &y, dtype)
            }

            in_f32!(@backward $op);
        }
        )*
        $(
        impl Run<$native> for Cuda {
            fn forward(&self, op: &$native, inputs: &[&Tensor]) -> Result<Tensor> {
                Native::<$native>::forward_native(self, op, inputs)
            }

            in_f32!(@backward $native);
        }
        )*
    };
    (@backward $op:ty) => {
        fn backward(
            &self,
            op: &$op,
            inputs: &[&Tensor],
            grad: &Tensor,
        ) -> Result<Vec<Option<Tensor>>> {
            let name = <$op>::NAME;
            op.infer_backward(inputs, grad)?;
            if let Some(grads) = InF32::<$op>::backward_native(self, op, inputs, grad) {
                return grads;
            }
            let wide = self.widened_all(name, inputs)?;
            let wide: Vec<&Tensor> = wide.iter().collect();
            let g = self.widened(name, grad)?;
            let grads = InF32::<$op>::backward_f32(self, op, &wide, &g)?;
            grads
                .into_iter()
                .zip(inputs)
                .map(|(g, x)| g.map(|g| self.narrowed(name, &g, x.dtype())).transpose())
                .collect()
        }
    };
}

/// An operation whose forward pass reads and writes `f32`, `bf16` and `f16`
/// directly, computing in `f32`: the same values as widening the inputs,
/// computing in `f32` and rounding the result, in one kernel.
trait Native<O> {
    fn forward_native(&self, op: &O, inputs: &[&Tensor]) -> Result<Tensor>;
}

in_f32!(
    Max,
    Softmax,
    LogSoftmax,
    ScatterAddRows;
    native: Sum,
    Mean,
    Unary,
    QuantizedMatmul,
    Glu,
    Affine,
    Binary,
    LayerNorm,
    RmsNorm,
    Rope,
);

impl Cuda {
    /// `scale ·` the sum of `x` over `dims`, read in its own type, as a
    /// tensor of `shape` and `dtype`.
    fn reduce_native(
        &self,
        op: &'static str,
        x: &Tensor,
        dims: &[usize],
        scale: f32,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        let x = Run::<Contiguous>::forward(self, &Contiguous, &[x])?;
        let size = x.dtype().block_bytes();
        let storage = self.storage(op, &x)?;
        let pointer = storage.pointer() + (x.layout().offset() * size) as u64;
        let sums = self.reduce_raw(
            op,
            float_code(op, x.dtype())?,
            pointer,
            x.numel() * size,
            x.shape(),
            dims,
            scale,
            shape,
        )?;
        self.narrowed(op, &sums, dtype)
    }
}

impl Native<Sum> for Cuda {
    fn forward_native(&self, op: &Sum, inputs: &[&Tensor]) -> Result<Tensor> {
        let s = op.infer(inputs)?;
        self.reduce_native(Sum::NAME, inputs[0], &op.dims, 1.0, &s.shape, s.dtype)
    }
}

impl Native<Mean> for Cuda {
    fn forward_native(&self, op: &Mean, inputs: &[&Tensor]) -> Result<Tensor> {
        let s = op.infer(inputs)?;
        let scale = 1.0 / count(inputs[0].shape(), &op.dims);
        self.reduce_native(Mean::NAME, inputs[0], &op.dims, scale, &s.shape, s.dtype)
    }
}

impl Native<Unary> for Cuda {
    fn forward_native(&self, op: &Unary, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = Unary::NAME;
        let signature = op.infer(inputs)?;
        let code = unary_code(name, op.kind)?;
        if let Some(kind) = half_kind(inputs[0].dtype())
            && signature.dtype == inputs[0].dtype()
            && let Some((px, wx)) = self.pieces_16(name, inputs[0])?
        {
            return self.output_as(&signature.shape, signature.dtype, false, |out, n| {
                self.launch(
                    "unary_16",
                    n / 8,
                    Args::new()
                        .u32(code)
                        .u32(kind)
                        .slice(px, wx)
                        .slice(out, n / 2),
                )
            });
        }
        let x = self.float_arg(name, inputs[0], inputs[0].layout())?;
        let ko = float_code(name, signature.dtype)?;
        self.output_as(&signature.shape, signature.dtype, false, |out, n| {
            self.launch(
                "unary_any",
                n,
                Args::new()
                    .u32(code)
                    .float(&x)
                    .u32(ko)
                    .slice(out, signature.dtype.storage_bytes(n)?),
            )
        })
    }
}

impl Native<Glu> for Cuda {
    fn forward_native(&self, op: &Glu, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = Glu::NAME;
        let signature = op.infer(inputs)?;
        let code = unary_code(name, op.activation)?;
        let x = self.contiguous_any(name, inputs[0])?;
        let xa = self.float_arg(name, &x, x.layout())?;
        let n = *signature.shape.last().expect("rank checked");
        let ko = float_code(name, signature.dtype)?;
        self.output_as(&signature.shape, signature.dtype, false, |out, count| {
            self.launch(
                "glu_any",
                count,
                Args::new()
                    .u32(code)
                    .u32(xa.code)
                    .slice(xa.base, xa.bytes)
                    .u32(small(name, n)?)
                    .u32(ko)
                    .slice(out, signature.dtype.storage_bytes(count)?),
            )
        })
    }
}

impl InF32<Glu> for Cuda {
    fn forward_f32(&self, op: &Glu, inputs: &[&Tensor]) -> Result<Tensor> {
        self.forward_native(op, inputs)
    }

    fn backward_f32(
        &self,
        op: &Glu,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let name = Glu::NAME;
        let signature = op.infer_backward(inputs, grad)?;
        let code = unary_code(name, op.activation)?;
        let x = self.flat(name, inputs[0])?;
        let g = self.flat(name, grad)?;
        let n = *signature.shape.last().expect("rank checked");
        let out = self.output(inputs[0].shape(), false, |out, count| {
            self.launch(
                "glu_backward",
                count,
                Args::new()
                    .u32(code)
                    .flat(&x)
                    .flat(&g)
                    .u32(small(name, n)?)
                    .slice(out, count),
            )
        })?;
        Ok(vec![Some(out)])
    }
}

impl Native<Affine> for Cuda {
    fn forward_native(&self, op: &Affine, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = Affine::NAME;
        let signature = op.infer(inputs)?;
        let x = self.float_arg(name, inputs[0], inputs[0].layout())?;
        let ko = float_code(name, signature.dtype)?;
        self.output_as(&signature.shape, signature.dtype, false, |out, n| {
            self.launch(
                "affine_any",
                n,
                Args::new()
                    .f32(op.scale as f32)
                    .f32(op.bias as f32)
                    .float(&x)
                    .u32(ko)
                    .slice(out, signature.dtype.storage_bytes(n)?),
            )
        })
    }
}

/// The code of a 16-bit float type, for the kernels that pack them in pairs.
fn half_kind(dtype: DType) -> Option<u32> {
    match dtype {
        DType::BF16 => Some(kapel_cuda_abi::half::BF16),
        DType::F16 => Some(kapel_cuda_abi::half::F16),
        _ => None,
    }
}

impl Cuda {
    /// `f(a, b)` eight values a thread (`binary_16`), when both are of one
    /// 16-bit type, `a` of the output's shape and `b` too or a row of its
    /// last dimension (a bias), all in whole aligned pieces.
    fn binary_16(
        &self,
        code: u32,
        a: &Tensor,
        b: &Tensor,
        shape: &[usize],
    ) -> Result<Option<Tensor>> {
        let name = Binary::NAME;
        let Some(kind) = half_kind(a.dtype()) else {
            return Ok(None);
        };
        if b.dtype() != a.dtype() || a.shape() != shape {
            return Ok(None);
        }
        let width = shape.last().copied().unwrap_or(0);
        let lead_ones = b.shape().iter().rev().skip(1).all(|&d| d == 1);
        let row = if b.shape() == shape {
            0
        } else if lead_ones && b.shape().last() == Some(&width) && width.is_multiple_of(8) {
            width / 8
        } else {
            return Ok(None);
        };
        let (Some((pa, wa)), Some((pb, wb))) = (self.pieces_16(name, a)?, self.pieces_16(name, b)?)
        else {
            return Ok(None);
        };
        let dtype = a.dtype();
        Ok(Some(self.output_as(shape, dtype, false, |out, n| {
            self.launch(
                "binary_16",
                n / 8,
                Args::new()
                    .u32(code)
                    .u32(kind)
                    .slice(pa, wa)
                    .slice(pb, wb)
                    .u32(small(name, row)?)
                    .slice(out, n / 2),
            )
        })?))
    }
}

impl Native<Binary> for Cuda {
    fn forward_native(&self, op: &Binary, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = Binary::NAME;
        let signature = op.infer(inputs)?;
        let code = binary_code(name, op.kind)?;
        let shape = &signature.shape;
        if let Some(y) = self.binary_16(code, inputs[0], inputs[1], shape)? {
            return Ok(y);
        }
        let a = self.float_arg(name, inputs[0], &inputs[0].layout().broadcast_to(shape)?)?;
        let b = self.float_arg(name, inputs[1], &inputs[1].layout().broadcast_to(shape)?)?;
        let ko = float_code(name, signature.dtype)?;
        self.output_as(shape, signature.dtype, false, |out, n| {
            self.launch(
                "binary_any",
                n,
                Args::new()
                    .u32(code)
                    .float(&a)
                    .float(&b)
                    .u32(ko)
                    .slice(out, signature.dtype.storage_bytes(n)?),
            )
        })
    }
}

impl InF32<Unary> for Cuda {
    fn forward_f32(&self, op: &Unary, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let code = unary_code(Unary::NAME, op.kind)?;
        let x = self.flat(Unary::NAME, inputs[0])?;
        self.output(&signature.shape, false, |out, n| {
            self.launch(
                "unary_forward",
                n,
                Args::new().u32(code).flat(&x).slice(out, n),
            )
        })
    }

    fn backward_f32(
        &self,
        op: &Unary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.unary_backward(op, inputs[0], grad)?)])
    }

    fn backward_native(
        &self,
        op: &Unary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Option<Result<Vec<Option<Tensor>>>> {
        Some(
            self.unary_backward(op, inputs[0], grad)
                .map(|g| vec![Some(g)]),
        )
    }
}

impl Cuda {
    /// `g · f'(x)` in the type of `x`, both read through their views in
    /// their own types.
    fn unary_backward(&self, op: &Unary, x: &Tensor, grad: &Tensor) -> Result<Tensor> {
        let name = Unary::NAME;
        let code = unary_code(name, op.kind)?;
        if let Some(kind) = half_kind(x.dtype())
            && grad.dtype() == x.dtype()
            && let (Some((px, wx)), Some((pg, wg))) =
                (self.pieces_16(name, x)?, self.pieces_16(name, grad)?)
        {
            return self.output_as(x.shape(), x.dtype(), false, |out, n| {
                self.launch(
                    "unary_backward_16",
                    n / 8,
                    Args::new()
                        .u32(code)
                        .u32(kind)
                        .slice(px, wx)
                        .slice(pg, wg)
                        .slice(out, n / 2),
                )
            });
        }
        let xa = self.float_arg(name, x, x.layout())?;
        let ga = self.float_arg(name, grad, grad.layout())?;
        let dtype = x.dtype();
        let ko = float_code(name, dtype)?;
        self.output_as(x.shape(), dtype, false, |out, n| {
            self.launch(
                "unary_backward",
                n,
                Args::new()
                    .u32(code)
                    .float(&xa)
                    .float(&ga)
                    .u32(ko)
                    .slice(out, dtype.storage_bytes(n)?),
            )
        })
    }

    fn affine(&self, x: &Tensor, scale: f64, bias: f64, shape: &[usize]) -> Result<Tensor> {
        let x = self.flat(Affine::NAME, x)?;
        self.output(shape, false, |out, n| {
            self.launch(
                "affine",
                n,
                Args::new()
                    .f32(scale as f32)
                    .f32(bias as f32)
                    .flat(&x)
                    .slice(out, n),
            )
        })
    }
}

impl InF32<Affine> for Cuda {
    fn forward_f32(&self, op: &Affine, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        self.affine(inputs[0], op.scale, op.bias, &signature.shape)
    }

    fn backward_f32(
        &self,
        op: &Affine,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.affine(
            grad,
            op.scale,
            0.0,
            &signature.shape,
        )?)])
    }
}

impl InF32<Binary> for Cuda {
    fn forward_f32(&self, op: &Binary, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let code = binary_code(Binary::NAME, op.kind)?;
        let shape = &signature.shape;
        let [a, b] = [inputs[0], inputs[1]]
            .map(|t| self.arg_as(Binary::NAME, t, &t.layout().broadcast_to(shape)?));
        let (a, b) = (a?, b?);
        self.output(shape, false, |out, n| {
            self.launch(
                "binary_forward",
                n,
                Args::new().u32(code).strided(&a).strided(&b).slice(out, n),
            )
        })
    }

    fn backward_f32(
        &self,
        op: &Binary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let code = binary_code(Binary::NAME, op.kind)?;
        let shape = &signature.shape;
        let [a, b] = [inputs[0], inputs[1]]
            .map(|t| self.arg_as(Binary::NAME, t, &t.layout().broadcast_to(shape)?));
        let (a, b) = (a?, b?);
        let g = self.flat(Binary::NAME, grad)?;
        let mut grads = Vec::new();
        for side in 0..2u32 {
            let full = self.output(shape, false, |out, n| {
                self.launch(
                    "binary_backward",
                    n,
                    Args::new()
                        .u32(code)
                        .u32(side)
                        .strided(&a)
                        .strided(&b)
                        .flat(&g)
                        .slice(out, n),
                )
            })?;
            let target = inputs[side as usize].shape();
            grads.push(Some(self.sum_to(Binary::NAME, &full, target)?));
        }
        Ok(grads)
    }
}

/// Whether conversions between `from` and `to` run on the device: between
/// `f32`, `bf16` and `f16`.
fn on_device(from: DType, to: DType) -> bool {
    let float = |d| matches!(d, DType::F32 | DType::BF16 | DType::F16);
    float(from) && float(to)
}

impl Cuda {
    /// `x` converted to `dtype`, rounding to nearest even as the reference
    /// does: on the device between `f32`, `bf16` and `f16` (through the exact
    /// `f32` value), otherwise through the host and each element's exact
    /// value. Always a new row-major tensor.
    fn cast(&self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        let op = Cast::NAME;
        if x.dtype() == dtype && dtype.element_size().is_some() {
            return self.materialize(op, x, x.layout());
        }
        if let Some(y) = self.decoded_on_device(x, dtype)? {
            return Ok(y);
        }
        if on_device(x.dtype(), dtype) {
            let wide = self.widened(op, x)?;
            return match dtype {
                // `widened` copies 16-bit inputs.
                DType::F32 if x.dtype() != DType::F32 => Ok(wide),
                // An `f32` one still needs its own.
                DType::F32 => self.materialize(op, &wide, wide.layout()),
                _ => self.narrowed(op, &wide, dtype),
            };
        }
        let values = self.values(x)?;
        self.encoded(&values, x.shape(), dtype)
    }
}

impl Run<Cast> for Cuda {
    fn forward(&self, op: &Cast, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.cast(inputs[0], op.dtype)
    }

    fn backward(
        &self,
        op: &Cast,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        // Between float types the gradient passes through, in the input's
        // type; an integer input or output receives none.
        let dtype = inputs[0].dtype();
        if !op.differentiable(0) || !dtype.is_float() {
            return Ok(vec![None]);
        }
        Ok(vec![Some(self.cast(grad, dtype)?)])
    }
}

impl Run<Concat> for Cuda {
    /// Each input is copied, through its own layout, into its slice of the
    /// output (`copy_into_*`), bit for bit.
    fn forward(&self, op: &Concat, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = Concat::NAME;
        let signature = op.infer(inputs)?;
        let dtype = signature.dtype;
        let kernel = match dtype.element_size() {
            Some(1) => "copy_into_8",
            Some(2) => "copy_into_16",
            Some(4) => "copy_into_32",
            _ => "copy_into_64",
        };
        let layout = Layout::contiguous(&signature.shape)?;
        self.output_as(&signature.shape, dtype, false, |out, n| {
            let mut start = 0;
            for x in inputs {
                let len = x.shape()[op.dim];
                let to = view_of(name, &layout.narrow(op.dim, start, len)?)?;
                start += len;
                let count = x.numel();
                let from = self.any_arg(name, x, x.layout())?;
                self.launch(
                    kernel,
                    count,
                    Args::new()
                        .strided(&from)
                        .slice(out, n)
                        .view(to)
                        .u32(small(name, count)?),
                )?;
            }
            Ok(())
        })
    }

    fn backward(
        &self,
        op: &Concat,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        // Each input receives its slice of the gradient, in its own type; an
        // integer input receives none.
        let mut start = 0;
        inputs
            .iter()
            .map(|x| {
                let len = x.shape()[op.dim];
                let slice = grad.narrow(op.dim, start, len)?;
                start += len;
                if x.dtype().is_float() {
                    Ok(Some(self.cast(&slice, x.dtype())?))
                } else {
                    Ok(None)
                }
            })
            .collect()
    }
}

impl Run<Contiguous> for Cuda {
    fn forward(&self, op: &Contiguous, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let x = inputs[0];
        self.storage(Contiguous::NAME, x)?;
        if x.is_contiguous() {
            return Ok(x.clone());
        }
        self.materialize(Contiguous::NAME, x, x.layout())
    }

    fn backward(
        &self,
        op: &Contiguous,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        // The gradient passes through, in the input's type; an integer input
        // receives none.
        let dtype = inputs[0].dtype();
        if !dtype.is_float() {
            return Ok(vec![None]);
        }
        Ok(vec![Some(self.cast(grad, dtype)?)])
    }
}

impl Run<Matmul> for Cuda {
    fn forward(&self, op: &Matmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        self.matmul(
            inputs[0],
            inputs[1],
            inputs.get(2).copied(),
            &signature.shape,
        )
    }

    /// `∂a = g · bᵀ` and `∂b = aᵀ · g`, summed over the batch dimensions each
    /// input was broadcast along.
    fn backward(
        &self,
        op: &Matmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let (a, b) = (inputs[0], inputs[1]);
        let (ra, rb) = (a.rank(), b.rank());
        let bt = b.transpose(rb - 2, rb - 1)?;
        let at = a.transpose(ra - 2, ra - 1)?;
        let ga = self.forward(op, &[grad, &bt])?;
        let gb = self.forward(op, &[&at, grad])?;
        let mut grads = vec![
            Some(self.summed_to(&ga, a.shape())?),
            Some(self.summed_to(&gb, b.shape())?),
        ];
        if let Some(bias) = inputs.get(2) {
            // The output's gradient summed over every row.
            let dims: Vec<usize> = (0..grad.rank() - 1).collect();
            let sum = Sum {
                dims,
                keepdim: false,
            };
            grads.push(Some(self.converted(
                &Run::<Sum>::forward(self, &sum, &[grad])?,
                bias.dtype(),
            )?));
        }
        Ok(grads)
    }
}

impl Cuda {
    /// The gradient of a product, of its type, summed back to `target` over the
    /// batch dimensions it was broadcast along: in `f32` (then rounded back),
    /// when there is anything to sum.
    fn summed_to(&self, full: &Tensor, target: &[usize]) -> Result<Tensor> {
        let op = Matmul::NAME;
        if full.shape() == target {
            return Ok(full.clone());
        }
        let wide = self.widened(op, full)?;
        self.narrowed(op, &self.sum_to(op, &wide, target)?, full.dtype())
    }
}

/// The product of all dimensions but the last.
fn leading(shape: &[usize]) -> usize {
    shape[..shape.len().saturating_sub(1)].iter().product()
}

impl Native<QuantizedMatmul> for Cuda {
    /// On the device for the formats with kernels; through the host (the
    /// weights dequantized there) for the others.
    fn forward_native(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = QuantizedMatmul::NAME;
        let signature = op.infer(inputs)?;
        if let Some(y) =
            self.quantized_product(inputs[0], &inputs[1..], &signature.shape, signature.dtype)?
        {
            return Ok(y);
        }
        let wide = self.widened_all(name, inputs)?;
        let wide: Vec<&Tensor> = wide.iter().collect();
        let y = InF32::<QuantizedMatmul>::forward_f32(self, op, &wide)?;
        self.narrowed(name, &y, signature.dtype)
    }
}

impl InF32<QuantizedMatmul> for Cuda {
    /// The weights are dequantized on the host, then multiplied. Kernels that
    /// read the codes directly come with the inference engine.
    fn forward_f32(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = inputs[0];
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        // The rows from the leading dimensions: with `K = 0`, `numel / K`
        // would lose them.
        let rows = leading(x.shape());
        let x = self.contiguous(QuantizedMatmul::NAME, x)?;
        let x = x.reshape(&[rows, k])?;
        let wt = self.stacked(&inputs[1..])?.transpose(0, 1)?;
        let shape = [x.shape()[0], n];
        self.matmul(&x, &wt, None, &shape)?
            .reshape(&signature.shape)
    }

    /// On the device for the formats with kernels.
    fn backward_native(
        &self,
        _op: &QuantizedMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Option<Result<Vec<Option<Tensor>>>> {
        match self.quantized_product_grad(inputs[0], &inputs[1..], grad) {
            Ok(Some(dx)) => {
                let mut grads = vec![Some(dx)];
                grads.resize(inputs.len(), None);
                Some(Ok(grads))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn backward_f32(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let x = inputs[0];
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        let rows = leading(grad.shape());
        let g = self.contiguous(QuantizedMatmul::NAME, grad)?;
        let g = g.reshape(&[rows, n])?;
        let w = self.stacked(&inputs[1..])?;
        let shape = [g.shape()[0], k];
        let mut grads = vec![Some(self.matmul(&g, &w, None, &shape)?.reshape(x.shape())?)];
        grads.resize(inputs.len(), None);
        Ok(grads)
    }
}

impl InF32<Sum> for Cuda {
    fn forward_f32(&self, op: &Sum, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = self.flat(Sum::NAME, inputs[0])?;
        self.reduce(
            Sum::NAME,
            &x,
            inputs[0].shape(),
            &op.dims,
            1.0,
            &signature.shape,
        )
    }

    fn backward_f32(
        &self,
        op: &Sum,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.spread(
            Sum::NAME,
            grad,
            inputs[0].shape(),
            &op.dims,
            1.0,
        )?)])
    }
}

fn count(shape: &[usize], dims: &[usize]) -> f32 {
    dims.iter().map(|&d| shape[d]).product::<usize>() as f32
}

impl InF32<Mean> for Cuda {
    fn forward_f32(&self, op: &Mean, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = self.flat(Mean::NAME, inputs[0])?;
        let shape = inputs[0].shape();
        self.reduce(
            Mean::NAME,
            &x,
            shape,
            &op.dims,
            1.0 / count(shape, &op.dims),
            &signature.shape,
        )
    }

    fn backward_f32(
        &self,
        op: &Mean,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let shape = inputs[0].shape();
        Ok(vec![Some(self.spread(
            Mean::NAME,
            grad,
            shape,
            &op.dims,
            1.0 / count(shape, &op.dims),
        )?)])
    }
}

impl Cuda {
    /// The maximum of each line along `dim`, as a tensor of `shape`, and the
    /// position of the first maximum of each line.
    /// Also gives the number of lines.
    fn line_max(
        &self,
        x: &Tensor,
        dim: usize,
        shape: &[usize],
    ) -> Result<(Tensor, CudaStorage, usize)> {
        let op = Max::NAME;
        let (s, step, len) = self.line_table(op, x.shape(), dim)?;
        let x = self.flat(op, x)?;
        let at = self.alloc(s.len.max(1) * 4, false)?;
        let value = self.output(shape, false, |out, n| {
            self.launch(
                "line_max",
                n,
                Args::new()
                    .flat(&x)
                    .table(&s)
                    .u32(step)
                    .u32(len)
                    .slice(out, n)
                    .list(&at, s.len),
            )
        })?;
        Ok((value, at, s.len))
    }

    /// The first element of each line of `shape` along `dim`, on the device,
    /// the step between the elements of a line, and their number.
    fn line_table(
        &self,
        op: &'static str,
        shape: &[usize],
        dim: usize,
    ) -> Result<(Arc<Table>, u32, u32)> {
        let step: usize = shape[dim + 1..].iter().product();
        let starts = self.table("line starts", &Self::key(shape, &[dim]), || {
            smalls(op, &line_starts(shape, dim))
        })?;
        Ok((starts, small(op, step)?, small(op, shape[dim])?))
    }
}

impl InF32<Max> for Cuda {
    fn forward_f32(&self, op: &Max, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        Ok(self.line_max(inputs[0], op.dim, &signature.shape)?.0)
    }

    fn backward_f32(
        &self,
        op: &Max,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let (_, at, lines) = self.line_max(inputs[0], op.dim, &signature.shape)?;
        // `infer_backward` checked that the gradient has one value per line.
        let g = self.flat(Max::NAME, grad)?;
        let out = self.output(inputs[0].shape(), true, |out, n| {
            self.launch(
                "place",
                lines,
                Args::new().flat(&g).list(&at, lines).slice(out, n),
            )
        })?;
        Ok(vec![Some(out)])
    }
}

impl Cuda {
    /// Softmax (`log` = 0) or log-softmax (`log` = 1) along `dim`; with `grad`,
    /// its backward pass instead.
    fn softmax(
        &self,
        op: &'static str,
        log: u32,
        x: &Tensor,
        dim: usize,
        grad: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shape = x.shape().to_vec();
        let (s, step, len) = self.line_table(op, &shape, dim)?;
        let x = self.flat(op, x)?;
        let g = grad.map(|g| self.flat(op, g)).transpose()?;
        self.output(&shape, false, |out, n| {
            let args = Args::new().u32(log).flat(&x);
            let (name, args) = match &g {
                None => ("softmax_lines", args),
                Some(g) => ("softmax_lines_backward", args.flat(g)),
            };
            self.launch(name, s.len, args.table(&s).u32(step).u32(len).slice(out, n))
        })
    }
}

impl InF32<Softmax> for Cuda {
    fn forward_f32(&self, op: &Softmax, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.softmax(Softmax::NAME, 0, inputs[0], op.dim, None)
    }

    fn backward_f32(
        &self,
        op: &Softmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.softmax(
            Softmax::NAME,
            0,
            inputs[0],
            op.dim,
            Some(grad),
        )?)])
    }
}

impl InF32<LogSoftmax> for Cuda {
    fn forward_f32(&self, op: &LogSoftmax, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.softmax(LogSoftmax::NAME, 1, inputs[0], op.dim, None)
    }

    fn backward_f32(
        &self,
        op: &LogSoftmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.softmax(
            LogSoftmax::NAME,
            1,
            inputs[0],
            op.dim,
            Some(grad),
        )?)])
    }
}

/// The threads of `count` warps, one per row of a normalization.
fn warps(op: &'static str, count: usize) -> Result<usize> {
    count.checked_mul(32).ok_or(Error::Overflow { op })
}

/// A normalization: its kind (`rms` = 0 for LayerNorm, 1 for RMSNorm), its
/// constants, and its optional vectors with what stands for them when absent.
struct Norm<'a> {
    op: &'static str,
    rms: u32,
    eps: f32,
    offset: f32,
    weight: Option<&'a Tensor>,
    default_weight: f32,
    bias: Option<&'a Tensor>,
}

impl Cuda {
    /// A vector of the normalization, in `f32` (converted if it is in another
    /// float type), or `width` copies of `default`.
    fn norm_vector(
        &self,
        op: &'static str,
        vector: Option<&Tensor>,
        width: usize,
        default: f32,
    ) -> Result<Flat> {
        match vector {
            Some(v) => self.flat(op, &self.converted(v, DType::F32)?),
            None => self.flat(op, &self.filled(op, default, &[width])?),
        }
    }

    /// `tensor` in `dtype`: itself, or a conversion.
    fn converted(&self, tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        if tensor.dtype() == dtype {
            Ok(tensor.clone())
        } else {
            self.cast(tensor, dtype)
        }
    }

    /// A normalization's weight or bias, of any float type, as a kernel
    /// argument; `default` everywhere (an `f32` table) without one.
    fn norm_vector_any(
        &self,
        op: &'static str,
        vector: Option<&Tensor>,
        width: usize,
        default: f32,
    ) -> Result<(FloatArg, Tensor)> {
        let tensor = match vector {
            Some(v) => self.contiguous_any(op, v)?,
            None => self.filled(op, default, &[width])?,
        };
        let arg = self.float_arg(op, &tensor, &Layout::contiguous(tensor.shape())?)?;
        Ok((arg, tensor))
    }

    /// A tensor of this backend, of any type, with its elements in row-major
    /// order from the start of its storage: itself if it already is,
    /// otherwise a copy.
    pub(crate) fn contiguous_any(&self, op: &'static str, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && x.layout().offset() == 0 {
            self.storage(op, x)?;
            return Ok(x.clone());
        }
        self.materialize(op, x, x.layout())
    }

    /// The forward pass of a normalization, its operands of any float types,
    /// its output of `dtype`: one warp per row, or one block of 256 threads
    /// per row when rows are wide.
    fn norm_native(&self, norm: &Norm<'_>, x: &Tensor, dtype: DType) -> Result<Tensor> {
        let op = norm.op;
        let shape = x.shape().to_vec();
        let width = *shape.last().expect("checked by infer");
        let rows = x.numel() / width.max(1);
        let x = self.contiguous_any(op, x)?;
        let xa = self.float_arg(op, &x, &Layout::contiguous(x.shape())?)?;
        let (w, _keep_w) = self.norm_vector_any(op, norm.weight, width, norm.default_weight)?;
        let (b, _keep_b) = self.norm_vector_any(op, norm.bias, width, 0.0)?;
        let ko = float_code(op, dtype)?;
        // Wide rows whose operands share one type: one block per row.
        let block = (512..=8192).contains(&width)
            && norm.weight.is_some()
            && w.code == xa.code
            && ko == xa.code
            && norm.bias.is_none_or(|_| b.code == xa.code);
        let half = matches!(dtype, DType::BF16 | DType::F16) && width.is_multiple_of(8);
        let (kernel, threads, first, types) = if block && half {
            // 16-byte pieces of 16-bit values, in pairs.
            let flags = norm.rms | u32::from(norm.bias.is_some()) << 1;
            let kind = if dtype == DType::BF16 {
                kapel_cuda_abi::half::BF16
            } else {
                kapel_cuda_abi::half::F16
            };
            let threads = rows.checked_mul(256).ok_or(Error::Overflow { op })?;
            return self.output_as(&shape, dtype, false, |out, n| {
                self.launch(
                    "norm_16",
                    threads,
                    Args::new()
                        .u32(flags)
                        .u32(kind)
                        .slice(xa.base, xa.bytes / 4)
                        .slice(w.base, w.bytes / 4)
                        .slice(b.base, b.bytes / 4)
                        .f32(norm.offset)
                        .f32(norm.eps)
                        .u32(small(op, width)?)
                        .slice(out, dtype.storage_bytes(n)? / 4),
                )
            });
        } else if block {
            let flags = norm.rms | u32::from(norm.bias.is_some()) << 1;
            ("norm_block_any", rows.checked_mul(256), flags, xa.code)
        } else {
            let types = xa.code | w.code << 8 | b.code << 16 | ko << 24;
            ("norm_any", rows.checked_mul(32), norm.rms, types)
        };
        let threads = threads.ok_or(Error::Overflow { op })?;
        self.output_as(&shape, dtype, false, |out, n| {
            self.launch(
                kernel,
                threads,
                Args::new()
                    .u32(first)
                    .u32(types)
                    .slice(xa.base, xa.bytes)
                    .slice(w.base, w.bytes)
                    .slice(b.base, b.bytes)
                    .f32(norm.offset)
                    .f32(norm.eps)
                    .u32(small(op, width)?)
                    .slice(out, dtype.storage_bytes(n)?),
            )
        })
    }

    fn norm_forward(&self, norm: &Norm<'_>, x: &Tensor) -> Result<Tensor> {
        let op = norm.op;
        let shape = x.shape().to_vec();
        let width = *shape.last().expect("checked by infer");
        let rows = x.numel() / width.max(1);
        let x = self.flat(op, x)?;
        let w = self.norm_vector(op, norm.weight, width, norm.default_weight)?;
        let b = self.norm_vector(op, norm.bias, width, 0.0)?;
        self.output(&shape, false, |out, n| {
            self.launch(
                "norm_rows",
                warps(op, rows)?,
                Args::new()
                    .u32(norm.rms)
                    .flat(&x)
                    .flat(&w)
                    .flat(&b)
                    .f32(norm.offset)
                    .f32(norm.eps)
                    .u32(small(op, width)?)
                    .slice(out, n),
            )
        })
    }

    /// The gradients of `x`, of the weight and of the bias.
    fn norm_backward(
        &self,
        norm: &Norm<'_>,
        x: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let op = norm.op;
        let shape = x.shape().to_vec();
        let width = *shape.last().expect("checked by infer");
        let rows = x.numel() / width.max(1);
        let numel = x.numel();
        let dtype = x.dtype();
        // Row-major, in their own types.
        let rows_of = |t: &Tensor| -> Result<(u32, u64, usize, Tensor)> {
            let t = Run::<Contiguous>::forward(self, &Contiguous, &[t])?;
            let size = t.dtype().block_bytes();
            let storage = self.storage(op, &t)?;
            let pointer = storage.pointer() + (t.layout().offset() * size) as u64;
            Ok((float_code(op, t.dtype())?, pointer, numel * size, t))
        };
        let (kx, px, bx, _keep_x) = rows_of(x)?;
        let (kg, pg, bg, _keep_g) = rows_of(grad)?;
        let w = self.norm_vector(op, norm.weight, width, norm.default_weight)?;
        let gw_rows = self.alloc(numel * 4, false)?;
        let gb_rows = self.alloc(numel * 4, false)?;
        let ko = float_code(op, dtype)?;
        let gx = self.output_as(&shape, dtype, false, |out, n| {
            self.launch(
                "norm_rows_backward",
                warps(op, rows)?,
                Args::new()
                    .u32(norm.rms)
                    .u32(kx)
                    .slice(px, bx)
                    .flat(&w)
                    .u32(kg)
                    .slice(pg, bg)
                    .f32(norm.offset)
                    .f32(norm.eps)
                    .u32(small(op, width)?)
                    .u32(ko)
                    .slice(out, dtype.storage_bytes(n)?)
                    .list(&gw_rows, n)
                    .list(&gb_rows, n),
            )
        })?;
        let column_sum = |rows_of: &CudaStorage| {
            self.reduce_raw(
                op,
                F32_CODE,
                rows_of.pointer(),
                numel * 4,
                &[rows, width],
                &[0],
                1.0,
                &[width],
            )
        };
        Ok((gx, column_sum(&gw_rows)?, column_sum(&gb_rows)?))
    }
}

impl Native<LayerNorm> for Cuda {
    fn forward_native(&self, op: &LayerNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        let dtype = op.infer(inputs)?.dtype;
        self.norm_native(&layer_norm(op, inputs), inputs[0], dtype)
    }
}

impl Native<RmsNorm> for Cuda {
    fn forward_native(&self, op: &RmsNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        let dtype = op.infer(inputs)?.dtype;
        self.norm_native(&rms_norm(op, inputs), inputs[0], dtype)
    }
}

impl InF32<LayerNorm> for Cuda {
    fn forward_f32(&self, op: &LayerNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.norm_forward(&layer_norm(op, inputs), inputs[0])
    }

    fn backward_f32(
        &self,
        op: &LayerNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let (gx, gw, gb) = self.norm_backward(&layer_norm(op, inputs), inputs[0], grad)?;
        let mut grads = vec![Some(gx)];
        // The vectors' gradients come in their own types.
        for (input, g) in inputs[1..].iter().zip([gw, gb]) {
            grads.push(Some(self.converted(&g, input.dtype())?));
        }
        Ok(grads)
    }

    fn backward_native(
        &self,
        op: &LayerNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Option<Result<Vec<Option<Tensor>>>> {
        Some((|| {
            let (gx, gw, gb) = self.norm_backward(&layer_norm(op, inputs), inputs[0], grad)?;
            let mut grads = vec![Some(gx)];
            for (input, g) in inputs[1..].iter().zip([gw, gb]) {
                grads.push(Some(self.converted(&g, input.dtype())?));
            }
            Ok(grads)
        })())
    }
}

fn layer_norm<'a>(op: &LayerNorm, inputs: &[&'a Tensor]) -> Norm<'a> {
    Norm {
        op: LayerNorm::NAME,
        rms: 0,
        eps: op.eps as f32,
        offset: 0.0,
        weight: inputs.get(1).copied(),
        default_weight: 1.0,
        bias: inputs.get(2).copied(),
    }
}

fn rms_norm<'a>(op: &RmsNorm, inputs: &[&'a Tensor]) -> Norm<'a> {
    let offset = op.weight_offset as f32;
    Norm {
        op: RmsNorm::NAME,
        rms: 1,
        eps: op.eps as f32,
        offset,
        weight: inputs.get(1).copied(),
        // Without a weight, the factor `offset + weight` is 1.
        default_weight: 1.0 - offset,
        bias: None,
    }
}

impl InF32<RmsNorm> for Cuda {
    fn forward_f32(&self, op: &RmsNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.norm_forward(&rms_norm(op, inputs), inputs[0])
    }

    fn backward_f32(
        &self,
        op: &RmsNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let (gx, gw, _) = self.norm_backward(&rms_norm(op, inputs), inputs[0], grad)?;
        let mut grads = vec![Some(gx)];
        if let Some(w) = inputs.get(1) {
            grads.push(Some(self.converted(&gw, w.dtype())?));
        }
        Ok(grads)
    }

    fn backward_native(
        &self,
        op: &RmsNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Option<Result<Vec<Option<Tensor>>>> {
        Some((|| {
            let (gx, gw, _) = self.norm_backward(&rms_norm(op, inputs), inputs[0], grad)?;
            let mut grads = vec![Some(gx)];
            if let Some(w) = inputs.get(1) {
                grads.push(Some(self.converted(&gw, w.dtype())?));
            }
            Ok(grads)
        })())
    }
}

impl Run<ArgMax> for Cuda {
    /// Rows along the last dimension, one block each; another dimension is
    /// first moved last.
    fn forward(&self, op: &ArgMax, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = ArgMax::NAME;
        let signature = op.infer(inputs)?;
        let x = inputs[0];
        let last = x.rank() - 1;
        let moved = if op.dim == last {
            x.clone()
        } else {
            x.transpose(op.dim, last)?
        };
        let rows = self.contiguous_any(name, &moved)?;
        let width = rows.shape()[last];
        let xa = self.float_arg(name, &rows, rows.layout())?;
        let count = rows.numel() / width;
        // Long rows (a vocabulary) in parts, merged after: few rows would
        // otherwise leave the device to a few blocks.
        let parts = (width / 4096).clamp(1, 128);
        if parts > 1 {
            let values = self.alloc(count * parts * 4, false)?;
            let at = self.alloc(count * parts * 4, false)?;
            self.launch(
                "argmax_parts",
                (count * parts)
                    .checked_mul(256)
                    .ok_or(Error::Overflow { op: name })?,
                Args::new()
                    .u32(xa.code)
                    .slice(xa.base, xa.bytes)
                    .u32(small(name, width)?)
                    .u32(small(name, parts)?)
                    .list(&values, count * parts)
                    .list(&at, count * parts),
            )?;
            return self.output_as(&signature.shape, DType::U32, false, |out, n| {
                self.launch(
                    "argmax_merge",
                    n,
                    Args::new()
                        .u32(small(name, parts)?)
                        .list(&values, count * parts)
                        .list(&at, count * parts)
                        .slice(out, n),
                )
            });
        }
        self.output_as(&signature.shape, DType::U32, false, |out, n| {
            self.launch(
                "argmax_rows",
                count.checked_mul(256).ok_or(Error::Overflow { op: name })?,
                Args::new()
                    .u32(xa.code)
                    .slice(xa.base, xa.bytes)
                    .u32(small(name, width)?)
                    .slice(out, n),
            )
        })
    }

    fn backward(
        &self,
        op: &ArgMax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None])
    }
}

impl Run<Attention> for Cuda {
    fn forward(&self, op: &Attention, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.flash_forward(op, inputs)
    }

    fn backward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let [gq, gk, gv] = self.flash_backward(op, inputs, grad)?;
        let mut grads = vec![Some(gq), Some(gk), Some(gv)];
        if inputs.len() == 4 {
            grads.push(None);
        }
        Ok(grads)
    }
}

impl Cuda {
    /// Turns every vector of `x` by its position's angles; `direction` is 1 for
    /// the rotation, −1 for its transpose (the backward pass).
    ///
    /// The angles are computed in `f64` (`rope_table`), once per position
    /// entry: positions in the hundreds of thousands times a frequency near 1
    /// would lose their phase in `f32`.
    /// Rotary embeddings of `x`, of any float type read through its layout,
    /// as a row-major tensor of `dtype`; `direction` −1 turns back (the
    /// gradient).
    fn rope(
        &self,
        op: &Rope,
        x: &Tensor,
        positions: &Tensor,
        direction: f64,
        dtype: DType,
    ) -> Result<Tensor> {
        let name = Rope::NAME;
        let shape = x.shape().to_vec();
        let d = *shape.last().expect("checked by infer");
        let r = op.rotated(d);
        // The factor, then the frequencies, as the halves of their bits.
        let constants: Vec<u64> = std::iter::once(op.attention_factor())
            .chain(op.frequencies(r))
            .map(f64::to_bits)
            .collect();
        let key: Vec<usize> = constants.iter().map(|&c| c as usize).collect();
        let key_constants = constants.clone();
        let constants = self.table("rope constants", &key, || {
            Ok(constants
                .iter()
                .flat_map(|&c| [c as u32, (c >> 32) as u32])
                .collect())
        })?;
        let signed = u32::from(positions.dtype() == DType::I32);
        let positions = if positions.is_contiguous() {
            positions.clone()
        } else {
            self.materialize(name, positions, positions.layout())?
        };
        let entries = positions.numel();
        let storage = self.storage(name, &positions)?;
        let p = storage.pointer() + positions.layout().offset() as u64 * 4;
        let half = r / 2;
        let compute = || {
            self.output(&[entries * half * 2], false, |out, n| {
                self.launch(
                    "rope_table",
                    entries * half,
                    Args::new()
                        .slice(p, entries)
                        .u32(signed)
                        .slice(constants.storage.pointer(), half + 1)
                        .f32(direction as f32)
                        .slice(out, n),
                )
            })
        };
        // The positions known on the host (an upload): the angles of the
        // same positions and constants are computed once, for every layer.
        let mut recorded_key = (storage.pointer() + positions.layout().offset() as u64 * 4)
            .to_le_bytes()
            .to_vec();
        recorded_key.extend((entries as u64).to_le_bytes());
        recorded_key.extend(direction.to_bits().to_le_bytes());
        for &c in &key_constants {
            recorded_key.extend(c.to_le_bytes());
        }
        let table = match storage.host() {
            // While recording, the same positions' storage gives the same
            // angles at every replay: computed once per pass.
            _ if self.arena().is_some() => self.recorded_angles(recorded_key, compute)?,
            Some(bytes) => {
                let start = positions.layout().offset() * 4;
                let mut key: Vec<u8> = bytes[start..start + entries * 4].to_vec();
                key.extend(direction.to_bits().to_le_bytes());
                key.push(signed as u8);
                for &c in &key_constants {
                    key.extend(c.to_le_bytes());
                }
                self.angles(key, compute)?
            }
            None => compute()?,
        };
        let vectors = &shape[..shape.len() - 1];
        let entry = self.table(
            "rope entries",
            &Self::key(positions.shape(), vectors),
            || {
                let entry_of: Vec<usize> = Layout::contiguous(positions.shape())?
                    .broadcast_to(vectors)?
                    .offsets()
                    .collect();
                smalls(name, &entry_of)
            },
        )?;
        let x = self.float_arg(name, x, x.layout())?;
        let table = self.flat(name, &table)?;
        let half_pairs = u32::from(op.layout == RopeLayout::Half);
        let ko = float_code(name, dtype)?;
        self.output_as(&shape, dtype, false, |out, n| {
            self.launch(
                "rope_any",
                n,
                Args::new()
                    .float(&x)
                    .flat(&table)
                    .table(&entry)
                    .u32(small(name, d)?)
                    .u32(small(name, r)?)
                    .u32(half_pairs)
                    .u32(ko)
                    .slice(out, dtype.storage_bytes(n)?),
            )
        })
    }
}

impl Native<Rope> for Cuda {
    fn forward_native(&self, op: &Rope, inputs: &[&Tensor]) -> Result<Tensor> {
        let dtype = op.infer(inputs)?.dtype;
        self.rope(op, inputs[0], inputs[1], 1.0, dtype)
    }
}

impl InF32<Rope> for Cuda {
    fn forward_f32(&self, op: &Rope, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.rope(op, inputs[0], inputs[1], 1.0, DType::F32)
    }

    fn backward_f32(
        &self,
        op: &Rope,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![
            Some(self.rope(op, grad, inputs[1], -1.0, DType::F32)?),
            None,
        ])
    }
}

impl Run<GatherRows> for Cuda {
    /// The rows are copied bit for bit, in `f32`, `bf16` or `f16`.
    fn forward(&self, op: &GatherRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = GatherRows::NAME;
        let signature = op.infer(inputs)?;
        let shape = inputs[0].shape();
        let rows = self.indices(name, inputs[1], shape[0])?;
        let width = shape[1..].iter().product();
        match inputs[0].dtype() {
            DType::BF16 | DType::F16 => {
                self.gather_16(inputs[0], inputs[1], &rows, width, &signature)
            }
            // A quantized table is dequantized on the host, whole: kernels
            // that read the codes come with the products by quantized weights.
            dtype if dtype.is_quantized() => {
                if let Some(y) =
                    self.quantized_rows(inputs[0], inputs[1], &rows, &signature.shape)?
                {
                    return Ok(y);
                }
                let table = self.dequantized(inputs[0])?;
                self.gather(name, &table, inputs[1], &rows, width, &signature.shape)
            }
            _ => self.gather(name, inputs[0], inputs[1], &rows, width, &signature.shape),
        }
    }

    fn backward(
        &self,
        op: &GatherRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let name = GatherRows::NAME;
        op.infer_backward(inputs, grad)?;
        if inputs[0].dtype().is_quantized() {
            return Ok(vec![None, None]);
        }
        let shape = inputs[0].shape();
        let rows = self.indices(name, inputs[1], shape[0])?;
        let g = self.widened(name, grad)?;
        let out = self.scatter(name, &g, &rows, shape[0], shape)?;
        Ok(vec![
            Some(self.narrowed(name, &out, inputs[0].dtype())?),
            None,
        ])
    }
}

impl Cuda {
    /// `gather` for a source in 16 bits.
    fn gather_16(
        &self,
        src: &Tensor,
        index: &Tensor,
        rows: &[usize],
        width: usize,
        signature: &kapel_ops::Signature,
    ) -> Result<Tensor> {
        let op = GatherRows::NAME;
        let src = if src.is_contiguous() {
            src.clone()
        } else {
            self.materialize(op, src, src.layout())?
        };
        let storage = self.storage(op, &src)?;
        let start = src.layout().offset();
        let pointer = storage.pointer() + start as u64 * 2;
        let (r, _keep) = self.row_list(op, index, rows)?;
        self.output_as(&signature.shape, signature.dtype, false, |out, n| {
            self.launch(
                "gather_rows_16",
                n,
                Args::new()
                    .slice(pointer, src.numel())
                    .slice(r, rows.len())
                    .u32(small(op, width)?)
                    .slice(out, n),
            )
        })
    }
}

impl InF32<ScatterAddRows> for Cuda {
    fn forward_f32(&self, op: &ScatterAddRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let rows = self.indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        self.scatter(
            ScatterAddRows::NAME,
            inputs[0],
            &rows,
            op.rows,
            &signature.shape,
        )
    }

    fn backward_f32(
        &self,
        op: &ScatterAddRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let rows = self.indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        let width = signature.shape[1..].iter().product();
        let out = self.gather(
            ScatterAddRows::NAME,
            grad,
            inputs[1],
            &rows,
            width,
            inputs[0].shape(),
        )?;
        Ok(vec![Some(out), None])
    }
}

impl Cuda {
    /// The targets of a cross-entropy, read on the host (from the copy an
    /// upload keeps, without waiting) and checked, as `u32` bits on the
    /// device: those of `targets` in place when they are row-major.
    fn targets(
        &self,
        targets: &Tensor,
        vocab: usize,
    ) -> Result<(u64, Arc<dyn kapel_core::Storage>)> {
        let op = CrossEntropy::NAME;
        let host = match self.storage(op, targets)?.host() {
            Some(bytes) => Tensor::from_storage(
                Arc::new(HostStorage::from_bytes(bytes)),
                targets.layout().clone(),
                targets.dtype(),
            )?,
            None => self.download(targets)?,
        };
        let values = host.to_f64_vec()?;
        for &v in &values {
            CrossEntropy::target(v as i64, vocab)?;
        }
        if targets.is_contiguous() {
            // `u32` and `i32` alike: −1 has the bits of `IGNORE`.
            let storage = self.storage(op, targets)?;
            let pointer = storage.pointer() + targets.layout().offset() as u64 * 4;
            return Ok((pointer, Arc::clone(targets.storage())));
        }
        let bits: Vec<u32> = values.iter().map(|&v| v as i64 as u32).collect();
        let uploaded = self.upload_u32(&bits)?;
        Ok((uploaded.pointer(), Arc::new(uploaded)))
    }
}

/// The logits of a cross-entropy as the kernels read them: rows of
/// consecutive values `stride` apart, from `offset`.
struct Logits {
    /// Keeps the storage alive.
    tensor: Tensor,
    code: u32,
    base: u64,
    bytes: usize,
    offset: usize,
    stride: usize,
}

impl Cuda {
    fn logits(&self, x: &Tensor) -> Result<Logits> {
        let name = CrossEntropy::NAME;
        let (n, vocab) = (x.shape()[0], x.shape()[1]);
        let x = if x.strides()[1] == 1 || vocab <= 1 {
            x.clone()
        } else {
            Run::<Contiguous>::forward(self, &Contiguous, &[x])?
        };
        let storage = self.storage(name, &x)?;
        let (base, bytes) = (storage.pointer(), kapel_core::Storage::len(storage));
        let stride = if n > 1 { x.strides()[0] } else { vocab };
        Ok(Logits {
            code: float_code(name, x.dtype())?,
            base,
            bytes,
            offset: x.layout().offset(),
            stride,
            tensor: x,
        })
    }
}

/// One block of 256 threads per row; the logits are read in their own type,
/// and their gradient written in it. Logits that are the first columns of
/// wider rows (a vocabulary padded for the products) get a gradient of the
/// same layout, the padding columns zero: the wider gradient is there for
/// the products that follow.
impl Run<CrossEntropy> for Cuda {
    fn forward(&self, op: &CrossEntropy, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = CrossEntropy::NAME;
        let signature = op.infer(inputs)?;
        let (n, vocab) = (inputs[0].shape()[0], inputs[0].shape()[1]);
        let (t, _keep) = self.targets(inputs[1], vocab)?;
        let x = self.logits(inputs[0])?;
        self.output(&signature.shape, false, |out, len| {
            self.launch(
                "cross_entropy_rows",
                n.checked_mul(256).ok_or(Error::Overflow { op: name })?,
                Args::new()
                    .u32(x.code)
                    .slice(x.base, x.bytes)
                    .u32(small(name, x.offset)?)
                    .u32(small(name, x.stride)?)
                    .slice(t, n)
                    .u32(small(name, vocab)?)
                    .slice(out, len),
            )
        })
    }

    fn backward(
        &self,
        op: &CrossEntropy,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let name = CrossEntropy::NAME;
        op.infer_backward(inputs, grad)?;
        let (n, vocab) = (inputs[0].shape()[0], inputs[0].shape()[1]);
        let (t, _keep) = self.targets(inputs[1], vocab)?;
        let x = self.logits(inputs[0])?;
        let g = self.widened(name, grad)?;
        let g = self.flat(name, &g)?;
        let dtype = inputs[0].dtype();
        let ko = float_code(name, dtype)?;
        let size = dtype.block_bytes();
        let padded = x.offset == 0 && x.stride > vocab && x.bytes >= n * x.stride * size;
        let width = if padded { x.stride } else { vocab };
        let gx = self.output_as(&[n, width], dtype, false, |out, len| {
            self.launch(
                "cross_entropy_rows_backward",
                n.checked_mul(256).ok_or(Error::Overflow { op: name })?,
                Args::new()
                    .u32(x.code)
                    .slice(x.base, x.bytes)
                    .u32(small(name, x.offset)?)
                    .u32(small(name, x.stride)?)
                    .slice(t, n)
                    .flat(&g)
                    .u32(small(name, vocab)?)
                    .u32(small(name, width)?)
                    .u32(ko)
                    .slice(out, dtype.storage_bytes(len)?),
            )
        })?;
        drop(x.tensor);
        let gx = if padded { gx.narrow(1, 0, vocab)? } else { gx };
        Ok(vec![Some(gx), None])
    }
}

impl Cuda {
    /// The device address of a tensor of this backend that may be written in
    /// place: its storage is its only handle's.
    fn writable(&self, op: &'static str, t: &mut Tensor) -> Result<u64> {
        let storage = t
            .storage_mut()
            .and_then(|s| s.as_any_mut())
            .and_then(|any| any.downcast_mut::<CudaStorage>())
            .ok_or_else(|| Error::InvalidArgument {
                op,
                message: "not a storage of this backend that can be written".into(),
            })?;
        Ok(storage.pointer())
    }
}

impl Cuda {
    /// The device address of a state written in place, its copy on the host
    /// forgotten.
    fn state(&self, op: &'static str, t: &mut Tensor) -> Result<(u64, usize)> {
        let storage = t
            .storage_mut()
            .and_then(|s| s.as_any_mut())
            .and_then(|any| any.downcast_mut::<CudaStorage>())
            .ok_or_else(|| Error::InvalidArgument {
                op,
                message: "the state is not a storage of this backend".into(),
            })?;
        storage.written();
        Ok((storage.pointer(), kapel_core::Storage::len(&*storage)))
    }
}

impl kapel_ops::Recurrent for Cuda {
    /// One thread per channel, the rows in order.
    fn causal_conv(
        &self,
        state: &mut Tensor,
        x: &Tensor,
        weight: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor> {
        let op = kapel_ops::CAUSAL_CONV;
        let (shape, dtype) = kapel_ops::check_conv(state, x, weight, slots)?;
        let n = state.shape()[0];
        let rows = self.indices(op, slots, n)?;
        let (c, taps) = (weight.shape()[0], weight.shape()[1]);
        let ks = float_code(op, state.dtype())?;
        let ko = float_code(op, dtype)?;
        let xa = self.float_arg(op, x, x.layout())?;
        let wa = self.float_arg(op, weight, weight.layout())?;
        let (list, _keep) = self.row_list(op, slots, &rows)?;
        let (pointer, bytes) = self.state(op, state)?;
        self.output_as(&shape, dtype, false, |out, count| {
            let out_bytes = dtype.storage_bytes(count)?;
            self.launch(
                "causal_conv",
                c,
                Args::new()
                    .float(&xa)
                    .float(&wa)
                    .slice(list, rows.len())
                    .u32(small(op, c)?)
                    .u32(small(op, taps)?)
                    .u32(ks)
                    .slice(pointer, bytes)
                    .u32(ko)
                    .slice(out, out_bytes),
            )
        })
    }

    /// One thread per value head and column of its states, the rows in
    /// order.
    fn delta_rule(
        &self,
        state: &mut Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        slots: &Tensor,
        layout: kapel_ops::HeadLayout,
    ) -> Result<Tensor> {
        let op = kapel_ops::DELTA_RULE;
        let (shape, dtype) = kapel_ops::check_delta(state, q, k, v, g, beta, slots)?;
        let n = state.shape()[0];
        let rows = self.indices(op, slots, n)?;
        let (hk, dk) = (q.shape()[1], q.shape()[2]);
        let (h, dv) = (v.shape()[1], v.shape()[2]);
        if dk > 256 {
            return Err(Error::InvalidArgument {
                op,
                message: format!("key heads of {dk}: at most 256 on the device"),
            });
        }
        let ko = float_code(op, dtype)?;
        let arg = |t: &Tensor| self.float_arg(op, t, t.layout());
        let (qa, ka, va, ga, ba) = (arg(q)?, arg(k)?, arg(v)?, arg(g)?, arg(beta)?);
        let (list, _keep) = self.row_list(op, slots, &rows)?;
        let (pointer, bytes) = self.state(op, state)?;
        self.output_as(&shape, dtype, false, |out, count| {
            let out_bytes = dtype.storage_bytes(count)?;
            self.launch(
                "delta_rule",
                h * dv.div_ceil(32) * 256,
                Args::new()
                    .float(&qa)
                    .float(&ka)
                    .float(&va)
                    .float(&ga)
                    .float(&ba)
                    .slice(list, rows.len())
                    .u32(small(op, h)?)
                    .u32(small(op, hk)?)
                    .u32(small(op, dk)?)
                    .u32(small(op, dv)?)
                    .u32(u32::from(layout == kapel_ops::HeadLayout::Tiled))
                    .slice(pointer, bytes / 4)
                    .u32(ko)
                    .slice(out, out_bytes),
            )
        })
    }
}

impl kapel_ops::Update for Cuda {
    /// One thread per element, the gradient read in its type through its
    /// view.
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &kapel_ops::AdamWStep,
    ) -> Result<()> {
        let op = kapel_ops::ADAMW;
        kapel_ops::check_adamw(param, m, v, grad)?;
        let n = param.numel();
        let g = self.float_arg(op, grad, grad.layout())?;
        let (pp, mp, vp) = (
            self.writable(op, param)?,
            self.writable(op, m)?,
            self.writable(op, v)?,
        );
        self.launch(
            "adamw",
            n,
            Args::new()
                .slice(pp, n)
                .slice(mp, n)
                .slice(vp, n)
                .float(&g)
                .f32(step.beta1 as f32)
                .f32(step.beta2 as f32)
                .f32((1.0 - step.lr * step.weight_decay) as f32)
                .f32((step.lr / step.correction1) as f32)
                .f32(step.correction2 as f32)
                .f32(step.eps as f32)
                .f32(step.grad_scale as f32),
        )
    }

    /// One block of 256 threads per block of the moments, one thread per
    /// element; the block's scales from its largest moments.
    fn adamw8bit(
        &self,
        param: &mut Tensor,
        moments: &mut kapel_ops::Moments8,
        grad: &Tensor,
        maps: &Tensor,
        step: &kapel_ops::AdamWStep,
    ) -> Result<()> {
        let op = kapel_ops::ADAMW8;
        kapel_ops::check_adamw8bit(param, moments, grad, maps)?;
        let n = param.numel();
        let blocks = n.div_ceil(kapel_ops::BLOCK8);
        let g = self.float_arg(op, grad, grad.layout())?;
        let table = self.flat(op, maps)?;
        let kapel_ops::Moments8 {
            m,
            v,
            m_scale,
            v_scale,
        } = moments;
        let (pp, mp, vp) = (
            self.writable(op, param)?,
            self.writable(op, m)?,
            self.writable(op, v)?,
        );
        let (msp, vsp) = (self.writable(op, m_scale)?, self.writable(op, v_scale)?);
        self.launch(
            "adamw8bit",
            blocks * kapel_ops::BLOCK8,
            Args::new()
                .slice(pp, n)
                .slice(mp, n)
                .slice(vp, n)
                .slice(msp, blocks)
                .slice(vsp, blocks)
                .float(&g)
                .flat(&table)
                .f32(step.beta1 as f32)
                .f32(step.beta2 as f32)
                .f32((1.0 - step.lr * step.weight_decay) as f32)
                .f32((step.lr / step.correction1) as f32)
                .f32(step.correction2 as f32)
                .f32(step.eps as f32)
                .f32(step.grad_scale as f32),
        )
    }
}
