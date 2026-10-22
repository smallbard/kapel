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
    Affine, Attention, Binary, BinaryKind, Cast, Concat, Contiguous, GatherRows, LayerNorm,
    LogSoftmax, Matmul, Max, Mean, Operation, QuantizedMatmul, RmsNorm, Rope, RopeLayout, Run,
    ScatterAddRows, Softmax, Sum, Transfer, Unary, UnaryKind,
};

use crate::device::{Args, Cuda, CudaStorage, Flat, Table, small, smalls, view_of};

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
        let (b, i) = self.reduction_tables(op, shape, dims)?;
        self.output(out_shape, false, |out, n| {
            self.launch(
                "reduce_sum",
                n,
                Args::new()
                    .flat(x)
                    .table(&b)
                    .table(&i)
                    .f32(scale)
                    .slice(out, n),
            )
        })
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
    fn indices(&self, op: &'static str, tensor: &Tensor, size: usize) -> Result<Vec<usize>> {
        self.download(tensor)?
            .to_f64_vec()?
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

    /// `out[i] = src[rows[i]]`, rows of the given width.
    fn gather(
        &self,
        op: &'static str,
        src: &Tensor,
        rows: &[usize],
        width: usize,
        shape: &[usize],
    ) -> Result<Tensor> {
        let src = self.flat(op, src)?;
        let r = self.upload_u32(&smalls(op, rows)?)?;
        self.output(shape, false, |out, n| {
            self.launch(
                "gather_rows",
                n,
                Args::new()
                    .flat(&src)
                    .list(&r, rows.len())
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
}

impl Cuda {
    /// Each input in `f32` if it is in `bf16` or `f16`; the others as they are.
    fn widened_all(&self, op: &'static str, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        inputs.iter().map(|x| self.widened(op, x)).collect()
    }
}

/// [`Run`] for operations computed in `f32`: inputs widened, the output
/// rounded to the type of its signature, and each gradient to its input's.
macro_rules! in_f32 {
    ($($op:ty),* $(,)?) => {$(
        impl Run<$op> for Cuda {
            fn forward(&self, op: &$op, inputs: &[&Tensor]) -> Result<Tensor> {
                let name = <$op>::NAME;
                let dtype = op.infer(inputs)?.dtype;
                let wide = self.widened_all(name, inputs)?;
                let wide: Vec<&Tensor> = wide.iter().collect();
                let y = InF32::<$op>::forward_f32(self, op, &wide)?;
                self.narrowed(name, &y, dtype)
            }

            fn backward(
                &self,
                op: &$op,
                inputs: &[&Tensor],
                grad: &Tensor,
            ) -> Result<Vec<Option<Tensor>>> {
                let name = <$op>::NAME;
                op.infer_backward(inputs, grad)?;
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
        }
    )*};
}

in_f32!(
    Unary,
    Affine,
    Binary,
    QuantizedMatmul,
    Sum,
    Mean,
    Max,
    Softmax,
    LogSoftmax,
    LayerNorm,
    RmsNorm,
    Rope,
    ScatterAddRows,
);

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
        let signature = op.infer_backward(inputs, grad)?;
        let code = unary_code(Unary::NAME, op.kind)?;
        let x = self.flat(Unary::NAME, inputs[0])?;
        let g = self.flat(Unary::NAME, grad)?;
        let out = self.output(&signature.shape, false, |out, n| {
            self.launch(
                "unary_backward",
                n,
                Args::new().u32(code).flat(&x).flat(&g).slice(out, n),
            )
        })?;
        Ok(vec![Some(out)])
    }
}

impl Cuda {
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
        if on_device(x.dtype(), dtype) {
            let wide = self.widened(op, x)?;
            return match dtype {
                // `widened` copies 16-bit inputs; an `f32` one still needs its own.
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
        self.matmul(inputs[0], inputs[1], &signature.shape)
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
        Ok(vec![
            Some(self.summed_to(&ga, a.shape())?),
            Some(self.summed_to(&gb, b.shape())?),
        ])
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
        self.matmul(&x, &wt, &shape)?.reshape(&signature.shape)
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
        let mut grads = vec![Some(self.matmul(&g, &w, &shape)?.reshape(x.shape())?)];
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
        let x = self.flat(op, x)?;
        let g = self.flat(op, grad)?;
        let w = self.norm_vector(op, norm.weight, width, norm.default_weight)?;
        let gw_rows = self.alloc(x.len * 4, false)?;
        let gb_rows = self.alloc(x.len * 4, false)?;
        let gx = self.output(&shape, false, |out, n| {
            self.launch(
                "norm_rows_backward",
                warps(op, rows)?,
                Args::new()
                    .u32(norm.rms)
                    .flat(&x)
                    .flat(&w)
                    .flat(&g)
                    .f32(norm.offset)
                    .f32(norm.eps)
                    .u32(small(op, width)?)
                    .slice(out, n)
                    .list(&gw_rows, n)
                    .list(&gb_rows, n),
            )
        })?;
        let column_sum = |rows_of: &CudaStorage| {
            self.output(&[width], false, |out, n| {
                self.launch(
                    "column_sum",
                    n,
                    Args::new()
                        .list(rows_of, x.len)
                        .u32(small(op, rows)?)
                        .slice(out, n),
                )
            })
        };
        Ok((gx, column_sum(&gw_rows)?, column_sum(&gb_rows)?))
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
    fn rope(&self, op: &Rope, x: &Tensor, positions: &Tensor, direction: f64) -> Result<Tensor> {
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
        let constants = self.table("rope constants", &key, || {
            Ok(constants
                .iter()
                .flat_map(|&c| [c as u32, (c >> 32) as u32])
                .collect())
        })?;
        let signed = u32::from(positions.dtype() == DType::I32);
        let positions = self.materialize(name, positions, positions.layout())?;
        let entries = positions.numel();
        let p = self.storage(name, &positions)?;
        let half = r / 2;
        let table = self.output(&[entries * half * 2], false, |out, n| {
            self.launch(
                "rope_table",
                entries * half,
                Args::new()
                    .list(p, entries)
                    .u32(signed)
                    .slice(constants.storage.pointer(), half + 1)
                    .f32(direction as f32)
                    .slice(out, n),
            )
        })?;
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
        let x = self.flat(name, x)?;
        let table = self.flat(name, &table)?;
        let half_pairs = u32::from(op.layout == RopeLayout::Half);
        self.output(&shape, false, |out, n| {
            self.launch(
                "rope",
                n,
                Args::new()
                    .flat(&x)
                    .flat(&table)
                    .table(&entry)
                    .u32(small(name, d)?)
                    .u32(small(name, r)?)
                    .u32(half_pairs)
                    .slice(out, n),
            )
        })
    }
}

impl InF32<Rope> for Cuda {
    fn forward_f32(&self, op: &Rope, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        self.rope(op, inputs[0], inputs[1], 1.0)
    }

    fn backward_f32(
        &self,
        op: &Rope,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![Some(self.rope(op, grad, inputs[1], -1.0)?), None])
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
            DType::BF16 | DType::F16 => self.gather_16(inputs[0], &rows, width, &signature),
            // A quantized table is dequantized on the host, whole: kernels
            // that read the codes come with the products by quantized weights.
            dtype if dtype.is_quantized() => {
                let table = self.dequantized(inputs[0])?;
                self.gather(name, &table, &rows, width, &signature.shape)
            }
            _ => self.gather(name, inputs[0], &rows, width, &signature.shape),
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
        let r = self.upload_u32(&smalls(op, rows)?)?;
        self.output_as(&signature.shape, signature.dtype, false, |out, n| {
            self.launch(
                "gather_rows_16",
                n,
                Args::new()
                    .slice(pointer, src.numel())
                    .list(&r, rows.len())
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
        let out = self.gather(ScatterAddRows::NAME, grad, &rows, width, inputs[0].shape())?;
        Ok(vec![Some(out), None])
    }
}
