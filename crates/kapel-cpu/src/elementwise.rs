//! Element-wise operations and conversions.

use std::f32::consts::PI;
use std::sync::Arc;

use kapel_core::{Error, HostStorage, Layout, Result, Tensor};
use kapel_ops::{
    Affine, Binary, BinaryKind, Cast, Concat, Contiguous, Glu, Operation, Run, Unary, UnaryKind,
};

use crate::backend::{CHUNK, Cpu};
use crate::io::{self, broadcast_offsets};
use crate::math;

/// Refuses, before any work starts, a function this backend does not know:
/// the enumerations of `kapel-ops` may grow.
fn check_unary(kind: UnaryKind) -> Result<()> {
    match kind {
        UnaryKind::Neg
        | UnaryKind::Exp
        | UnaryKind::Log
        | UnaryKind::Sqrt
        | UnaryKind::Rsqrt
        | UnaryKind::Square
        | UnaryKind::Abs
        | UnaryKind::Sigmoid
        | UnaryKind::Tanh
        | UnaryKind::Relu
        | UnaryKind::Silu
        | UnaryKind::GeluTanh
        | UnaryKind::GeluErf => Ok(()),
        kind => Err(Error::InvalidArgument {
            op: Unary::NAME,
            message: format!("the CPU backend does not implement {kind:?}"),
        }),
    }
}

/// As [`check_unary`], for binary functions.
fn check_binary(kind: BinaryKind) -> Result<()> {
    match kind {
        BinaryKind::Add
        | BinaryKind::Sub
        | BinaryKind::Mul
        | BinaryKind::Div
        | BinaryKind::Max
        | BinaryKind::Min => Ok(()),
        kind => Err(Error::InvalidArgument {
            op: Binary::NAME,
            message: format!("the CPU backend does not implement {kind:?}"),
        }),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn unary_value(kind: UnaryKind, x: f32) -> f32 {
    match kind {
        UnaryKind::Neg => -x,
        UnaryKind::Exp => x.exp(),
        UnaryKind::Log => x.ln(),
        UnaryKind::Sqrt => x.sqrt(),
        UnaryKind::Rsqrt => 1.0 / x.sqrt(),
        UnaryKind::Square => x * x,
        UnaryKind::Abs => x.abs(),
        UnaryKind::Sigmoid => sigmoid(x),
        UnaryKind::Tanh => x.tanh(),
        UnaryKind::Relu => x.max(0.0),
        UnaryKind::Silu => x * sigmoid(x),
        UnaryKind::GeluTanh => {
            let c = (2.0 / PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
        }
        UnaryKind::GeluErf => (f64::from(x) * kapel_ops::special::normal_cdf(f64::from(x))) as f32,
        _ => unreachable!("refused by `check_unary`"),
    }
}

fn unary_derivative(kind: UnaryKind, x: f32) -> f32 {
    match kind {
        UnaryKind::Neg => -1.0,
        UnaryKind::Exp => x.exp(),
        UnaryKind::Log => 1.0 / x,
        UnaryKind::Sqrt => 0.5 / x.sqrt(),
        UnaryKind::Rsqrt => -0.5 / (x * x.sqrt()),
        UnaryKind::Square => 2.0 * x,
        UnaryKind::Abs => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryKind::Sigmoid => {
            let s = sigmoid(x);
            s * (1.0 - s)
        }
        UnaryKind::Tanh => 1.0 - x.tanh().powi(2),
        UnaryKind::Relu => f32::from(u8::from(x > 0.0)),
        UnaryKind::Silu => {
            let s = sigmoid(x);
            s + x * s * (1.0 - s)
        }
        UnaryKind::GeluTanh => {
            let c = (2.0 / PI).sqrt();
            let t = (c * (x + 0.044_715 * x * x * x)).tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044_715 * x * x)
        }
        UnaryKind::GeluErf => {
            let x = f64::from(x);
            let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            (kapel_ops::special::normal_cdf(x) + x * phi) as f32
        }
        _ => unreachable!("refused by `check_unary`"),
    }
}

impl Run<Unary> for Cpu {
    fn forward(&self, op: &Unary, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        check_unary(op.kind)?;
        let x = io::contiguous(Unary::NAME, inputs[0])?;
        let isa = self.isa;
        io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
            self.fill(out, CHUNK, |start, part| {
                let x = &x[start..start + part.len()];
                match op.kind {
                    UnaryKind::Exp => {
                        part.copy_from_slice(x);
                        math::exp_in_place(isa, part);
                    }
                    UnaryKind::Silu => math::silu(isa, x, part),
                    UnaryKind::GeluTanh => math::gelu(isa, x, part),
                    kind => part
                        .iter_mut()
                        .zip(x)
                        .for_each(|(y, &x)| *y = unary_value(kind, x)),
                }
            });
        })
    }

    fn backward(
        &self,
        op: &Unary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        check_unary(op.kind)?;
        let x = io::contiguous(Unary::NAME, inputs[0])?;
        let g = io::contiguous(Unary::NAME, grad)?;
        let mut out = vec![0.0; x.len()];
        let isa = self.isa;
        self.fill(&mut out, CHUNK, |start, part| {
            let range = start..start + part.len();
            let (x, g) = (&x[range.clone()], &g[range]);
            match op.kind {
                UnaryKind::Silu => math::silu_backward(isa, x, g, part),
                UnaryKind::GeluTanh => math::gelu_backward(isa, x, g, part),
                kind => {
                    for ((y, &x), &g) in part.iter_mut().zip(x).zip(g) {
                        *y = g * unary_derivative(kind, x);
                    }
                }
            }
        });
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            &signature.shape,
            signature.dtype,
        )?)])
    }
}

impl Run<Glu> for Cpu {
    /// One task per block of rows; the activation of the gates is vectorized
    /// for SiLU, as for [`Unary`].
    fn forward(&self, op: &Glu, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        check_unary(op.activation)?;
        let x = io::contiguous(Glu::NAME, inputs[0])?;
        let n = *signature.shape.last().expect("rank checked");
        let isa = self.isa;
        io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
            if n == 0 {
                return;
            }
            let rows = (CHUNK / n).max(1);
            self.fill(out, rows * n, |start, part| {
                let first = start / n;
                for (r, y) in part.chunks_exact_mut(n).enumerate() {
                    let row = &x[(first + r) * 2 * n..][..2 * n];
                    let (gate, value) = row.split_at(n);
                    match op.activation {
                        UnaryKind::Silu => math::silu(isa, gate, y),
                        kind => y
                            .iter_mut()
                            .zip(gate)
                            .for_each(|(y, &a)| *y = unary_value(kind, a)),
                    }
                    y.iter_mut().zip(value).for_each(|(y, &b)| *y *= b);
                }
            });
        })
    }

    fn backward(&self, op: &Glu, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        check_unary(op.activation)?;
        let x = io::contiguous(Glu::NAME, inputs[0])?;
        let g = io::contiguous(Glu::NAME, grad)?;
        let n = *signature.shape.last().expect("rank checked");
        let mut out = vec![0.0; x.len()];
        if n > 0 {
            for ((row, y), g) in x
                .chunks_exact(2 * n)
                .zip(out.chunks_exact_mut(2 * n))
                .zip(g.chunks_exact(n))
            {
                for j in 0..n {
                    let a = row[j];
                    y[j] = g[j] * row[n + j] * unary_derivative(op.activation, a);
                    y[n + j] = g[j] * unary_value(op.activation, a);
                }
            }
        }
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            inputs[0].dtype(),
        )?)])
    }
}

fn binary(kind: BinaryKind, a: f32, b: f32) -> (f32, f32, f32) {
    match kind {
        BinaryKind::Add => (a + b, 1.0, 1.0),
        BinaryKind::Sub => (a - b, 1.0, -1.0),
        BinaryKind::Mul => (a * b, b, a),
        BinaryKind::Div => (a / b, 1.0 / b, -a / (b * b)),
        BinaryKind::Max if a >= b => (a, 1.0, 0.0),
        BinaryKind::Max => (b, 0.0, 1.0),
        BinaryKind::Min if a <= b => (a, 1.0, 0.0),
        BinaryKind::Min => (b, 0.0, 1.0),
        _ => unreachable!("refused by `check_binary`"),
    }
}

/// `y[i] = f(a[i], b[i])`.
#[inline(always)]
fn same_shape(y: &mut [f32], a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) {
    for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
        *y = f(a, b);
    }
}

/// Sums a full-size gradient back to the shape of a broadcast input, in a fixed
/// order: the same bits whatever the number of threads.
fn reduce_to(full: &[f32], offsets: &[usize], len: usize) -> Vec<f32> {
    let mut out = vec![0.0; len];
    for (&value, &at) in full.iter().zip(offsets) {
        out[at] += value;
    }
    out
}

impl Run<Binary> for Cpu {
    fn forward(&self, op: &Binary, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        check_binary(op.kind)?;
        // Two inputs of the output's shape, in order: the common case of the
        // residual sums and gated products, without the offsets.
        if inputs
            .iter()
            .all(|t| t.shape() == signature.shape.as_slice())
        {
            let a = io::contiguous(Binary::NAME, inputs[0])?;
            let b = io::contiguous(Binary::NAME, inputs[1])?;
            return io::output(&self.buffers, &signature.shape, signature.dtype, |out| {
                self.fill(out, CHUNK, |start, part| {
                    let (a, b) = (&a[start..][..part.len()], &b[start..][..part.len()]);
                    match op.kind {
                        BinaryKind::Add => same_shape(part, a, b, |a, b| a + b),
                        BinaryKind::Sub => same_shape(part, a, b, |a, b| a - b),
                        BinaryKind::Mul => same_shape(part, a, b, |a, b| a * b),
                        BinaryKind::Div => same_shape(part, a, b, |a, b| a / b),
                        kind => same_shape(part, a, b, |a, b| binary(kind, a, b).0),
                    }
                });
            });
        }
        let (a, b) = (
            io::strided(Binary::NAME, inputs[0])?,
            io::strided(Binary::NAME, inputs[1])?,
        );
        let (ia, ib) = (
            broadcast_offsets(&a.layout, &signature.shape)?,
            broadcast_offsets(&b.layout, &signature.shape)?,
        );
        let mut out = vec![0.0; ia.len()];
        self.fill(&mut out, CHUNK, |start, part| {
            for (n, y) in part.iter_mut().enumerate() {
                *y = binary(op.kind, a.data[ia[start + n]], b.data[ib[start + n]]).0;
            }
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Binary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        check_binary(op.kind)?;
        let g = io::contiguous(Binary::NAME, grad)?;
        let (a, b) = (
            io::strided(Binary::NAME, inputs[0])?,
            io::strided(Binary::NAME, inputs[1])?,
        );
        let (ia, ib) = (
            broadcast_offsets(&a.layout, &signature.shape)?,
            broadcast_offsets(&b.layout, &signature.shape)?,
        );
        let mut grads = Vec::new();
        for (side, input) in inputs.iter().take(2).enumerate() {
            // Each input's gradient, computed at full size, then summed back.
            let mut full = vec![0.0; g.len()];
            self.fill(&mut full, CHUNK, |start, part| {
                for (n, y) in part.iter_mut().enumerate() {
                    let at = start + n;
                    let (_, da, db) = binary(op.kind, a.data[ia[at]], b.data[ib[at]]);
                    *y = g[at] * if side == 0 { da } else { db };
                }
            });
            let shape = input.shape();
            let values = if shape == signature.shape.as_slice() {
                full
            } else {
                let compact = Layout::contiguous(shape)?;
                reduce_to(
                    &full,
                    &broadcast_offsets(&compact, &signature.shape)?,
                    compact.numel(),
                )
            };
            grads.push(Some(io::write(
                &self.buffers,
                &values,
                shape,
                signature.dtype,
            )?));
        }
        Ok(grads)
    }
}

impl Run<Affine> for Cpu {
    fn forward(&self, op: &Affine, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Affine::NAME, inputs[0])?;
        let (scale, bias) = (op.scale as f32, op.bias as f32);
        let mut out = vec![0.0; x.len()];
        self.fill(&mut out, CHUNK, |start, part| {
            for (y, &x) in part.iter_mut().zip(&x[start..]) {
                *y = scale * x + bias;
            }
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Affine,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let g = io::contiguous(Affine::NAME, grad)?;
        let scale = op.scale as f32;
        let out: Vec<f32> = g.iter().map(|g| scale * g).collect();
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            &signature.shape,
            signature.dtype,
        )?)])
    }
}

impl Run<Cast> for Cpu {
    /// Conversions go through the exact `f64` value of each element, as the
    /// reference does: they are rare, and must round exactly as it does.
    fn forward(&self, op: &Cast, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let values = inputs[0].to_f64_vec()?;
        let size = op.dtype.block_bytes();
        let mut storage = HostStorage::new(values.len() * size);
        for (&value, out) in values
            .iter()
            .zip(storage.bytes_mut().chunks_exact_mut(size))
        {
            op.dtype.encode(value, out)?;
        }
        Tensor::from_storage(
            Arc::new(storage),
            Layout::contiguous(&signature.shape)?,
            op.dtype,
        )
    }

    fn backward(
        &self,
        op: &Cast,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        // Between float types the gradient passes through, converted to the
        // input's type; an integer input or output receives none.
        let dtype = inputs[0].dtype();
        if !op.differentiable(0) || !dtype.is_float() {
            return Ok(vec![None]);
        }
        let g = io::contiguous(Cast::NAME, grad)?;
        io::supported(Cast::NAME, dtype)?;
        Ok(vec![Some(io::write(
            &self.buffers,
            &g,
            inputs[0].shape(),
            dtype,
        )?)])
    }
}

impl Run<Contiguous> for Cpu {
    fn forward(&self, op: &Contiguous, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        inputs[0].contiguous()
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
        let g = io::contiguous(Contiguous::NAME, grad)?;
        io::supported(Contiguous::NAME, dtype)?;
        Ok(vec![Some(io::write(
            &self.buffers,
            &g,
            inputs[0].shape(),
            dtype,
        )?)])
    }
}

impl Run<Concat> for Cpu {
    fn forward(&self, op: &Concat, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        Tensor::concat(inputs, op.dim)
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
                if !x.dtype().is_float() {
                    return Ok(None);
                }
                let g = io::contiguous(Concat::NAME, &slice)?;
                io::supported(Concat::NAME, x.dtype())?;
                Ok(Some(io::write(&self.buffers, &g, x.shape(), x.dtype())?))
            })
            .collect()
    }
}
