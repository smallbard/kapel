//! Element-wise operations and conversions.

use std::f64::consts::PI;
use std::sync::Arc;

use kapel_core::{HostStorage, Layout, Result, Tensor};
use kapel_ops::{Affine, Binary, BinaryKind, Cast, Operation, Run, Unary, UnaryKind};

use crate::Reference;
use crate::array::{Array, broadcast_map, sum_to};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `f(x)` and `f'(x)`.
fn unary(kind: UnaryKind, x: f64) -> (f64, f64) {
    match kind {
        UnaryKind::Neg => (-x, -1.0),
        UnaryKind::Exp => (x.exp(), x.exp()),
        UnaryKind::Log => (x.ln(), 1.0 / x),
        UnaryKind::Sqrt => (x.sqrt(), 0.5 / x.sqrt()),
        UnaryKind::Rsqrt => (1.0 / x.sqrt(), -0.5 / (x * x.sqrt())),
        UnaryKind::Square => (x * x, 2.0 * x),
        UnaryKind::Abs => (
            x.abs(),
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            },
        ),
        UnaryKind::Sigmoid => {
            let s = sigmoid(x);
            (s, s * (1.0 - s))
        }
        UnaryKind::Tanh => {
            let t = x.tanh();
            (t, 1.0 - t * t)
        }
        UnaryKind::Relu => (x.max(0.0), if x > 0.0 { 1.0 } else { 0.0 }),
        UnaryKind::Silu => {
            let s = sigmoid(x);
            (x * s, s + x * s * (1.0 - s))
        }
        UnaryKind::GeluErf => {
            let phi = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
            let cdf = kapel_ops::special::normal_cdf(x);
            (x * cdf, cdf + x * phi)
        }
        UnaryKind::GeluTanh => {
            let c = (2.0 / PI).sqrt();
            let t = (c * (x + 0.044_715 * x * x * x)).tanh();
            let dt = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044_715 * x * x);
            (0.5 * x * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * x * dt)
        }
        _ => unreachable!("every unary function is defined above"),
    }
}

impl Run<Unary> for Reference {
    fn forward(&self, op: &Unary, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let data = x.data.iter().map(|&v| unary(op.kind, v).0).collect();
        Array::new(&x.shape, data).tensor()
    }

    fn backward(
        &self,
        op: &Unary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (x, g) = (Array::of(inputs[0])?, Array::of(grad)?);
        let data = x
            .data
            .iter()
            .zip(&g.data)
            .map(|(&v, &g)| g * unary(op.kind, v).1)
            .collect();
        Ok(vec![Some(Array::new(&x.shape, data).tensor()?)])
    }
}

/// `f(a, b)`, `∂f/∂a` and `∂f/∂b`.
fn binary(kind: BinaryKind, a: f64, b: f64) -> (f64, f64, f64) {
    match kind {
        BinaryKind::Add => (a + b, 1.0, 1.0),
        BinaryKind::Sub => (a - b, 1.0, -1.0),
        BinaryKind::Mul => (a * b, b, a),
        BinaryKind::Div => (a / b, 1.0 / b, -a / (b * b)),
        BinaryKind::Max if a >= b => (a, 1.0, 0.0),
        BinaryKind::Max => (b, 0.0, 1.0),
        BinaryKind::Min if a <= b => (a, 1.0, 0.0),
        BinaryKind::Min => (b, 0.0, 1.0),
        _ => unreachable!("every binary function is defined above"),
    }
}

/// The two inputs of a binary operation, and for each output element, the
/// element of each input it reads.
struct Pair {
    shape: Vec<usize>,
    a: Array,
    b: Array,
    map_a: Vec<usize>,
    map_b: Vec<usize>,
}

fn broadcast_pair(op: &Binary, inputs: &[&Tensor]) -> Result<Pair> {
    let shape = op.infer(inputs)?.shape;
    let (a, b) = (Array::of(inputs[0])?, Array::of(inputs[1])?);
    let (map_a, map_b) = (
        broadcast_map(&a.shape, &shape),
        broadcast_map(&b.shape, &shape),
    );
    Ok(Pair {
        shape,
        a,
        b,
        map_a,
        map_b,
    })
}

impl Run<Binary> for Reference {
    fn forward(&self, op: &Binary, inputs: &[&Tensor]) -> Result<Tensor> {
        let Pair {
            shape,
            a,
            b,
            map_a,
            map_b,
        } = broadcast_pair(op, inputs)?;
        let data = map_a
            .iter()
            .zip(&map_b)
            .map(|(&i, &j)| binary(op.kind, a.data[i], b.data[j]).0)
            .collect();
        Array::new(&shape, data).tensor()
    }

    fn backward(
        &self,
        op: &Binary,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let Pair {
            shape,
            a,
            b,
            map_a,
            map_b,
        } = broadcast_pair(op, inputs)?;
        let g = Array::of(grad)?;
        let (mut ga, mut gb) = (Array::zeros(&shape), Array::zeros(&shape));
        for (n, (&i, &j)) in map_a.iter().zip(&map_b).enumerate() {
            let (_, da, db) = binary(op.kind, a.data[i], b.data[j]);
            ga.data[n] = g.data[n] * da;
            gb.data[n] = g.data[n] * db;
        }
        Ok(vec![
            Some(sum_to(&ga, &a.shape).tensor()?),
            Some(sum_to(&gb, &b.shape).tensor()?),
        ])
    }
}

impl Run<Affine> for Reference {
    fn forward(&self, op: &Affine, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let data = x.data.iter().map(|&v| op.scale * v + op.bias).collect();
        Array::new(&x.shape, data).tensor()
    }

    fn backward(
        &self,
        op: &Affine,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let g = Array::of(grad)?;
        let data = g.data.iter().map(|&v| op.scale * v).collect();
        Ok(vec![Some(Array::new(&g.shape, data).tensor()?)])
    }
}

impl Run<Cast> for Reference {
    fn forward(&self, op: &Cast, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let size = op.dtype.block_bytes();
        let mut storage = HostStorage::new(x.data.len() * size);
        for (&value, out) in x
            .data
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
        op.infer(inputs)?;
        if !op.differentiable(0) {
            return Ok(vec![None]);
        }
        Ok(vec![Some(Array::of(grad)?.tensor()?)])
    }
}
