//! Reductions.

use kapel_core::{Result, Tensor};
use kapel_ops::{Max, Mean, Operation, Run, Sum, shape};

use crate::Reference;
use crate::array::{Array, lines, strides};

/// For each element of `shape`, its position in the result of reducing `dims`
/// with `keepdim` (the positions are the same without it).
fn reduction_map(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let kept = shape::reduced(shape, dims, true);
    let (from, to) = (strides(shape), strides(&kept));
    (0..shape.iter().product())
        .map(|flat: usize| {
            (0..shape.len())
                .filter(|d| !dims.contains(d))
                .map(|d| flat / from[d] % shape[d] * to[d])
                .sum()
        })
        .collect()
}

/// The sum over `dims`, and the number of elements summed into each result.
fn sum(x: &Array, dims: &[usize], out_shape: &[usize]) -> (Array, f64) {
    let mut out = Array::zeros(out_shape);
    for (value, at) in x.data.iter().zip(reduction_map(&x.shape, dims)) {
        out.data[at] += value;
    }
    let count = dims.iter().map(|&d| x.shape[d]).product::<usize>() as f64;
    (out, count)
}

/// Each element of `x` receives `factor` times the gradient of its result.
fn spread(x: &[usize], dims: &[usize], grad: &Tensor, factor: f64) -> Result<Tensor> {
    let g = Array::of(grad)?;
    let data = reduction_map(x, dims)
        .into_iter()
        .map(|at| g.data[at] * factor)
        .collect();
    Array::new(x, data).tensor()
}

impl Run<Sum> for Reference {
    fn forward(&self, op: &Sum, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        sum(&Array::of(inputs[0])?, &op.dims, &signature.shape)
            .0
            .tensor()
    }

    fn backward(&self, op: &Sum, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        Ok(vec![Some(spread(inputs[0].shape(), &op.dims, grad, 1.0)?)])
    }
}

impl Run<Mean> for Reference {
    fn forward(&self, op: &Mean, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let (mut out, count) = sum(&Array::of(inputs[0])?, &op.dims, &signature.shape);
        out.data.iter_mut().for_each(|v| *v /= count);
        out.tensor()
    }

    fn backward(
        &self,
        op: &Mean,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let count = op
            .dims
            .iter()
            .map(|&d| inputs[0].shape()[d])
            .product::<usize>() as f64;
        Ok(vec![Some(spread(
            inputs[0].shape(),
            &op.dims,
            grad,
            1.0 / count,
        )?)])
    }
}

/// The position of the first maximum of each line along `dim`.
fn argmax(x: &Array, dim: usize) -> Vec<usize> {
    let len = x.shape[dim];
    lines(&x.shape, dim)
        .map(|(start, step)| {
            (0..len).map(|i| start + i * step).fold(start, |best, at| {
                if x.data[at] > x.data[best] { at } else { best }
            })
        })
        .collect()
}

impl Run<Max> for Reference {
    fn forward(&self, op: &Max, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let data = argmax(&x, op.dim)
            .into_iter()
            .map(|at| x.data[at])
            .collect();
        Array::new(&signature.shape, data).tensor()
    }

    fn backward(&self, op: &Max, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (x, g) = (Array::of(inputs[0])?, Array::of(grad)?);
        let mut gx = Array::zeros(&x.shape);
        for (line, at) in argmax(&x, op.dim).into_iter().enumerate() {
            gx.data[at] = g.data[line];
        }
        Ok(vec![Some(gx.tensor()?)])
    }
}
