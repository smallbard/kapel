//! Softmax and normalizations.

use kapel_core::{Result, Tensor};
use kapel_ops::{LayerNorm, LogSoftmax, Operation, RmsNorm, Run, Softmax};

use crate::Reference;
use crate::array::{Array, lines};

/// The softmax of every line along `dim`, computed after subtracting the line's
/// maximum.
fn softmax(x: &Array, dim: usize) -> Array {
    let mut out = x.clone();
    let len = x.shape[dim];
    for (start, step) in lines(&x.shape, dim) {
        let at = |i: usize| start + i * step;
        let max = (0..len)
            .map(|i| x.data[at(i)])
            .fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = (0..len).map(|i| (x.data[at(i)] - max).exp()).sum();
        for i in 0..len {
            out.data[at(i)] = (x.data[at(i)] - max).exp() / total;
        }
    }
    out
}

impl Run<Softmax> for Reference {
    fn forward(&self, op: &Softmax, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        softmax(&Array::of(inputs[0])?, op.dim).tensor()
    }

    /// `∂x = y ⊙ (g − Σ g·y)` along each line.
    fn backward(
        &self,
        op: &Softmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (y, g) = (softmax(&Array::of(inputs[0])?, op.dim), Array::of(grad)?);
        let mut gx = y.clone();
        let len = y.shape[op.dim];
        for (start, step) in lines(&y.shape, op.dim) {
            let at = |i: usize| start + i * step;
            let dot: f64 = (0..len).map(|i| g.data[at(i)] * y.data[at(i)]).sum();
            for i in 0..len {
                gx.data[at(i)] = y.data[at(i)] * (g.data[at(i)] - dot);
            }
        }
        Ok(vec![Some(gx.tensor()?)])
    }
}

impl Run<LogSoftmax> for Reference {
    fn forward(&self, op: &LogSoftmax, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let mut y = softmax(&Array::of(inputs[0])?, op.dim);
        // Computed directly rather than as ln(softmax), to keep tiny
        // probabilities exact.
        let x = Array::of(inputs[0])?;
        let len = x.shape[op.dim];
        for (start, step) in lines(&x.shape, op.dim) {
            let at = |i: usize| start + i * step;
            let max = (0..len)
                .map(|i| x.data[at(i)])
                .fold(f64::NEG_INFINITY, f64::max);
            let log_total = (0..len)
                .map(|i| (x.data[at(i)] - max).exp())
                .sum::<f64>()
                .ln();
            for i in 0..len {
                y.data[at(i)] = x.data[at(i)] - max - log_total;
            }
        }
        y.tensor()
    }

    /// `∂x = g − softmax(x) · Σ g` along each line.
    fn backward(
        &self,
        op: &LogSoftmax,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (p, g) = (softmax(&Array::of(inputs[0])?, op.dim), Array::of(grad)?);
        let mut gx = g.clone();
        let len = p.shape[op.dim];
        for (start, step) in lines(&p.shape, op.dim) {
            let at = |i: usize| start + i * step;
            let total: f64 = (0..len).map(|i| g.data[at(i)]).sum();
            for i in 0..len {
                gx.data[at(i)] = g.data[at(i)] - p.data[at(i)] * total;
            }
        }
        Ok(vec![Some(gx.tensor()?)])
    }
}

/// The optional vector input `index`, or a constant.
fn vector(inputs: &[&Tensor], index: usize, width: usize, absent: f64) -> Result<Vec<f64>> {
    match inputs.get(index) {
        Some(tensor) => tensor.to_f64_vec(),
        None => Ok(vec![absent; width]),
    }
}

/// Per row: the normalized values `x̂` and the factor `1/σ` (or `1/rms`).
struct Normalized {
    hat: Vec<f64>,
    inverse: Vec<f64>,
    width: usize,
}

fn layer_normalize(x: &Array, eps: f64) -> Normalized {
    let width = *x.shape.last().expect("checked by infer");
    let mut hat = x.data.clone();
    let mut inverse = Vec::new();
    for row in hat.chunks_mut(width.max(1)) {
        let mean = row.iter().sum::<f64>() / width as f64;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (variance + eps).sqrt();
        row.iter_mut().for_each(|v| *v = (*v - mean) * inv);
        inverse.push(inv);
    }
    Normalized {
        hat,
        inverse,
        width,
    }
}

fn rms_normalize(x: &Array, eps: f64) -> Normalized {
    let width = *x.shape.last().expect("checked by infer");
    let mut hat = x.data.clone();
    let mut inverse = Vec::new();
    for row in hat.chunks_mut(width.max(1)) {
        let mean_square = row.iter().map(|v| v * v).sum::<f64>() / width as f64;
        let inv = 1.0 / (mean_square + eps).sqrt();
        row.iter_mut().for_each(|v| *v *= inv);
        inverse.push(inv);
    }
    Normalized {
        hat,
        inverse,
        width,
    }
}

impl Run<LayerNorm> for Reference {
    fn forward(&self, op: &LayerNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let n = layer_normalize(&x, op.eps);
        let (w, b) = (
            vector(inputs, 1, n.width, 1.0)?,
            vector(inputs, 2, n.width, 0.0)?,
        );
        let data = n
            .hat
            .iter()
            .enumerate()
            .map(|(i, h)| h * w[i % n.width] + b[i % n.width]);
        Array::new(&x.shape, data.collect()).tensor()
    }

    /// With `ĝ = g · weight`: `∂x = (ĝ − mean(ĝ) − x̂ · mean(ĝ·x̂)) / σ`,
    /// `∂weight = Σ g·x̂`, `∂bias = Σ g`.
    fn backward(
        &self,
        op: &LayerNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (x, g) = (Array::of(inputs[0])?, Array::of(grad)?);
        let n = layer_normalize(&x, op.eps);
        let width = n.width;
        let w = vector(inputs, 1, width, 1.0)?;
        let (mut gx, mut gw, mut gb) =
            (vec![0.0; x.data.len()], vec![0.0; width], vec![0.0; width]);
        for (row, inv) in n.inverse.iter().enumerate() {
            let range = row * width..(row + 1) * width;
            let (hat, g) = (&n.hat[range.clone()], &g.data[range.clone()]);
            let gh: Vec<f64> = (0..width).map(|c| g[c] * w[c]).collect();
            let mean_gh = gh.iter().sum::<f64>() / width as f64;
            let mean_ghh = gh.iter().zip(hat).map(|(a, b)| a * b).sum::<f64>() / width as f64;
            for c in 0..width {
                gx[range.start + c] = inv * (gh[c] - mean_gh - hat[c] * mean_ghh);
                gw[c] += g[c] * hat[c];
                gb[c] += g[c];
            }
        }
        let mut grads = vec![Some(Array::new(&x.shape, gx).tensor()?)];
        if inputs.len() > 1 {
            grads.push(Some(Array::new(&[width], gw).tensor()?));
        }
        if inputs.len() > 2 {
            grads.push(Some(Array::new(&[width], gb).tensor()?));
        }
        Ok(grads)
    }
}

impl Run<RmsNorm> for Reference {
    fn forward(&self, op: &RmsNorm, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let n = rms_normalize(&x, op.eps);
        let w = vector(inputs, 1, n.width, 1.0 - op.weight_offset)?;
        let data = n
            .hat
            .iter()
            .enumerate()
            .map(|(i, h)| h * (op.weight_offset + w[i % n.width]));
        Array::new(&x.shape, data.collect()).tensor()
    }

    /// With `ĝ = g · (offset + weight)`: `∂x = (ĝ − x̂ · mean(ĝ·x̂)) / rms`,
    /// `∂weight = Σ g·x̂`.
    fn backward(
        &self,
        op: &RmsNorm,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let (x, g) = (Array::of(inputs[0])?, Array::of(grad)?);
        let n = rms_normalize(&x, op.eps);
        let width = n.width;
        let w = vector(inputs, 1, width, 1.0 - op.weight_offset)?;
        let (mut gx, mut gw) = (vec![0.0; x.data.len()], vec![0.0; width]);
        for (row, inv) in n.inverse.iter().enumerate() {
            let range = row * width..(row + 1) * width;
            let (hat, g) = (&n.hat[range.clone()], &g.data[range.clone()]);
            let gh: Vec<f64> = (0..width)
                .map(|c| g[c] * (op.weight_offset + w[c]))
                .collect();
            let mean_ghh = gh.iter().zip(hat).map(|(a, b)| a * b).sum::<f64>() / width as f64;
            for c in 0..width {
                gx[range.start + c] = inv * (gh[c] - hat[c] * mean_ghh);
                gw[c] += g[c] * hat[c];
            }
        }
        let mut grads = vec![Some(Array::new(&x.shape, gx).tensor()?)];
        if inputs.len() > 1 {
            grads.push(Some(Array::new(&[width], gw).tensor()?));
        }
        Ok(grads)
    }
}
