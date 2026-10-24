//! Losses.

use kapel_core::{Result, Tensor};
use kapel_ops::{CrossEntropy, Operation, Run};

use crate::Reference;
use crate::array::Array;

/// The column of each row's target, `None` for an ignored row.
fn targets(t: &Tensor, vocab: usize) -> Result<Vec<Option<usize>>> {
    t.to_f64_vec()?
        .into_iter()
        .map(|v| CrossEntropy::target(v as i64, vocab))
        .collect()
}

/// The softmax of row `x`, and the logarithm of its normalizer, after
/// subtracting the row's maximum.
fn softmax(x: &[f64]) -> (Vec<f64>, f64) {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = x.iter().map(|v| (v - max).exp()).sum();
    let p = x.iter().map(|v| (v - max).exp() / total).collect();
    (p, max + total.ln())
}

impl Run<CrossEntropy> for Reference {
    fn forward(&self, op: &CrossEntropy, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let vocab = x.shape[1];
        let t = targets(inputs[1], vocab)?;
        let y = t
            .iter()
            .enumerate()
            .map(|(n, t)| match t {
                None => 0.0,
                Some(t) => {
                    let row = &x.data[n * vocab..(n + 1) * vocab];
                    softmax(row).1 - row[*t]
                }
            })
            .collect();
        Array::new(&signature.shape, y).tensor()
    }

    /// `∂x[n, j] = g[n] · (softmax(x[n])[j] − [j = t[n]])`.
    fn backward(
        &self,
        op: &CrossEntropy,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let x = Array::of(inputs[0])?;
        let vocab = x.shape[1];
        let t = targets(inputs[1], vocab)?;
        let g = grad.to_f64_vec()?;
        let mut gx = Array::zeros(&x.shape);
        for (n, t) in t.iter().enumerate() {
            let Some(t) = t else { continue };
            let (p, _) = softmax(&x.data[n * vocab..(n + 1) * vocab]);
            for (j, p) in p.into_iter().enumerate() {
                let onehot = if j == *t { 1.0 } else { 0.0 };
                gx.data[n * vocab + j] = g[n] * (p - onehot);
            }
        }
        Ok(vec![Some(gx.tensor()?), None])
    }
}
