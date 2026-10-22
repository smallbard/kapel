//! Matrix products.

use kapel_core::{Result, Tensor};
use kapel_ops::{Matmul, Operation, QuantizedMatmul, Run, shape};

use crate::Reference;
use crate::array::{Array, broadcast_map, sum_to};

/// The batched product `a · b`, `[…, M, K] · […, K, N]`, batch dimensions
/// broadcast.
fn product(a: &Array, b: &Array) -> Result<Array> {
    let (ra, rb) = (a.shape.len(), b.shape.len());
    let (m, k, n) = (a.shape[ra - 2], a.shape[ra - 1], b.shape[rb - 1]);
    let batch = shape::broadcast(Matmul::NAME, &a.shape[..ra - 2], &b.shape[..rb - 2])?;
    let (map_a, map_b) = (
        broadcast_map(&a.shape[..ra - 2], &batch),
        broadcast_map(&b.shape[..rb - 2], &batch),
    );
    let mut shape = batch.clone();
    shape.extend([m, n]);
    let mut out = Array::zeros(&shape);
    for (t, (&ia, &ib)) in map_a.iter().zip(&map_b).enumerate() {
        let (a, b) = (
            &a.data[ia * m * k..][..m * k],
            &b.data[ib * k * n..][..k * n],
        );
        let out = &mut out.data[t * m * n..][..m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
    }
    Ok(out)
}

impl Run<Matmul> for Reference {
    fn forward(&self, op: &Matmul, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        product(&Array::of(inputs[0])?, &Array::of(inputs[1])?)?.tensor()
    }

    /// `∂a = g · bᵀ` and `∂b = aᵀ · g`, summed over broadcast batch dimensions.
    fn backward(
        &self,
        op: &Matmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let (a, b, g) = (
            Array::of(inputs[0])?,
            Array::of(inputs[1])?,
            Array::of(grad)?,
        );
        let ga = product(&g, &b.transpose_last())?;
        let gb = product(&a.transpose_last(), &g)?;
        Ok(vec![
            Some(sum_to(&ga, &a.shape).tensor()?),
            Some(sum_to(&gb, &b.shape).tensor()?),
        ])
    }
}

/// The weights of a quantized product, dequantized and their rows stacked.
fn stacked(weights: &[&Tensor]) -> Result<Array> {
    let k = weights[0].shape()[1];
    let mut data = Vec::new();
    for w in weights {
        data.extend(Array::of(w)?.data);
    }
    Ok(Array::new(&[data.len() / k.max(1), k], data))
}

impl Run<QuantizedMatmul> for Reference {
    fn forward(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        let w = stacked(&inputs[1..])?;
        let mut out = Array::zeros(&signature.shape);
        // Indexed rather than chunked: any of the sizes may be 0.
        let rows: usize = x.shape[..x.shape.len() - 1].iter().product();
        for r in 0..rows {
            let row = &x.data[r * k..][..k];
            for j in 0..n {
                out.data[r * n + j] = row
                    .iter()
                    .zip(&w.data[j * k..][..k])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        out.tensor()
    }

    /// `∂x = g · w`; the weights receive nothing.
    fn backward(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let (x, w, g) = (
            Array::of(inputs[0])?,
            stacked(&inputs[1..])?,
            Array::of(grad)?,
        );
        let k = inputs[1].shape()[1];
        let n = *signature.shape.last().expect("an output dimension");
        let mut gx = Array::zeros(&x.shape);
        // Indexed rather than chunked: any of the sizes may be 0.
        let rows: usize = x.shape[..x.shape.len() - 1].iter().product();
        for r in 0..rows {
            let gx = &mut gx.data[r * k..][..k];
            for j in 0..n {
                let gy = g.data[r * n + j];
                for (gx, &w) in gx.iter_mut().zip(&w.data[j * k..][..k]) {
                    *gx += gy * w;
                }
            }
        }
        let mut grads = vec![Some(gx.tensor()?)];
        grads.resize(inputs.len(), None);
        Ok(grads)
    }
}
