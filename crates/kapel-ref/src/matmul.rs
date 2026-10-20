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
    let batch = shape::broadcast(&a.shape[..ra - 2], &b.shape[..rb - 2])?;
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
        op.infer(inputs)?;
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

impl Run<QuantizedMatmul> for Reference {
    fn forward(&self, op: &QuantizedMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let (x, w) = (Array::of(inputs[0])?, Array::of(inputs[1])?);
        let (n, k) = (w.shape[0], w.shape[1]);
        let mut out = Array::zeros(&signature.shape);
        for (row, y) in x.data.chunks(k).zip(out.data.chunks_mut(n)) {
            for (j, y) in y.iter_mut().enumerate() {
                *y = row
                    .iter()
                    .zip(&w.data[j * k..(j + 1) * k])
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
        op.infer(inputs)?;
        let (x, w, g) = (
            Array::of(inputs[0])?,
            Array::of(inputs[1])?,
            Array::of(grad)?,
        );
        let (n, k) = (w.shape[0], w.shape[1]);
        let mut gx = Array::zeros(&x.shape);
        for (gy, gx) in g.data.chunks(n).zip(gx.data.chunks_mut(k)) {
            for (j, &gy) in gy.iter().enumerate() {
                for (gx, &w) in gx.iter_mut().zip(&w.data[j * k..(j + 1) * k]) {
                    *gx += gy * w;
                }
            }
        }
        Ok(vec![Some(gx.tensor()?), None])
    }
}
