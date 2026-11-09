//! Mixtures of experts, as their definitions say.

use kapel_core::{Result, Tensor};
use kapel_ops::{ExpertMatmul, GatherColumns, Operation, Run, TopK};

use crate::Reference;
use crate::array::Array;
use crate::paged::index_values;

impl Run<TopK> for Reference {
    fn forward(&self, op: &TopK, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let e = *x.shape.last().expect("checked by infer");
        let mut out = Vec::with_capacity(signature.shape.iter().product());
        if e > 0 {
            for row in x.data.chunks_exact(e) {
                // The values compared as the backends compare them, in `f32`.
                let row: Vec<f32> = row.iter().map(|&v| v as f32).collect();
                out.extend(TopK::choose(op.k, &row));
            }
        }
        Tensor::from_slice(&out, &signature.shape)
    }

    fn backward(
        &self,
        op: &TopK,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None])
    }
}

impl Run<GatherColumns> for Reference {
    fn forward(&self, op: &GatherColumns, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Array::of(inputs[0])?;
        let idx = index_values(GatherColumns::NAME, inputs[1])?;
        let e = *x.shape.last().expect("checked by infer");
        GatherColumns::check(&idx, e)?;
        let k = *signature.shape.last().expect("a last dimension");
        let data = idx
            .iter()
            .enumerate()
            .map(|(i, &c)| x.data[i / k.max(1) * e + c])
            .collect();
        Array::new(&signature.shape, data).tensor()
    }

    fn backward(
        &self,
        op: &GatherColumns,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None, None])
    }
}

impl Run<ExpertMatmul> for Reference {
    fn forward(&self, op: &ExpertMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let (x, w) = (Array::of(inputs[0])?, Array::of(inputs[1])?);
        let idx = index_values(ExpertMatmul::NAME, inputs[2])?;
        op.check(&idx)?;
        let [t, k, n] = [signature.shape[0], signature.shape[1], signature.shape[2]];
        let width = w.shape[1];
        let per_route = x.shape.len() == 3;
        let mut y = Array::zeros(&signature.shape);
        for token in 0..t {
            for j in 0..k {
                let input = if per_route {
                    &x.data[(token * k + j) * width..][..width]
                } else {
                    &x.data[token * width..][..width]
                };
                let expert = idx[token * k + j];
                for row in 0..n {
                    let weights = &w.data[(expert * n + row) * width..][..width];
                    y.data[(token * k + j) * n + row] =
                        input.iter().zip(weights).map(|(a, b)| a * b).sum();
                }
            }
        }
        y.tensor()
    }

    fn backward(
        &self,
        op: &ExpertMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None, None, None])
    }
}
