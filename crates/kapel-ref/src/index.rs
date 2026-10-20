//! Gathering and scattering rows.

use kapel_core::{Result, Tensor};
use kapel_ops::{GatherRows, Operation, Run, ScatterAddRows};

use crate::Reference;
use crate::array::{Array, indices};

/// `y[i] = src[idx[i]]`, rows of `width` elements.
fn gather(src: &Array, rows: &[usize], width: usize, shape: &[usize]) -> Array {
    let data = rows
        .iter()
        .flat_map(|&r| src.data[r * width..(r + 1) * width].iter().copied());
    Array::new(shape, data.collect())
}

/// `y[idx[i]] += src[i]`, rows of `width` elements.
fn scatter(src: &Array, rows: &[usize], width: usize, shape: &[usize]) -> Array {
    let mut out = Array::zeros(shape);
    for (i, &r) in rows.iter().enumerate() {
        for c in 0..width {
            out.data[r * width + c] += src.data[i * width + c];
        }
    }
    out
}

impl Run<GatherRows> for Reference {
    fn forward(&self, op: &GatherRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let src = Array::of(inputs[0])?;
        let rows = indices(GatherRows::NAME, inputs[1], src.shape[0])?;
        let width = src.shape[1..].iter().product();
        gather(&src, &rows, width, &signature.shape).tensor()
    }

    fn backward(
        &self,
        op: &GatherRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let shape = inputs[0].shape();
        let rows = indices(GatherRows::NAME, inputs[1], shape[0])?;
        let width = shape[1..].iter().product();
        Ok(vec![
            Some(scatter(&Array::of(grad)?, &rows, width, shape).tensor()?),
            None,
        ])
    }
}

impl Run<ScatterAddRows> for Reference {
    fn forward(&self, op: &ScatterAddRows, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let src = Array::of(inputs[0])?;
        let rows = indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        let width = signature.shape[1..].iter().product();
        scatter(&src, &rows, width, &signature.shape).tensor()
    }

    fn backward(
        &self,
        op: &ScatterAddRows,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let rows = indices(ScatterAddRows::NAME, inputs[1], op.rows)?;
        let width = signature.shape[1..].iter().product();
        let gathered = gather(&Array::of(grad)?, &rows, width, inputs[0].shape());
        Ok(vec![Some(gathered.tensor()?), None])
    }
}
