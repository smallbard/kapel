//! Rotary position embeddings.

use kapel_core::{Result, Tensor};
use kapel_ops::{Operation, Rope, RopeLayout, Run};

use crate::Reference;
use crate::array::{Array, broadcast_map};

/// Turns every vector of `x` by its position's angles; `direction` is 1 for the
/// rotation, −1 for its inverse (the backward pass, which also applies the
/// factor `s`).
fn rotate(op: &Rope, x: &Array, positions: &Tensor, direction: f64) -> Result<Array> {
    let d = *x.shape.last().expect("checked by infer");
    let r = op.rotated(d);
    let frequencies = op.frequencies(r);
    let factor = op.attention_factor();
    let lead = &x.shape[..x.shape.len() - 1];
    let p = Array::of(positions)?;
    let map = broadcast_map(&p.shape, lead);
    let mut out = x.clone();
    for (vector, &at) in out.data.chunks_mut(d).zip(&map) {
        let position = p.data[at];
        for (j, &w) in frequencies.iter().enumerate() {
            let (a, b) = match op.layout {
                RopeLayout::Half => (j, j + r / 2),
                RopeLayout::Interleaved => (2 * j, 2 * j + 1),
            };
            let (sin, cos) = (direction * position * w).sin_cos();
            let (va, vb) = (vector[a], vector[b]);
            vector[a] = factor * (va * cos - vb * sin);
            vector[b] = factor * (vb * cos + va * sin);
        }
    }
    Ok(out)
}

impl Run<Rope> for Reference {
    fn forward(&self, op: &Rope, inputs: &[&Tensor]) -> Result<Tensor> {
        op.infer(inputs)?;
        rotate(op, &Array::of(inputs[0])?, inputs[1], 1.0)?.tensor()
    }

    /// The rotation is orthogonal up to the factor `s`: its transpose turns the
    /// other way, and keeps `s`.
    fn backward(
        &self,
        op: &Rope,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        Ok(vec![
            Some(rotate(op, &Array::of(grad)?, inputs[1], -1.0)?.tensor()?),
            None,
        ])
    }
}
