//! Rotary position embeddings.

use kapel_core::{Layout, Result, Tensor};
use kapel_ops::{Operation, Rope, RopeLayout, Run};

use crate::backend::Cpu;
use crate::io;

/// Turns every vector of `x` by its position's angles; `direction` is 1 for the
/// rotation, −1 for its transpose (the backward pass).
///
/// Angles are computed in `f64` once per distinct position entry: positions in
/// the hundreds of thousands times a frequency near 1 would lose their phase in
/// `f32`.
fn rotate(
    cpu: &Cpu,
    op: &Rope,
    x: &[f32],
    shape: &[usize],
    positions: &Tensor,
    direction: f64,
) -> Result<Vec<f32>> {
    let d = *shape.last().expect("checked by infer");
    let r = op.rotated(d);
    let frequencies = op.frequencies(r);
    let factor = op.attention_factor();
    let entries = io::indices(Rope::NAME, positions, usize::MAX)?;
    let table: Vec<(f32, f32)> = entries
        .iter()
        .flat_map(|&p| {
            frequencies.iter().map(move |&w| {
                let (sin, cos) = (direction * p as f64 * w).sin_cos();
                ((factor * sin) as f32, (factor * cos) as f32)
            })
        })
        .collect();
    let lead = &shape[..shape.len() - 1];
    let entry_of: Vec<usize> = Layout::contiguous(positions.shape())?
        .broadcast_to(lead)?
        .offsets()
        .collect();
    let half = r / 2;
    let mut out = x.to_vec();
    cpu.fill(
        &mut out,
        (crate::backend::CHUNK / d.max(1)).max(1) * d,
        |start, part| {
            for (n, vector) in part.chunks_mut(d).enumerate() {
                let angles = &table[entry_of[start / d + n] * half..][..half];
                for (j, &(sin, cos)) in angles.iter().enumerate() {
                    let (a, b) = match op.layout {
                        RopeLayout::Half => (j, j + half),
                        RopeLayout::Interleaved => (2 * j, 2 * j + 1),
                    };
                    let (va, vb) = (vector[a], vector[b]);
                    vector[a] = va * cos - vb * sin;
                    vector[b] = vb * cos + va * sin;
                }
            }
        },
    );
    Ok(out)
}

impl Run<Rope> for Cpu {
    fn forward(&self, op: &Rope, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Rope::NAME, inputs[0])?;
        let out = rotate(self, op, &x, &signature.shape, inputs[1], 1.0)?;
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Rope,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer(inputs)?;
        let g = io::contiguous(Rope::NAME, grad)?;
        let out = rotate(self, op, &g, &signature.shape, inputs[1], -1.0)?;
        Ok(vec![
            Some(io::write(
                &self.buffers,
                &out,
                &signature.shape,
                signature.dtype,
            )?),
            None,
        ])
    }
}
