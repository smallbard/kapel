//! Losses.

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{CrossEntropy, Operation, Run};

use crate::backend::{CHUNK, Cpu};
use crate::io;
use crate::math;
use crate::pool::SharedMut;

/// The column of each row's target, `None` for an ignored row.
fn targets(t: &Tensor, vocab: usize) -> Result<Vec<Option<usize>>> {
    let bytes = io::host_bytes(CrossEntropy::NAME, t)?;
    let signed = t.dtype() == DType::I32;
    t.layout()
        .offsets()
        .map(|at| {
            let raw = [
                bytes[4 * at],
                bytes[4 * at + 1],
                bytes[4 * at + 2],
                bytes[4 * at + 3],
            ];
            let value = if signed {
                i64::from(i32::from_le_bytes(raw))
            } else {
                i64::from(u32::from_le_bytes(raw))
            };
            CrossEntropy::target(value, vocab)
        })
        .collect()
}

/// `f(n, row, out)` for every row `n` of `x [N, V]`, rows shared between
/// threads; `out` is row `n` of an `[N, width]` result.
fn rows(cpu: &Cpu, n: usize, width: usize, f: impl Fn(usize, &mut [f32]) + Sync) -> Vec<f32> {
    let mut out = vec![0.0; n * width];
    let per_task = (CHUNK / width.max(1)).max(1);
    let shared = SharedMut::new(&mut out);
    cpu.pool.run(n.div_ceil(per_task), |task| {
        for r in task * per_task..((task + 1) * per_task).min(n) {
            // SAFETY: each row is written by one task alone, and `out`
            // outlives the call to `run`.
            f(r, unsafe { shared.slice(r * width, width) });
        }
    });
    out
}

impl Run<CrossEntropy> for Cpu {
    fn forward(&self, op: &CrossEntropy, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let (n, vocab) = (inputs[0].shape()[0], inputs[0].shape()[1]);
        let x = io::contiguous(CrossEntropy::NAME, inputs[0])?;
        let t = targets(inputs[1], vocab)?;
        let isa = self.isa;
        let y = rows(self, n, 1, |r, out| {
            out[0] = match t[r] {
                None => 0.0,
                Some(t) => {
                    let row = &x[r * vocab..(r + 1) * vocab];
                    let max = math::max(isa, row);
                    max + math::exp_shifted_total(isa, row, max).ln() - row[t]
                }
            };
        });
        io::write(&self.buffers, &y, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &CrossEntropy,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        let (n, vocab) = (inputs[0].shape()[0], inputs[0].shape()[1]);
        let x = io::contiguous(CrossEntropy::NAME, inputs[0])?;
        let g = io::contiguous(CrossEntropy::NAME, grad)?;
        let t = targets(inputs[1], vocab)?;
        let isa = self.isa;
        let gx = rows(self, n, vocab, |r, out| {
            let Some(t) = t[r] else {
                out.fill(0.0);
                return;
            };
            let row = &x[r * vocab..(r + 1) * vocab];
            let max = math::max(isa, row);
            out.copy_from_slice(row);
            let total = math::exp_shifted_sum(isa, out, max);
            math::scale(isa, out, g[r] / total);
            out[t] -= g[r];
        });
        Ok(vec![
            Some(io::write(
                &self.buffers,
                &gx,
                inputs[0].shape(),
                inputs[0].dtype(),
            )?),
            None,
        ])
    }
}
