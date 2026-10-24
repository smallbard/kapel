//! Updates in place.

use kapel_core::{Error, HostStorage, Result, Tensor};
use kapel_ops::{ADAMW, AdamWStep, Update, check_adamw};

use crate::backend::{CHUNK, Cpu};
use crate::io;
use crate::pool::SharedMut;

/// The `f32` values of a host tensor that can be written.
fn host_f32(t: &mut Tensor) -> Result<&mut [f32]> {
    let storage = t
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| Error::InvalidArgument {
            op: ADAMW,
            message: "the tensor is not in host memory that can be written".into(),
        })?;
    Ok(io::as_f32_mut(storage.bytes_mut()))
}

impl Update for Cpu {
    /// [`AdamWStep::apply`], element by element, pieces of the tensors
    /// shared between threads.
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &AdamWStep,
    ) -> Result<()> {
        check_adamw(param, m, v, grad)?;
        let g = io::contiguous(ADAMW, grad)?;
        let n = g.len();
        let (pm, mm, vm) = (host_f32(param)?, host_f32(m)?, host_f32(v)?);
        let (ps, ms, vs) = (
            SharedMut::new(&mut pm[..n]),
            SharedMut::new(&mut mm[..n]),
            SharedMut::new(&mut vm[..n]),
        );
        self.pool.run(n.div_ceil(CHUNK), |task| {
            let start = task * CHUNK;
            let len = CHUNK.min(n - start);
            // SAFETY: the pieces are disjoint, each updated by one task; the
            // tensors outlive the call to `run`.
            let (p, m, v) = unsafe {
                (
                    ps.slice(start, len),
                    ms.slice(start, len),
                    vs.slice(start, len),
                )
            };
            for i in 0..len {
                (p[i], m[i], v[i]) = step.apply(p[i], m[i], v[i], g[start + i]);
            }
        });
        Ok(())
    }
}
