//! Working line by line along one dimension of a row-major tensor.

use crate::backend::{CHUNK, Cpu};
use crate::pool::SharedMut;

/// The lines of `shape` along `dim`: for each, the position of its first element
/// and the step between its elements.
pub(crate) fn lines(shape: &[usize], dim: usize) -> (usize, impl Fn(usize) -> (usize, usize)) {
    let len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let count = outer * inner;
    (count, move |line: usize| {
        (
            (line / inner.max(1)) * len * inner + line % inner.max(1),
            inner,
        )
    })
}

/// `out = f(x)` line by line along `dim`; each line is copied into a contiguous
/// scratch buffer when its elements are not adjacent. Lines are shared between
/// threads in a fixed way.
pub(crate) fn map_lines(
    cpu: &Cpu,
    inputs: &[&[f32]],
    shape: &[usize],
    dim: usize,
    f: impl Fn(&[&[f32]], &mut [f32]) + Sync,
) -> Vec<f32> {
    let total: usize = shape.iter().product();
    let mut out = vec![0.0; total];
    if total == 0 {
        return out;
    }
    let len = shape[dim];
    let (count, line) = lines(shape, dim);
    let per_task = (CHUNK / len.max(1)).max(1);
    let shared = SharedMut::new(&mut out);
    cpu.pool.run(count.div_ceil(per_task), |task| {
        let mut scratch_in: Vec<Vec<f32>> = inputs.iter().map(|_| vec![0.0; len]).collect();
        let mut scratch_out = vec![0.0; len];
        for l in task * per_task..((task + 1) * per_task).min(count) {
            let (start, step) = line(l);
            if step == 1 {
                let views: Vec<&[f32]> = inputs.iter().map(|x| &x[start..start + len]).collect();
                // SAFETY: lines are disjoint; each is written by one task, and
                // `out` outlives the call to `run`.
                let target = unsafe { shared.slice(start, len) };
                f(&views, target);
            } else {
                for (scratch, x) in scratch_in.iter_mut().zip(inputs) {
                    for (i, v) in scratch.iter_mut().enumerate() {
                        *v = x[start + i * step];
                    }
                }
                let views: Vec<&[f32]> = scratch_in.iter().map(Vec::as_slice).collect();
                f(&views, &mut scratch_out);
                for (i, &v) in scratch_out.iter().enumerate() {
                    // SAFETY: as above; the elements of one line belong to it alone.
                    unsafe { *shared.ptr(start + i * step) = v };
                }
            }
        }
    });
    out
}
