//! Reductions.

use kapel_core::{Result, Tensor};
use kapel_ops::{Max, Mean, Operation, Run, Sum};

use crate::backend::{CHUNK, Cpu};
use crate::lines::lines;
use crate::{io, math};

/// Row-major strides.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Every combination of the coordinates of `dims`, as offsets in `shape`.
fn reduced_offsets(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let mut offsets = vec![0];
    for &d in dims {
        let (size, stride) = (shape[d], s[d]);
        offsets = offsets
            .iter()
            .flat_map(|&o| (0..size).map(move |i| o + i * stride))
            .collect();
    }
    offsets
}

/// For each result of reducing `dims`, the offset of its first element.
fn kept_offsets(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let kept: Vec<usize> = (0..shape.len()).filter(|d| !dims.contains(d)).collect();
    // The kept dimensions in increasing order: the results come out row-major.
    reduced_offsets(shape, &kept)
}

/// Elements summed at once by [`math::sum`]: sixteen running sums of 64
/// elements each, enough to amortize the reduction of the lanes.
const SUM_BLOCK: usize = 1024;

/// The sum of `values`, halves first: the error grows with the logarithm of
/// the length, not with the length.
fn pairwise(values: &[f32]) -> f32 {
    match values {
        [] => 0.0,
        [value] => *value,
        _ => {
            let (low, high) = values.split_at(values.len() / 2);
            pairwise(low) + pairwise(high)
        }
    }
}

/// The sum over `dims`. Each result is the pairwise sum of the sums of its
/// blocks of `SUM_BLOCK` elements, each taken by [`math::sum`]: the order of
/// the additions depends only on the shape. The blocks of every result are
/// spread over the threads, so a single large sum uses them all.
fn sum(cpu: &Cpu, x: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    // Reducing the last dimensions: each result reads a contiguous range,
    // and the offsets within it need not be listed.
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    let contiguous = sorted
        .iter()
        .copied()
        .eq(shape.len() - dims.len()..shape.len());
    let len: usize = dims.iter().map(|&d| shape[d]).product();
    let inner = if contiguous {
        Vec::new()
    } else {
        reduced_offsets(shape, dims)
    };
    let bases = kept_offsets(shape, dims);
    let blocks = len.div_ceil(SUM_BLOCK);
    let isa = cpu.isa;
    let mut sums = vec![0.0; bases.len() * blocks];
    cpu.fill(&mut sums, (CHUNK / SUM_BLOCK).max(1), |start, part| {
        let mut gathered = [0.0f32; SUM_BLOCK];
        for (n, y) in part.iter_mut().enumerate() {
            let (result, block) = ((start + n) / blocks, (start + n) % blocks);
            let range = block * SUM_BLOCK..len.min((block + 1) * SUM_BLOCK);
            let base = bases[result];
            *y = if contiguous {
                math::sum(isa, &x[base + range.start..base + range.end])
            } else {
                let values = &mut gathered[..range.len()];
                for (value, &o) in values.iter_mut().zip(&inner[range]) {
                    *value = x[base + o];
                }
                math::sum(isa, values)
            };
        }
    });
    let mut out = vec![0.0; bases.len()];
    cpu.fill(&mut out, (CHUNK / blocks.max(1)).max(1), |start, part| {
        for (n, y) in part.iter_mut().enumerate() {
            *y = pairwise(&sums[(start + n) * blocks..][..blocks]);
        }
    });
    out
}

/// Each input element receives `factor` times the gradient of its result.
fn spread(cpu: &Cpu, g: &[f32], shape: &[usize], dims: &[usize], factor: f32) -> Vec<f32> {
    let (inner, bases) = (reduced_offsets(shape, dims), kept_offsets(shape, dims));
    let total: usize = shape.iter().product();
    let mut out = vec![0.0; total];
    // Results are spread over disjoint sets of input elements.
    let shared = crate::pool::SharedMut::new(&mut out);
    let chunk = (CHUNK / inner.len().max(1)).max(1);
    cpu.pool.run(bases.len().div_ceil(chunk), |task| {
        for r in task * chunk..((task + 1) * chunk).min(bases.len()) {
            for &o in &inner {
                // SAFETY: each input element belongs to exactly one result, and
                // each result to one task.
                unsafe { *shared.ptr(bases[r] + o) = factor * g[r] };
            }
        }
    });
    out
}

impl Run<Sum> for Cpu {
    fn forward(&self, op: &Sum, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Sum::NAME, inputs[0])?;
        let out = sum(self, &x, inputs[0].shape(), &op.dims);
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(&self, op: &Sum, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let g = io::contiguous(Sum::NAME, grad)?;
        let out = spread(self, &g, inputs[0].shape(), &op.dims, 1.0);
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}

fn count(shape: &[usize], dims: &[usize]) -> f32 {
    dims.iter().map(|&d| shape[d]).product::<usize>() as f32
}

impl Run<Mean> for Cpu {
    fn forward(&self, op: &Mean, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Mean::NAME, inputs[0])?;
        let n = count(inputs[0].shape(), &op.dims);
        let out: Vec<f32> = sum(self, &x, inputs[0].shape(), &op.dims)
            .iter()
            .map(|v| v / n)
            .collect();
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &Mean,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let g = io::contiguous(Mean::NAME, grad)?;
        let n = count(inputs[0].shape(), &op.dims);
        let out = spread(self, &g, inputs[0].shape(), &op.dims, 1.0 / n);
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}

/// The position of the first maximum of each line along `dim`.
fn argmax(cpu: &Cpu, x: &[f32], shape: &[usize], dim: usize) -> Vec<usize> {
    let len = shape[dim];
    let (count, line) = lines(shape, dim);
    let mut out = vec![0usize; count];
    let shared = crate::pool::SharedMut::new(&mut out);
    let chunk = (CHUNK / len.max(1)).max(1);
    cpu.pool.run(count.div_ceil(chunk), |task| {
        for l in task * chunk..((task + 1) * chunk).min(count) {
            let (start, step) = line(l);
            let best = (0..len)
                .map(|i| start + i * step)
                .fold(start, |best, at| if x[at] > x[best] { at } else { best });
            // SAFETY: one slot per line, one line per task.
            unsafe { *shared.ptr(l) = best };
        }
    });
    out
}

impl Run<Max> for Cpu {
    fn forward(&self, op: &Max, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(Max::NAME, inputs[0])?;
        let out: Vec<f32> = argmax(self, &x, inputs[0].shape(), op.dim)
            .iter()
            .map(|&at| x[at])
            .collect();
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(&self, op: &Max, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let signature = op.infer_backward(inputs, grad)?;
        let x = io::contiguous(Max::NAME, inputs[0])?;
        let g = io::contiguous(Max::NAME, grad)?;
        let mut out = vec![0.0; x.len()];
        for (l, at) in argmax(self, &x, inputs[0].shape(), op.dim)
            .into_iter()
            .enumerate()
        {
            out[at] = g[l];
        }
        Ok(vec![Some(io::write(
            &self.buffers,
            &out,
            inputs[0].shape(),
            signature.dtype,
        )?)])
    }
}

#[cfg(test)]
mod tests {
    use kapel_core::Tensor;
    use kapel_ops::{Mean, Run, Sum};

    use crate::Cpu;

    #[test]
    fn long_sums_stay_accurate() {
        // 0.1 is not exact in binary: a sequential sum of ten million of them
        // in `f32` is off by several percent.
        let n = 10_000_000;
        let x = Tensor::from_slice(&vec![0.1f32; n], &[n]).unwrap();
        let op = Sum {
            dims: vec![0],
            keepdim: false,
        };
        let got = Cpu::with_threads(4)
            .forward(&op, &[&x])
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[0];
        let want = n as f64 * f64::from(0.1f32);
        assert!(
            ((got as f64 - want) / want).abs() < 1e-6,
            "{got} instead of {want}"
        );
    }

    #[test]
    fn sums_do_not_depend_on_the_number_of_threads() {
        let shape = [37, 3001, 5];
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 1000) as f32 / 300.0 - 1.5)
            .collect();
        let x = Tensor::from_slice(&values, &shape).unwrap();
        let run = |threads: usize| {
            let cpu = Cpu::with_threads(threads);
            let mut all = Vec::new();
            for dims in [vec![2], vec![1], vec![0, 2], vec![1, 2], vec![0, 1, 2]] {
                let sum = Sum {
                    dims: dims.clone(),
                    keepdim: false,
                };
                let mean = Mean {
                    dims,
                    keepdim: true,
                };
                for y in [cpu.forward(&sum, &[&x]), cpu.forward(&mean, &[&x])] {
                    all.push(y.unwrap().to_vec::<f32>().unwrap());
                }
            }
            all
        };
        let one = run(1);
        for threads in [2, 3, 8] {
            assert_eq!(run(threads), one, "{threads} threads");
        }
    }
}
