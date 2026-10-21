//! Measures the matrix product with few rows, as during generation: `m` rows of
//! activations times a `4096 × 4096` weight, stored row by row (`x · W`) or
//! transposed (`x · Wᵀ`, the layout of a linear layer). The product is bound by
//! the memory bandwidth: the rate of the weights read, in GB/s, says how close
//! it gets.
//!
//! `cargo run --release -p kapel-cpu --example gemm_few_rows`

use std::time::Instant;

use kapel_core::Tensor;
use kapel_cpu::Cpu;
use kapel_ops::{Matmul, Run};

/// The best of `runs` timings of `f`, in seconds, after one warm-up call.
fn best(
    runs: usize,
    mut f: impl FnMut() -> Result<(), kapel_core::Error>,
) -> Result<f64, kapel_core::Error> {
    f()?;
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let start = Instant::now();
        f()?;
        best = best.min(start.elapsed().as_secs_f64());
    }
    Ok(best)
}

fn main() -> Result<(), kapel_core::Error> {
    let cpu = Cpu::new();
    println!("{} threads, {:?}", cpu.threads(), cpu.isa());
    let (k, n) = (4096, 4096);
    let w = Tensor::from_slice(&vec![0.25f32; k * n], &[k, n])?;
    let wt = w.transpose(0, 1)?;
    for m in [1, 16, 256, 257] {
        let x = Tensor::from_slice(&vec![0.5f32; m * k], &[m, k])?;
        for (name, weight) in [("x · W ", &w), ("x · Wᵀ", &wt)] {
            let time = best(20, || cpu.forward(&Matmul, &[&x, weight]).map(drop))?;
            let ops = 2.0 * (m * k * n) as f64;
            let bytes = 4.0 * (k * n) as f64;
            println!(
                "m = {m:3}, {name}: {:7.3} ms, {:5.0} Gop/s, {:4.0} GB/s of weights",
                time * 1e3,
                ops / time / 1e9,
                bytes / time / 1e9
            );
        }
    }
    Ok(())
}
