//! Measures the sum: over the last dimension (contiguous lines), over the
//! first one (strided lines), and over every element.
//!
//! `cargo run --release -p kapel-cpu --example reduce`

use std::time::Instant;

use kapel_core::Tensor;
use kapel_cpu::Cpu;
use kapel_ops::{Run, Sum};

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
    let (rows, cols) = (4096, 4096);
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    let x = Tensor::from_slice(&values, &[rows, cols])?;
    let flat = Tensor::from_slice(&values, &[rows * cols])?;
    for (name, tensor, dims) in [
        ("4096 × 4096 over dim 1", &x, vec![1]),
        ("4096 × 4096 over dim 0", &x, vec![0]),
        ("16 777 216 over dim 0 ", &flat, vec![0]),
    ] {
        let op = Sum {
            dims,
            keepdim: false,
        };
        let time = best(10, || cpu.forward(&op, &[tensor]).map(drop))?;
        println!(
            "sum {name}: {:6.2} ms, {:5.1} GB/s",
            time * 1e3,
            4.0 * (rows * cols) as f64 / time / 1e9
        );
    }
    Ok(())
}
