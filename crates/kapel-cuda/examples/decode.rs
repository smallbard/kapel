//! Measures the products of decoding on the card: activations `[M, K]` times
//! weights `[N, K]` read transposed in place, as a model computes them, for
//! the shapes of Llama 3.2 3B and a few row counts. Prints the time and the
//! bandwidth reached on the weights, and, for comparison, a plain copy.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example decode`

use std::time::Instant;

use kapel_core::{BF16, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{Cast, Matmul, Run, Transfer};

fn matrix(rows: usize, columns: usize, seed: usize) -> Tensor {
    let values: Vec<BF16> = (0..rows * columns)
        .map(|i| BF16::from_f64(((i * 7919 + seed) % 1000) as f64 / 500.0 - 1.0))
        .collect();
    Tensor::from_slice(&values, &[rows, columns]).expect("sizes agree")
}

fn time(cuda: &Cuda, runs: u32, mut f: impl FnMut()) -> f64 {
    f();
    cuda.synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..runs {
        f();
    }
    cuda.synchronize().unwrap();
    start.elapsed().as_secs_f64() / f64::from(runs)
}

fn main() -> Result<(), kapel_core::Error> {
    let cuda = Cuda::new(0)?;
    let big = cuda.upload(&matrix(8192, 8192, 0))?;
    let copy = time(&cuda, 20, || {
        cuda.forward(
            &Cast {
                dtype: kapel_core::DType::BF16,
            },
            &[&big],
        )
        .unwrap();
    });
    let bytes = 2.0 * big.numel() as f64 * 2.0;
    println!(
        "copy of 128 MiB: {:.3} ms, {:.0} GB/s",
        copy * 1e3,
        bytes / copy / 1e9
    );
    let filter: Option<usize> = std::env::var("M").ok().and_then(|m| m.parse().ok());
    for m in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096] {
        if filter.is_some_and(|f| f != m) {
            continue;
        }
        for (k, n) in [
            (3072, 3072),
            (3072, 1024),
            (3072, 8192),
            (8192, 3072),
            (3072, 128_256),
        ] {
            let x = cuda.upload(&matrix(m, k, 1))?;
            let w = cuda.upload(&matrix(n, k, 2))?;
            let wt = w.transpose(0, 1)?;
            let t = time(&cuda, 50, || {
                cuda.forward(&Matmul, &[&x, &wt]).unwrap();
            });
            let bytes = (n * k * 2) as f64;
            println!(
                "M {m:>3}  {k:>5} → {n:>6}: {:>8.1} µs, {:>4.0} GB/s, {:>5.1} Top/s",
                t * 1e6,
                bytes / t / 1e9,
                2.0 * (m * n * k) as f64 / t / 1e12
            );
        }
    }
    Ok(())
}
