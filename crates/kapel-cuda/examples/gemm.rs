//! Measures the matrix product of the CUDA backend, in trillions of operations
//! per second (two per multiply-add). Measure on an idle device: another
//! program sharing it takes a varying share of its time.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example gemm`

use std::time::Instant;

use kapel_core::{BF16, DType, Element, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{Matmul, Run, Transfer};

fn matrix<T: Element>(rows: usize, columns: usize, value: impl Fn(usize) -> T) -> Tensor {
    let values: Vec<T> = (0..rows * columns).map(value).collect();
    Tensor::from_slice(&values, &[rows, columns]).expect("sizes agree")
}

fn main() -> Result<(), kapel_core::Error> {
    let cuda = Cuda::new(0)?;
    for dtype in [DType::BF16, DType::F32] {
        for (m, k, n) in [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 512, 1536),
        ] {
            let value = |i: usize| ((i * 7919) % 1000) as f64 / 500.0 - 1.0;
            let (a, b) = match dtype {
                DType::BF16 => (
                    matrix(m, k, |i| BF16::from_f64(value(i))),
                    matrix(k, n, |i| BF16::from_f64(value(i + 1))),
                ),
                _ => (
                    matrix(m, k, |i| value(i) as f32),
                    matrix(k, n, |i| value(i + 1) as f32),
                ),
            };
            let (a, b) = (cuda.upload(&a)?, cuda.upload(&b)?);
            // The first run compiles the kernel, or reads it from the cache.
            cuda.forward(&Matmul, &[&a, &b])?;
            cuda.synchronize()?;
            let runs = 20;
            let start = Instant::now();
            for _ in 0..runs {
                cuda.forward(&Matmul, &[&a, &b])?;
            }
            cuda.synchronize()?;
            let time = start.elapsed().as_secs_f64() / f64::from(runs);
            let ops = 2.0 * (m * k * n) as f64;
            println!(
                "{dtype} {m} × {k} × {n}: {:.2} ms, {:.1} Top/s",
                time * 1e3,
                ops / time / 1e12
            );
        }
    }
    Ok(())
}
