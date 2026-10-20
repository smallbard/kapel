//! Measures the matrix product of the CPU backend, in billions of operations per
//! second (two per multiply-add).
//!
//! `cargo run --release -p kapel-cpu --example gemm`

use std::time::Instant;

use kapel_core::Tensor;
use kapel_cpu::Cpu;
use kapel_ops::{Matmul, Run};

fn main() -> Result<(), kapel_core::Error> {
    let cpu = Cpu::new();
    println!("{} threads, {:?}", cpu.threads(), cpu.isa());
    for (m, k, n) in [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 512, 1536),
    ] {
        let a = Tensor::from_slice(&vec![0.5f32; m * k], &[m, k])?;
        let b = Tensor::from_slice(&vec![0.25f32; k * n], &[k, n])?;
        cpu.forward(&Matmul, &[&a, &b])?;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            cpu.forward(&Matmul, &[&a, &b])?;
            best = best.min(start.elapsed().as_secs_f64());
        }
        let ops = 2.0 * (m * k * n) as f64;
        println!(
            "{m} × {k} × {n}: {:.1} ms, {:.0} Gop/s",
            best * 1e3,
            ops / best / 1e9
        );
    }
    Ok(())
}
