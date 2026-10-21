//! Measures the mixed-precision product (`bf16` operands, `f32` sums) next to
//! the `f32` one, through the low-level API, and the product of `bf16` tensors
//! through the backend. Without `avx512bf16`, the mixed product falls back to
//! `f32`.
//!
//! `cargo run --release -p kapel-cpu --example gemm_bf16`

use std::time::Instant;

use kapel_core::{DType, Tensor};
use kapel_cpu::gemm::{self, Matrix};
use kapel_cpu::pool::Pool;
use kapel_cpu::{Cpu, default_threads};
use kapel_ops::{Cast, Matmul, Run};

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
    let pool = Pool::new(default_threads());
    let isa = cpu.isa();
    println!(
        "{} threads, {isa:?}, vdpbf16ps: {}",
        cpu.threads(),
        gemm::bf16_available()
    );
    for (m, k, n) in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 / 16.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 / 12.0).collect();
        let mut c = vec![0.0f32; m * n];
        let ops = 2.0 * (m * k * n) as f64;
        let (left, right) = (Matrix::new(&a, m, k)?, Matrix::new(&b, k, n)?);
        let f32_time = best(5, || {
            gemm::gemm(&pool, isa, left, right, &mut c, false);
            Ok(())
        })?;
        let mixed_time = best(5, || {
            gemm::gemm_mixed(&pool, isa, left, right, &mut c, false);
            Ok(())
        })?;
        let ta = Tensor::from_slice(&a, &[m, k])?;
        let tb = Tensor::from_slice(&b, &[k, n])?;
        let cast = Cast { dtype: DType::BF16 };
        let (ta, tb) = (cpu.forward(&cast, &[&ta])?, cpu.forward(&cast, &[&tb])?);
        let tensor_time = best(5, || cpu.forward(&Matmul, &[&ta, &tb]).map(drop))?;
        println!(
            "{m} × {k} × {n}: gemm f32 {:5.0} Gop/s, gemm_mixed {:5.0} Gop/s, Matmul on bf16 tensors {:5.0} Gop/s",
            ops / f32_time / 1e9,
            ops / mixed_time / 1e9,
            ops / tensor_time / 1e9
        );
    }
    Ok(())
}
