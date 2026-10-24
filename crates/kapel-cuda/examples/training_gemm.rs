//! Measures the products of a training step of GPT-2 124M (16 × 1024 tokens),
//! in bf16, as the tape computes them: the forward product `x · w`, the
//! gradient of the input `g · wᵀ` and that of the weight `xᵀ · g`, the
//! transposed operands as views. In trillions of operations per second.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example training_gemm`

use std::time::Instant;

use kapel_core::{BF16, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{Matmul, Run, Transfer};

fn matrix(rows: usize, columns: usize, seed: usize) -> Tensor {
    let values: Vec<BF16> = (0..rows * columns)
        .map(|i| BF16::from_f64((((i + seed) * 7919) % 1000) as f64 / 500.0 - 1.0))
        .collect();
    Tensor::from_slice(&values, &[rows, columns]).expect("sizes agree")
}

fn main() -> Result<(), kapel_core::Error> {
    let cuda = Cuda::new(0)?;
    let n = 16 * 1024;
    let mut total = 0.0;
    // (in, out) of each projection of a block, and the output table by a
    // piece of 1024 rows.
    for (k, m, rows, what) in [
        (768, 2304, n, "c_attn"),
        (768, 768, n, "attn.c_proj"),
        (768, 3072, n, "c_fc"),
        (3072, 768, n, "mlp.c_proj"),
        (768, 50257, 1024, "lm head (a piece)"),
        (768, 50304, 1024, "lm head, padded"),
    ] {
        let x = cuda.upload(&matrix(rows, k, 1))?;
        let w = cuda.upload(&matrix(k, m, 2))?;
        let g = cuda.upload(&matrix(rows, m, 3))?;
        let xt = cuda.forward(&kapel_ops::Contiguous, &[&x.transpose(0, 1)?])?;
        let wt = cuda.forward(&kapel_ops::Contiguous, &[&w.transpose(0, 1)?])?;
        let cases: [(&str, Tensor, Tensor); 5] = [
            ("x · w", x.clone(), w.clone()),
            ("g · wᵀ", g.clone(), w.transpose(0, 1)?),
            ("g · (wᵀ)", g.clone(), wt),
            ("xᵀ · g", x.transpose(0, 1)?, g.clone()),
            ("(xᵀ) · g", xt, g.clone()),
        ];
        for (name, a, b) in cases {
            cuda.forward(&Matmul, &[&a, &b])?;
            cuda.synchronize()?;
            let runs = 10;
            let start = Instant::now();
            for _ in 0..runs {
                cuda.forward(&Matmul, &[&a, &b])?;
            }
            cuda.synchronize()?;
            let time = start.elapsed().as_secs_f64() / f64::from(runs);
            let ops = 2.0 * (rows * k * m) as f64;
            total += time;
            println!(
                "{what:>18} {name:>9} [{:>5} × {:>5}] · [{:>5} × {:>5}]: {:6.2} ms, {:5.1} Top/s",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1],
                time * 1e3,
                ops / time / 1e12
            );
        }
    }
    // The forms of the fused loss: the table `[V, C]` as stored.
    let x = cuda.upload(&matrix(1024, 768, 1))?;
    let w = cuda.upload(&matrix(50304, 768, 2))?;
    let g = cuda.upload(&matrix(1024, 50304, 3))?;
    for (name, a, b) in [
        ("x · wᵀ", x.clone(), w.transpose(0, 1)?),
        ("g · w", g.clone(), w.clone()),
        ("gᵀ · x", g.transpose(0, 1)?, x.clone()),
    ] {
        cuda.forward(&Matmul, &[&a, &b])?;
        cuda.synchronize()?;
        let start = Instant::now();
        for _ in 0..10 {
            cuda.forward(&Matmul, &[&a, &b])?;
        }
        cuda.synchronize()?;
        let time = start.elapsed().as_secs_f64() / 10.0;
        let ops = 2.0 * 1024.0 * 768.0 * 50304.0;
        println!(
            "{:>18} {name:>9}: {:6.2} ms, {:5.1} Top/s",
            "loss",
            time * 1e3,
            ops / time / 1e12
        );
    }
    println!("total {:.1} ms", total * 1e3);
    Ok(())
}
