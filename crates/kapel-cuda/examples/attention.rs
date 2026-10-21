//! Measures the causal attention of the CUDA backend, forward and backward, in
//! trillions of operations per second (two per multiply-add, the hidden half
//! of the scores not counted), then with a padding mask instead (all scores
//! counted). Measure on an idle device.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example attention`

use std::time::Instant;

use kapel_core::{BF16, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{Attention, Run, Transfer};

fn main() -> Result<(), kapel_core::Error> {
    let cuda = Cuda::new(0)?;
    let op = Attention::causal();
    for (b, h, t, d) in [
        (4, 16, 1024, 64),
        (4, 16, 2048, 128),
        (1, 32, 4096, 128),
        (2, 8, 2048, 256),
    ] {
        let tensor = |seed: usize| -> Result<Tensor, kapel_core::Error> {
            let values: Vec<BF16> = (0..b * h * t * d)
                .map(|i| BF16::from_f64(((i * 7919 + seed) % 1000) as f64 / 500.0 - 1.0))
                .collect();
            cuda.upload(&Tensor::from_slice(&values, &[b, h, t, d])?)
        };
        let (q, k, v, g) = (tensor(1)?, tensor(2)?, tensor(3)?, tensor(4)?);
        let time =
            |f: &dyn Fn() -> Result<(), kapel_core::Error>| -> Result<f64, kapel_core::Error> {
                f()?;
                cuda.synchronize()?;
                let runs = 10;
                let start = Instant::now();
                for _ in 0..runs {
                    f()?;
                }
                cuda.synchronize()?;
                Ok(start.elapsed().as_secs_f64() / f64::from(runs))
            };
        let forward = time(&|| cuda.forward(&op, &[&q, &k, &v]).map(drop))?;
        let backward = time(&|| cuda.backward(&op, &[&q, &k, &v], &g).map(drop))?;
        // Two products of `T × T × D` per head, half of them hidden.
        let ops = 2.0 * 2.0 * (b * h * t * t * d) as f64 / 2.0;
        println!(
            "bf16 [{b}, {h}, {t}, {d}]: forward {:.2} ms, {:.1} Top/s; backward {:.2} ms, {:.1} Top/s",
            forward * 1e3,
            ops / forward / 1e12,
            backward * 1e3,
            // The backward pass: five products, the forward pass recomputed.
            3.5 * ops / backward / 1e12,
        );
        // The same with a padding mask instead of causality: every key seen,
        // the scores of the last eighth of them lowered to `f32::MIN`.
        let mask: Vec<f32> = (0..t * t)
            .map(|n| if n % t >= t - t / 8 { f32::MIN } else { 0.0 })
            .collect();
        let mask = cuda.upload(&Tensor::from_slice(&mask, &[1, 1, t, t])?)?;
        let free = Attention {
            causal: false,
            ..op
        };
        let inputs = [&q, &k, &v, &mask];
        let forward = time(&|| cuda.forward(&free, &inputs).map(drop))?;
        let backward = time(&|| cuda.backward(&free, &inputs, &g).map(drop))?;
        println!(
            "bf16 [{b}, {h}, {t}, {d}] masked: forward {:.2} ms, {:.1} Top/s; backward {:.2} ms, {:.1} Top/s",
            forward * 1e3,
            2.0 * ops / forward / 1e12,
            backward * 1e3,
            7.0 * ops / backward / 1e12,
        );
    }
    Ok(())
}
