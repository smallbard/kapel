//! Measures attention, forward and backward: prefill with grouped heads, a
//! batch of one with as many query heads as key heads, and one generation step.
//!
//! `cargo run --release -p kapel-cpu --example attention`

use std::time::Instant;

use kapel_core::Tensor;
use kapel_cpu::Cpu;
use kapel_ops::{Attention, Run};

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

fn values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn main() -> Result<(), kapel_core::Error> {
    let cpu = Cpu::new();
    println!("{} threads, {:?}", cpu.threads(), cpu.isa());
    // (batch, query heads, key heads, query length, key length, head size)
    for (b, hq, hkv, tq, tk, d) in [
        (1, 32, 8, 1024, 1024, 128),
        (1, 8, 8, 2048, 2048, 64),
        (1, 32, 8, 1, 4096, 128),
    ] {
        let q = Tensor::from_slice(&values(b * hq * tq * d), &[b, hq, tq, d])?;
        let k = Tensor::from_slice(&values(b * hkv * tk * d), &[b, hkv, tk, d])?;
        let v = Tensor::from_slice(&values(b * hkv * tk * d), &[b, hkv, tk, d])?;
        let g = Tensor::from_slice(&values(b * hq * tq * d), &[b, hq, tq, d])?;
        let op = Attention::causal();
        let forward = best(5, || cpu.forward(&op, &[&q, &k, &v]).map(drop))?;
        let backward = best(3, || cpu.backward(&op, &[&q, &k, &v], &g).map(drop))?;
        println!(
            "b {b}, heads {hq}/{hkv}, t {tq}×{tk}, d {d}: forward {:8.2} ms, backward {:8.2} ms",
            forward * 1e3,
            backward * 1e3
        );
    }
    Ok(())
}
