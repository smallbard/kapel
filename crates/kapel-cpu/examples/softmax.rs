//! Measures the softmax over rows and the exponential, in billions of elements
//! per second.
//!
//! `cargo run --release -p kapel-cpu --example softmax`

use std::time::Instant;

use kapel_core::Tensor;
use kapel_cpu::Cpu;
use kapel_ops::{Run, Softmax, Unary, UnaryKind};

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
    for (rows, cols) in [(4096, 4096), (64, 32_000), (32_768, 128)] {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| (i % 1000) as f32 / 100.0 - 5.0)
            .collect();
        let x = Tensor::from_slice(&values, &[rows, cols])?;
        let elements = (rows * cols) as f64;
        let time = best(10, || cpu.forward(&Softmax { dim: 1 }, &[&x]).map(drop))?;
        println!(
            "softmax {rows} × {cols}: {:6.2} ms, {:5.2} G elements/s",
            time * 1e3,
            elements / time / 1e9
        );
        let time = best(10, || {
            cpu.forward(
                &Unary {
                    kind: UnaryKind::Exp,
                },
                &[&x],
            )
            .map(drop)
        })?;
        println!(
            "exp     {rows} × {cols}: {:6.2} ms, {:5.2} G elements/s",
            time * 1e3,
            elements / time / 1e9
        );
    }
    // The vectorized exponential alone, on one thread: a copy into an L1-sized
    // buffer, then the exponential in place.
    let source: Vec<f32> = (0..4096).map(|i| (i % 1000) as f32 / 100.0 - 5.0).collect();
    let mut row = source.clone();
    let time = best(10_000, || {
        row.copy_from_slice(&source);
        kapel_cpu::math::exp_in_place(cpu.isa(), &mut row);
        std::hint::black_box(&row);
        Ok(())
    })?;
    println!(
        "math::exp_in_place, one thread, 4096 elements: {:6.2} µs, {:5.2} G elements/s",
        time * 1e6,
        4096.0 / time / 1e9
    );
    Ok(())
}
