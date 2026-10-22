//! Measures the products by quantized weights during generation: `m` rows of
//! activations times weights `[N, K]` read in place, larger than the caches.
//! The product is bound by the memory bandwidth: the rate of the weight bytes
//! read, in GB/s, says how close it gets.
//!
//! `cargo run --release -p kapel-cpu --example quantized`

use std::time::Instant;

use kapel_core::{DType, F16, QuantFormat, Tensor};
use kapel_cpu::Cpu;
use kapel_ops::{Matmul, QuantizedMatmul, Run};

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

/// Blocks of `dtype` with arbitrary codes and small scales.
fn weights(dtype: DType, n: usize, k: usize) -> Result<Tensor, kapel_core::Error> {
    let bytes = dtype.storage_bytes(n * k)?;
    let mut data: Vec<u8> = (0..bytes).map(|i| ((i * 2654435761) >> 13) as u8).collect();
    let scale = F16::from_f64(0.01).0.to_le_bytes();
    let halves: &[usize] = match dtype {
        DType::Quantized(QuantFormat::Q8_0 | QuantFormat::Q4_0) => &[0],
        DType::Quantized(QuantFormat::Q4K | QuantFormat::Q5K) => &[0, 2],
        DType::Quantized(QuantFormat::Q6K) => &[208],
        _ => &[],
    };
    for block in data.chunks_exact_mut(dtype.block_bytes()) {
        for &at in halves {
            block[at..at + 2].copy_from_slice(&scale);
        }
        if matches!(
            dtype,
            DType::Quantized(QuantFormat::Int8 { .. } | QuantFormat::Int4 { .. })
        ) {
            block[..4].copy_from_slice(&0.01f32.to_le_bytes());
        }
    }
    Tensor::from_bytes(dtype, &[n, k], &data)
}

fn main() -> Result<(), kapel_core::Error> {
    // `BYTES=1`: activations in bytes.
    let bytes = std::env::var("BYTES").is_ok_and(|v| v == "1");
    let cpu = Cpu::new().with_byte_activations(bytes);
    println!(
        "{} threads, {:?}, activations in bytes: {bytes}",
        cpu.threads(),
        cpu.isa()
    );
    // Weights of 0.3 to 2 GB, far beyond the caches.
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|k| k.parse().ok())
        .unwrap_or(4096);
    let n = 131072 * 4096 / k;
    let formats = [
        DType::F32,
        DType::int8(32)?,
        DType::int4(32)?,
        DType::Quantized(QuantFormat::Q8_0),
        DType::Quantized(QuantFormat::Q4_0),
        DType::Quantized(QuantFormat::Q4K),
        DType::Quantized(QuantFormat::Q5K),
        DType::Quantized(QuantFormat::Q6K),
    ];
    for dtype in formats {
        let w = if dtype == DType::F32 {
            Tensor::from_slice(&vec![0.01f32; n * k], &[n, k])?
        } else {
            weights(dtype, n, k)?
        };
        let bytes = dtype.storage_bytes(n * k)? as f64;
        for m in [1, 4, 16] {
            let x = Tensor::from_slice(&vec![0.5f32; m * k], &[m, k])?;
            let time = if dtype == DType::F32 {
                let wt = w.transpose(0, 1)?;
                best(10, || cpu.forward(&Matmul, &[&x, &wt]).map(drop))?
            } else {
                best(10, || cpu.forward(&QuantizedMatmul, &[&x, &w]).map(drop))?
            };
            println!(
                "{:>9}, m = {m:2}: {:7.2} ms, {:5.1} GB/s of weights, {:6.1} G weights/s",
                dtype.to_string(),
                time * 1e3,
                bytes / time / 1e9,
                (n * k) as f64 / time / 1e9,
            );
        }
    }
    Ok(())
}
