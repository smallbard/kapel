//! Measures the products by quantized weights on the card, one row of
//! activations in `bf16` (decoding), for the shapes of Llama 3.2 1B and 3B:
//! the time of a product and the bandwidth reached on the codes. Each kernel
//! is timed alone, after one product to compile and warm it.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example quantized
//! [-- ROWS]`

use std::time::Instant;

use kapel_core::{BF16, DType, F16, QuantFormat, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{QuantizedMatmul, Run, Transfer};

/// Random bytes in `format`, with valid scales (as the conformity suite
/// draws them).
fn weights(format: QuantFormat, n: usize, k: usize) -> Tensor {
    let dtype = DType::Quantized(format);
    let blocks = n * k / dtype.block_len();
    let block_bytes = dtype.block_bytes();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut bytes = vec![0u8; blocks * block_bytes];
    for block in bytes.chunks_exact_mut(block_bytes) {
        for b in block.iter_mut() {
            *b = next() as u8;
        }
        let scale = |b: &mut [u8], at: usize| {
            b[at..at + 2].copy_from_slice(&F16::from_f64(0.01).0.to_le_bytes());
        };
        match format {
            QuantFormat::Q8_0 | QuantFormat::Q4_0 => scale(block, 0),
            QuantFormat::Q4K => {
                scale(block, 0);
                scale(block, 2);
            }
            QuantFormat::Q6K => scale(block, 208),
            _ => block[..4].copy_from_slice(&0.01f32.to_le_bytes()),
        }
    }
    Tensor::from_bytes(dtype, &[n, k], &bytes).expect("whole blocks")
}

fn main() -> Result<(), kapel_core::Error> {
    let rows: usize = std::env::args()
        .nth(1)
        .and_then(|r| r.parse().ok())
        .unwrap_or(1);
    let cuda = Cuda::new(0)?;
    let formats = [
        QuantFormat::Q4K,
        QuantFormat::Q6K,
        QuantFormat::Q8_0,
        QuantFormat::Int8 { group: 128 },
        QuantFormat::Int4 { group: 64 },
    ];
    for format in formats {
        for (k, n) in [
            (2048, 2048),
            (2048, 8192),
            (8192, 2048),
            (3072, 3072),
            (3072, 8192),
        ] {
            let w = cuda.upload(&weights(format, n, k))?;
            let values: Vec<BF16> = (0..rows * k)
                .map(|i| BF16::from_f64(((i * 7919) % 1000) as f64 / 500.0 - 1.0))
                .collect();
            let x = cuda.upload(&Tensor::from_slice(&values, &[rows, k])?)?;
            cuda.forward(&QuantizedMatmul, &[&x, &w])?;
            cuda.synchronize()?;
            let runs = 100;
            let start = Instant::now();
            for _ in 0..runs {
                cuda.forward(&QuantizedMatmul, &[&x, &w])?;
            }
            cuda.synchronize()?;
            let time = start.elapsed().as_secs_f64() / f64::from(runs);
            let bytes = w.dtype().storage_bytes(n * k)? as f64;
            println!(
                "{:>8} {k:>5} → {n:>5}: {:>7.1} µs, {:>4.0} GB/s",
                DType::Quantized(format).to_string(),
                time * 1e6,
                bytes / time / 1e9
            );
        }
    }
    Ok(())
}
