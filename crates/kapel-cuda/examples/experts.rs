//! Measures the products of a mixture of experts on the card, decoding
//! (`ExpertMatmul`, one token and its experts, activations in `bf16`), for
//! the shapes of Qwen3.6-35B-A3B (256 experts of 512, a token taking 8) and
//! Qwen3.8-Flash-Next (512 of 640, 10): the time of a product and the
//! bandwidth reached on the codes of the experts it reads. Each product is
//! timed alone, after one to compile and warm it.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example experts
//! [-- TOKENS]`

use std::time::Instant;

use kapel_core::{BF16, DType, F16, QuantFormat, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{ExpertMatmul, Run, Transfer};

/// Random bytes in `format`, with valid `f16` scales at the start of each
/// block (as the conformity suite draws them).
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
        block[..2].copy_from_slice(&F16::from_f64(0.01).0.to_le_bytes());
    }
    Tensor::from_bytes(dtype, &[n, k], &bytes).expect("whole blocks")
}

fn main() -> Result<(), kapel_core::Error> {
    let tokens: usize = std::env::args()
        .nth(1)
        .and_then(|r| r.parse().ok())
        .unwrap_or(1);
    let cuda = Cuda::new(0)?;
    // (model, experts, per token, hidden, width of an expert)
    let shapes: [(&str, usize, usize, usize, usize); 2] = [
        ("qwen3.6-35b-a3b", 256, 8, 2048, 512),
        ("qwen3.8-flash-next", 512, 10, 2560, 640),
    ];
    for (model, experts, k, hidden, width) in shapes {
        let products: [(QuantFormat, &str, usize, usize); 3] = [
            (QuantFormat::Iq3S, "gate", width, hidden),
            (QuantFormat::Iq4Xs, "down", hidden, width),
            (QuantFormat::Q8_0, "gate", width, hidden),
        ];
        for (format, what, n, kk) in products {
            if !kk.is_multiple_of(DType::Quantized(format).block_len()) {
                continue;
            }
            let w = cuda.upload(&weights(format, experts * n, kk))?;
            let values: Vec<BF16> = (0..tokens * kk)
                .map(|i| BF16::from_f64(((i * 7919) % 1000) as f64 / 500.0 - 1.0))
                .collect();
            let x = cuda.upload(&Tensor::from_slice(&values, &[tokens, kk])?)?;
            let chosen: Vec<u32> = (0..tokens * k).map(|i| (i * 37 % experts) as u32).collect();
            let idx = cuda.upload(&Tensor::from_slice(&chosen, &[tokens, k])?)?;
            let op = ExpertMatmul { experts };
            cuda.forward(&op, &[&x, &w, &idx])?;
            cuda.synchronize()?;
            let runs = 200;
            let start = Instant::now();
            for _ in 0..runs {
                cuda.forward(&op, &[&x, &w, &idx])?;
            }
            cuda.synchronize()?;
            let time = start.elapsed().as_secs_f64() / f64::from(runs);
            let bytes = DType::Quantized(format).storage_bytes(tokens * k * n * kk)? as f64;
            println!(
                "{model:>20} {what} {:>7} {kk:>5} → {n:>5} × {k}: {:>6.1} µs, {:>4.0} GB/s",
                DType::Quantized(format).to_string(),
                time * 1e6,
                bytes / time / 1e9
            );
        }
    }
    Ok(())
}
