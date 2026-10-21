//! Measures a model's forward pass over a prompt on the card (the prefill), in
//! tokens per second, in `f32` and in `bf16`. Measure on an idle device.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example prefill -- MODEL_DIR`

use std::time::Instant;

use kapel_core::DType;
use kapel_cuda::Cuda;
use kapel_hub::Checkpoint;
use kapel_models::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("usage: prefill MODEL_DIR")?;
    let cuda = Cuda::new(0)?;
    let model = Model::load(&Checkpoint::open(&dir)?)?.upload(&cuda)?;
    let vocab = model.config().vocab as u32;
    for (dtype, model) in [
        (DType::F32, model.cast(&cuda, DType::F32)?),
        (DType::BF16, model.cast(&cuda, DType::BF16)?),
    ] {
        for t in [128, 512, 2048] {
            let ids: Vec<u32> = (0..t as u32).map(|i| (i * 7919 + 13) % vocab).collect();
            // The first run compiles the tile kernels, or reads them from the cache.
            model.forward(&cuda, &ids)?;
            cuda.synchronize()?;
            let runs = 5;
            let start = Instant::now();
            for _ in 0..runs {
                model.forward(&cuda, &ids)?;
            }
            cuda.synchronize()?;
            let time = start.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "{dtype} {t} tokens: {:.1} ms, {:.0} tokens/s",
                time * 1e3,
                t as f64 / time
            );
        }
    }
    Ok(())
}
