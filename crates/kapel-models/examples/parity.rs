//! Compares Kapel's logits with those of `transformers`, saved by
//! `tools/parity/reference.py` in `data/parity/<model>/`.
//!
//! `cargo run --release -p kapel-models --example parity [models...]`

use std::path::Path;

use kapel_hub::{Checkpoint, SafeTensors};
use kapel_models::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let mut names: Vec<String> = std::env::args().skip(1).collect();
    if names.is_empty() {
        names = std::fs::read_dir(&root)?
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        names.sort();
    }
    let cpu = kapel_cpu::Cpu::new();
    let mut worst_all = 0.0f64;
    for name in names {
        let dir = root.join(&name);
        let model = match Checkpoint::open(&dir)
            .map_err(Into::into)
            .and_then(|c| Model::load(&c))
        {
            Ok(model) => model,
            Err(error) => {
                println!("{name:10} skipped: {error}");
                continue;
            }
        };
        let reference = SafeTensors::open(dir.join("reference.safetensors"))?;
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for i in 0.. {
            let Ok(ids) = reference.tensor(&format!("ids_{i}")) else {
                break;
            };
            let ids: Vec<u32> = ids.to_vec::<i32>()?.into_iter().map(|v| v as u32).collect();
            let want = reference.tensor(&format!("logits_{i}"))?.to_vec::<f32>()?;
            let got = model.forward(&cpu, &ids)?.to_vec::<f32>()?;
            for (a, b) in got.iter().zip(&want) {
                worst = worst.max(f64::from((a - b).abs()));
                scale = scale.max(f64::from(b.abs()));
            }
        }
        let relative = worst / scale.max(1e-30);
        worst_all = worst_all.max(relative);
        println!(
            "{name:10} {:9} max |Δ| {worst:.2e}, max |logit| {scale:.2e}, relative {relative:.2e}",
            model.config().model_type
        );
    }
    println!("worst relative difference: {worst_all:.2e}");
    Ok(())
}
