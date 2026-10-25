//! LoRA adapters against PEFT's (`tools/train/peft_parity.py`):
//!
//! ```text
//! cargo run --release -p kapel-train --example lora -- write MODEL OUT
//! cargo run --release -p kapel-train --example lora -- read MODEL ADAPTER
//! ```
//!
//! `write` adds adapters of rank 4 to every projection and to the
//! embeddings of the model directory MODEL, draws all their factors, writes
//! them into OUT as PEFT does, and the logits of the model with them into
//! `OUT/kapel_logits.json`. `read` reads the adapter PEFT (or `write`) wrote
//! into ADAPTER, and prints the logits of the model with it. The logits are
//! those of the ids of [`IDS`], in `f32` on the processor.

use std::collections::BTreeMap;
use std::path::Path;

use kapel_core::{DType, Tensor};
use kapel_models::{Config, Model};
use kapel_train::Tape;
use kapel_train::lora::{self, LoraConfig};
use kapel_train::model::{Batch, Decoder, Options, Parameters, Rng};

/// The ids whose logits are compared: one row.
const IDS: [u32; 8] = [3, 17, 4, 4, 22, 0, 9, 13];

fn logits(config: &Config, params: &Parameters, scale: f64) -> Vec<f64> {
    let cpu = kapel_cpu::Cpu::new();
    let tape = Tape::new(cpu);
    let leaves = params.leaves(&tape);
    let targets = [kapel_train::IGNORE; IDS.len()];
    let batch = Batch {
        ids: &IDS,
        targets: &targets,
        rows: 1,
        segments: None,
    };
    let decoder = Decoder::new(config.clone())
        .expect("a decoder")
        .with_lora(scale);
    decoder
        .logits(&leaves, &batch, &Options::default())
        .and_then(|l| l.value().to_f64_vec())
        .expect("the logits")
}

fn base(dir: &Path) -> (Config, Parameters) {
    let model = Model::load(&kapel_hub::Checkpoint::open(dir).expect("a model")).expect("a model");
    let config = model.config().clone();
    let params = Parameters::from_model(&model)
        .to(&kapel_cpu::Cpu::new(), DType::F32)
        .expect("f32 weights");
    (config, params)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [what, model, dir] = &args[..] else {
        return Err("usage: lora write MODEL OUT | lora read MODEL ADAPTER".into());
    };
    let (config, mut params) = base(Path::new(model));
    match what.as_str() {
        "write" => {
            let settings = LoraConfig {
                r: 4,
                alpha: 8.0,
                embedding: true,
                ..LoraConfig::default()
            };
            let adapted = lora::add_adapters(&mut params, &config, &settings, 1)?;
            let mut rng = Rng::new(5);
            let mut factors = BTreeMap::new();
            for (name, p) in params.iter_mut() {
                if lora::is_factor(name) {
                    let v: Vec<f32> = (0..p.value.numel())
                        .map(|_| (rng.normal() * 0.2) as f32)
                        .collect();
                    p.value = Tensor::from_slice(&v, p.value.shape())?;
                    factors.insert(name.to_string(), p.value.clone());
                }
            }
            let peft = lora::peft_config(&settings, &config, &adapted, model);
            lora::save(Path::new(dir), &factors, &peft)?;
            let out = logits(&config, &params, settings.scale());
            std::fs::write(
                Path::new(dir).join("kapel_logits.json"),
                serde_json::to_vec(&out)?,
            )?;
            eprintln!(
                "{} factors of {} projections written to {dir}",
                factors.len(),
                adapted.len()
            );
        }
        "read" => {
            let adapter = lora::load(Path::new(dir))?;
            for (name, t) in adapter.factors {
                let mut p = kapel_train::model::Parameter::new(t);
                p.sparse = name.ends_with(".lora_embedding_A");
                params.insert(name, p);
            }
            let out = logits(&config, &params, adapter.config.scale());
            println!("{}", serde_json::to_string(&out)?);
        }
        other => return Err(format!("{other}: write or read").into()),
    }
    Ok(())
}
