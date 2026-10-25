//! LoRA adapters: the model unchanged at the start, the merged model equal
//! to the adapted one, gradients only for the adapters and right by finite
//! differences, a quantized base read in its codes, and PEFT's files.

use std::collections::BTreeMap;

use kapel_core::{DType, Tensor};
use kapel_models::Config;
use kapel_ref::Reference;
use kapel_train::Tape;
use kapel_train::lora::{self, LoraConfig};
use kapel_train::model::{Batch, Decoder, Options, Parameters, Rng, init};
use serde_json::{Value, json};

fn gpt2() -> Value {
    json!({
        "model_type": "gpt2", "n_embd": 16, "n_head": 2, "n_layer": 2, "vocab_size": 23,
        "n_positions": 12, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
    })
}

fn llama() -> Value {
    json!({
        "model_type": "llama", "hidden_size": 16, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 23,
        "intermediate_size": 24, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
    })
}

fn qwen3() -> Value {
    json!({
        "model_type": "qwen3", "hidden_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8, "num_hidden_layers": 2, "vocab_size": 23,
        "intermediate_size": 24, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true, "rope_theta": 1000.0,
    })
}

const IDS: [u32; 10] = [3, 17, 4, 4, 22, 0, 9, 13, 1, 5];

fn targets() -> Vec<u32> {
    IDS.iter().map(|&t| (t * 7 + 3) % 23).collect()
}

fn batch(t: &[u32]) -> Batch<'_> {
    Batch {
        ids: &IDS,
        targets: t,
        rows: 2,
        segments: None,
    }
}

fn logits(decoder: &Decoder, params: &Parameters, options: &Options) -> Vec<f64> {
    let tape = Tape::new(kapel_cpu::Cpu::new());
    let leaves = params.leaves(&tape);
    let t = targets();
    decoder
        .logits(&leaves, &batch(&t), options)
        .unwrap()
        .value()
        .to_f64_vec()
        .unwrap()
}

fn adapted(raw: &Value, embedding: bool) -> (Config, Parameters, LoraConfig) {
    let config = Config::from_json(raw).unwrap();
    let mut params = init(&config, 0.2, 1).unwrap();
    let lora = LoraConfig {
        r: 4,
        alpha: 8.0,
        embedding,
        ..LoraConfig::default()
    };
    lora::add_adapters(&mut params, &config, &lora, 7).unwrap();
    (config, params, lora)
}

/// Every factor drawn anew: the adapters then change the model.
fn randomize(params: &mut Parameters) {
    let mut rng = Rng::new(11);
    for (name, p) in params.iter_mut() {
        if lora::is_factor(name) {
            let values: Vec<f32> = (0..p.value.numel())
                .map(|_| (rng.normal() * 0.3) as f32)
                .collect();
            p.value = Tensor::from_slice(&values, p.value.shape()).unwrap();
        }
    }
}

fn close(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tolerance, "{x} and {y}");
    }
}

#[test]
fn new_adapters_leave_the_model_as_it_was() {
    for raw in [gpt2(), llama(), qwen3()] {
        let config = Config::from_json(&raw).unwrap();
        let plain = init(&config, 0.2, 1).unwrap();
        let (_, params, lora) = adapted(&raw, true);
        let options = Options::default();
        let before = logits(&Decoder::new(config.clone()).unwrap(), &plain, &options);
        let decoder = Decoder::new(config).unwrap().with_lora(lora.scale());
        assert_eq!(logits(&decoder, &params, &options), before);
        // Every base parameter is frozen, every factor trains.
        for (name, p) in params.iter() {
            assert_eq!(p.trainable, lora::is_factor(name), "{name}");
        }
    }
}

#[test]
fn the_merged_model_computes_what_the_adapted_one_does() {
    for raw in [gpt2(), llama(), qwen3()] {
        let (config, mut params, lora) = adapted(&raw, true);
        randomize(&mut params);
        let options = Options::default();
        let decoder = Decoder::new(config.clone())
            .unwrap()
            .with_lora(lora.scale());
        let with_adapters = logits(&decoder, &params, &options);
        let merged = lora::merge(&kapel_cpu::Cpu::new(), &params, &config, lora.scale()).unwrap();
        assert!(merged.iter().all(|(n, _)| !lora::is_factor(n)));
        // A tied model's output keeps the table as it was: no longer tied.
        let untied = lora::merged_config(&raw, &params, &config);
        assert_eq!(lora::unties(&params, &config), config.tie_embeddings);
        let untied = Config::from_json(&untied).unwrap();
        assert!(!untied.tie_embeddings);
        let plain = Decoder::new(untied).unwrap();
        let from_merged = logits(&plain, &merged, &options);
        close(&with_adapters, &from_merged, 1e-4);
        // And the adapters do change the model.
        let base = init(&config, 0.2, 1).unwrap();
        let before = logits(&Decoder::new(config.clone()).unwrap(), &base, &options);
        assert!(
            before
                .iter()
                .zip(&with_adapters)
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }
}

#[test]
fn only_the_adapters_receive_gradients_and_they_are_right() {
    for raw in [gpt2(), qwen3()] {
        let (config, mut params, lora) = adapted(&raw, true);
        randomize(&mut params);
        let params = params.to(&Reference, DType::F64).unwrap();
        let options = Options {
            dtype: DType::F64,
            chunk: 3,
            ..Options::default()
        };
        let decoder = Decoder::new(config).unwrap().with_lora(lora.scale());
        let t = targets();
        let loss_of = |params: &Parameters| -> f64 {
            let tape = Tape::new(Reference);
            let leaves = params.leaves(&tape);
            let loss = decoder.loss(&leaves, &batch(&t), &options).unwrap();
            loss.value().to_f64_vec().unwrap()[0]
        };
        let tape = Tape::new(Reference);
        let leaves = params.leaves(&tape);
        let loss = decoder.loss(&leaves, &batch(&t), &options).unwrap();
        let mut grads = tape.backward(&loss).unwrap();
        drop(loss);
        let grads: BTreeMap<String, Vec<f64>> = leaves
            .gradients(&mut grads)
            .into_iter()
            .map(|(n, g)| {
                let shape = params.get(&n).unwrap().value.shape().to_vec();
                (
                    n,
                    g.dense(&Reference, &shape).unwrap().to_f64_vec().unwrap(),
                )
            })
            .collect();
        let factors: Vec<&str> = params
            .iter()
            .map(|(n, _)| n)
            .filter(|n| lora::is_factor(n))
            .collect();
        assert_eq!(
            grads.keys().map(String::as_str).collect::<Vec<_>>(),
            factors
        );
        let h = 1e-6;
        for name in factors {
            let p = params.get(name).unwrap();
            let base = p.value.to_f64_vec().unwrap();
            for j in (0..base.len()).step_by(base.len().div_ceil(4)) {
                let at = |delta: f64| {
                    let mut moved = base.clone();
                    moved[j] += delta;
                    let mut ps = params.clone();
                    ps.get_mut(name).unwrap().value =
                        Tensor::from_slice(&moved, p.value.shape()).unwrap();
                    loss_of(&ps)
                };
                let numeric = (at(h) - at(-h)) / (2.0 * h);
                let analytic = grads[name][j];
                assert!(
                    (numeric - analytic).abs() < 1e-6 * (1.0 + numeric.abs()),
                    "{name}[{j}]: {analytic} by the tape, {numeric} by differences"
                );
            }
        }
    }
}

#[test]
fn a_quantized_base_computes_with_its_codes() {
    // Projections whose inputs are multiples of the blocks of 32.
    let raw = json!({
        "model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "num_hidden_layers": 2, "vocab_size": 23,
        "intermediate_size": 64, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true, "rope_theta": 1000.0,
    });
    let (config, mut params, lora) = adapted(&raw, false);
    randomize(&mut params);
    let dtype = DType::Quantized(kapel_core::QuantFormat::Q8_0);
    let mut quantized = params.clone();
    let count = lora::quantize_frozen(&mut quantized, &config, dtype).unwrap();
    assert_eq!(count, 2 * 7);
    // The same model with its weights dequantized, in floats.
    let mut dequantized = quantized.clone();
    for (_, p) in dequantized.iter_mut() {
        if p.value.dtype().is_quantized() {
            let values: Vec<f32> = p
                .value
                .to_f64_vec()
                .unwrap()
                .into_iter()
                .map(|v| v as f32)
                .collect();
            p.value = Tensor::from_slice(&values, p.value.shape()).unwrap();
        }
    }
    let options = Options::default();
    let decoder = Decoder::new(config).unwrap().with_lora(lora.scale());
    close(
        &logits(&decoder, &quantized, &options),
        &logits(&decoder, &dequantized, &options),
        1e-4,
    );
    // Its adapters still train.
    let tape = Tape::new(kapel_cpu::Cpu::new());
    let leaves = quantized.leaves(&tape);
    let t = targets();
    let loss = decoder.loss(&leaves, &batch(&t), &options).unwrap();
    let mut grads = tape.backward(&loss).unwrap();
    drop(loss);
    let names: Vec<String> = leaves
        .gradients(&mut grads)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names.len(), 2 * 7 * 2);
}

#[test]
fn an_adapter_written_as_peft_reads_back() {
    let (config, mut params, lora) = adapted(&qwen3(), true);
    randomize(&mut params);
    let adapted_modules = lora::projections(&params, &config, &lora.targets);
    let factors: BTreeMap<String, Tensor> = params
        .iter()
        .filter(|(n, _)| lora::is_factor(n))
        .map(|(n, p)| (n.to_string(), p.value.clone()))
        .collect();
    let dir = std::env::temp_dir().join(format!("kapel-peft-{}", std::process::id()));
    let peft = lora::peft_config(&lora, &config, &adapted_modules, "base");
    lora::save(&dir, &factors, &peft).unwrap();
    let file =
        kapel_hub::safetensors::SafeTensors::open(dir.join("adapter_model.safetensors")).unwrap();
    // PEFT's names, and its layout of the embeddings' A.
    let a = "base_model.model.model.embed_tokens.lora_embedding_A";
    assert_eq!(file.tensor(a).unwrap().shape(), &[4, 23]);
    assert!(
        file.names()
            .any(|n| n == "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight")
    );
    let back = lora::load(&dir).unwrap();
    assert_eq!(back.config.r, 4);
    assert_eq!(back.config.alpha, 8.0);
    assert!(back.config.embedding);
    assert_eq!(back.factors.len(), factors.len());
    for (n, t) in &factors {
        assert_eq!(back.factors[n].shape(), t.shape(), "{n}");
        assert_eq!(
            back.factors[n].to_f64_vec().unwrap(),
            t.to_f64_vec().unwrap()
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
