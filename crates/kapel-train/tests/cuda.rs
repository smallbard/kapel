//! Training on the card: the processor's gradients, and checkpointing that
//! changes no bit.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use kapel_core::DType;
use kapel_models::Config;
use kapel_train::model::{Batch, Decoder, Options, Parameters, init};
use kapel_train::{Compute, Tape};
use serde_json::json;

fn configs() -> Vec<Config> {
    [
        json!({
            "model_type": "gpt2", "n_embd": 64, "n_head": 4, "n_layer": 2, "vocab_size": 101,
            "n_positions": 64, "activation_function": "gelu_new",
        }),
        json!({
            "model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 2,
            "vocab_size": 101, "intermediate_size": 96, "tie_word_embeddings": true,
        }),
    ]
    .iter()
    .map(|raw| Config::from_json(raw).unwrap())
    .collect()
}

fn run<B: Compute>(
    backend: B,
    config: &Config,
    params: &Parameters,
    options: &Options,
) -> (f64, BTreeMap<String, Vec<f64>>) {
    let params = params.to(&backend, DType::F32).unwrap();
    let decoder = Decoder::new(config.clone()).unwrap();
    let ids: Vec<u32> = (0..96).map(|i| (i * 37 + 11) % 101).collect();
    let mut targets: Vec<u32> = ids.iter().map(|t| (t * 3 + 1) % 101).collect();
    targets[5] = kapel_train::IGNORE;
    let tape = Tape::new(backend.clone());
    let leaves = params.leaves(&tape);
    let batch = Batch {
        ids: &ids,
        targets: &targets,
        rows: 3,
    };
    let loss = decoder.loss(&leaves, &batch, options).unwrap();
    let value = backend
        .download(&loss.value())
        .unwrap()
        .to_f64_vec()
        .unwrap()[0];
    let mut grads = tape.backward(&loss).unwrap();
    drop(loss);
    let grads = leaves
        .gradients(&mut grads)
        .into_iter()
        .map(|(n, g)| {
            let shape = params.get(&n).unwrap().value.shape().to_vec();
            let g = g.dense(&backend, &shape).unwrap();
            (n, backend.download(&g).unwrap().to_f64_vec().unwrap())
        })
        .collect();
    (value, grads)
}

#[test]
fn the_card_gives_the_gradients_of_the_processor() {
    let cuda = kapel_cuda::Cuda::new(0).unwrap();
    for config in configs() {
        let params = init(&config, 0.1, 7).unwrap();
        let options = Options {
            chunk: 40,
            ..Options::default()
        };
        let (a, ga) = run(kapel_cpu::Cpu::new(), &config, &params, &options);
        let (b, gb) = run(cuda.clone(), &config, &params, &options);
        assert!((a - b).abs() < 1e-5 * a.abs(), "{a} and {b}");
        assert_eq!(ga.len(), gb.len());
        for (name, x) in &ga {
            let y = &gb[name];
            let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let diff = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                diff <= 1e-4 * norm.max(1e-6),
                "{}: {name}: {diff} of {norm}",
                config.model_type
            );
        }
    }
}

#[test]
fn checkpointing_on_the_card_changes_no_bit() {
    let cuda = kapel_cuda::Cuda::new(0).unwrap();
    for config in configs() {
        let params = init(&config, 0.1, 8).unwrap();
        let options = Options {
            dtype: DType::BF16,
            chunk: 40,
            ..Options::default()
        };
        let plain = run(cuda.clone(), &config, &params, &options);
        let saved = run(
            cuda.clone(),
            &config,
            &params,
            &Options {
                checkpoint: true,
                ..options
            },
        );
        assert_eq!(plain, saved);
    }
}

#[test]
fn attentions_kept_for_their_backward_change_no_bit() {
    let cuda = kapel_cuda::Cuda::new(0).unwrap();
    let config = &configs()[1];
    let params = init(config, 0.1, 9).unwrap();
    let options = Options {
        dtype: DType::BF16,
        chunk: 40,
        ..Options::default()
    };
    let computed = run(cuda.clone(), config, &params, &options);
    cuda.keep_attention_for_backward(true);
    let kept = run(cuda.clone(), config, &params, &options);
    cuda.keep_attention_for_backward(false);
    assert_eq!(computed, kept);
}
