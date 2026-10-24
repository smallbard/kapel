//! The decoders on a tape: the logits of the inference pass, gradients by
//! finite differences, and the same gradients with checkpointing.

use std::collections::BTreeMap;

use kapel_core::{DType, Tensor};
use kapel_models::{Config, Model};
use kapel_ops::Run;
use kapel_ref::Reference;
use kapel_train::model::{Batch, Decoder, Options, Parameters, init};
use kapel_train::{Compute, Tape};
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

fn weights(params: &Parameters, dtype: DType) -> BTreeMap<String, Tensor> {
    params
        .iter()
        .map(|(n, p)| {
            let v = Reference
                .forward(&kapel_ops::Cast { dtype }, &[&p.value])
                .unwrap();
            (n.to_string(), v)
        })
        .collect()
}

const IDS: [u32; 10] = [3, 17, 4, 4, 22, 0, 9, 13, 1, 5];

fn targets() -> Vec<u32> {
    IDS.iter().map(|&t| (t * 7 + 3) % 23).collect()
}

/// The logits of the two sequences of `IDS` on the processor, by the
/// decoder in one batch and by the inference pass one at a time.
fn compare_with_inference(raw: Value) {
    let config = Config::from_json(&raw).unwrap();
    let params = init(&config, 0.2, 1).unwrap();
    let model = Model::new(raw, weights(&params, DType::F32)).unwrap();
    let cpu = kapel_cpu::Cpu::new();
    let decoder = Decoder::new(config).unwrap();
    let tape = Tape::new(cpu.clone());
    let leaves = params.leaves(&tape);
    let t = targets();
    let batch = Batch {
        ids: &IDS,
        targets: &t,
        rows: 2,
    };
    let logits = decoder
        .logits(&leaves, &batch, &Options::default())
        .unwrap()
        .value()
        .to_f64_vec()
        .unwrap();
    let mut expected = Vec::new();
    for row in IDS.chunks(5) {
        expected.extend(model.forward(&cpu, row).unwrap().to_f64_vec().unwrap());
    }
    assert_eq!(logits.len(), expected.len());
    for (a, b) in logits.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "{a} and {b}");
    }
}

#[test]
fn the_decoders_give_the_logits_of_inference() {
    compare_with_inference(gpt2());
    compare_with_inference(llama());
    compare_with_inference(qwen3());
}

/// The loss and the gradients of every parameter on `backend`, dense.
fn loss_and_gradients<B: Compute>(
    backend: B,
    config: &Config,
    params: &Parameters,
    options: &Options,
) -> (f64, BTreeMap<String, Vec<f64>>) {
    let decoder = Decoder::new(config.clone()).unwrap();
    let tape = Tape::new(backend.clone());
    let leaves = params.leaves(&tape);
    let t = targets();
    let batch = Batch {
        ids: &IDS,
        targets: &t,
        rows: 2,
    };
    let loss = decoder.loss(&leaves, &batch, options).unwrap();
    let value = backend
        .download(&loss.value())
        .unwrap()
        .to_f64_vec()
        .unwrap()[0];
    let mut grads = tape.backward(&loss).unwrap();
    drop(loss);
    let named = leaves
        .gradients(&mut grads)
        .into_iter()
        .map(|(n, g)| {
            let shape = params.get(&n).unwrap().value.shape().to_vec();
            let g = g.dense(&backend, &shape).unwrap();
            (n, backend.download(&g).unwrap().to_f64_vec().unwrap())
        })
        .collect();
    (value, named)
}

fn in_f64(params: &Parameters) -> Parameters {
    params.to(&Reference, DType::F64).unwrap()
}

fn check_by_differences(raw: Value) {
    let config = Config::from_json(&raw).unwrap();
    let params = in_f64(&init(&config, 0.3, 2).unwrap());
    let options = Options {
        dtype: DType::F64,
        chunk: 3,
        ..Options::default()
    };
    let (_, grads) = loss_and_gradients(Reference, &config, &params, &options);
    assert_eq!(
        grads.len(),
        params.iter().count(),
        "a parameter has no gradient"
    );
    let h = 1e-6;
    for (name, p) in params.iter() {
        let base = p.value.to_f64_vec().unwrap();
        // A few elements of each parameter.
        for j in (0..base.len()).step_by(base.len().div_ceil(3)) {
            let at = |delta: f64| {
                let mut moved = base.clone();
                moved[j] += delta;
                let mut params = params.clone();
                params.get_mut(name).unwrap().value =
                    Tensor::from_slice(&moved, p.value.shape()).unwrap();
                loss_and_gradients(Reference, &config, &params, &options).0
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            let analytic = grads[name][j];
            assert!(
                (numeric - analytic).abs() <= 1e-6 * (1.0 + numeric.abs()),
                "{name}[{j}]: {analytic} by the tape, {numeric} by differences"
            );
        }
    }
}

#[test]
fn the_gradients_of_the_decoders_match_finite_differences() {
    check_by_differences(gpt2());
    check_by_differences(llama());
    check_by_differences(qwen3());
}

#[test]
fn checkpointing_changes_no_gradient() {
    for raw in [gpt2(), llama()] {
        let config = Config::from_json(&raw).unwrap();
        let params = init(&config, 0.2, 3).unwrap();
        let cpu = kapel_cpu::Cpu::new();
        let plain = loss_and_gradients(cpu.clone(), &config, &params, &Options::default());
        let saved = loss_and_gradients(
            cpu,
            &config,
            &params,
            &Options {
                checkpoint: true,
                ..Options::default()
            },
        );
        assert_eq!(plain, saved);
    }
}

#[test]
fn a_trained_model_is_a_model_again() {
    let raw = gpt2();
    let config = Config::from_json(&raw).unwrap();
    let params = init(&config, 0.02, 4).unwrap();
    let like = Model::new(raw, weights(&params, DType::F32)).unwrap();
    let model = params
        .to_model(&kapel_cpu::Cpu::new(), &like, DType::BF16)
        .unwrap();
    assert_eq!(
        model
            .tensor("transformer.h.1.mlp.c_fc.weight")
            .unwrap()
            .dtype(),
        DType::BF16
    );
    assert_eq!(model.names().count(), params.iter().count());
}

#[test]
fn a_frozen_parameter_receives_no_gradient_and_changes_no_other() {
    let raw = llama();
    let config = Config::from_json(&raw).unwrap();
    let params = init(&config, 0.2, 5).unwrap();
    let mut frozen = params.clone();
    frozen
        .get_mut("model.embed_tokens.weight")
        .unwrap()
        .trainable = false;
    let cpu = kapel_cpu::Cpu::new();
    let (a, ga) = loss_and_gradients(cpu.clone(), &config, &params, &Options::default());
    let (b, gb) = loss_and_gradients(cpu, &config, &frozen, &Options::default());
    assert_eq!(a, b);
    assert!(!gb.contains_key("model.embed_tokens.weight"));
    for (name, g) in &gb {
        assert_eq!(g, &ga[name], "{name}");
    }
}
