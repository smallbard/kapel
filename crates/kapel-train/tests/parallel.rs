//! Tensor parallelism on the processor: two and four ranks, each with its
//! heads and its part of the MLP, compute the loss and the gradients of the
//! whole model, within `f32`'s rounding.

use std::collections::BTreeMap;
use std::sync::Arc;

use kapel_core::Tensor;
use kapel_models::Config;
use kapel_train::Tape;
use kapel_train::dist::{Collective, Threads};
use kapel_train::model::{Batch, Decoder, Options, Parameters, init};
use kapel_train::parallel::{Group, join, local_config, partial, shard};
use serde_json::json;

const IDS: [u32; 12] = [3, 17, 4, 4, 22, 0, 9, 13, 1, 5, 11, 2];

/// The loss and the gradients of `params` with `decoder`, on the processor.
fn loss_and_grads(decoder: &Decoder, params: &Parameters) -> (f64, BTreeMap<String, Tensor>) {
    let cpu = kapel_cpu::Cpu::new();
    let tape = Tape::new(cpu.clone());
    let leaves = params.leaves(&tape);
    let targets: Vec<u32> = IDS.iter().map(|&t| (t * 7 + 3) % 23).collect();
    let batch = Batch {
        ids: &IDS,
        targets: &targets,
        rows: 2,
        segments: None,
    };
    let options = Options {
        dtype: kapel_core::DType::F32,
        checkpoint: true,
        chunk: 64,
    };
    let loss = decoder.loss(&leaves, &batch, &options).unwrap();
    let value = loss.value().to_f64_vec().unwrap()[0];
    let mut grads = tape.backward(&loss).unwrap();
    drop(loss);
    let grads = leaves
        .gradients(&mut grads)
        .into_iter()
        .map(|(n, g)| {
            let shape = params.get(&n).unwrap().value.shape().to_vec();
            (n, g.dense(&cpu, &shape).unwrap())
        })
        .collect();
    (value, grads)
}

fn close(name: &str, a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape(), "{name}");
    let (a, b) = (a.to_f64_vec().unwrap(), b.to_f64_vec().unwrap());
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-6);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-5 * scale, "{name}: {x} and {y}");
    }
}

#[test]
fn ranks_sharing_the_blocks_compute_the_gradients_of_the_whole_model() {
    let raw = json!({
        "model_type": "qwen3", "hidden_size": 32, "num_attention_heads": 4,
        "num_key_value_heads": 4, "head_dim": 8, "num_hidden_layers": 2, "vocab_size": 23,
        "intermediate_size": 48, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false, "rope_theta": 1000.0, "attention_bias": true,
    });
    let config = Config::from_json(&raw).unwrap();
    let params = init(&config, 0.2, 1).unwrap();
    let (whole_loss, whole) = loss_and_grads(&Decoder::new(config.clone()).unwrap(), &params);
    for ranks in [2, 4] {
        let handles: Vec<_> = Threads::group(ranks)
            .into_iter()
            .map(|g| {
                let (config, params) = (config.clone(), params.clone());
                std::thread::spawn(move || {
                    let rank = g.rank();
                    let group = Group(Arc::new(g));
                    let decoder = Decoder::new(local_config(&config, ranks).unwrap())
                        .unwrap()
                        .with_tensor_parallel(group)
                        .unwrap();
                    loss_and_grads(&decoder, &shard(&params, rank, ranks).unwrap())
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (loss, _) in &results {
            assert!(
                (loss - whole_loss).abs() <= 1e-6 * whole_loss.abs(),
                "{loss} and {whole_loss}"
            );
        }
        for (name, expected) in &whole {
            let parts: Vec<Tensor> = results.iter().map(|(_, g)| g[name].clone()).collect();
            let got = if partial(name) {
                let cpu = kapel_cpu::Cpu::new();
                parts.iter().skip(1).fold(parts[0].clone(), |acc, p| {
                    use kapel_ops::Run;
                    cpu.forward(
                        &kapel_ops::Binary {
                            kind: kapel_ops::BinaryKind::Add,
                        },
                        &[&acc, p],
                    )
                    .unwrap()
                })
            } else {
                join(name, &parts).unwrap()
            };
            close(&format!("{ranks} ranks, {name}"), &got, expected);
        }
    }
}
