//! The operations a training step asks for, counted call by call
//! ([`kapel_train::profile::Profiled`]), against what [`kapel_train::flops`]
//! predicts from the configuration alone.

use kapel_models::Config;
use kapel_train::model::{Batch, Decoder, Options, init};
use kapel_train::profile::Profiled;
use kapel_train::{Tape, flops};
use serde_json::{Value, json};

/// One forward and backward pass of `rows × length` tokens: the
/// operations counted, the scores' forward ones among them, and those
/// [`flops::step_flops`] predicts.
fn counted(raw: &Value, rows: usize, length: usize) -> (f64, f64, f64) {
    let config = Config::from_json(raw).unwrap();
    let params = init(&config, 0.02, 1).unwrap();
    let cpu = Profiled::new(kapel_cpu::Cpu::new(), || {});
    let decoder = Decoder::new(config.clone()).unwrap();
    let tape = Tape::new(cpu.clone());
    let leaves = params.leaves(&tape);
    let ids: Vec<u32> = (0..rows * length).map(|i| (i * 7 % 50) as u32).collect();
    let targets: Vec<u32> = ids.iter().map(|t| (t + 1) % 50).collect();
    let batch = Batch {
        ids: &ids,
        targets: &targets,
        rows,
        segments: None,
    };
    let loss = decoder.loss(&leaves, &batch, &Options::default()).unwrap();
    tape.backward(&loss).unwrap();
    let attention = cpu.spent().get("attention").map_or(0.0, |s| s.operations);
    let predicted = flops::step_flops(&config, rows * length, length) as f64;
    (cpu.operations(), attention, predicted)
}

/// Every product a step asks for is predicted, to the operation, for each
/// kind of block: but the scores, which the backward pass of the attention
/// computes again (it keeps their statistics, not the probabilities): half
/// a forward attention more, which the count leaves out, as llm.c's and
/// Smallem's do.
#[test]
fn a_step_asks_for_the_operations_predicted() {
    let models = [
        json!({
            "model_type": "gpt2", "n_embd": 32, "n_head": 4, "n_layer": 2, "vocab_size": 64,
            "n_positions": 32,
        }),
        json!({
            "model_type": "llama", "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 64,
            "intermediate_size": 48, "hidden_act": "silu", "tie_word_embeddings": true,
            "attention_bias": true, "mlp_bias": true,
        }),
        json!({
            "model_type": "qwen3", "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 1, "head_dim": 16, "num_hidden_layers": 2,
            "vocab_size": 64, "intermediate_size": 48, "hidden_act": "silu",
            "tie_word_embeddings": false,
        }),
        // A window of 5 keys on every other layer, as Gemma 2's.
        json!({
            "model_type": "gemma2", "hidden_size": 32, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "num_hidden_layers": 2,
            "vocab_size": 64, "intermediate_size": 48, "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 5, "query_pre_attn_scalar": 16,
        }),
    ];
    for raw in &models {
        let (counted, scores, predicted) = counted(raw, 2, 16);
        assert!(
            (counted - predicted - scores / 2.0).abs() < 0.5,
            "{}: counted {counted}, predicted {predicted}, scores {scores}",
            raw["model_type"]
        );
    }
}
