//! Counting the computation of a step, for the model's FLOP utilization
//! (MFU): the operations its definition asks for, over the time the card
//! took, against the card's peak.
//!
//! Nearly all the computation of a transformer is in its products: `[M, K]
//! · [K, N]` costs `M · K · N` multiply-adds, two operations each. Norms,
//! activations and softmax cost a few operations per element and are not
//! counted. The attention counts its whole score matrix, as Smallem and
//! llm.c count it, though a causal kernel computes half of it. The backward
//! pass computes two products for each product of the forward pass: a step
//! costs three forward passes.

use kapel_models::{Block, Config};

/// The multiply-adds of a forward pass over `rows` sequences of `length`
/// tokens.
pub fn forward_macs(config: &Config, rows: usize, length: usize) -> u64 {
    let c = config;
    let tokens = (rows * length) as u64;
    let t = length as u64;
    let (h, i, v) = (c.hidden as u64, c.intermediate as u64, c.vocab as u64);
    let (q, kv) = (
        (c.heads * c.head_dim) as u64,
        (c.kv_heads * c.head_dim) as u64,
    );
    let projections = tokens * h * (2 * q + 2 * kv);
    let mlp = match c.block {
        Block::Gpt2 | Block::Parallel => 2 * tokens * h * i,
        Block::Sequential | Block::Sandwich => 3 * tokens * h * i,
    };
    // Scores `q · kᵀ` and the mix of the values: `T² · H · D` each.
    let attention = 2 * rows as u64 * t * t * q;
    c.layers as u64 * (projections + mlp + attention) + tokens * h * v
}

/// The operations of a training step over `tokens` tokens in sequences of
/// `length`: three forward passes, two operations per multiply-add.
pub fn step_flops(config: &Config, tokens: usize, length: usize) -> u64 {
    6 * forward_macs(config, tokens / length.max(1), length)
}

/// The peak of a card in the given type, in Top/s, when Kapel knows it: the
/// dense tensor operations of NVIDIA's data sheets, accumulated in `f32`.
pub fn peak_tflops(name: &str, dtype: kapel_core::DType) -> Option<f64> {
    use kapel_core::DType;
    let n = name.to_lowercase();
    let half = matches!(dtype, DType::BF16 | DType::F16);
    let table: [(&str, f64, f64); 8] = [
        ("h100", 989.0, 67.0),
        ("a100", 312.0, 19.5),
        ("l40s", 362.0, 91.6),
        ("rtx 4090", 165.2, 82.6),
        ("rtx 3090", 71.0, 35.6),
        ("a6000", 154.8, 38.7),
        ("rtx 4080", 97.5, 48.7),
        ("a10", 125.0, 31.2),
    ];
    table
        .iter()
        .find(|(k, _, _)| n.contains(k))
        .map(|&(_, h16, f32)| if half { h16 } else { f32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_small_costs_what_nanogpt_counts() {
        let config = Config::from_json(&serde_json::json!({
            "model_type": "gpt2", "n_embd": 768, "n_head": 12, "n_layer": 12,
            "vocab_size": 50257, "n_positions": 1024,
        }))
        .unwrap();
        // nanoGPT's estimate: 6 N + 12 L H Q T per token, N the parameters
        // but the position table (and the output table counted once).
        let per_token = step_flops(&config, 1024, 1024) as f64 / 1024.0;
        let n = 85_000_000.0 + 768.0 * 50257.0;
        let nanogpt = 6.0 * n + 12.0 * 12.0 * 768.0 * 1024.0;
        assert!(
            (per_token / nanogpt - 1.0).abs() < 0.01,
            "{per_token} {nanogpt}"
        );
    }
}
