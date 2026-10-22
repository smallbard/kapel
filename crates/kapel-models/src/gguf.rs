//! Models read from GGUF, the format of llama.cpp.
//!
//! A GGUF file holds the configuration as metadata (`llama.block_count`, …)
//! and the weights under llama.cpp's names (`blk.0.attn_q.weight`, …). Both
//! are translated to what a model directory of `transformers` holds: a
//! `config.json` and the names of its checkpoint. The weights stay in the
//! mapped file, quantized or not, and are read in place.
//!
//! Two conversions of llama.cpp are undone in the configuration rather than
//! in the weights:
//!
//! - For Llama (and Mistral, which GGUF files as Llama), the query and key
//!   projections are stored with the components of each head reordered so
//!   that RoPE pairs neighbours (`2j` with `2j + 1`) instead of halves (`j`
//!   with `j + D/2`). Attention only sees products of queries and keys, so
//!   the reordered weights with the interleaved RoPE give the original model.
//! - The RoPE scaling of Llama 3 is stored as its per-frequency factors
//!   (`rope_freqs.weight`); the parameters that give them are recovered and
//!   checked against them.

use std::collections::BTreeMap;

use kapel_hub::Gguf;
use kapel_hub::gguf::Value as Meta;
use kapel_ops::RopeLayout;
use serde_json::{Value, json};

use crate::ModelError;
use crate::config::Config;
use crate::model::Model;

type Result<T> = std::result::Result<T, ModelError>;

/// llama.cpp's names of the weights of a block, and those of `transformers`.
const BLOCK_NAMES: [(&str, &str); 14] = [
    ("attn_norm", "input_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("post_attention_norm", "post_attention_layernorm"),
    ("pre_feedforward_norm", "pre_feedforward_layernorm"),
    ("post_ffw_norm", "post_feedforward_layernorm"),
];

/// The `transformers` name of a GGUF tensor, if the model uses it.
fn hf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (base, kind) = rest.rsplit_once('.')?;
    let (_, hf) = BLOCK_NAMES.iter().find(|(gguf, _)| *gguf == base)?;
    Some(format!("model.layers.{layer}.{hf}.{kind}"))
}

/// The metadata of a GGUF file.
struct Metadata<'a> {
    gguf: &'a Gguf,
    arch: String,
}

impl Metadata<'_> {
    fn get(&self, key: &str) -> Option<&Meta> {
        self.gguf.metadata.get(&format!("{}.{key}", self.arch))
    }

    fn number(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Meta::Uint(v) => Some(*v as f64),
            Meta::Int(v) => Some(*v as f64),
            Meta::Float(v) => Some(*v),
            _ => None,
        }
    }

    fn count(&self, key: &str) -> Result<u64> {
        match self.get(key) {
            Some(Meta::Uint(v)) => Ok(*v),
            Some(Meta::Int(v)) if *v >= 0 => Ok(*v as u64),
            // Some files give one count per layer: they must agree.
            Some(Meta::Array(values)) => {
                let mut counts = values.iter().map(|v| match v {
                    Meta::Uint(v) => Some(*v),
                    Meta::Int(v) if *v >= 0 => Some(*v as u64),
                    _ => None,
                });
                let first = counts.next().flatten();
                match first {
                    Some(n) if counts.all(|c| c == Some(n)) => Ok(n),
                    _ => Err(ModelError::Config(format!(
                        "{}.{key}: counts that differ by layer are not supported",
                        self.arch
                    ))),
                }
            }
            _ => Err(ModelError::Config(format!(
                "the GGUF file has no {}.{key}",
                self.arch
            ))),
        }
    }
}

/// The parameters of the RoPE scaling of Llama 3 whose factors are `factors`
/// (`rope_freqs.weight`, one per pair of components), or an error if no such
/// parameters give them.
fn llama3_scaling(factors: &[f64], theta: f64, rotary: usize, original: f64) -> Result<Value> {
    let factor = factors.iter().copied().fold(1.0, f64::max);
    // The factors of `transformers`' `_compute_llama3_parameters`, as
    // llama.cpp's converter writes them.
    let (low, high) = (1.0, 4.0);
    let expected = |j: usize| {
        let frequency = theta.powf(-2.0 * j as f64 / rotary as f64);
        let wavelength = 2.0 * std::f64::consts::PI / frequency;
        if wavelength < original / high {
            1.0
        } else if wavelength > original / low {
            factor
        } else {
            let smooth = (original / wavelength - low) / (high - low);
            1.0 / ((1.0 - smooth) / factor + smooth)
        }
    };
    let matches = factors.len() == rotary / 2
        && factors
            .iter()
            .enumerate()
            .all(|(j, &f)| (f - expected(j)).abs() <= 1e-5 * f.abs().max(1.0));
    if !matches {
        return Err(ModelError::Config(
            "rope_freqs.weight: factors that are not those of Llama 3 are not supported".into(),
        ));
    }
    Ok(json!({
        "rope_type": "llama3",
        "factor": factor,
        "low_freq_factor": low,
        "high_freq_factor": high,
        "original_max_position_embeddings": original as u64,
    }))
}

impl Model {
    /// Reads a model from a GGUF file: its configuration from the metadata,
    /// its weights mapped in place, quantized or not.
    ///
    /// Llama (with Mistral), Qwen 2 and Qwen 3 are read.
    ///
    /// # Errors
    ///
    /// For another architecture, missing metadata, or weights of a type
    /// Kapel does not read.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let arch = match gguf.metadata.get("general.architecture") {
            Some(Meta::String(arch)) => arch.clone(),
            _ => {
                return Err(ModelError::Config(
                    "the GGUF file has no architecture".into(),
                ));
            }
        };
        let model_type = match arch.as_str() {
            "llama" | "qwen2" | "qwen3" => arch.clone(),
            other => {
                return Err(ModelError::Config(format!(
                    "GGUF architecture {other:?} is not supported"
                )));
            }
        };
        let meta = Metadata { gguf, arch };
        let mut weights = BTreeMap::new();
        for name in gguf.names() {
            if let Some(hf) = hf_name(name) {
                weights.insert(hf, gguf.tensor(name)?);
            }
        }
        let embed = weights
            .get("model.embed_tokens.weight")
            .ok_or_else(|| ModelError::MissingWeight("token_embd.weight".into()))?;
        let vocab = embed.shape()[0];
        let heads = meta.count("attention.head_count")?;
        let hidden = meta.count("embedding_length")?;
        let head_dim = match meta.count("attention.key_length") {
            Ok(d) => d,
            Err(_) => hidden / heads.max(1),
        };
        let theta = meta.number("rope.freq_base").unwrap_or(10_000.0);
        let mut config = json!({
            "model_type": model_type,
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": meta.count("block_count")?,
            "num_attention_heads": heads,
            "num_key_value_heads": meta.count("attention.head_count_kv").unwrap_or(heads),
            "head_dim": head_dim,
            "intermediate_size": meta.count("feed_forward_length")?,
            "hidden_act": "silu",
            "rms_norm_eps": meta.number("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            "rope_theta": theta,
            "max_position_embeddings": meta.count("context_length").unwrap_or(4096),
            "tie_word_embeddings": !weights.contains_key("lm_head.weight"),
            "attention_bias": weights.contains_key("model.layers.0.self_attn.q_proj.bias"),
        });
        if let Some(factors) = gguf.names().find(|n| *n == "rope_freqs.weight") {
            let factors = gguf.tensor(factors)?.to_f64_vec()?;
            let rotary = meta
                .count("rope.dimension_count")
                .map_or(head_dim as usize, |r| r as usize);
            let original = meta
                .number("rope.scaling.original_context_length")
                .unwrap_or(8192.0);
            config["rope_scaling"] = llama3_scaling(&factors, theta, rotary, original)?;
        }
        let mut parsed = Config::from_json(&config)?;
        if parsed.model_type == "llama" {
            parsed.rope_layout = RopeLayout::Interleaved;
        }
        Ok(Self::from_parts(parsed, config, weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_transformers() {
        assert_eq!(
            hf_name("blk.3.attn_q.weight").as_deref(),
            Some("model.layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(
            hf_name("blk.0.attn_k.bias").as_deref(),
            Some("model.layers.0.self_attn.k_proj.bias")
        );
        assert_eq!(
            hf_name("blk.11.ffn_down.weight").as_deref(),
            Some("model.layers.11.mlp.down_proj.weight")
        );
        assert_eq!(
            hf_name("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(hf_name("rope_freqs.weight"), None);
        assert_eq!(hf_name("blk.x.attn_q.weight"), None);
    }

    #[test]
    fn llama3_factors_give_their_parameters() {
        // Llama 3.2: θ = 500 000, 64 rotated components, factor 32.
        let (theta, rotary, original) = (500_000.0, 64, 8192.0);
        let factors: Vec<f64> = (0..rotary / 2)
            .map(|j| {
                let frequency: f64 = f64::powf(theta, -2.0 * j as f64 / rotary as f64);
                let wavelength = 2.0 * std::f64::consts::PI / frequency;
                if wavelength < original / 4.0 {
                    1.0
                } else if wavelength > original {
                    32.0
                } else {
                    let smooth = (original / wavelength - 1.0) / 3.0;
                    1.0 / ((1.0 - smooth) / 32.0 + smooth)
                }
            })
            .collect();
        let scaling = llama3_scaling(&factors, theta, rotary, original).unwrap();
        assert_eq!(scaling["factor"], 32.0);
        let mut wrong = factors.clone();
        wrong[10] *= 1.5;
        assert!(llama3_scaling(&wrong, theta, rotary, original).is_err());
    }
}
