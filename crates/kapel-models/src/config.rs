//! Architectures described by configuration.
//!
//! `config.json` names the architecture (`model_type`) and its sizes. Most
//! decoders of the Hub are the same transformer with a few switches: where the
//! norms sit, whether the projections have biases, how the positions enter, the
//! activation of the MLP. [`Config`] holds these switches, read from the file;
//! the forward pass in [`crate::Model`] follows them.

use kapel_ops::{RopeLayout, RopeScaling, UnaryKind};
use serde_json::Value;

use crate::ModelError;

/// The shape of a transformer block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// `h = x + attn(norm(x))`, then `h + mlp(norm(h))`, with a gated MLP
    /// `down(act(gate(x)) · up(x))`: Llama, Mistral, Qwen 2 and 3, SmolLM.
    Sequential,
    /// As `Sequential`, with a norm after the attention and after the MLP too,
    /// before they are added back: Gemma 2.
    Sandwich,
    /// `x + attn(norm(x)) + mlp(norm(x))`, one norm for both, and an MLP
    /// `fc2(act(fc1(x)))`: Phi.
    Parallel,
    /// GPT-2: learned positions, a fused query-key-value projection stored as
    /// `[in, out]`, and an MLP `c_proj(act(c_fc(x)))`.
    Gpt2,
}

/// The normalization of the hidden states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Norm {
    /// RMSNorm; the weight is used as `offset + weight`.
    Rms {
        /// 0, or 1 for Gemma, which stores `weight − 1`.
        offset: f64,
    },
    /// LayerNorm with a bias.
    Layer,
}

/// Everything the forward pass needs to know about an architecture.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// `model_type` in `config.json`.
    pub model_type: String,
    /// The shape of the blocks.
    pub block: Block,
    /// The normalization.
    pub norm: Norm,
    /// Its epsilon.
    pub eps: f64,
    /// The vocabulary size.
    pub vocab: usize,
    /// The width of the hidden states.
    pub hidden: usize,
    /// The number of blocks.
    pub layers: usize,
    /// The number of query heads.
    pub heads: usize,
    /// The number of key and value heads.
    pub kv_heads: usize,
    /// The width of a head.
    pub head_dim: usize,
    /// The width of the MLP.
    pub intermediate: usize,
    /// The activation of the MLP.
    pub activation: UnaryKind,
    /// Whether the query, key and value projections have biases.
    pub qkv_bias: bool,
    /// Whether the output projection and the MLP have biases.
    pub out_bias: bool,
    /// Whether queries and keys are RMS-normalized per head before RoPE (Qwen 3).
    pub qk_norm: bool,
    /// Rotary embeddings: base, scaling, rotated components; `None` for learned
    /// positions (GPT-2).
    pub rope: Option<(f64, RopeScaling, usize)>,
    /// How rotated components pair.
    pub rope_layout: RopeLayout,
    /// The factor of the attention scores; `None` for `1/√head_dim`.
    pub query_scale: Option<f64>,
    /// The cap of the attention scores (Gemma 2).
    pub attn_softcap: Option<f64>,
    /// The cap of the final logits (Gemma 2).
    pub final_softcap: Option<f64>,
    /// For each layer, the sliding window of its attention, if any.
    pub windows: Vec<Option<usize>>,
    /// The factor of the embeddings (Gemma: `√hidden`).
    pub embed_scale: Option<f64>,
    /// Whether the output projection is the embedding table.
    pub tie_embeddings: bool,
    /// Whether the output projection has a bias (Phi).
    pub lm_head_bias: bool,
    /// The positions the model was trained for (`max_position_embeddings`),
    /// if the configuration says.
    pub max_positions: Option<usize>,
}

// Bounds far above any model, so that a hostile file cannot make Kapel allocate
// without limit, or overflow when sizes are multiplied together: with these, a
// product of two of them fits in 48 bits.

/// The most layers a configuration may have.
pub const MAX_LAYERS: usize = 4096;
/// The most heads a configuration may have.
pub const MAX_HEADS: usize = 4096;
/// The widest hidden states a configuration may have (the largest models have
/// a few tens of thousands).
pub const MAX_HIDDEN: usize = 1 << 20;
/// The widest head a configuration may have (models use 64 to 256).
pub const MAX_HEAD_DIM: usize = 1 << 14;
/// The widest MLP a configuration may have.
pub const MAX_INTERMEDIATE: usize = 1 << 24;
/// The largest vocabulary a configuration may have (token ids are `u32`;
/// models have up to a few hundred thousand).
pub const MAX_VOCAB: usize = 1 << 24;

/// A field of the configuration, under any of its names.
fn field<'a>(json: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names
        .iter()
        .find_map(|n| json.get(*n).filter(|v| !v.is_null()))
}

fn usize_field(json: &Value, names: &[&str]) -> Result<usize, ModelError> {
    field(json, names)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| ModelError::Config(format!("missing integer field {}", names.join(" or "))))
}

fn f64_field(json: &Value, names: &[&str]) -> Option<f64> {
    field(json, names).and_then(Value::as_f64)
}

/// The activation named in `config.json`, as `transformers` defines it.
fn activation(name: &str) -> Result<UnaryKind, ModelError> {
    Ok(match name {
        "silu" | "swish" => UnaryKind::Silu,
        "gelu" | "gelu_python" => UnaryKind::GeluErf,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => UnaryKind::GeluTanh,
        "relu" => UnaryKind::Relu,
        other => {
            return Err(ModelError::Config(format!(
                "unsupported activation {other:?}"
            )));
        }
    })
}

/// The RoPE scaling of `rope_parameters` (transformers 5) or `rope_scaling`.
fn scaling(json: &Value) -> Result<RopeScaling, ModelError> {
    let params = field(json, &["rope_parameters", "rope_scaling"]);
    let Some(p) = params else {
        return Ok(RopeScaling::None);
    };
    let kind = field(p, &["rope_type", "type"])
        .and_then(Value::as_str)
        .unwrap_or("default");
    let factor = || f64_field(p, &["factor"]).unwrap_or(1.0);
    let original = || usize_field(p, &["original_max_position_embeddings"]);
    Ok(match kind {
        "default" => RopeScaling::None,
        "linear" => RopeScaling::Linear { factor: factor() },
        "dynamic" => RopeScaling::Ntk { factor: factor() },
        "llama3" => RopeScaling::Llama3 {
            factor: factor(),
            low_freq_factor: f64_field(p, &["low_freq_factor"]).unwrap_or(1.0),
            high_freq_factor: f64_field(p, &["high_freq_factor"]).unwrap_or(4.0),
            original_max_position: original()?,
        },
        "yarn" => RopeScaling::Yarn {
            factor: factor(),
            original_max_position: original()?,
            beta_fast: f64_field(p, &["beta_fast"]).unwrap_or(32.0),
            beta_slow: f64_field(p, &["beta_slow"]).unwrap_or(1.0),
            attention_factor: f64_field(p, &["attention_factor"]),
        },
        other => {
            return Err(ModelError::Config(format!(
                "unsupported RoPE type {other:?}"
            )));
        }
    })
}

impl Config {
    /// Reads `config.json`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Config`] for an architecture Kapel does not have, or a
    /// missing field.
    pub fn from_json(json: &Value) -> Result<Self, ModelError> {
        let model_type = field(json, &["model_type"])
            .and_then(Value::as_str)
            .ok_or_else(|| ModelError::Config("no model_type".into()))?
            .to_string();
        let (block, norm) = match model_type.as_str() {
            "llama" | "mistral" | "qwen2" | "qwen3" => {
                (Block::Sequential, Norm::Rms { offset: 0.0 })
            }
            "gemma2" => (Block::Sandwich, Norm::Rms { offset: 1.0 }),
            "phi" => (Block::Parallel, Norm::Layer),
            "gpt2" => (Block::Gpt2, Norm::Layer),
            other => {
                return Err(ModelError::Config(format!(
                    "unsupported architecture {other:?}"
                )));
            }
        };
        let hidden = usize_field(json, &["hidden_size", "n_embd"])?;
        let heads = usize_field(json, &["num_attention_heads", "n_head"])?;
        let layers = usize_field(json, &["num_hidden_layers", "n_layer"])?;
        let kv_heads = usize_field(json, &["num_key_value_heads"]).unwrap_or(heads);
        let head_dim =
            usize_field(json, &["head_dim"]).unwrap_or(hidden.checked_div(heads).unwrap_or(0));
        let intermediate = match usize_field(json, &["intermediate_size", "n_inner"]) {
            Ok(n) => n,
            Err(_) => hidden
                .checked_mul(4)
                .ok_or_else(|| ModelError::Config(format!("an MLP of 4 × {hidden} is too wide")))?,
        };
        let vocab = usize_field(json, &["vocab_size"])?;
        let act_name = field(
            json,
            &["hidden_activation", "hidden_act", "activation_function"],
        )
        .and_then(Value::as_str)
        .unwrap_or("silu");
        let eps = f64_field(
            json,
            &["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"],
        )
        .unwrap_or(1e-5);
        let rope = if block == Block::Gpt2 {
            None
        } else {
            let params = field(json, &["rope_parameters"]);
            let theta = params
                .and_then(|p| f64_field(p, &["rope_theta"]))
                .or_else(|| f64_field(json, &["rope_theta"]))
                .unwrap_or(10_000.0);
            let partial = params
                .and_then(|p| f64_field(p, &["partial_rotary_factor"]))
                .or_else(|| f64_field(json, &["partial_rotary_factor"]))
                .unwrap_or(1.0);
            let rotary = ((head_dim as f64 * partial) as usize) & !1;
            Some((theta, scaling(json)?, rotary))
        };
        for (name, value, limit) in [
            ("layers", layers, MAX_LAYERS),
            ("heads", heads, MAX_HEADS),
            ("key heads", kv_heads, MAX_HEADS),
            ("hidden width", hidden, MAX_HIDDEN),
            ("head width", head_dim, MAX_HEAD_DIM),
            ("MLP width", intermediate, MAX_INTERMEDIATE),
            ("vocabulary size", vocab, MAX_VOCAB),
        ] {
            if value > limit {
                return Err(ModelError::Config(format!(
                    "{value} {name}: at most {limit} are accepted"
                )));
            }
        }
        let sliding = usize_field(json, &["sliding_window"]).ok();
        let windows: Vec<Option<usize>> =
            match field(json, &["layer_types"]).and_then(Value::as_array) {
                Some(types) => types
                    .iter()
                    .map(|t| {
                        (t.as_str() == Some("sliding_attention"))
                            .then_some(sliding)
                            .flatten()
                    })
                    .collect(),
                // Gemma 2 alternates, starting with a sliding layer; Mistral slides
                // everywhere. Others do not slide.
                None => (0..layers)
                    .map(|i| match model_type.as_str() {
                        "gemma2" if i % 2 == 0 => sliding,
                        "mistral" => sliding,
                        _ => None,
                    })
                    .collect(),
            };
        let invalid = |message: String| Err(ModelError::Config(message));
        if hidden == 0 || layers == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 {
            return invalid(format!(
                "sizes must be positive: hidden {hidden}, layers {layers}, heads {heads}, \
                 key heads {kv_heads}, head width {head_dim}"
            ));
        }
        if heads % kv_heads != 0 {
            return invalid(format!(
                "{heads} heads are not a multiple of {kv_heads} key heads"
            ));
        }
        if let Some((_, _, rotary)) = rope
            && (rotary == 0 || rotary > head_dim)
        {
            return invalid(format!(
                "{rotary} rotated components for heads of {head_dim}"
            ));
        }
        if windows.len() != layers {
            return invalid(format!("{} layer types for {layers} layers", windows.len()));
        }
        let is = |name: &str| field(json, &[name]).and_then(Value::as_bool);
        Ok(Self {
            block,
            norm,
            eps,
            vocab,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            activation: activation(act_name)?,
            qkv_bias: match model_type.as_str() {
                "qwen2" | "phi" | "gpt2" => true,
                _ => is("attention_bias").unwrap_or(false),
            },
            out_bias: matches!(model_type.as_str(), "phi" | "gpt2"),
            qk_norm: model_type == "qwen3",
            rope,
            rope_layout: RopeLayout::Half,
            query_scale: f64_field(json, &["query_pre_attn_scalar"]).map(|s| 1.0 / s.sqrt()),
            attn_softcap: f64_field(json, &["attn_logit_softcapping"]),
            final_softcap: f64_field(json, &["final_logit_softcapping"]),
            windows,
            embed_scale: (model_type == "gemma2").then(|| (hidden as f64).sqrt()),
            tie_embeddings: is("tie_word_embeddings").unwrap_or(model_type == "gpt2"),
            lm_head_bias: model_type == "phi",
            max_positions: usize_field(json, &["max_position_embeddings", "n_positions"]).ok(),
            model_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama(change: impl FnOnce(&mut Value)) -> Result<Config, ModelError> {
        let mut json = serde_json::json!({
            "model_type": "llama", "hidden_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 100,
            "intermediate_size": 128, "hidden_act": "silu",
        });
        change(&mut json);
        Config::from_json(&json)
    }

    #[test]
    fn a_valid_configuration_is_read() {
        let config = llama(|_| {}).unwrap();
        assert_eq!((config.head_dim, config.kv_heads), (16, 2));
    }

    #[test]
    fn impossible_sizes_are_errors_not_panics() {
        for change in [
            |j: &mut Value| j["num_attention_heads"] = 0.into(),
            |j: &mut Value| j["num_key_value_heads"] = 3.into(),
            |j: &mut Value| j["hidden_size"] = 0.into(),
            |j: &mut Value| j["hidden_act"] = "magic".into(),
            |j: &mut Value| j["model_type"] = "unknown".into(),
            |j: &mut Value| j["layer_types"] = serde_json::json!(["full_attention"]),
            |j: &mut Value| j["num_hidden_layers"] = 1_200_011_111u64.into(),
        ] {
            assert!(llama(change).is_err());
        }
    }

    /// Sizes near `usize::MAX` once overflowed `4 * hidden`; every size has a
    /// bound now.
    #[test]
    fn huge_sizes_are_errors_not_overflows() {
        let huge = u64::MAX / 2;
        for key in [
            "hidden_size",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
        ] {
            for value in [huge, u64::MAX, 1 << 40] {
                let error = llama(|j| {
                    j[key] = value.into();
                    if key == "hidden_size" {
                        j.as_object_mut().unwrap().remove("intermediate_size");
                    }
                })
                .unwrap_err()
                .to_string();
                assert!(
                    error.contains("accepted") || error.contains("too wide"),
                    "{key} = {value}: {error}"
                );
            }
        }
        // GPT-2 derives its MLP width from the hidden one.
        let gpt2 = serde_json::json!({
            "model_type": "gpt2", "n_embd": u64::MAX / 2, "n_head": 1, "n_layer": 1,
            "vocab_size": 10,
        });
        assert!(Config::from_json(&gpt2).is_err());
        let gpt2 = serde_json::json!({
            "model_type": "gpt2", "n_embd": 64, "n_head": 4, "n_layer": 1, "vocab_size": 10,
        });
        assert_eq!(Config::from_json(&gpt2).unwrap().intermediate, 256);
    }
}
