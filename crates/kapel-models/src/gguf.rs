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
//!
//! [`Model::to_gguf`] goes the other way, as llama.cpp's converter
//! (`convert_hf_to_gguf.py`) does: the same names and metadata, the query
//! and key rows of Llama reordered, the norms in `f32`, the factors of Llama
//! 3's scaling computed. The tokenizer's metadata is `kapel-infer`'s part.

use std::collections::BTreeMap;

use kapel_core::{DType, QuantFormat, Tensor};
use kapel_hub::Gguf;
use kapel_hub::gguf::{Entry, Value as Meta};
use kapel_ops::{Backend, Cast, RopeLayout, RopeScaling};
use serde_json::{Value, json};

use crate::ModelError;
use crate::config::{Block, Config};
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

/// llama.cpp's names of the weights of Qwen 3.5's layers of Gated DeltaNet
/// and of its experts (`qwen35moe`), and those of `transformers` (after
/// `model.layers.{i}.`). `linear_attn.A` is not `transformers`' `A_log`:
/// llama.cpp stores `−exp(A_log)`.
const QWEN35_NAMES: [(&str, &str); 17] = [
    ("attn_qkv.weight", "linear_attn.in_proj_qkv.weight"),
    ("attn_gate.weight", "linear_attn.in_proj_z.weight"),
    ("ssm_conv1d.weight", "linear_attn.conv1d.weight"),
    ("ssm_dt.bias", "linear_attn.dt_bias"),
    ("ssm_a", "linear_attn.A"),
    ("ssm_alpha.weight", "linear_attn.in_proj_a.weight"),
    ("ssm_beta.weight", "linear_attn.in_proj_b.weight"),
    ("ssm_norm.weight", "linear_attn.norm.weight"),
    ("ssm_out.weight", "linear_attn.out_proj.weight"),
    ("ffn_gate_inp.weight", "mlp.gate.weight"),
    ("ffn_gate_exps.weight", "mlp.experts.gate_proj"),
    ("ffn_up_exps.weight", "mlp.experts.up_proj"),
    ("ffn_down_exps.weight", "mlp.experts.down_proj"),
    ("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"),
    (
        "ffn_gate_shexp.weight",
        "mlp.shared_expert.gate_proj.weight",
    ),
    ("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"),
    (
        "ffn_down_shexp.weight",
        "mlp.shared_expert.down_proj.weight",
    ),
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
    if let Some((_, hf)) = QWEN35_NAMES.iter().find(|(gguf, _)| *gguf == rest) {
        return Some(format!("model.layers.{layer}.{hf}"));
    }
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
    /// Llama (with Mistral), Qwen 2, Qwen 3 and Qwen 3.5's mixtures of
    /// experts (`qwen35moe`) are read.
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
            "qwen35moe" => "qwen3_5_moe".to_string(),
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
        let feed_forward = match meta.count("feed_forward_length") {
            Ok(n) => n,
            Err(_) if model_type == "qwen3_5_moe" => meta.count("expert_feed_forward_length")?,
            Err(e) => return Err(e),
        };
        let mut config = json!({
            "model_type": model_type,
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": meta.count("block_count")?,
            "num_attention_heads": heads,
            "num_key_value_heads": meta.count("attention.head_count_kv").unwrap_or(heads),
            "head_dim": head_dim,
            "intermediate_size": feed_forward,
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
        if model_type == "qwen3_5_moe" {
            qwen35_config(&meta, &mut config, head_dim)?;
        }
        let mut parsed = Config::from_json(&config)?;
        if parsed.model_type == "llama" {
            parsed.rope_layout = RopeLayout::Interleaved;
        }
        if parsed.model_type == "qwen3_5_moe" {
            // llama.cpp stores the norms centred on zero with their 1 added
            // (`1 + w`), and lays the value heads of linear attention out
            // tiled.
            parsed.norm = crate::config::Norm::Rms { offset: 0.0 };
            if let Some(linear) = parsed.linear.as_mut() {
                linear.tiled = true;
            }
        }
        Ok(Self::from_parts(parsed, config, weights))
    }
}

/// The fields of Qwen 3.5's configuration (`qwen35moe`) that GGUF keeps
/// apart: its layers of Gated DeltaNet, its experts, its partial RoPE.
fn qwen35_config(meta: &Metadata<'_>, config: &mut Value, head_dim: u64) -> Result<()> {
    let layers = meta.count("block_count")?;
    let interval = meta.count("full_attention_interval")?.max(1);
    let types: Vec<&str> = (0..layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let value_heads = meta.count("ssm.time_step_rank")?;
    let rotary = meta.count("rope.dimension_count")?;
    let fields = json!({
        "layer_types": types,
        "linear_num_key_heads": meta.count("ssm.group_count")?,
        "linear_num_value_heads": value_heads,
        "linear_key_head_dim": meta.count("ssm.state_size")?,
        "linear_value_head_dim": meta.count("ssm.inner_size")? / value_heads.max(1),
        "linear_conv_kernel_dim": meta.count("ssm.conv_kernel")?,
        "num_experts": meta.count("expert_count")?,
        "num_experts_per_tok": meta.count("expert_used_count")?,
        "moe_intermediate_size": meta.count("expert_feed_forward_length")?,
        "shared_expert_intermediate_size": meta.count("expert_shared_feed_forward_length")?,
        "partial_rotary_factor": rotary as f64 / head_dim.max(1) as f64,
        "attn_output_gate": true,
    });
    for (k, v) in fields.as_object().expect("an object") {
        config[k] = v.clone();
    }
    Ok(())
}

/// The GGUF name of a weight of `transformers`, if llama.cpp has one.
fn gguf_name(name: &str) -> Option<String> {
    match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (base, kind) = rest.rsplit_once('.')?;
    // `post_attention_layernorm` is the norm of the MLP in a Llama block.
    let (gguf, _) = BLOCK_NAMES.iter().find(|(_, hf)| *hf == base)?;
    Some(format!("blk.{layer}.{gguf}.{kind}"))
}

/// The metadata and tensors of a GGUF file of llama.cpp: what
/// [`Model::to_gguf`] gives, before the tokenizer's metadata is added.
#[derive(Debug)]
pub struct GgufModel {
    /// The metadata of the architecture, in llama.cpp's order.
    pub metadata: Vec<(String, Entry)>,
    /// The tensors, under llama.cpp's names.
    pub tensors: Vec<(String, Tensor)>,
}

/// llama.cpp's `general.file_type` of a model whose projections are mostly
/// of type `dtype`.
fn file_type(dtype: DType) -> Option<u32> {
    Some(match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Quantized(QuantFormat::Q4_0) => 2,
        DType::Quantized(QuantFormat::Q4_1) => 3,
        DType::Quantized(QuantFormat::Q8_0) => 7,
        DType::Quantized(QuantFormat::Q5_0) => 8,
        DType::Quantized(QuantFormat::Q5_1) => 9,
        DType::Quantized(QuantFormat::Q2K) => 10,
        DType::Quantized(QuantFormat::Q4K) => 15,
        DType::Quantized(QuantFormat::Q5K) => 17,
        DType::Quantized(QuantFormat::Q6K) => 18,
        DType::BF16 => 32,
        _ => return None,
    })
}

/// The rows of `w` `[N, K]` reordered: row `i` of the result is row
/// `order[i]` of `w`. Quantized rows move as their blocks.
fn reorder_rows(w: &Tensor, order: &[usize]) -> Result<Tensor> {
    let w = w.contiguous()?;
    let (n, k) = (w.shape()[0], w.shape()[1]);
    let row = w.dtype().storage_bytes(k)?;
    let bytes = w
        .storage()
        .host_bytes()
        .ok_or_else(|| ModelError::Config("the weights must be in host memory".into()))?;
    let from = w.dtype().storage_bytes(w.layout().offset())?;
    let mut out = Vec::with_capacity(n * row);
    for &r in order {
        out.extend_from_slice(&bytes[from + r * row..from + (r + 1) * row]);
    }
    Ok(Tensor::from_bytes(w.dtype(), &[n, k], &out)?)
}

/// The order of llama.cpp's converter for the query or key rows of Llama,
/// `heads` heads of `d` components: within each head, the two halves
/// interleaved, so that RoPE pairs neighbours.
fn interleaved(heads: usize, d: usize) -> Vec<usize> {
    (0..heads * d)
        .map(|r| {
            let (h, j) = (r / d, r % d);
            h * d + (j % 2) * (d / 2) + j / 2
        })
        .collect()
}

impl Model {
    /// The model as llama.cpp's converter writes it to GGUF: Llama (and
    /// Mistral, which GGUF files as Llama), Qwen 2 and Qwen 3, under
    /// llama.cpp's names, with its metadata. Weights keep their types,
    /// quantized or not; one-dimensional ones (norms, biases) are converted
    /// to `f32` on `backend`, as llama.cpp keeps them. The weights must be in
    /// host memory, and not [fused](Model::fused).
    ///
    /// # Errors
    ///
    /// For another architecture, a weight llama.cpp has no name for, a RoPE
    /// scaling other than Llama 3's, or a weight outside host memory.
    pub fn to_gguf<B: Backend>(&self, backend: &B) -> Result<GgufModel> {
        let c = &self.config;
        let arch = match c.model_type.as_str() {
            "llama" | "mistral" => "llama",
            "qwen2" => "qwen2",
            "qwen3" => "qwen3",
            other => {
                return Err(ModelError::Config(format!(
                    "{other:?} is not written to GGUF: Llama, Mistral, Qwen 2 and Qwen 3 are"
                )));
            }
        };
        if c.block != Block::Sequential {
            return Err(ModelError::Config(
                "only blocks of Llama's kind are written to GGUF".into(),
            ));
        }
        let Some((theta, scaling, rotary)) = c.rope else {
            return Err(ModelError::Config("a model without RoPE".into()));
        };
        // Llama's queries and keys pair halves in `transformers`, neighbours
        // in llama.cpp. A model read from GGUF already pairs neighbours.
        let reorder = arch == "llama" && c.rope_layout == RopeLayout::Half;
        let mut tensors = Vec::new();
        let mut projections: Vec<(DType, usize)> = Vec::new();
        if let RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } = scaling
        {
            let original = original_max_position as f64;
            let factors: Vec<f32> = (0..rotary / 2)
                .map(|j| {
                    // In `f32`, as llama.cpp's converter computes them.
                    let exponent = (2 * j) as f32 / rotary as f32;
                    let frequency = 1.0 / (theta as f32).powf(exponent);
                    let wavelength = std::f32::consts::TAU / frequency;
                    let (factor, low, high) = (
                        factor as f32,
                        low_freq_factor as f32,
                        high_freq_factor as f32,
                    );
                    let original = original as f32;
                    if wavelength < original / high {
                        1.0
                    } else if wavelength > original / low {
                        factor
                    } else {
                        let smooth = (original / wavelength - low) / (high - low);
                        1.0 / ((1.0 - smooth) / factor + smooth)
                    }
                })
                .collect();
            tensors.push((
                "rope_freqs.weight".to_string(),
                Tensor::from_slice(&factors, &[rotary / 2])?,
            ));
        } else if scaling != RopeScaling::None {
            return Err(ModelError::Config(format!(
                "RoPE scaling {scaling:?} is not written to GGUF"
            )));
        }
        for (name, w) in &self.weights {
            if name.contains(".qkv_proj.") || name.contains(".gate_up_proj.") {
                return Err(ModelError::Config(
                    "a fused model is not written to GGUF: write it before fusing".into(),
                ));
            }
            let gguf = gguf_name(name).ok_or_else(|| {
                ModelError::Config(format!("weight {name:?} has no name in llama.cpp"))
            })?;
            let mut w = w.clone();
            if w.rank() == 1 {
                if w.dtype() != DType::F32 {
                    w = backend.forward(&Cast { dtype: DType::F32 }, &[&w])?;
                }
            } else if name.ends_with("_proj.weight") {
                match projections.iter_mut().find(|(d, _)| *d == w.dtype()) {
                    Some((_, n)) => *n += w.numel(),
                    None => projections.push((w.dtype(), w.numel())),
                }
            }
            if reorder {
                if name.ends_with("self_attn.q_proj.weight") {
                    w = reorder_rows(&w, &interleaved(c.heads, c.head_dim))?;
                } else if name.ends_with("self_attn.k_proj.weight") {
                    w = reorder_rows(&w, &interleaved(c.kv_heads, c.head_dim))?;
                }
            }
            tensors.push((gguf, w));
        }
        let mostly = projections
            .iter()
            .max_by_key(|(_, n)| *n)
            .map_or(DType::F32, |(d, _)| *d);
        let u32_of = |v: usize| Entry::U32(u32::try_from(v).unwrap_or(u32::MAX));
        let key = |k: &str| format!("{arch}.{k}");
        let mut metadata = vec![
            (
                "general.architecture".to_string(),
                Entry::String(arch.into()),
            ),
            ("general.type".to_string(), Entry::String("model".into())),
        ];
        if let Some(name) = self.raw_config.get("_name_or_path").and_then(Value::as_str) {
            let name = name.rsplit('/').next().unwrap_or(name);
            metadata.push(("general.name".into(), Entry::String(name.into())));
        }
        metadata.extend([
            (key("block_count"), u32_of(c.layers)),
            (
                key("context_length"),
                u32_of(c.max_positions.unwrap_or(4096)),
            ),
            (key("embedding_length"), u32_of(c.hidden)),
            (key("feed_forward_length"), u32_of(c.intermediate)),
            (key("attention.head_count"), u32_of(c.heads)),
            (key("attention.head_count_kv"), u32_of(c.kv_heads)),
            (key("rope.freq_base"), Entry::F32(theta as f32)),
            (
                key("attention.layer_norm_rms_epsilon"),
                Entry::F32(c.eps as f32),
            ),
            (key("attention.key_length"), u32_of(c.head_dim)),
            (key("attention.value_length"), u32_of(c.head_dim)),
        ]);
        if let Some(t) = file_type(mostly) {
            metadata.push(("general.file_type".into(), Entry::U32(t)));
        }
        if arch == "llama" {
            metadata.push((key("vocab_size"), u32_of(c.vocab)));
            metadata.push((key("rope.dimension_count"), u32_of(rotary)));
        }
        metadata.push(("general.quantization_version".into(), Entry::U32(2)));
        Ok(GgufModel { metadata, tensors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_go_back_to_llama_cpp() {
        for gguf in [
            "blk.3.attn_q.weight",
            "blk.0.attn_k.bias",
            "blk.11.ffn_down.weight",
            "blk.2.ffn_norm.weight",
            "blk.5.attn_q_norm.weight",
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
        ] {
            assert_eq!(gguf_name(&hf_name(gguf).unwrap()).as_deref(), Some(gguf));
        }
        assert_eq!(gguf_name("model.mtp_layers.0.input_proj.weight"), None);
    }

    #[test]
    fn interleaving_pairs_neighbours() {
        // One head of 4 components: [0, 1, 2, 3] → [0, 2, 1, 3], so that
        // components 0 and 2 (paired by halves) become neighbours.
        assert_eq!(interleaved(1, 4), [0, 2, 1, 3]);
        assert_eq!(interleaved(2, 4), [0, 2, 1, 3, 4, 6, 5, 7]);
    }

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
