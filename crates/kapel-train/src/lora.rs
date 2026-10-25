//! LoRA (Hu et al., 2021): the model's weights frozen, and beside each chosen
//! projection `W` a product of rank `r`, the only parameters trained:
//!
//! ```text
//! y = x · Wᵀ + s · (x · Aᵀ) · Bᵀ        A [r, in], B [out, r], s = α / r
//! ```
//!
//! `B` starts at zero: the adapted model starts as the model. `A` starts as
//! PEFT draws it (Kaiming's uniform of `torch.nn.Linear`, bounds `±1/√in`).
//! With rsLoRA, `s = α / √r`.
//!
//! The table of embeddings may have its adapter too: its rows are
//! `E[i] + s · Aᵀ[i] · Bᵀ`, `Aᵀ [V, r]` gathered as the table is (its
//! gradient stays rows), `B [C, r]`. PEFT starts `A` at zero and draws `B`
//! from a standard normal. Smallem needed it: tokens added to a frozen table
//! start identical, and stay so.
//!
//! QLoRA (Dettmers et al., 2023) quantizes the frozen projections; their
//! products read the codes, dequantized by pieces on the device, forward
//! and backward ([`kapel_ops::QuantizedMatmul`]).
//!
//! The adapters are named as PEFT names them, without its prefix
//! `base_model.model.`: `model.layers.0.self_attn.q_proj.lora_A.weight`. Only
//! the embedding's `A` is kept transposed, `[V, r]`, for its rows; it is
//! written `[r, V]`, as PEFT has it ([`to_peft`]). [`save`] writes an
//! adapter that PEFT loads (`adapter_config.json`,
//! `adapter_model.safetensors`); [`load`] reads one, PEFT's own included.

use std::collections::BTreeMap;
use std::path::Path;

use kapel_core::{DType, Error, Result, Tensor};
use kapel_models::{Block, Config};
use kapel_ops::{Binary, BinaryKind, Cast, Matmul, Run as _};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::model::{Parameter, Parameters, Rng};
use crate::tape::Compute;

/// PEFT's prefix of the names of the adapters.
pub const PEFT_PREFIX: &str = "base_model.model.";

/// The adapters of a run.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraConfig {
    /// The rank of the products.
    #[serde(default = "rank_default")]
    pub r: usize,
    /// `α`: the scale of the products is `α / r`.
    #[serde(default = "alpha_default")]
    pub alpha: f64,
    /// The projections adapted, by the last part of their names
    /// (`q_proj`, `v_proj`, `c_attn`…), or `"all-linear"` for all those of
    /// the blocks.
    #[serde(default = "targets_default")]
    pub targets: Vec<String>,
    /// An adapter on the table of embeddings too.
    #[serde(default)]
    pub embedding: bool,
    /// rsLoRA: the scale is `α / √r`.
    #[serde(default)]
    pub rslora: bool,
    /// The format the frozen projections are quantized to, for QLoRA:
    /// `"q4_0"`, `"int4"` (groups of 64), `"q8_0"` or `"int8"`; absent, they
    /// stay in the computation's type.
    #[serde(default)]
    pub quantize: Option<String>,
}

fn rank_default() -> usize {
    16
}

fn alpha_default() -> f64 {
    32.0
}

fn targets_default() -> Vec<String> {
    vec!["all-linear".into()]
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: rank_default(),
            alpha: alpha_default(),
            targets: targets_default(),
            embedding: false,
            rslora: false,
            quantize: None,
        }
    }
}

impl LoraConfig {
    /// The scale of the products.
    pub fn scale(&self) -> f64 {
        if self.rslora {
            self.alpha / (self.r as f64).sqrt()
        } else {
            self.alpha / self.r as f64
        }
    }

    /// The type of the frozen projections, for QLoRA.
    ///
    /// # Errors
    ///
    /// For a format not listed.
    pub fn quantized_dtype(&self) -> std::result::Result<Option<DType>, String> {
        use kapel_core::QuantFormat;
        let format = match self.quantize.as_deref() {
            None => return Ok(None),
            Some("q4_0") => QuantFormat::Q4_0,
            Some("q8_0") => QuantFormat::Q8_0,
            Some("int4") => QuantFormat::Int4 { group: 64 },
            Some("int8") => QuantFormat::Int8 { group: 64 },
            Some(other) => {
                return Err(format!(
                    "lora.quantize: \"q4_0\", \"int4\", \"q8_0\" or \"int8\", got {other:?}"
                ));
            }
        };
        Ok(Some(DType::Quantized(format)))
    }

    /// Checks the values.
    ///
    /// # Errors
    ///
    /// A message naming the first value refused.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.r == 0 {
            return Err("lora.r: at least 1".into());
        }
        if !(self.alpha > 0.0 && self.alpha.is_finite()) {
            return Err(format!("lora.alpha: positive, got {}", self.alpha));
        }
        if self.targets.is_empty() && !self.embedding {
            return Err("lora.targets: none, and no adapter on the embeddings".into());
        }
        self.quantized_dtype().map(drop)
    }
}

/// The projections of the blocks, by the last part of their names.
const PROJECTIONS: [&str; 10] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",
];

/// A projection of the blocks: its module's name (its weight's without
/// `.weight`), its inputs and its outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// `model.layers.0.self_attn.q_proj`.
    pub module: String,
    /// The width of its input.
    pub inputs: usize,
    /// The width of its output.
    pub outputs: usize,
}

/// Whether the model stores its projections `[in, out]` (GPT-2's
/// `Conv1D`), not `[out, in]` (`torch.nn.Linear`): PEFT's `fan_in_fan_out`.
pub fn fan_in_fan_out(config: &Config) -> bool {
    config.block == Block::Gpt2
}

/// The name of the table of embeddings.
pub fn embedding_name(config: &Config) -> &'static str {
    match config.block {
        Block::Gpt2 => "transformer.wte",
        _ => "model.embed_tokens",
    }
}

/// The projections of the blocks that `targets` chooses.
pub fn projections(params: &Parameters, config: &Config, targets: &[String]) -> Vec<Projection> {
    let all = targets.iter().any(|t| t == "all-linear");
    let blocks = match config.block {
        Block::Gpt2 => "transformer.h.",
        _ => "model.layers.",
    };
    params
        .iter()
        .filter_map(|(name, p)| {
            let module = name.strip_prefix(blocks)?.strip_suffix(".weight")?;
            let last = module.rsplit('.').next()?;
            let chosen = if all {
                PROJECTIONS.contains(&last)
            } else {
                targets.iter().any(|t| t == last)
            };
            if !chosen || p.value.rank() != 2 {
                return None;
            }
            let (a, b) = (p.value.shape()[0], p.value.shape()[1]);
            let (inputs, outputs) = if fan_in_fan_out(config) {
                (a, b)
            } else {
                (b, a)
            };
            Some(Projection {
                module: format!("{blocks}{module}"),
                inputs,
                outputs,
            })
        })
        .collect()
}

/// The names of the two factors of `module`'s adapter.
pub fn factor_names(module: &str) -> (String, String) {
    (
        format!("{module}.lora_A.weight"),
        format!("{module}.lora_B.weight"),
    )
}

/// The names of the two factors of the embeddings' adapter.
pub fn embedding_factor_names(config: &Config) -> (String, String) {
    let e = embedding_name(config);
    (
        format!("{e}.lora_embedding_A"),
        format!("{e}.lora_embedding_B"),
    )
}

/// Whether `name` is a factor of an adapter.
pub fn is_factor(name: &str) -> bool {
    name.contains(".lora_A.")
        || name.contains(".lora_B.")
        || name.ends_with(".lora_embedding_A")
        || name.ends_with(".lora_embedding_B")
}

/// Freezes every parameter of `params`, and adds the adapters `lora` asks
/// for, drawn from `seed` (each factor from a seed of its own name). Returns
/// the projections adapted.
///
/// # Errors
///
/// If no projection matches the targets.
pub fn add_adapters(
    params: &mut Parameters,
    config: &Config,
    lora: &LoraConfig,
    seed: u64,
) -> Result<Vec<Projection>> {
    let chosen = projections(params, config, &lora.targets);
    if chosen.is_empty() && !lora.targets.is_empty() {
        return Err(Error::InvalidArgument {
            op: "lora",
            message: format!("no projection of the blocks is named {:?}", lora.targets),
        });
    }
    for (_, p) in params.iter_mut() {
        p.trainable = false;
    }
    let r = lora.r;
    for p in &chosen {
        let (a, b) = factor_names(&p.module);
        let bound = 1.0 / (p.inputs as f64).sqrt();
        let mut rng = Rng::new(crate::model::seed_of(seed, &a));
        let values: Vec<f32> = (0..r * p.inputs)
            .map(|_| ((2.0 * rng.uniform() - 1.0) * bound) as f32)
            .collect();
        params.insert(
            a,
            Parameter::new(Tensor::from_slice(&values, &[r, p.inputs])?),
        );
        params.insert(
            b,
            Parameter::new(Tensor::zeros(DType::F32, &[p.outputs, r])?),
        );
    }
    if lora.embedding {
        let table = params
            .get(&format!("{}.weight", embedding_name(config)))
            .ok_or_else(|| Error::InvalidArgument {
                op: "lora",
                message: "no table of embeddings".into(),
            })?;
        let (vocab, hidden) = (table.value.shape()[0], table.value.shape()[1]);
        let (a, b) = embedding_factor_names(config);
        let mut rng = Rng::new(crate::model::seed_of(seed, &b));
        let values: Vec<f32> = (0..hidden * r).map(|_| rng.normal() as f32).collect();
        let mut at = Parameter::new(Tensor::zeros(DType::F32, &[vocab, r])?);
        at.sparse = true;
        params.insert(a, at);
        params.insert(
            b,
            Parameter::new(Tensor::from_slice(&values, &[hidden, r])?),
        );
    }
    Ok(chosen)
}

/// Quantizes the frozen projections of the blocks to `dtype`, for QLoRA:
/// those whose input width is a multiple of the format's blocks.
///
/// # Errors
///
/// If a weight is not in host memory.
pub fn quantize_frozen(params: &mut Parameters, config: &Config, dtype: DType) -> Result<usize> {
    let all = projections(params, config, &["all-linear".to_string()]);
    let mut count = 0;
    for p in all {
        let name = format!("{}.weight", p.module);
        let Some(param) = params.get_mut(&name) else {
            continue;
        };
        // The codes run along the input of `[out, in]`: GPT-2's `[in, out]`
        // is left as it is.
        if param.trainable
            || fan_in_fan_out(config)
            || !param.value.shape()[1].is_multiple_of(dtype.block_len())
        {
            continue;
        }
        param.value = kapel_models::quantize_matrix(&param.value, dtype)?;
        count += 1;
    }
    Ok(count)
}

/// `params` with each adapter merged into its weight, `W ← W + s · B · A`
/// (`Aᵀ · Bᵀ` for GPT-2's `[in, out]`), and the embeddings' into the table,
/// the adapters removed: the model the adapted one computes, for inference
/// or for the Hub. The merged weights keep their types; quantized ones are
/// dequantized first, to `f32`.
///
/// The embeddings' adapter changes the table, not the output of a model
/// that ties them: its merged model has the table as it was for its output,
/// `lm_head.weight`, and is no longer tied ([`unties`], [`merged_config`]).
///
/// # Errors
///
/// Those of the backend.
pub fn merge<B: Compute>(
    backend: &B,
    params: &Parameters,
    config: &Config,
    scale: f64,
) -> Result<Parameters> {
    let mut out = Parameters::default();
    // Quantized weights are decoded by the cast, on the device.
    let wide = |t: &Tensor| -> Result<Tensor> {
        if t.dtype() == DType::F32 {
            Ok(t.clone())
        } else {
            backend.forward(&Cast { dtype: DType::F32 }, &[t])
        }
    };
    let add = Binary {
        kind: BinaryKind::Add,
    };
    let (ea, eb) = embedding_factor_names(config);
    for (name, p) in params.iter() {
        if is_factor(name) {
            continue;
        }
        let module = name.strip_suffix(".weight").unwrap_or(name);
        let (a, b) = factor_names(module);
        let delta = match (params.get(&a), params.get(&b)) {
            (Some(a), Some(b)) => {
                let (a, b) = (wide(&a.value)?, wide(&b.value)?);
                // B · A: [out, in].
                let d = backend.forward(&Matmul, &[&b, &a])?;
                Some(if fan_in_fan_out(config) {
                    backend.forward(&kapel_ops::Contiguous, &[&d.transpose(0, 1)?])?
                } else {
                    d
                })
            }
            _ if module == embedding_name(config) => match (params.get(&ea), params.get(&eb)) {
                // Aᵀ · Bᵀ: [V, C].
                (Some(a), Some(b)) => {
                    let (a, b) = (wide(&a.value)?, wide(&b.value)?);
                    Some(backend.forward(&Matmul, &[&a, &b.transpose(0, 1)?])?)
                }
                _ => None,
            },
            _ => None,
        };
        let value = match delta {
            Some(d) => {
                let d = backend.forward(&kapel_ops::Affine { scale, bias: 0.0 }, &[&d])?;
                let w = backend.forward(&add, &[&wide(&p.value)?, &d])?;
                let dtype = if p.value.dtype().is_float() {
                    p.value.dtype()
                } else {
                    DType::F32
                };
                if dtype == DType::F32 {
                    w
                } else {
                    backend.forward(&Cast { dtype }, &[&w])?
                }
            }
            None => p.value.clone(),
        };
        out.insert(name, Parameter { value, ..p.clone() });
    }
    if unties(params, config)
        && let Some(table) = params.get(&format!("{}.weight", embedding_name(config)))
    {
        out.insert("lm_head.weight", table.clone());
    }
    Ok(out)
}

/// Whether merging `params` unties the output from the table: a model that
/// ties them, with an adapter on the embeddings.
pub fn unties(params: &Parameters, config: &Config) -> bool {
    config.tie_embeddings && params.get(&embedding_factor_names(config).0).is_some()
}

/// The `config.json` of the model [`merge`] gives: `raw`, untied if merging
/// unties it.
pub fn merged_config(raw: &Value, params: &Parameters, config: &Config) -> Value {
    let mut raw = raw.clone();
    if unties(params, config)
        && let Some(o) = raw.as_object_mut()
    {
        o.insert("tie_word_embeddings".into(), Value::Bool(false));
    }
    raw
}

/// A factor as PEFT stores it: its name with PEFT's prefix, and the
/// embeddings' `A` transposed back to `[r, V]`; `t` in host memory.
///
/// # Errors
///
/// If `t` cannot be copied.
pub fn to_peft(name: &str, t: &Tensor) -> Result<(String, Tensor)> {
    let t = if name.ends_with(".lora_embedding_A") {
        t.transpose(0, 1)?.contiguous()?
    } else {
        t.clone()
    };
    Ok((format!("{PEFT_PREFIX}{name}"), t))
}

/// A factor PEFT stored, by the name Kapel trains it under; `None` for a
/// tensor that is not a factor of a LoRA adapter.
///
/// # Errors
///
/// If `t` cannot be copied.
pub fn from_peft(name: &str, t: &Tensor) -> Result<Option<(String, Tensor)>> {
    let name = name.strip_prefix(PEFT_PREFIX).unwrap_or(name);
    // PEFT's names in memory carry the adapter's name, `default`; the
    // files drop it, as here.
    let name = name.replace(".default.", ".");
    let name = name.strip_suffix(".default").unwrap_or(&name).to_string();
    if !is_factor(&name) {
        return Ok(None);
    }
    let t = if name.ends_with(".lora_embedding_A") {
        t.transpose(0, 1)?.contiguous()?
    } else {
        t.clone()
    };
    Ok(Some((name, t)))
}

/// `adapter_config.json` for these adapters of the model at `base`.
pub fn peft_config(
    lora: &LoraConfig,
    config: &Config,
    adapted: &[Projection],
    base: &str,
) -> Value {
    let mut targets: Vec<String> = adapted
        .iter()
        .filter_map(|p| p.module.rsplit('.').next().map(str::to_string))
        .collect();
    if lora.embedding {
        let e = embedding_name(config);
        targets.push(e.rsplit('.').next().unwrap_or(e).to_string());
    }
    targets.sort();
    targets.dedup();
    json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": 0.0,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": fan_in_fan_out(config),
        "use_rslora": lora.rslora,
        "use_dora": false,
        "init_lora_weights": true,
        "inference_mode": true,
        "modules_to_save": null,
        "layers_to_transform": null,
        "layers_pattern": null,
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": null,
    })
}

/// Writes the adapters among `factors` (host tensors, by Kapel's names)
/// into `dir` as PEFT does: `adapter_model.safetensors` and
/// `adapter_config.json` (`peft_config`).
///
/// # Errors
///
/// If a file cannot be written.
pub fn save(
    dir: &Path,
    factors: &BTreeMap<String, Tensor>,
    peft_config: &Value,
) -> std::result::Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let peft: Vec<(String, Tensor)> = factors
        .iter()
        .filter(|(n, _)| is_factor(n))
        .map(|(n, t)| to_peft(n, t))
        .collect::<Result<_>>()
        .map_err(|e| e.to_string())?;
    let refs: Vec<(&str, &Tensor)> = peft.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
    kapel_hub::safetensors::save(dir.join("adapter_model.safetensors"), &refs, &metadata)
        .map_err(|e| e.to_string())?;
    let json = serde_json::to_vec_pretty(peft_config).expect("JSON serializes");
    std::fs::write(dir.join("adapter_config.json"), json)
        .map_err(|e| format!("{}: {e}", dir.display()))
}

/// An adapter read back: its settings, and its factors by Kapel's names.
#[derive(Clone, Debug)]
pub struct Adapter {
    /// The rank, `α` and rsLoRA of `adapter_config.json`; its targets.
    pub config: LoraConfig,
    /// The factors, in host memory.
    pub factors: BTreeMap<String, Tensor>,
}

/// Reads the adapter PEFT (or [`save`]) wrote into `dir`.
///
/// # Errors
///
/// If a file is missing or unreadable, or the adapter is not LoRA.
pub fn load(dir: &Path) -> std::result::Result<Adapter, String> {
    let at = |e: &dyn std::fmt::Display| format!("{}: {e}", dir.display());
    let json: Value = serde_json::from_slice(
        &std::fs::read(dir.join("adapter_config.json")).map_err(|e| at(&e))?,
    )
    .map_err(|e| at(&e))?;
    if json.get("peft_type").and_then(Value::as_str) != Some("LORA") {
        return Err(at(&"not a LoRA adapter"));
    }
    if json.get("use_dora").and_then(Value::as_bool) == Some(true) {
        return Err(at(&"a DoRA adapter: not read yet"));
    }
    let number = |key: &str| json.get(key).and_then(Value::as_f64);
    let targets = match json.get("target_modules") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    };
    let config = LoraConfig {
        r: number("r").map_or(8, |r| r as usize),
        alpha: number("lora_alpha").unwrap_or(8.0),
        targets,
        embedding: false,
        rslora: json.get("use_rslora").and_then(Value::as_bool) == Some(true),
        quantize: None,
    };
    let file = kapel_hub::safetensors::SafeTensors::open(dir.join("adapter_model.safetensors"))
        .map_err(|e| e.to_string())?;
    let mut factors = BTreeMap::new();
    for name in file.names() {
        let t = file.tensor(name).map_err(|e| e.to_string())?;
        let t = Tensor::from_slice(
            &t.to_f64_vec()
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
            t.shape(),
        )
        .map_err(|e| e.to_string())?;
        if let Some((n, t)) = from_peft(name, &t).map_err(|e| e.to_string())? {
            factors.insert(n, t);
        }
    }
    Ok(Adapter {
        config: LoraConfig {
            embedding: factors.keys().any(|n| n.ends_with(".lora_embedding_A")),
            ..config
        },
        factors,
    })
}

/// The model of the directory `base` with the adapter of `adapter` merged
/// into it, in host memory, its float weights in `dtype` (absent, those of
/// the base's table of embeddings): a model for `kapel run`, or the Hub.
///
/// # Errors
///
/// If the model or the adapter cannot be read, or an adapter's factor fits
/// no projection of the model.
pub fn merge_dirs(
    base: &Path,
    adapter: &Path,
    dtype: Option<DType>,
) -> std::result::Result<kapel_models::Model, String> {
    let checkpoint = kapel_hub::Checkpoint::open(base).map_err(|e| e.to_string())?;
    let model = kapel_models::Model::load(&checkpoint).map_err(|e| e.to_string())?;
    let config = model.config().clone();
    let adapter = load(adapter)?;
    let mut params = Parameters::from_model(&model);
    let dtype = dtype.unwrap_or_else(|| {
        params
            .get(&format!("{}.weight", embedding_name(&config)))
            .map_or(DType::F32, |p| p.value.dtype())
    });
    for (name, t) in adapter.factors {
        let module = name
            .strip_suffix(".lora_A.weight")
            .or_else(|| name.strip_suffix(".lora_B.weight"));
        if let Some(m) = module
            && params.get(&format!("{m}.weight")).is_none()
        {
            return Err(format!(
                "the adapter's {name} fits no projection of the model"
            ));
        }
        params.insert(name, Parameter::new(t));
    }
    let cpu = kapel_cpu::Cpu::new();
    let merged =
        merge(&cpu, &params, &config, adapter.config.scale()).map_err(|e| e.to_string())?;
    let raw = merged_config(model.raw_config(), &params, &config);
    let mut weights = BTreeMap::new();
    for (name, p) in merged.iter() {
        let v = if p.value.dtype().is_float() && p.value.dtype() != dtype {
            cpu.forward(&Cast { dtype }, &[&p.value])
                .map_err(|e| e.to_string())?
        } else {
            p.value.clone()
        };
        weights.insert(name.to_string(), v);
    }
    kapel_models::Model::new(raw, weights).map_err(|e| e.to_string())
}
