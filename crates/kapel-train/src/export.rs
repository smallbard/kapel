//! Exports: a checkpoint of a run as a model, or an adapter, to publish on
//! the Hugging Face Hub.
//!
//! A checkpoint is already a model `transformers` loads (or an adapter PEFT
//! loads), but it holds what a run needs to go on and the Hub does not: the
//! weights in `f32`, the optimizer's moments, the run's state. [`export`]
//! writes the directory a repository of the Hub holds:
//!
//! - a model: `model.safetensors` in the type asked (`bf16` by default) and
//!   `config.json` naming it (`torch_dtype`);
//! - or an adapter: `adapter_model.safetensors` and `adapter_config.json`,
//!   its base named by the Hub's name when the run started from the Hub;
//!   or, with `merge`, the model with its adapter merged ([`crate::lora::merge_dirs`]);
//! - the tokenizer's files and `generation_config.json`, from the model the
//!   run started from or from the directory given;
//! - `README.md`, a model card: what the model is, how it was trained, its
//!   last losses and the run's configuration, the paths of this machine
//!   reduced to their file names.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kapel_core::{DType, Tensor};
use kapel_ops::{Cast, Run as _};
use serde_json::{Value, json};

use crate::config::TrainConfig;

/// The files of a tokenizer, and the settings of generation, copied with a
/// model.
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
];

/// What [`export`] does.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// The type of the float weights of a model (adapters keep theirs).
    pub dtype: DType,
    /// A directory whose tokenizer goes with the model, instead of the
    /// tokenizer of the model the run started from.
    pub tokenizer: Option<PathBuf>,
    /// An adapter merged into its model, written as a model.
    pub merge: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            dtype: DType::BF16,
            tokenizer: None,
            merge: false,
        }
    }
}

/// What [`export`] wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exported {
    /// The checkpoint exported.
    pub checkpoint: PathBuf,
    /// An adapter (rather than a model).
    pub adapter: bool,
    /// The files written, by name.
    pub files: Vec<String>,
    /// The name of the model the run started from on the Hub, if any.
    pub base: Option<String>,
}

fn at(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", path.display())
}

/// The checkpoint `path` names: a checkpoint, or the last of a run's
/// directory.
///
/// # Errors
///
/// If `path` is neither.
pub fn checkpoint_of(path: &Path) -> Result<PathBuf, String> {
    if path.join("state.json").exists() {
        return Ok(path.to_path_buf());
    }
    crate::checkpoint::list(path)
        .pop()
        .map(|(_, p)| p)
        .ok_or_else(|| at(path, "neither a checkpoint nor a run's directory with one"))
}

/// The name on the Hub of a model in Kapel's cache
/// (`…/hub/OWNER--NAME/REVISION`), if `dir` is one.
pub fn hub_name_of_cache(dir: &Path) -> Option<String> {
    let repo = dir.parent()?;
    if repo.parent()?.file_name()? != "hub" {
        return None;
    }
    let (owner, name) = repo.file_name()?.to_str()?.split_once("--")?;
    Some(format!("{owner}/{name}"))
}

/// The `torch_dtype` of `config.json` for a type.
fn torch_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        _ => "float32",
    }
}

/// Exports the checkpoint `path` (or the last of the run's directory
/// `path`) into `out`, as the module says.
///
/// # Errors
///
/// If the checkpoint cannot be read, `out` is the checkpoint, or a file
/// cannot be written.
pub fn export(path: &Path, out: &Path, options: &ExportOptions) -> Result<Exported, String> {
    let checkpoint = checkpoint_of(path)?;
    if out.exists() && std::fs::canonicalize(out).ok() == std::fs::canonicalize(&checkpoint).ok() {
        return Err(at(
            out,
            "the checkpoint itself: export into another directory",
        ));
    }
    if !matches!(options.dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Err(format!(
            "the weights in bf16, f16 or f32, not {:?}",
            options.dtype
        ));
    }
    let toml_path = checkpoint.join("config.toml");
    let run =
        TrainConfig::parse(&std::fs::read_to_string(&toml_path).map_err(|e| at(&toml_path, e))?)
            .map_err(|e| at(&toml_path, e))?;
    // The model the run started from: a directory, a config.json, or the
    // Hub's name of a model in the cache.
    let source = &run.model.config;
    let hub = run
        .model
        .hub
        .clone()
        .or_else(|| {
            crate::config::is_hub_name(source)
                .then(|| source.to_string_lossy().into_owned())
                .filter(|_| !source.exists())
        })
        .or_else(|| hub_name_of_cache(source));
    let source_dir = if source.is_dir() {
        Some(source.clone())
    } else if source.is_file() {
        source.parent().map(Path::to_path_buf)
    } else if let Some(name) = &hub {
        Some(
            kapel_hub::download::snapshot(name, "main", &kapel_hub::download::default_cache())
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let fine_tuned = run.model.init == "weights";
    let base = hub.clone().filter(|_| fine_tuned);

    std::fs::create_dir_all(out).map_err(|e| at(out, e))?;
    let mut files = Vec::new();
    let adapter = checkpoint.join("adapter_config.json").exists();
    if adapter && checkpoint.join("trainable.safetensors").exists() {
        return Err(at(
            &checkpoint,
            "the run trained parameters beside its adapters (trainable.safetensors): \
             PEFT has no place for them, and merging leaves them out",
        ));
    }
    let model_json: Value;
    if adapter && options.merge {
        let dir = source_dir
            .as_deref()
            .filter(|d| d.join("config.json").exists())
            .ok_or_else(|| at(source, "the adapter's model is not a model directory"))?;
        let merged = crate::lora::merge_dirs(dir, &checkpoint, Some(options.dtype))?;
        merged.save(out).map_err(|e| e.to_string())?;
        let mut raw = merged.raw_config().clone();
        set_dtype(&mut raw, options.dtype);
        write_json(&out.join("config.json"), &raw)?;
        model_json = raw;
        files.extend(["config.json".to_string(), "model.safetensors".to_string()]);
    } else if adapter {
        let config_path = checkpoint.join("adapter_config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|e| at(&config_path, e))?)
                .map_err(|e| at(&config_path, e))?;
        // The Hub's name of the base; a path of this machine reduced to
        // its last name.
        if let Some(o) = config.as_object_mut() {
            let local = o
                .get("base_model_name_or_path")
                .and_then(Value::as_str)
                .filter(|p| Path::new(p).is_absolute())
                .and_then(|p| Path::new(p).file_name())
                .map(|n| n.to_string_lossy().into_owned());
            if let Some(name) = hub.clone().or(local) {
                o.insert("base_model_name_or_path".into(), json!(name));
            }
        }
        write_json(&out.join("adapter_config.json"), &config)?;
        let from = checkpoint.join("adapter_model.safetensors");
        std::fs::copy(&from, out.join("adapter_model.safetensors")).map_err(|e| at(&from, e))?;
        model_json = read_json(&checkpoint.join("config.json"))?;
        files.extend([
            "adapter_config.json".to_string(),
            "adapter_model.safetensors".to_string(),
        ]);
    } else {
        let weights = checkpoint.join("model.safetensors");
        let file =
            kapel_hub::safetensors::SafeTensors::open(&weights).map_err(|e| e.to_string())?;
        let cpu = kapel_cpu::Cpu::new();
        let tensors: Vec<(String, Tensor)> = file
            .names()
            .map(|name| {
                let t = file.tensor(name).map_err(|e| e.to_string())?;
                let t = if t.dtype().is_float() && t.dtype() != options.dtype {
                    cpu.forward(
                        &Cast {
                            dtype: options.dtype,
                        },
                        &[&t],
                    )
                    .map_err(|e| e.to_string())?
                } else {
                    t
                };
                Ok((name.to_string(), t))
            })
            .collect::<Result<_, String>>()?;
        let refs: Vec<(&str, &Tensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        kapel_hub::safetensors::save(out.join("model.safetensors"), &refs, &metadata)
            .map_err(|e| e.to_string())?;
        let mut raw = read_json(&checkpoint.join("config.json"))?;
        set_dtype(&mut raw, options.dtype);
        write_json(&out.join("config.json"), &raw)?;
        model_json = raw;
        files.extend(["config.json".to_string(), "model.safetensors".to_string()]);
    }

    // The tokenizer, and the settings of generation.
    let tokenizer_dir = options.tokenizer.clone().or_else(|| source_dir.clone());
    if let Some(dir) = &tokenizer_dir {
        for name in TOKENIZER_FILES {
            let from = dir.join(name);
            if from.is_file() {
                std::fs::copy(&from, out.join(name)).map_err(|e| at(&from, e))?;
                files.push((*name).to_string());
            }
        }
    }
    if !adapter || options.merge {
        let generation = source_dir
            .iter()
            .chain(tokenizer_dir.iter())
            .map(|d| d.join("generation_config.json"))
            .find(|p| p.is_file());
        match generation {
            Some(from) => {
                std::fs::copy(&from, out.join("generation_config.json"))
                    .map_err(|e| at(&from, e))?;
                files.push("generation_config.json".to_string());
            }
            None => {
                let mut g = serde_json::Map::new();
                for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
                    if let Some(v) = model_json.get(key).filter(|v| !v.is_null()) {
                        g.insert(key.into(), v.clone());
                    }
                }
                if !g.is_empty() {
                    write_json(&out.join("generation_config.json"), &Value::Object(g))?;
                    files.push("generation_config.json".to_string());
                }
            }
        }
    }

    let card = model_card(
        &checkpoint,
        &run,
        &model_json,
        base.as_deref(),
        adapter && !options.merge,
        out,
    )?;
    std::fs::write(out.join("README.md"), card).map_err(|e| at(out, e))?;
    files.push("README.md".to_string());
    files.sort();
    files.dedup();
    Ok(Exported {
        checkpoint,
        adapter: adapter && !options.merge,
        files,
        base,
    })
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| at(path, e))?).map_err(|e| at(path, e))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut text = serde_json::to_string_pretty(value).expect("JSON serializes");
    text.push('\n');
    std::fs::write(path, text).map_err(|e| at(path, e))
}

fn set_dtype(config: &mut Value, dtype: DType) {
    if let Some(o) = config.as_object_mut() {
        o.insert("torch_dtype".into(), json!(torch_dtype(dtype)));
    }
}

/// The run's configuration as the card shows it: each absolute path reduced
/// to its file name (the paths of this machine are nobody else's business),
/// a model of the Hub by its name.
pub fn public_config(run: &TrainConfig, hub: Option<&str>) -> String {
    let mut value = toml::Value::try_from(run).expect("a configuration is TOML");
    fn reduce(v: &mut toml::Value) {
        match v {
            toml::Value::String(s) if Path::new(s.as_str()).is_absolute() => {
                if let Some(name) = Path::new(s.as_str()).file_name() {
                    *s = name.to_string_lossy().into_owned();
                }
            }
            toml::Value::Array(a) => a.iter_mut().for_each(reduce),
            toml::Value::Table(t) => t.iter_mut().for_each(|(_, v)| reduce(v)),
            _ => {}
        }
    }
    if let (Some(name), Some(model)) = (
        hub,
        value.get_mut("model").and_then(toml::Value::as_table_mut),
    ) {
        model.insert("config".into(), toml::Value::String(name.to_string()));
    }
    reduce(&mut value);
    toml::to_string_pretty(&value).expect("TOML serializes")
}

/// The last losses and accuracies the run's log shows up to the checkpoint's
/// step: the means of the last 20 steps logged (the training loss, and
/// GRPO's reward or DPO's accuracy), and the numbers of the last evaluation,
/// a task's under its name.
fn last_measures(run_dir: &Path, step: usize) -> Vec<(String, f64)> {
    let Ok(text) = std::fs::read_to_string(run_dir.join("log.jsonl")) else {
        return Vec::new();
    };
    let (mut steps, mut evals) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let Ok(Value::Object(o)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match o.get("step").and_then(Value::as_u64) {
            Some(s) if s as usize <= step => {}
            _ => continue,
        }
        if o.contains_key("loss") {
            steps.push(o);
        } else {
            evals.push(o);
        }
    }
    let mut out = Vec::new();
    let tail = &steps[steps.len().saturating_sub(20)..];
    for (key, label) in [
        ("loss", "training loss"),
        ("reward", "reward"),
        ("accuracy", "accuracy of the pairs"),
    ] {
        let values: Vec<f64> = tail.iter().filter_map(|o| o.get(key)?.as_f64()).collect();
        if !values.is_empty() {
            out.push((
                format!("{label} (mean of the last {} steps logged)", values.len()),
                values.iter().sum::<f64>() / values.len() as f64,
            ));
        }
    }
    let last = evals.last().and_then(|o| o.get("step")).cloned();
    for o in evals.iter().filter(|o| o.get("step") == last.as_ref()) {
        let task = o.get("task").and_then(Value::as_str);
        for (k, v) in o {
            let Some(v) = v.as_f64().filter(|_| k != "step" && k != "tokens") else {
                continue;
            };
            let label = match (task, k.as_str()) {
                (Some(t), k) => format!("{t} {}", k.replace('_', " ")),
                (None, "val_loss") => "validation loss".to_string(),
                (None, "control_loss") => "control loss".to_string(),
                (None, k) => k.replace("val_", "validation ").replace('_', " "),
            };
            out.push((label, v));
        }
    }
    out
}

fn model_card(
    checkpoint: &Path,
    run: &TrainConfig,
    model: &Value,
    base: Option<&str>,
    adapter: bool,
    out: &Path,
) -> Result<String, String> {
    let state: crate::checkpoint::State =
        serde_json::from_value(read_json(&checkpoint.join("state.json"))?)
            .map_err(|e| at(checkpoint, e))?;
    let name = out
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "model".into());
    let kind = model
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("transformer");
    let method = if run.grpo.is_some() {
        "GRPO"
    } else if run.dpo.is_some() {
        "DPO"
    } else if run.model.init == "weights" {
        "supervised fine-tuning"
    } else {
        "pre-training from scratch"
    };
    let mut card = String::from("---\n");
    card += if adapter {
        "library_name: peft\n"
    } else {
        "library_name: transformers\n"
    };
    if let Some(b) = base {
        card += &format!("base_model: {b}\n");
    }
    card += "pipeline_tag: text-generation\ntags:\n- kapel\n";
    if adapter {
        card += "- lora\n";
    }
    card += "---\n\n";
    card += &format!("# {name}\n\n");
    let what = if adapter { "A LoRA adapter of a" } else { "A" };
    let from = match (base, run.model.init == "weights") {
        (Some(b), _) => format!(" from `{b}`"),
        (None, true) => " from a published model".to_string(),
        (None, false) => String::new(),
    };
    card += &format!(
        "{what} {kind} model trained with Kapel: {method}{from}{}.\n\n",
        run.lora
            .as_ref()
            .map(|l| format!(", with LoRA adapters of rank {} (α = {})", l.r, l.alpha))
            .filter(|_| !adapter)
            .unwrap_or_default()
    );
    card += "| | |\n| --- | --- |\n";
    card += &format!("| Steps | {} |\n", state.step);
    if state.tokens > 0 {
        card += &format!("| Tokens seen | {} |\n", state.tokens);
    }
    if let Some(run_dir) = checkpoint.parent() {
        for (k, v) in last_measures(run_dir, state.step) {
            card += &format!("| {k} | {v:.4} |\n");
        }
    }
    card += "\n## Use\n\n```python\n";
    if adapter {
        let b = base.unwrap_or("BASE_MODEL");
        card += &format!(
            "from peft import PeftModel\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n\
             model = AutoModelForCausalLM.from_pretrained(\"{b}\")\n\
             model = PeftModel.from_pretrained(model, \"OWNER/{name}\")\n\
             tokenizer = AutoTokenizer.from_pretrained(\"{b}\")\n"
        );
    } else {
        card += &format!(
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n\
             model = AutoModelForCausalLM.from_pretrained(\"OWNER/{name}\")\n\
             tokenizer = AutoTokenizer.from_pretrained(\"OWNER/{name}\")\n"
        );
    }
    card += "```\n\n";
    if !adapter {
        card += &format!("Or with Kapel: `kapel run OWNER/{name}`.\n\n");
    }
    card += "## Training configuration\n\nThe configuration of `kapel train`, its paths reduced to their file names:\n\n```toml\n";
    card += &public_config(run, base);
    card += "```\n";
    Ok(card)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_in_the_cache_is_known_by_its_name() {
        let dir = Path::new("/home/me/.cache/kapel/hub/Qwen--Qwen3-0.6B/c1899de2");
        assert_eq!(hub_name_of_cache(dir).as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(hub_name_of_cache(Path::new("/data/models/qwen/abc")), None);
    }

    #[test]
    fn the_card_shows_no_path_of_the_machine() {
        let mut run = TrainConfig::parse(
            r#"
            [model]
            config = "Qwen/Qwen3-0.6B"
            init = "weights"
            [data]
            train = "data"
            [batch]
            rows = 2
            length = 8
            [optimizer]
            kind = "adamw"
            lr = 1e-4
            [schedule]
            steps = 3
            "#,
        )
        .unwrap();
        run.resolve(Path::new("/home/someone/secret/recipes"));
        let text = public_config(&run, Some("Qwen/Qwen3-0.6B"));
        assert!(!text.contains("/home"), "{text}");
        assert!(text.contains("config = \"Qwen/Qwen3-0.6B\""), "{text}");
        assert!(text.contains("train = \"data\""), "{text}");
    }
}
