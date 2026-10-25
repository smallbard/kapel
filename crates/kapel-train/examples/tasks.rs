//! The accuracy of a model on prompts whose answers a program checks:
//!
//! ```text
//! cargo run --release -p kapel-train --features cuda --example tasks -- \
//!     MODEL PROMPTS.jsonl [--adapter DIR] [--limit N] [--max-tokens N] \
//!     [--reward math|exact] [--field answer] [--no-thinking] [--cpu] \
//!     [--answers OUT.jsonl] [--cache-gib 6]
//! ```
//!
//! MODEL is a model directory; with `--adapter`, a LoRA adapter (PEFT's or
//! Kapel's) merged into it first. Each record of PROMPTS has its prompt in
//! `prompt` and its expected answer in the field of the reward. The answers
//! are greedy, written by the inference engine in `bf16` on the card (`f32`
//! on the processor); `--no-thinking` gives the chat template
//! `enable_thinking = false` (Qwen 3). Prints the accuracy, and writes each
//! answer with its reward to OUT if asked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kapel_core::DType;
use kapel_infer::{Pipeline, Request, Sampling};
use kapel_models::Model;
use kapel_train::Compute;
use kapel_train::chat::Encoder;
use kapel_train::lora;
use kapel_train::model::{Parameter, Parameters};
use kapel_train::rewards::{RewardConfig, Rewards};

struct Args {
    model: PathBuf,
    prompts: String,
    adapter: Option<PathBuf>,
    limit: Option<usize>,
    max_tokens: usize,
    reward: String,
    field: String,
    thinking: bool,
    cpu: bool,
    answers: Option<PathBuf>,
    cache_gib: f64,
}

fn args() -> Result<Args, String> {
    let mut all = std::env::args().skip(1);
    let usage = "usage: tasks MODEL PROMPTS [--adapter DIR] [--limit N] [--max-tokens N] \
                 [--reward math|exact] [--field NAME] [--no-thinking] [--cpu] [--answers OUT] \
                 [--cache-gib G]";
    let model = PathBuf::from(all.next().ok_or(usage)?);
    let prompts = all.next().ok_or(usage)?;
    let mut a = Args {
        model,
        prompts,
        adapter: None,
        limit: None,
        max_tokens: 512,
        reward: "math".into(),
        field: "answer".into(),
        thinking: true,
        cpu: false,
        answers: None,
        cache_gib: 6.0,
    };
    while let Some(arg) = all.next() {
        let mut value = || all.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--adapter" => a.adapter = Some(PathBuf::from(value()?)),
            "--limit" => a.limit = Some(value()?.parse().map_err(|_| "--limit: a number")?),
            "--max-tokens" => {
                a.max_tokens = value()?.parse().map_err(|_| "--max-tokens: a number")?;
            }
            "--reward" => a.reward = value()?,
            "--field" => a.field = value()?,
            "--answers" => a.answers = Some(PathBuf::from(value()?)),
            "--cache-gib" => a.cache_gib = value()?.parse().map_err(|_| "--cache-gib: a number")?,
            "--no-thinking" => a.thinking = false,
            "--cpu" => a.cpu = true,
            other => return Err(format!("unknown option {other}\n{usage}")),
        }
    }
    Ok(a)
}

/// The model of `dir` on `backend` in `dtype`, the adapter merged.
fn model<B: Compute>(
    backend: &B,
    dir: &Path,
    adapter: Option<&Path>,
    dtype: DType,
) -> Result<Model, Box<dyn std::error::Error>> {
    let base = Model::load(&kapel_hub::Checkpoint::open(dir)?)?;
    let Some(adapter) = adapter else {
        return Ok(base
            .upload(backend)?
            .cast(backend, dtype)?
            .into_fused(backend)?);
    };
    let adapter = lora::load(adapter)?;
    let config = base.config().clone();
    let mut params = Parameters::from_model(&base);
    for (name, t) in adapter.factors {
        params.insert(name, Parameter::new(t));
    }
    let params = params.to(backend, DType::F32)?;
    let merged = lora::merge(backend, &params, &config, adapter.config.scale())?;
    let raw = lora::merged_config(base.raw_config(), &params, &config);
    let mut weights = BTreeMap::new();
    for (n, p) in merged.iter() {
        let v = if p.value.dtype().is_float() {
            backend.forward(&kapel_ops::Cast { dtype }, &[&p.value])?
        } else {
            p.value.clone()
        };
        weights.insert(n.to_string(), v);
    }
    Ok(Model::new(raw, weights)?.into_fused(backend)?)
}

fn evaluate<B: Compute>(
    backend: B,
    a: &Args,
    dtype: DType,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let encoder = Encoder::from_dir(&a.model)?;
    let mut variables = BTreeMap::new();
    if !a.thinking {
        variables.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
    }
    let prompts = kapel_train::grpo::read_prompts(&a.prompts, &encoder, &variables, a.limit)?;
    let m = model(&backend, &a.model, a.adapter.as_deref(), dtype)?;
    let pipeline = Pipeline::with_model(&a.model, backend, m)?;
    let loaded = started.elapsed().as_secs_f64();
    let request = Request {
        sampling: Sampling::greedy(),
        max_tokens: a.max_tokens,
        ..Request::default()
    };
    let asked: Vec<&[u32]> = prompts.iter().map(|p| p.tokens.as_slice()).collect();
    let cache = (a.cache_gib * f64::from(1u32 << 30)) as usize;
    let answers = kapel_train::grpo::generate(&pipeline, &asked, &request, 0, cache)?;
    let writing = started.elapsed().as_secs_f64() - loaded;
    let rewards = Rewards::new(&[RewardConfig {
        kind: a.reward.clone(),
        weight: 1.0,
        field: a.field.clone(),
        pattern: None,
        command: Vec::new(),
    }])?;
    let scored: Vec<(&str, &serde_json::Value)> = answers
        .iter()
        .zip(&prompts)
        .map(|(c, p)| (c.text.as_str(), &p.record))
        .collect();
    let (scores, _) = rewards.score(&scored)?;
    let n = scores.len() as f64;
    let tokens: usize = answers.iter().map(|c| c.tokens.len()).sum();
    println!(
        "{:.2} % of {} right ({:.0} tokens an answer, {:.1} % ended; {:.1} s to load, {:.1} s to write, {:.0} tokens/s)",
        100.0 * scores.iter().sum::<f64>() / n,
        scores.len(),
        tokens as f64 / n,
        100.0 * answers.iter().filter(|c| c.ended).count() as f64 / n,
        loaded,
        writing,
        tokens as f64 / writing
    );
    if let Some(out) = &a.answers {
        let lines: Vec<String> = answers
            .iter()
            .zip(&prompts)
            .zip(&scores)
            .map(|((c, p), s)| {
                serde_json::json!({"answer": c.text, "reward": s, "expected": p.record[&a.field]})
                    .to_string()
            })
            .collect();
        std::fs::write(out, lines.join("\n") + "\n")?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = args()?;
    if a.cpu {
        return evaluate(kapel_cpu::Cpu::new(), &a, DType::F32);
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = kapel_cuda::Cuda::new(0)?;
        evaluate(cuda, &a, DType::BF16)
    }
    #[cfg(not(feature = "cuda"))]
    Err("built without the feature `cuda`: --cpu".into())
}
