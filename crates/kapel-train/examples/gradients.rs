//! The loss and the gradients of a model on a batch of tokens, their time,
//! and the peak memory of the card: what T0's validation compares with
//! PyTorch (`tools/train/gradients.py`).
//!
//! `cargo run --release -p kapel-train [--features cuda] --example gradients --
//! MODEL TOKENS [--rows 16] [--length 1024] [--device cpu|cuda]
//! [--dtype f32|bf16] [--checkpoint] [--chunk 1024] [--repeat 3]
//! [--output FILE] [--profile]`
//!
//! `TOKENS` holds token ids as little-endian `u32`: row `r` reads
//! `length` of them from `r · length`, and predicts the next ones. With
//! `--output`, the gradients are written in `f32` to a safetensors file,
//! under the names of the parameters. With `--profile`, the time of each
//! operation of the last run, the device waited for around each.

use std::collections::BTreeMap;
use std::time::Instant;

use kapel_core::{DType, Tensor};
use kapel_models::Model;
use kapel_train::model::{Batch, Decoder, Options, Parameters};
use kapel_train::{Compute, Tape};

struct Args {
    model: String,
    tokens: String,
    rows: usize,
    length: usize,
    device: String,
    dtype: DType,
    checkpoint: bool,
    chunk: usize,
    repeat: usize,
    output: Option<String>,
    profile: bool,
}

const USAGE: &str = "usage: gradients MODEL TOKENS [--rows 16] [--length 1024] \
                     [--device cpu|cuda] [--dtype f32|bf16] [--checkpoint] [--chunk 1024] \
                     [--repeat 3] [--output FILE] [--profile]";

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        tokens: String::new(),
        rows: 16,
        length: 1024,
        device: "cpu".into(),
        dtype: DType::F32,
        checkpoint: false,
        chunk: 1024,
        repeat: 3,
        output: None,
        profile: false,
    };
    let mut args = std::env::args().skip(1);
    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| USAGE.to_string());
        let number = |v: String| v.parse::<usize>().map_err(|_| USAGE.to_string());
        match arg.as_str() {
            "--rows" => a.rows = number(value()?)?.max(1),
            "--length" => a.length = number(value()?)?.max(1),
            "--device" => a.device = value()?,
            "--dtype" => {
                a.dtype = match value()?.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    _ => return Err(USAGE.into()),
                }
            }
            "--checkpoint" => a.checkpoint = true,
            "--chunk" => a.chunk = number(value()?)?.max(1),
            "--repeat" => a.repeat = number(value()?)?.max(1),
            "--output" => a.output = Some(value()?),
            "--profile" => a.profile = true,
            _ if !arg.starts_with("--") => positional.push(arg),
            _ => return Err(USAGE.into()),
        }
    }
    let [model, tokens] = <[String; 2]>::try_from(positional).map_err(|_| USAGE.to_string())?;
    a.model = model;
    a.tokens = tokens;
    Ok(a)
}

/// What one run measured.
struct Run {
    loss: f64,
    seconds: f64,
    grads: Vec<(String, Tensor)>,
}

fn run<B: Compute>(
    backend: &B,
    synchronize: &dyn Fn(),
    decoder: &Decoder,
    params: &Parameters,
    batch: &Batch<'_>,
    options: &Options,
) -> Result<Run, String> {
    let error = |e: kapel_core::Error| e.to_string();
    synchronize();
    let start = Instant::now();
    let tape = Tape::new(backend.clone());
    let leaves = params.leaves(&tape);
    let loss = decoder.loss(&leaves, batch, options).map_err(error)?;
    let value = backend.download(&loss.value()).map_err(error)?;
    let mut grads = tape.backward(&loss).map_err(error)?;
    drop(loss);
    let named = leaves.gradients(&mut grads);
    let mut dense = Vec::new();
    for (name, g) in named {
        let shape = params
            .get(&name)
            .expect("a parameter")
            .value
            .shape()
            .to_vec();
        dense.push((name, g.dense(backend, &shape).map_err(error)?));
    }
    synchronize();
    Ok(Run {
        loss: value.to_f64_vec().map_err(error)?[0],
        seconds: start.elapsed().as_secs_f64(),
        grads: dense,
    })
}

fn measure<B: Compute>(
    backend: B,
    synchronize: &dyn Fn(),
    peak: &dyn Fn(bool) -> Option<usize>,
    a: &Args,
    model: &Model,
    tokens: &[u32],
) -> Result<(), String> {
    let error = |e: kapel_core::Error| e.to_string();
    let decoder = Decoder::new(model.config().clone()).map_err(error)?;
    let params = Parameters::from_model(model)
        .to(&backend, DType::F32)
        .map_err(error)?;
    let n = a.rows * a.length;
    let batch = Batch {
        ids: &tokens[..n],
        targets: &tokens[1..=n],
        rows: a.rows,
        segments: None,
    };
    let options = Options {
        dtype: a.dtype,
        checkpoint: a.checkpoint,
        chunk: a.chunk,
    };
    eprintln!(
        "gradients: {} parameters, {} × {} tokens, {} on {}{}",
        params.count(),
        a.rows,
        a.length,
        a.dtype,
        a.device,
        if a.checkpoint { ", checkpointed" } else { "" }
    );
    let mut best = f64::INFINITY;
    let mut last = None;
    for i in 0..a.repeat {
        peak(true);
        let r = run(&backend, synchronize, &decoder, &params, &batch, &options)?;
        let p = peak(false);
        eprintln!(
            "gradients: run {}: loss {:.6}, {:.3} s{}",
            i + 1,
            r.loss,
            r.seconds,
            p.map(|p| format!(", peak {:.3} GiB", p as f64 / (1u64 << 30) as f64))
                .unwrap_or_default()
        );
        best = best.min(r.seconds);
        last = Some((r, p));
    }
    let (r, p) = last.expect("one run at least");
    println!(
        "{}",
        serde_json::json!({
            "loss": r.loss,
            "seconds": best,
            "tokens_per_s": n as f64 / best,
            "peak_bytes": p,
        })
    );
    if let Some(path) = &a.output {
        let mut host = Vec::new();
        for (name, g) in &r.grads {
            let g = backend
                .forward(&kapel_ops::Cast { dtype: DType::F32 }, &[g])
                .map_err(error)?;
            host.push((name.clone(), backend.download(&g).map_err(error)?));
        }
        let refs: Vec<(&str, &Tensor)> = host.iter().map(|(n, t)| (n.as_str(), t)).collect();
        kapel_hub::safetensors::save(path, &refs, &BTreeMap::new()).map_err(|e| e.to_string())?;
        eprintln!("gradients: written to {path}");
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let a = parse()?;
    let bytes = std::fs::read(&a.tokens).map_err(|e| format!("{}: {e}", a.tokens))?;
    let tokens: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if tokens.len() <= a.rows * a.length {
        return Err(format!(
            "{} tokens: {} × {} + 1 are needed",
            tokens.len(),
            a.rows,
            a.length
        ));
    }
    let checkpoint = kapel_hub::Checkpoint::open(&a.model).map_err(|e| e.to_string())?;
    let model = Model::load(&checkpoint).map_err(|e| e.to_string())?;
    match a.device.as_str() {
        "cpu" if a.profile => {
            let cpu = kapel_train::profile::Profiled::new(kapel_cpu::Cpu::new(), || {});
            measure(cpu.clone(), &|| {}, &|_| None, &a, &model, &tokens)?;
            eprint!("{}", cpu.report());
            Ok(())
        }
        "cpu" => measure(
            kapel_cpu::Cpu::new(),
            &|| {},
            &|_| None,
            &a,
            &model,
            &tokens,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => {
            let cuda = kapel_cuda::Cuda::new(0).map_err(|e| e.to_string())?;
            let sync = cuda.clone();
            let counter = cuda.clone();
            let profiled = a.profile.then(|| {
                let s = cuda.clone();
                kapel_train::profile::Profiled::new(cuda.clone(), move || {
                    s.synchronize().expect("the card");
                })
            });
            if let Some(p) = &profiled {
                let measured = measure(
                    p.clone(),
                    &|| {},
                    &move |reset| {
                        if reset {
                            p.reset();
                            counter.reset_peak_memory();
                        }
                        Some(counter.peak_memory())
                    },
                    &a,
                    &model,
                    &tokens,
                );
                eprint!("{}", p.report());
                return measured;
            }
            measure(
                cuda,
                &move || sync.synchronize().expect("the card"),
                &move |reset| {
                    if reset {
                        counter.reset_peak_memory();
                    }
                    Some(counter.peak_memory())
                },
                &a,
                &model,
                &tokens,
            )
        }
        _ => Err(format!("unknown or unbuilt device {}", a.device)),
    }
}
