//! A few steps of training: a model from its configuration or its weights,
//! batches read in order from a file of tokens, AdamW, the loss of each
//! step. The training loop of T1 is the full version; this one shows that
//! the pieces of T0 train.
//!
//! `cargo run --release -p kapel-train [--features cuda] --example steps --
//! MODEL TOKENS [--scratch] [--steps 20] [--rows 16] [--length 1024]
//! [--lr 6e-4] [--device cpu|cuda] [--dtype bf16] [--checkpoint]`
//!
//! With `--scratch`, the weights start from GPT-2's initialization
//! ([`kapel_train::model::init`]) instead of `MODEL`'s.

use std::time::Instant;

use kapel_core::DType;
use kapel_models::Model;
use kapel_train::model::{Batch, Decoder, Options, Parameters, init};
use kapel_train::optim::AdamW;
use kapel_train::{Compute, Tape};

struct Args {
    model: String,
    tokens: String,
    scratch: bool,
    steps: usize,
    rows: usize,
    length: usize,
    lr: f64,
    device: String,
    dtype: DType,
    checkpoint: bool,
}

const USAGE: &str = "usage: steps MODEL TOKENS [--scratch] [--steps 20] [--rows 16] \
                     [--length 1024] [--lr 6e-4] [--device cpu|cuda] [--dtype f32|bf16] [--checkpoint]";

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        tokens: String::new(),
        scratch: false,
        steps: 20,
        rows: 16,
        length: 1024,
        lr: 6e-4,
        device: "cpu".into(),
        dtype: DType::BF16,
        checkpoint: false,
    };
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| USAGE.to_string());
        let number = |v: String| v.parse::<usize>().map_err(|_| USAGE.to_string());
        match arg.as_str() {
            "--scratch" => a.scratch = true,
            "--steps" => a.steps = number(value()?)?,
            "--rows" => a.rows = number(value()?)?.max(1),
            "--length" => a.length = number(value()?)?.max(1),
            "--lr" => a.lr = value()?.parse().map_err(|_| USAGE.to_string())?,
            "--device" => a.device = value()?,
            "--dtype" => {
                a.dtype = match value()?.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    _ => return Err(USAGE.into()),
                }
            }
            "--checkpoint" => a.checkpoint = true,
            _ if !arg.starts_with("--") => positional.push(arg),
            _ => return Err(USAGE.into()),
        }
    }
    let [model, tokens] = <[String; 2]>::try_from(positional).map_err(|_| USAGE.to_string())?;
    a.model = model;
    a.tokens = tokens;
    Ok(a)
}

fn train<B: Compute>(
    backend: B,
    sync: &dyn Fn(),
    a: &Args,
    model: &Model,
    tokens: &[u32],
) -> Result<(), String> {
    let error = |e: kapel_core::Error| e.to_string();
    let config = model.config().clone();
    let decoder = Decoder::new(config.clone()).map_err(error)?;
    let params = if a.scratch {
        init(&config, 0.02, 1).map_err(error)?
    } else {
        Parameters::from_model(model)
    };
    let mut params = params.to(&backend, DType::F32).map_err(error)?;
    let mut optimizer = AdamW::new(a.lr, 0.1);
    let options = Options {
        dtype: a.dtype,
        checkpoint: a.checkpoint,
        chunk: 1024,
    };
    let n = a.rows * a.length;
    eprintln!("steps: {} parameters, {} tokens a step", params.count(), n);
    for step in 0..a.steps {
        let at = (step * n) % (tokens.len() - n - 1);
        let batch = Batch {
            ids: &tokens[at..at + n],
            targets: &tokens[at + 1..=at + n],
            rows: a.rows,
            segments: None,
        };
        sync();
        let start = Instant::now();
        let tape = Tape::new(backend.clone());
        let leaves = params.leaves(&tape);
        let loss = decoder.loss(&leaves, &batch, &options).map_err(error)?;
        let value = backend
            .download(&loss.value())
            .map_err(error)?
            .to_f64_vec()
            .map_err(error)?[0];
        let mut grads = tape.backward(&loss).map_err(error)?;
        drop(loss);
        let grads = leaves.gradients(&mut grads);
        drop(tape);
        optimizer
            .step(&backend, &mut params, grads, 1.0)
            .map_err(error)?;
        sync();
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "step {:>3}: loss {value:.4}, {:.3} s, {:.0} tokens/s",
            step + 1,
            seconds,
            n as f64 / seconds
        );
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
    if tokens.len() <= a.rows * a.length + 1 {
        return Err("too few tokens for one batch".into());
    }
    let checkpoint = kapel_hub::Checkpoint::open(&a.model).map_err(|e| e.to_string())?;
    let model = Model::load(&checkpoint).map_err(|e| e.to_string())?;
    match a.device.as_str() {
        "cpu" => train(kapel_cpu::Cpu::new(), &|| {}, &a, &model, &tokens),
        #[cfg(feature = "cuda")]
        "cuda" => {
            let cuda = kapel_cuda::Cuda::new(0).map_err(|e| e.to_string())?;
            let sync = cuda.clone();
            train(
                cuda,
                &move || sync.synchronize().expect("the card"),
                &a,
                &model,
                &tokens,
            )
        }
        _ => Err(format!("unknown or unbuilt device {}", a.device)),
    }
}
