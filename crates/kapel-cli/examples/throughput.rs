//! The throughput of generation by batches, as vLLM's `vllm bench throughput`
//! measures it: `--sequences` prompts of `--input` random tokens, each
//! continued by `--output` tokens (greedy, past the end tokens), all in one
//! batch. Prints the time of the prefill and of the decoding, and the tokens
//! generated per second.
//!
//! `cargo run --release -p kapel-cli [--features cuda] --example throughput --
//! MODEL [--device cpu|cuda] [--sequences 64] [--input 128] [--output 512]
//! [--dtype bf16] [--kv-cache f8e4m3]`

use std::time::Instant;

use kapel_core::DType;
use kapel_infer::{LoadOptions, Pipeline, Request};
use kapel_ops::{Backend, Transfer};

struct Options {
    model: String,
    device: String,
    sequences: usize,
    input: usize,
    output: usize,
    dtype: Option<DType>,
    cache: Option<DType>,
}

fn parse() -> Result<Options, String> {
    let usage = "usage: throughput MODEL [--device cpu|cuda] [--sequences N] [--input N] \
                 [--output N] [--dtype f32|bf16|f16] [--kv-cache bf16|f16|f8e4m3|f8e5m2]";
    let mut options = Options {
        model: String::new(),
        device: "cpu".into(),
        sequences: 64,
        input: 128,
        output: 512,
        dtype: None,
        cache: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| usage.to_string());
        match arg.as_str() {
            "--device" => options.device = value()?,
            "--sequences" => options.sequences = value()?.parse().map_err(|_| usage)?,
            "--input" => options.input = value()?.parse().map_err(|_| usage)?,
            "--output" => options.output = value()?.parse().map_err(|_| usage)?,
            "--dtype" => {
                options.dtype = Some(match value()?.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    _ => return Err(usage.into()),
                });
            }
            "--kv-cache" => {
                options.cache = Some(match value()?.as_str() {
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    "f8e4m3" => DType::F8E4M3,
                    "f8e5m2" => DType::F8E5M2,
                    _ => return Err(usage.into()),
                });
            }
            _ if options.model.is_empty() => options.model = arg,
            _ => return Err(usage.into()),
        }
    }
    if options.model.is_empty() {
        return Err(usage.into());
    }
    Ok(options)
}

/// `count` token ids drawn from the middle of the vocabulary, away from the
/// special tokens, by a fixed generator.
fn random_prompt(seed: u64, count: usize, vocab: usize) -> Vec<u32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (1000 + state % (vocab as u64 - 2000)) as u32
        })
        .collect()
}

fn run<B: Backend + Transfer>(pipeline: &Pipeline<B>, options: &Options) -> Result<(), String> {
    let vocab = pipeline.model().config().vocab;
    let request = Request {
        max_tokens: options.output,
        ignore_eos: true,
        ..Request::default()
    };
    let requests: Vec<(Vec<u32>, Request)> = (0..options.sequences)
        .map(|i| {
            (
                random_prompt(i as u64 + 1, options.input, vocab),
                request.clone(),
            )
        })
        .collect();
    let mut batch = pipeline.batch(&requests).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut prefill = None;
    let mut steps = 0;
    while !batch.is_finished() {
        let pieces = batch.step().map_err(|e| e.to_string())?;
        steps += 1;
        // The prompts are computed when every sequence has its first token.
        if prefill.is_none() && (0..options.sequences).all(|i| !batch.tokens(i).is_empty()) {
            prefill = Some(start.elapsed().as_secs_f64());
        }
        debug_assert!(!pieces.is_empty());
    }
    let total = start.elapsed().as_secs_f64();
    let prefill = prefill.unwrap_or(total);
    let generated = options.sequences * options.output;
    println!(
        "{} sequences, {} + {} tokens: {steps} passes, prefill {prefill:.2} s, decoding {:.2} s",
        options.sequences,
        options.input,
        options.output,
        total - prefill
    );
    println!(
        "{:.0} generated tokens/s overall, {:.0} in decoding, {:.0} tokens/s with the prompts",
        generated as f64 / total,
        (generated - options.sequences) as f64 / (total - prefill),
        (generated + options.sequences * options.input) as f64 / total
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let options = parse()?;
    match options.device.as_str() {
        "cpu" => {
            let load = LoadOptions {
                dtype: options.dtype.or(Some(DType::F32)),
                cache_dtype: options.cache,
                ..LoadOptions::default()
            };
            let pipeline = Pipeline::load_with(&options.model, kapel_cpu::Cpu::new(), &load)
                .map_err(|e| e.to_string())?;
            run(&pipeline, &options)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let cuda = kapel_cuda::Cuda::new(0).map_err(|e| e.to_string())?;
            let load = LoadOptions {
                dtype: options.dtype.or(Some(DType::BF16)),
                fuse: true,
                cache_dtype: options.cache,
                ..LoadOptions::default()
            };
            let pipeline =
                Pipeline::load_with(&options.model, cuda, &load).map_err(|e| e.to_string())?;
            run(&pipeline, &options)
        }
        device => Err(format!("unknown device {device}")),
    }
}
