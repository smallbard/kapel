//! The measures Kapel publishes and tracks: the speed of a prompt, of
//! decoding one sequence and, on request, of a batch, and a perplexity that
//! shows whether the numbers changed. Writes them as JSON, for
//! `tools/bench/compare.py` to compare with a baseline.
//!
//! `cargo run --release -p kapel-cli [--features cuda] --example bench --
//! MODEL --name NAME [--device cpu|cuda] [--dtype f32|bf16|f16]
//! [--quantize FORMAT] [--quantize-output] [--prompt 512] [--decode 128] [--batch N]
//! [--text FILE] [--chunks 8] [--repeat 3] [--output FILE]`
//!
//! - **prefill**: a prompt of `--prompt` random tokens computed at once, in
//!   tokens per second;
//! - **decode**: `--decode` tokens generated one at a time after a prompt of
//!   32, greedy and past the end tokens, in tokens per second (the first token
//!   aside, which comes with the prompt);
//! - **batch**: with `--batch N`, `N` sequences of 128 + 256 tokens in one
//!   batch, as the example `throughput`, in generated tokens per second;
//! - **perplexity**: with `--text`, `--chunks` chunks of 512 tokens of the
//!   text, each scored on its second half after the first, the first token
//!   replaced by the beginning of sequence when the tokenizer has one, as
//!   llama.cpp's `llama-perplexity` does.
//!
//! Each speed is the best of `--repeat` runs, after one run that warms the
//! caches (and, on a card, records the passes).

use std::time::Instant;

use kapel_core::{DType, QuantFormat};
use kapel_infer::{LoadOptions, Pipeline, Request, Scorer};
use kapel_ops::{Backend, Transfer};
use serde_json::{Value, json};

struct Options {
    model: String,
    name: String,
    device: String,
    dtype: Option<DType>,
    quantize: Option<DType>,
    quantize_output: bool,
    prompt: usize,
    decode: usize,
    batch: Option<usize>,
    text: Option<String>,
    chunks: usize,
    repeat: usize,
    output: Option<String>,
}

const USAGE: &str = "usage: bench MODEL --name NAME [--device cpu|cuda] [--dtype f32|bf16|f16] \
                     [--quantize int8|int4|q8_0|q4_0] [--quantize-output] [--prompt 512] [--decode 128] [--batch N] \
                     [--text FILE] [--chunks 8] [--repeat 3] [--output FILE]";

fn parse() -> Result<Options, String> {
    let mut o = Options {
        model: String::new(),
        name: String::new(),
        device: "cpu".into(),
        dtype: None,
        quantize: None,
        quantize_output: false,
        prompt: 512,
        decode: 128,
        batch: None,
        text: None,
        chunks: 8,
        repeat: 3,
        output: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| USAGE.to_string());
        let number = |v: String| v.parse::<usize>().map_err(|_| USAGE.to_string());
        match arg.as_str() {
            "--name" => o.name = value()?,
            "--device" => o.device = value()?,
            "--dtype" => {
                o.dtype = Some(match value()?.as_str() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    _ => return Err(USAGE.into()),
                });
            }
            "--quantize" => {
                o.quantize = Some(match value()?.as_str() {
                    "int8" => DType::int8(128).map_err(|e| e.to_string())?,
                    "int4" => DType::int4(64).map_err(|e| e.to_string())?,
                    "q8_0" => DType::Quantized(QuantFormat::Q8_0),
                    "q4_0" => DType::Quantized(QuantFormat::Q4_0),
                    _ => return Err(USAGE.into()),
                });
            }
            "--quantize-output" => o.quantize_output = true,
            "--prompt" => o.prompt = number(value()?)?.max(1),
            "--decode" => o.decode = number(value()?)?.max(2),
            "--batch" => o.batch = Some(number(value()?)?.max(1)),
            "--text" => o.text = Some(value()?),
            "--chunks" => o.chunks = number(value()?)?.max(1),
            "--repeat" => o.repeat = number(value()?)?.max(1),
            "--output" => o.output = Some(value()?),
            _ if o.model.is_empty() && !arg.starts_with("--") => o.model = arg,
            _ => return Err(USAGE.into()),
        }
    }
    if o.model.is_empty() || o.name.is_empty() {
        return Err(USAGE.into());
    }
    Ok(o)
}

/// `count` token ids drawn from the middle of the vocabulary, away from the
/// special tokens, by a fixed generator.
fn random_tokens(seed: u64, count: usize, vocab: usize) -> Vec<u32> {
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

/// The best of `repeat` runs of `run` (a speed), after one to warm up.
fn best(repeat: usize, mut run: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    run()?;
    let mut best: f64 = 0.0;
    for _ in 0..repeat {
        best = best.max(run()?);
    }
    Ok(best)
}

fn measure<B: Backend + Transfer>(pipeline: &Pipeline<B>, o: &Options) -> Result<Value, String> {
    let error = |e: kapel_infer::InferError| e.to_string();
    let vocab = pipeline.model().config().vocab;
    let mut results = serde_json::Map::new();

    let prompt = random_tokens(1, o.prompt, vocab);
    let prefill = best(o.repeat, || {
        let mut cache = pipeline
            .cache_dtype()
            .map_or_else(kapel_models::Cache::new, kapel_models::Cache::with_dtype);
        let start = Instant::now();
        let logits = pipeline
            .model()
            .step(pipeline.backend(), &prompt, &mut cache)
            .map_err(|e| e.to_string())?;
        // Wait for the device: the logits come back to the host.
        pipeline
            .backend()
            .download(&logits)
            .map_err(|e| e.to_string())?;
        Ok(o.prompt as f64 / start.elapsed().as_secs_f64())
    })?;
    eprintln!("bench: prefill {prefill:.1} tokens/s");
    results.insert("prefill_tokens_per_s".into(), json!(prefill));

    let short = random_tokens(2, 32, vocab);
    let request = Request {
        max_tokens: o.decode,
        ignore_eos: true,
        sampling: kapel_infer::Sampling {
            temperature: 0.0,
            ..kapel_infer::Sampling::default()
        },
        ..Request::default()
    };
    let decode = best(o.repeat, || {
        let mut generation = pipeline.generate(&short, &request).map_err(error)?;
        generation.next().transpose().map_err(error)?;
        let start = Instant::now();
        let mut count = 0;
        for piece in generation {
            piece.map_err(error)?;
            count += 1;
        }
        Ok(count as f64 / start.elapsed().as_secs_f64())
    })?;
    eprintln!("bench: decode {decode:.1} tokens/s");
    results.insert("decode_tokens_per_s".into(), json!(decode));

    if let Some(sequences) = o.batch {
        let request = Request {
            max_tokens: 256,
            ..request.clone()
        };
        let requests: Vec<(Vec<u32>, Request)> = (0..sequences)
            .map(|i| (random_tokens(10 + i as u64, 128, vocab), request.clone()))
            .collect();
        let batch = best(o.repeat, || {
            let mut batch = pipeline.batch(&requests).map_err(error)?;
            let start = Instant::now();
            while !batch.is_finished() {
                batch.step().map_err(error)?;
            }
            Ok((sequences * 256) as f64 / start.elapsed().as_secs_f64())
        })?;
        eprintln!("bench: batch of {sequences}: {batch:.0} tokens/s");
        results.insert("batch_tokens_per_s".into(), json!(batch));
    }

    if let Some(text) = &o.text {
        let text = std::fs::read_to_string(text).map_err(|e| format!("{text}: {e}"))?;
        let tokenizer = pipeline.tokenizer();
        let tokens = tokenizer.encode(&text, true).map_err(|e| e.to_string())?;
        let bos = tokenizer
            .encode("", true)
            .map_err(|e| e.to_string())?
            .first()
            .copied();
        let chunks = o.chunks.min(tokens.len() / 512);
        let mut scorer = Scorer::new(pipeline, 512);
        let (mut nll, mut count) = (0.0, 0);
        for chunk in 0..chunks {
            let mut ids = tokens[chunk * 512..(chunk + 1) * 512].to_vec();
            if let Some(bos) = bos {
                ids[0] = bos;
            }
            // Positions 256 to 510 predict tokens 257 to 511.
            let score = scorer.score(&ids[..257], &ids[257..]).map_err(error)?;
            nll -= score.log_likelihood;
            count += score.tokens;
        }
        let perplexity = (nll / count.max(1) as f64).exp();
        eprintln!("bench: perplexity {perplexity:.4} over {count} tokens");
        results.insert("perplexity".into(), json!(perplexity));
    }
    Ok(Value::Object(results))
}

/// The processor's name, from `/proc/cpuinfo` on Linux.
fn processor() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|n| n.trim().to_string())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string())
}

fn main() -> Result<(), String> {
    let o = parse()?;
    let dir = {
        let path = std::path::PathBuf::from(&o.model);
        if path.exists() {
            path
        } else {
            kapel_hub::download::snapshot(&o.model, "main", &kapel_hub::download::default_cache())
                .map_err(|e| e.to_string())?
        }
    };
    let cuda = o.device.starts_with("cuda");
    let load = LoadOptions {
        dtype: o
            .dtype
            .or(Some(if cuda { DType::BF16 } else { DType::F32 })),
        quantize: o.quantize,
        quantize_output: o.quantize_output,
        fuse: cuda,
        ..LoadOptions::default()
    };
    let results = match o.device.as_str() {
        "cpu" => {
            let pipeline = Pipeline::load_with(&dir, kapel_cpu::Cpu::new(), &load)
                .map_err(|e| e.to_string())?;
            measure(&pipeline, &o)?
        }
        #[cfg(feature = "cuda")]
        device => {
            let ordinal = device
                .strip_prefix("cuda")
                .map(|n| n.trim_start_matches(':'))
                .map_or(Ok(0), |n| if n.is_empty() { Ok(0) } else { n.parse() })
                .map_err(|_| USAGE.to_string())?;
            let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
            let pipeline = Pipeline::load_with(&dir, cuda, &load).map_err(|e| e.to_string())?;
            measure(&pipeline, &o)?
        }
        #[cfg(not(feature = "cuda"))]
        _ => return Err("built without the feature `cuda`".into()),
    };
    let report = json!({
        "name": o.name,
        "model": o.model,
        "device": o.device,
        "dtype": load.dtype.map(|d| d.to_string()),
        "quantize": o.quantize.map(|q| q.to_string()),
        "quantize_output": o.quantize_output,
        "processor": processor(),
        "threads": kapel_cpu::default_threads(),
        "kapel": env!("CARGO_PKG_VERSION"),
        "results": results,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n";
    match &o.output {
        Some(path) => std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?,
        None => print!("{text}"),
    }
    Ok(())
}
