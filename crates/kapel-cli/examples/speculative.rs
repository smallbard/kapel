//! Speculative decoding against decoding alone: the same greedy answers to a
//! few chat prompts, the tokens per second of each, and how many proposals
//! were accepted.
//!
//! `cargo run --release -p kapel-cli --features cuda --example speculative --
//! MODEL [--draft DRAFT] [--tokens 4,5] [--ngram] [--multi-token]
//! [--max-tokens 256] [--device cuda]`
//!
//! With `--draft`, the smaller model proposes; `--ngram`, the tokens of the
//! sequence (prompt lookup); `--multi-token`, the model's own layer (MiMo).

use std::time::Instant;

use kapel_core::DType;
use kapel_infer::{Draft, LoadOptions, Pipeline, Request};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::Message;

const PROMPTS: [&str; 6] = [
    "Explain how a hash map works, with an example in Rust.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between TCP and UDP? Answer with a list.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number, then explain it.",
    "Translate into French: The weather is nice today, so we will walk to the market.",
];

struct Options {
    model: String,
    draft: Option<String>,
    tokens: Vec<usize>,
    ngram: bool,
    multi_token: bool,
    max_tokens: usize,
    device: String,
}

fn parse() -> Result<Options, String> {
    let usage = "usage: speculative MODEL [--draft DIR] [--tokens 4,5] [--ngram] \
                 [--multi-token] [--max-tokens N] [--device cpu|cuda]";
    let mut o = Options {
        model: String::new(),
        draft: None,
        tokens: vec![4],
        ngram: false,
        multi_token: false,
        max_tokens: 256,
        device: "cuda".into(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| usage.to_string());
        match arg.as_str() {
            "--draft" => o.draft = Some(value()?),
            "--tokens" => {
                o.tokens = value()?
                    .split(',')
                    .map(|t| t.parse().map_err(|_| usage.to_string()))
                    .collect::<Result<_, _>>()?;
            }
            "--ngram" => o.ngram = true,
            "--multi-token" => o.multi_token = true,
            "--max-tokens" => o.max_tokens = value()?.parse().map_err(|_| usage)?,
            "--device" => o.device = value()?,
            _ if o.model.is_empty() => o.model = arg,
            _ => return Err(usage.into()),
        }
    }
    if o.model.is_empty() {
        return Err(usage.into());
    }
    Ok(o)
}

fn run<'d, B: Backend + Transfer>(
    target: &Pipeline<B>,
    draft: Option<&'d Pipeline<B>>,
    o: &Options,
) -> Result<(), String> {
    let error = |e: kapel_infer::InferError| e.to_string();
    let request = Request {
        max_tokens: o.max_tokens,
        ..Request::default()
    };
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| {
            target
                .chat_prompt(&[Message::new("user", *p)])
                .map_err(error)
        })
        .collect::<Result<_, _>>()?;
    // A first answer, to compile the kernels and record the passes.
    for piece in target
        .generate(&prompts[0], &request)
        .map_err(error)?
        .take(8)
    {
        piece.map_err(error)?;
    }
    let mut alone = Vec::new();
    let start = Instant::now();
    for prompt in &prompts {
        let mut generation = target.generate(prompt, &request).map_err(error)?;
        for piece in generation.by_ref() {
            piece.map_err(error)?;
        }
        alone.push(generation.tokens().to_vec());
    }
    let time = start.elapsed().as_secs_f64();
    let count: usize = alone.iter().map(Vec::len).sum();
    println!(
        "{:<22} {:>7.1} tokens/s ({count} tokens in {time:.2} s)",
        "alone",
        count as f64 / time
    );
    type Make<'d, B> = Box<dyn Fn() -> Draft<'d, B> + 'd>;
    let mut proposers: Vec<(String, Make<'d, B>)> = Vec::new();
    for &k in &o.tokens {
        if let Some(draft) = draft {
            proposers.push((
                format!("draft, {k} tokens"),
                Box::new(move || Draft::Model {
                    pipeline: draft,
                    tokens: k,
                }),
            ));
        }
        if o.ngram {
            proposers.push((
                format!("prompt lookup, {k}"),
                Box::new(move || Draft::Ngram {
                    tokens: k,
                    longest: 4,
                    shortest: 2,
                }),
            ));
        }
    }
    if o.multi_token {
        proposers.push(("multi-token layer".into(), Box::new(|| Draft::MultiToken)));
    }
    for (name, proposer) in &proposers {
        // Warm the proposer too.
        for piece in target
            .speculate(&prompts[0], &request, proposer())
            .map_err(error)?
            .take(8)
        {
            piece.map_err(error)?;
        }
        let start = Instant::now();
        let (mut same, mut passes, mut proposed, mut accepted, mut count) = (0, 0, 0, 0, 0);
        for (prompt, want) in prompts.iter().zip(&alone) {
            let mut s = target
                .speculate(prompt, &request, proposer())
                .map_err(error)?;
            for piece in s.by_ref() {
                piece.map_err(error)?;
            }
            same += usize::from(s.tokens() == &want[..]);
            let stats = s.stats();
            passes += stats.passes;
            proposed += stats.proposed;
            accepted += stats.accepted;
            count += s.tokens().len();
        }
        let spec = start.elapsed().as_secs_f64();
        println!(
            "{name:<22} {:>7.1} tokens/s, {:.2}× ; {same}/{} answers the same, {:.0} % \
             accepted, {:.2} tokens per pass",
            count as f64 / spec,
            time / spec * count as f64 / alone.iter().map(Vec::len).sum::<usize>() as f64,
            prompts.len(),
            100.0 * accepted as f64 / proposed.max(1) as f64,
            count as f64 / passes.max(1) as f64
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let o = parse()?;
    match o.device.as_str() {
        "cpu" => {
            let load = LoadOptions {
                dtype: Some(DType::F32),
                ..LoadOptions::default()
            };
            let cpu = kapel_cpu::Cpu::new;
            let target = Pipeline::load_with(&o.model, cpu(), &load).map_err(|e| e.to_string())?;
            let draft = o
                .draft
                .as_ref()
                .map(|d| Pipeline::load_with(d, cpu(), &load).map_err(|e| e.to_string()))
                .transpose()?;
            run(&target, draft.as_ref(), &o)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let cuda = kapel_cuda::Cuda::new(0).map_err(|e| e.to_string())?;
            let load = LoadOptions {
                dtype: Some(DType::BF16),
                fuse: true,
                ..LoadOptions::default()
            };
            let target =
                Pipeline::load_with(&o.model, cuda.clone(), &load).map_err(|e| e.to_string())?;
            let draft = o
                .draft
                .as_ref()
                .map(|d| Pipeline::load_with(d, cuda.clone(), &load).map_err(|e| e.to_string()))
                .transpose()?;
            run(&target, draft.as_ref(), &o)
        }
        device => Err(format!("unknown device {device}")),
    }
}
