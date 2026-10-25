//! The `kapel` command line.

mod args;
mod eval;
mod merge;
mod prepare;
mod quantize;
mod train;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use kapel_core::DType;
use kapel_infer::{Draft, EngineOptions, Finish, LoadOptions, Piece, Pipeline, Request};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::Message;

use crate::args::{Device, Load, Run, Serve};

const USAGE: &str = "\
usage: kapel run MODEL [options]
       kapel serve MODEL [options]
       kapel quantize MODEL --output PATH [options]   (kapel quantize --help)
       kapel eval MODEL --tasks FILE [options]        (kapel eval --help)
       kapel prepare INPUT... --tokenizer PATH --output DIR   (kapel prepare --help)
       kapel train CONFIG.toml [options]              (kapel train --help)
       kapel merge MODEL ADAPTER --output DIR         (kapel merge --help)

Generates text with MODEL: a model directory, a GGUF file, or a repository of the
Hugging Face Hub (owner/name, downloaded once into ~/.cache/kapel/hub, or
$KAPEL_CACHE; a gated model needs HF_TOKEN). Without --prompt, holds a conversation: one message per line
on the standard input, an empty line to end.

options:
  --prompt TEXT            answer TEXT, then exit
  --raw                    continue the prompt as is, without the chat template
  --system TEXT            a system message first
  --device cpu|cuda[:N]    where to compute (default cpu; cuda needs the feature)
  --dtype f32|bf16|f16     the type of the float weights (default f32 on cpu,
                           bf16 on cuda); quantized weights keep their format
  --max-tokens N           the most tokens per answer (default 512)
  --greedy                 always the most likely token
  --temperature T          sampling (defaults: the model's generation_config.json)
  --top-k K, --top-p P, --min-p P, --repetition-penalty R
  --seed S                 the seed of the draws (default 0)
  --stop TEXT              end the answer at TEXT (repeatable)
  --revision REV           the revision of a Hub repository (default main)
  --stats                  speeds on the standard error
  --quantize FORMAT        quantize the linear projections: int8, int4 (groups of
                           128 and 64, or int8/G, int4/G), q8_0, q4_0
  --quantize-output        quantize the output table too
  --kv-cache TYPE          the type of the cache of keys and values: f32, bf16,
                           f16, f8e4m3 or f8e5m2 (default the type of the keys;
                           f8 halves its memory)
  --byte-activations       on cpu, quantize the activations of the products by
                           quantized weights to bytes, as llama.cpp does: faster,
                           and the results change a little
  --draft MODEL            speculative decoding: MODEL, smaller, of the same
                           tokenizer, proposes tokens that MODEL checks several at
                           a time; the answers do not change
  --prompt-lookup          the same, the tokens of the conversation proposing
  --multi-token            the same, the model's multi-token layer proposing (MiMo)
  --speculate K            the proposals per pass (default 4)

kapel serve serves MODEL behind an HTTP API compatible with OpenAI's
(/v1/chat/completions, /v1/completions, /v1/models, /health, /metrics), the
requests computed together. It takes --device, --dtype, --revision, --quantize,
--quantize-output, --kv-cache and --byte-activations as kapel run does, and:
  --host HOST              the address to listen on (default 127.0.0.1)
  --port PORT              (default 8000)
  --served-model-name NAME the model's name in the API (default from MODEL)
  --max-model-len N        the most positions of a sequence, prompt and answer
                           (default the model's, at most what the cache holds)
  --gpu-memory-utilization F
                           on cuda, the share of the card's memory to use, the
                           cache taking what the weights leave (default 0.85)
  --cache-memory GIB       the memory of the cache instead (default 4 on cpu)
  --max-sequences N        the most answers computed at once (default 256)
  --max-pass-tokens N      the most tokens of one pass (default 2048)
  --max-requests N         the most answers in flight; beyond, 429 (default 1024)
  --no-prefix-sharing      do not share the cache of common prompt beginnings
  --timeout SECONDS        the longest a request may take (default 600)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => exit(args::parse(&args[1..]).and_then(|run| execute(&run))),
        Some("serve") => {
            exit(args::parse_serve(&args[1..]).and_then(|serve| execute_serve(&serve)))
        }
        Some("quantize") if args.get(1).is_some_and(|a| a == "--help" || a == "-h") => {
            print!("{}", quantize::USAGE);
            ExitCode::SUCCESS
        }
        Some("quantize") => exit(quantize::execute(&args[1..])),
        Some("eval") if args.get(1).is_some_and(|a| a == "--help" || a == "-h") => {
            print!("{}", eval::USAGE);
            ExitCode::SUCCESS
        }
        Some("eval") => exit(eval::execute(&args[1..])),
        Some("prepare") if args.get(1).is_some_and(|a| a == "--help" || a == "-h") => {
            print!("{}", prepare::USAGE);
            ExitCode::SUCCESS
        }
        Some("prepare") => exit(prepare::execute(&args[1..])),
        Some("train") if args.get(1).is_some_and(|a| a == "--help" || a == "-h") => {
            print!("{}", train::USAGE);
            ExitCode::SUCCESS
        }
        Some("train") => exit(train::execute(&args[1..])),
        Some("merge") if args.get(1).is_some_and(|a| a == "--help" || a == "-h") => {
            print!("{}", merge::USAGE);
            ExitCode::SUCCESS
        }
        Some("merge") => exit(merge::execute(&args[1..])),
        Some("--version" | "-V") => {
            println!("kapel {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("--help" | "-h" | "help") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => {
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn exit(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("kapel: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The model directory: `model` itself, or its snapshot from the Hub.
fn directory(model: &str, revision: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(model);
    // A model directory, or a GGUF file.
    if path.is_dir() || path.is_file() {
        return Ok(path);
    }
    eprintln!("kapel: fetching {model} from the Hub…");
    kapel_hub::download::snapshot(model, revision, &kapel_hub::download::default_cache())
        .map_err(|e| e.to_string())
}

/// How `load` prepares the weights on its device.
fn load_options(load: &Load) -> LoadOptions {
    let cuda = matches!(load.device, Device::Cuda(_));
    LoadOptions {
        dtype: load
            .dtype
            .or(Some(if cuda { DType::BF16 } else { DType::F32 })),
        quantize: load.quantize,
        quantize_output: load.quantize_output,
        fuse: cuda,
        cache_dtype: load.kv_cache,
    }
}

fn execute(run: &Run) -> Result<(), String> {
    let dir = directory(&run.load.model, &run.load.revision)?;
    let draft = run
        .draft
        .as_ref()
        .map(|d| directory(d, &run.load.revision))
        .transpose()?;
    let options = load_options(&run.load);
    match run.load.device {
        Device::Cpu => {
            let cpu = || kapel_cpu::Cpu::new().with_byte_activations(run.load.byte_activations);
            let load = |dir: &PathBuf| {
                Pipeline::load_with(dir, cpu(), &options).map_err(|e| e.to_string())
            };
            let draft = draft.as_ref().map(load).transpose()?;
            converse(run, load(&dir)?, draft.as_ref())
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
            let load = |dir: &PathBuf| {
                Pipeline::load_with(dir, cuda.clone(), &options).map_err(|e| e.to_string())
            };
            let draft = draft.as_ref().map(load).transpose()?;
            converse(run, load(&dir)?, draft.as_ref())
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err("this kapel was built without the feature `cuda`".into()),
    }
}

/// The name of a model in the API: its Hub repository, from the name given
/// or from the path of a snapshot of the Hub's cache, else the last part of
/// its path.
fn model_name(model: &str) -> String {
    let path = std::path::Path::new(model);
    for part in path.components() {
        if let Some(repo) = part
            .as_os_str()
            .to_str()
            .and_then(|p| p.strip_prefix("models--"))
        {
            return repo.replacen("--", "/", 1);
        }
    }
    if !path.exists() {
        return model.to_string();
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(model)
        .to_string()
}

/// The bytes of one page of the cache of `pipeline`'s model.
fn page_bytes<B: Backend + Transfer>(pipeline: &Pipeline<B>, load: &Load) -> usize {
    let config = pipeline.model().config();
    let dtype = pipeline
        .cache_dtype()
        .or(load_options(load).dtype)
        .unwrap_or(DType::F32);
    let element = dtype.element_size().unwrap_or(4);
    config.layers * 2 * config.kv_heads * config.head_dim * element * kapel_models::PAGE
}

fn execute_serve(serve: &Serve) -> Result<(), String> {
    let dir = directory(&serve.load.model, &serve.load.revision)?;
    let options = load_options(&serve.load);
    match serve.load.device {
        Device::Cpu => {
            let cpu = kapel_cpu::Cpu::new().with_byte_activations(serve.load.byte_activations);
            let pipeline = Pipeline::load_with(&dir, cpu, &options).map_err(|e| e.to_string())?;
            let memory = serve.cache_memory.unwrap_or(4.0);
            let pages =
                (memory * f64::from(1u32 << 30)) as usize / page_bytes(&pipeline, &serve.load);
            start_server(serve, pipeline, pages)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
            let pipeline =
                Pipeline::load_with(&dir, cuda.clone(), &options).map_err(|e| e.to_string())?;
            let page = page_bytes(&pipeline, &serve.load);
            let pages = match serve.cache_memory {
                Some(memory) => (memory * f64::from(1u32 << 30)) as usize / page,
                None => {
                    cuda.clear_pool();
                    let (free, total) = cuda.memory().map_err(|e| e.to_string())?;
                    let config = pipeline.model().config();
                    // What the passes need besides the cache: activations of
                    // the widest pass, logits, and the recorded passes, which
                    // keep their own buffers (one per size of batch).
                    let activations =
                        serve.max_pass_tokens * (config.hidden * 8 + config.intermediate * 4) * 4
                            + serve.max_sequences * config.vocab * 8
                            + (5usize << 29);
                    let budget = (total as f64 * serve.gpu_memory_utilization) as usize;
                    let used = total - free;
                    let left = budget.saturating_sub(used + activations);
                    if left < 64 * page {
                        return Err(format!(
                            "the card has room for {} pages of cache; raise \
                             --gpu-memory-utilization or use --cache-memory",
                            left / page
                        ));
                    }
                    left / page
                }
            };
            start_server(serve, pipeline, pages)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err("this kapel was built without the feature `cuda`".into()),
    }
}

fn start_server<B>(serve: &Serve, pipeline: Pipeline<B>, pages: usize) -> Result<(), String>
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let held = pages * kapel_models::PAGE;
    let max_model_len = serve.max_model_len.unwrap_or_else(|| {
        pipeline
            .model()
            .config()
            .max_positions
            .unwrap_or(held)
            .min(held)
    });
    let page = page_bytes(&pipeline, &serve.load);
    eprintln!(
        "kapel: a cache of {pages} pages ({held} positions, {:.1} GiB); sequences of up to \
         {max_model_len} positions",
        (pages * page) as f64 / f64::from(1u32 << 30)
    );
    let address = format!("{}:{}", serve.host, serve.port)
        .parse()
        .map_err(|e| format!("--host, --port: {e}"))?;
    let mut options = kapel_serve::ServeOptions::new(
        serve
            .name
            .clone()
            .unwrap_or_else(|| model_name(&serve.load.model)),
        max_model_len,
    );
    options.address = address;
    options.max_requests = serve.max_requests;
    options.request_timeout = std::time::Duration::from_secs(serve.timeout);
    let engine = EngineOptions {
        pages: Some(pages),
        prefix_sharing: serve.prefix_sharing,
        max_pass_tokens: serve.max_pass_tokens,
        max_sequences: serve.max_sequences,
        reserve: 0.0,
    };
    kapel_serve::serve(pipeline, engine, options).map_err(|e| e.to_string())
}

/// Answers the prompt, or holds the conversation.
fn converse<B: Backend + Transfer>(
    run: &Run,
    pipeline: Pipeline<B>,
    draft: Option<&Pipeline<B>>,
) -> Result<(), String> {
    let mut request = Request {
        sampling: pipeline.default_sampling().clone(),
        max_tokens: run.max_tokens,
        stop: run.stop.clone(),
        ..Request::default()
    };
    run.sampling.apply(&mut request.sampling);
    let chat = !run.raw && pipeline.has_chat_template();
    let mut messages = Vec::new();
    if let Some(system) = &run.system {
        messages.push(Message::new("system", system.clone()));
    }
    if let Some(prompt) = &run.prompt {
        answer(run, &pipeline, draft, &request, chat, &mut messages, prompt)?;
        return Ok(());
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("> ");
        let Some(line) = lines.next() else { break };
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            break;
        }
        answer(run, &pipeline, draft, &request, chat, &mut messages, &line)?;
    }
    Ok(())
}

/// Prints the pieces as they come; the answer, and when its first token came.
fn print_pieces(
    pieces: &mut dyn Iterator<Item = kapel_infer::Result<Piece>>,
) -> Result<(String, Option<Instant>), String> {
    let mut first = None;
    let mut reply = String::new();
    let mut out = std::io::stdout().lock();
    for piece in pieces {
        let piece = piece.map_err(|e| e.to_string())?;
        first.get_or_insert_with(Instant::now);
        reply.push_str(&piece.text);
        out.write_all(piece.text.as_bytes())
            .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;
    Ok((reply, first))
}

/// Generates the answer to `text`, printing it as it comes.
#[allow(clippy::too_many_arguments)]
fn answer<B: Backend + Transfer>(
    run: &Run,
    pipeline: &Pipeline<B>,
    draft: Option<&Pipeline<B>>,
    request: &Request,
    chat: bool,
    messages: &mut Vec<Message>,
    text: &str,
) -> Result<(), String> {
    let error = |e: kapel_infer::InferError| e.to_string();
    let prompt = if chat {
        messages.push(Message::new("user", text));
        pipeline.chat_prompt(messages).map_err(error)?
    } else {
        pipeline.prompt(text).map_err(error)?
    };
    let start = Instant::now();
    let proposer = match draft {
        Some(pipeline) => Some(Draft::Model {
            pipeline,
            tokens: run.speculate,
        }),
        None if run.prompt_lookup => Some(Draft::Ngram {
            tokens: run.speculate,
            longest: 4,
            shortest: 2,
        }),
        None if run.multi_token => Some(Draft::MultiToken),
        None => None,
    };
    let (reply, first, n, finish, speculation) = match proposer {
        Some(proposer) => {
            let mut s = pipeline
                .speculate(&prompt, request, proposer)
                .map_err(error)?;
            let (reply, first) = print_pieces(&mut s)?;
            (
                reply,
                first,
                s.tokens().len(),
                s.finish().cloned(),
                Some(s.stats()),
            )
        }
        None => {
            let mut g = pipeline.generate(&prompt, request).map_err(error)?;
            let (reply, first) = print_pieces(&mut g)?;
            (reply, first, g.tokens().len(), g.finish().cloned(), None)
        }
    };
    if chat {
        messages.push(Message::new("assistant", reply));
    }
    if run.stats {
        let prefill = first.map_or(0.0, |f| (f - start).as_secs_f64());
        let decode = first.map_or(0.0, |f| f.elapsed().as_secs_f64());
        let finish = match finish {
            Some(Finish::Eos) => "end of sequence".to_string(),
            Some(Finish::Stop(s)) => format!("stop {s:?}"),
            Some(Finish::Length) | None => "length".to_string(),
            Some(Finish::Error(e)) => format!("error: {e}"),
        };
        let speculation = speculation.map_or(String::new(), |s| {
            format!(
                "; {:.0} % of {} proposals accepted, {:.2} tokens per pass",
                100.0 * s.acceptance(),
                s.proposed,
                n as f64 / s.passes.max(1) as f64
            )
        });
        eprintln!(
            "[{} prompt tokens in {:.2} s ({:.0}/s); {n} tokens in {:.2} s ({:.1}/s); \
             {finish}{speculation}]",
            prompt.len(),
            prefill,
            prompt.len() as f64 / prefill.max(1e-9),
            decode,
            n.saturating_sub(1) as f64 / decode.max(1e-9),
        );
    }
    Ok(())
}
