//! The `kapel` command line.

mod args;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use kapel_infer::{Finish, Pipeline, Request};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::Message;

use crate::args::{Device, Run};

const USAGE: &str = "\
usage: kapel run MODEL [options]

Generates text with MODEL, a model directory or a repository of the Hugging Face
Hub (owner/name, downloaded once into ~/.cache/kapel/hub, or $KAPEL_CACHE; a gated
model needs HF_TOKEN). Without --prompt, holds a conversation: one message per line
on the standard input, an empty line to end.

options:
  --prompt TEXT            answer TEXT, then exit
  --raw                    continue the prompt as is, without the chat template
  --system TEXT            a system message first
  --device cpu|cuda[:N]    where to compute (default cpu; cuda needs the feature)
  --dtype f32|bf16|f16     the type of the weights (default f32 on cpu, bf16 on
                           cuda)
  --max-tokens N           the most tokens per answer (default 512)
  --greedy                 always the most likely token
  --temperature T          sampling (defaults: the model's generation_config.json)
  --top-k K, --top-p P, --min-p P, --repetition-penalty R
  --seed S                 the seed of the draws (default 0)
  --stop TEXT              end the answer at TEXT (repeatable)
  --revision REV           the revision of a Hub repository (default main)
  --stats                  speeds on the standard error
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => match args::parse(&args[1..]).and_then(|run| execute(&run)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("kapel: {message}");
                ExitCode::FAILURE
            }
        },
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

/// The model directory: `model` itself, or its snapshot from the Hub.
fn directory(model: &str, revision: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }
    eprintln!("kapel: fetching {model} from the Hub…");
    kapel_hub::download::snapshot(model, revision, &kapel_hub::download::default_cache())
        .map_err(|e| e.to_string())
}

fn execute(run: &Run) -> Result<(), String> {
    let dir = directory(&run.model, &run.revision)?;
    match run.device {
        Device::Cpu => serve(
            run,
            Pipeline::load(&dir, kapel_cpu::Cpu::new(), run.dtype).map_err(|e| e.to_string())?,
        ),
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
            let dtype = run.dtype.or(Some(kapel_core::DType::BF16));
            serve(
                run,
                Pipeline::load(&dir, cuda, dtype).map_err(|e| e.to_string())?,
            )
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err("this kapel was built without the feature `cuda`".into()),
    }
}

/// Answers the prompt, or holds the conversation.
fn serve<B: Backend + Transfer>(run: &Run, pipeline: Pipeline<B>) -> Result<(), String> {
    let mut request = Request {
        sampling: pipeline.default_sampling().clone(),
        max_tokens: run.max_tokens,
        stop: run.stop.clone(),
    };
    run.sampling.apply(&mut request.sampling);
    let chat = !run.raw && pipeline.has_chat_template();
    let mut messages = Vec::new();
    if let Some(system) = &run.system {
        messages.push(Message::new("system", system.clone()));
    }
    if let Some(prompt) = &run.prompt {
        answer(run, &pipeline, &request, chat, &mut messages, prompt)?;
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
        answer(run, &pipeline, &request, chat, &mut messages, &line)?;
    }
    Ok(())
}

/// Generates the answer to `text`, printing it as it comes.
fn answer<B: Backend + Transfer>(
    run: &Run,
    pipeline: &Pipeline<B>,
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
    let mut first = None;
    let mut generation = pipeline.generate(&prompt, request).map_err(error)?;
    let mut reply = String::new();
    let mut out = std::io::stdout().lock();
    for piece in generation.by_ref() {
        let piece = piece.map_err(error)?;
        first.get_or_insert_with(Instant::now);
        reply.push_str(&piece.text);
        out.write_all(piece.text.as_bytes())
            .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;
    if chat {
        messages.push(Message::new("assistant", reply));
    }
    if run.stats {
        let n = generation.tokens().len();
        let prefill = first.map_or(0.0, |f| (f - start).as_secs_f64());
        let decode = first.map_or(0.0, |f| f.elapsed().as_secs_f64());
        let finish = match generation.finish() {
            Some(Finish::Eos) => "end of sequence".to_string(),
            Some(Finish::Stop(s)) => format!("stop {s:?}"),
            Some(Finish::Length) | None => "length".to_string(),
        };
        eprintln!(
            "[{} prompt tokens in {:.2} s ({:.0}/s); {n} tokens in {:.2} s ({:.1}/s); {finish}]",
            prompt.len(),
            prefill,
            prompt.len() as f64 / prefill.max(1e-9),
            decode,
            n.saturating_sub(1) as f64 / decode.max(1e-9),
        );
    }
    Ok(())
}
