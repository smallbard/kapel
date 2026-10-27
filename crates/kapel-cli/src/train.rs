//! `kapel train`: a training run described by a TOML file.

use kapel_train::config::TrainConfig;
use std::sync::Arc;

use kapel_train::dist::{Collective, Tcp};
use kapel_train::trainer::{RunOptions, Summary, run, run_threads, run_with};

pub(crate) const USAGE: &str = "\
usage: kapel train CONFIG.toml [options]

Trains a model as CONFIG.toml describes it (see The Kapel Book, chapters 15 to
17): the model (a directory, or a repository of the Hub), what it learns from,
the batch, the optimizer, the schedule of learning rates, the evaluations and the
checkpoints. A run learns from tokens ([data]: pre-training, and supervised
fine-tuning on conversations prepared with kapel prepare --chat), from pairs of
answers ([dpo]), or from answers it writes and their rewards ([grpo]); with
[lora], the model is frozen and adapters beside it train (QLoRA with
lora.quantize). Each step goes to log.jsonl in the run's directory, with its loss,
learning rate, gradient norm, speed and model FLOP utilization, and what the
method shows (rewards, accuracy of preferences...); every eval.every steps, the
validation and control losses and the accuracy of [[eval.tasks]]. Each checkpoint
is a model directory that transformers loads, or with adapters an adapter that
PEFT loads (kapel merge makes it a model).

options:
  --resume                 go on from the last checkpoint of the run's directory
                           (the run computes what it would have computed without
                           stopping)
  --steps N                stop after step N, as if killed there
  --dir DIR                the run's directory (its log, checkpoints and
                           summary) instead of checkpoint.dir
  --quiet                  print nothing but errors
  --profile                time each operation (the device waited for around
                           each, so slower) and print the times at the end
  --ranks N                N ranks in this process, one thread each, each its own
                           backend on the device named, the passes of a step
                           shared between them and their gradients summed

Several processes: as torchrun starts them, each with RANK, WORLD_SIZE and
MASTER_ADDR:MASTER_PORT (127.0.0.1:29500 by default) in its environment, joined
by TCP; with training.device = \"cuda\", LOCAL_RANK names the card. Rank 0 alone
writes the log, the checkpoints and the summary.
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let (mut path, mut dir) = (None, None);
    let mut options = RunOptions::default();
    let mut ranks = 1usize;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--resume" => options.resume = true,
            "--quiet" => options.quiet = true,
            "--profile" => options.profile = true,
            "--dir" => dir = Some(rest.next().ok_or("--dir needs a directory")?.clone()),
            "--ranks" => {
                let n = rest.next().ok_or("--ranks needs a number")?;
                ranks = n
                    .parse()
                    .ok()
                    .filter(|&r| r > 0)
                    .ok_or_else(|| format!("--ranks: not a positive number: {n}"))?;
            }
            "--steps" => {
                let n = rest.next().ok_or("--steps needs a number")?;
                options.stop_after = Some(
                    n.parse()
                        .map_err(|_| format!("--steps: not a number: {n}"))?,
                );
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other if path.is_none() => path = Some(other.to_string()),
            other => return Err(format!("one configuration, got {other} too")),
        }
    }
    let path = path.ok_or(USAGE)?;
    let mut config = TrainConfig::load(&path).map_err(|e| e.to_string())?;
    if let Some(dir) = dir {
        config.checkpoint.dir = dir.into();
    }
    let summary = match world()? {
        Some((rank, size, address)) => {
            if ranks > 1 {
                return Err("--ranks and WORLD_SIZE: one or the other".into());
            }
            if config.training.device == "cuda"
                && let Ok(local) = std::env::var("LOCAL_RANK")
            {
                config.training.device = format!("cuda:{local}");
            }
            let group: Arc<dyn Collective> =
                Arc::new(Tcp::connect(&address, rank, size).map_err(|e| e.to_string())?);
            let result = run_with(&config, &options, Some(Arc::clone(&group)));
            if result.is_err() {
                group.abort();
            }
            result.map_err(|e| e.to_string())?
        }
        None if ranks > 1 => first(run_threads(&config, &options, ranks))?,
        None => run(&config, &options).map_err(|e| e.to_string())?,
    };
    if options.quiet || rank_env() > 0 {
        return Ok(());
    }
    {
        eprintln!(
            "train: {} steps, loss {:.4}{}",
            summary.steps,
            summary.loss,
            summary
                .val_loss
                .map(|v| format!(", validation {v:.4}"))
                .unwrap_or_default()
        );
    }
    Ok(())
}

/// Rank 0's summary of a run of several ranks.
fn first(
    summaries: Result<Vec<Summary>, kapel_train::trainer::TrainError>,
) -> Result<Summary, String> {
    summaries
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "no rank".into())
}

/// This process's rank in torchrun's environment, 0 without it.
fn rank_env() -> usize {
    std::env::var("RANK")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(0)
}

/// `(rank, size, address)` from torchrun's environment, if `WORLD_SIZE` is
/// above 1.
fn world() -> Result<Option<(usize, usize, String)>, String> {
    let var = |name: &str| std::env::var(name).ok();
    let Some(size) = var("WORLD_SIZE") else {
        return Ok(None);
    };
    let size: usize = size
        .parse()
        .map_err(|_| format!("WORLD_SIZE: not a number: {size}"))?;
    if size <= 1 {
        return Ok(None);
    }
    let rank: usize = var("RANK")
        .ok_or("WORLD_SIZE without RANK")?
        .parse()
        .map_err(|_| "RANK: not a number".to_string())?;
    if rank >= size {
        return Err(format!("RANK {rank} of WORLD_SIZE {size}"));
    }
    let address = format!(
        "{}:{}",
        var("MASTER_ADDR").unwrap_or_else(|| "127.0.0.1".into()),
        var("MASTER_PORT").unwrap_or_else(|| "29500".into())
    );
    Ok(Some((rank, size, address)))
}
