//! `kapel train`: a training run described by a TOML file.

use kapel_train::config::TrainConfig;
use kapel_train::trainer::{RunOptions, run};

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
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let (mut path, mut dir) = (None, None);
    let mut options = RunOptions::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--resume" => options.resume = true,
            "--quiet" => options.quiet = true,
            "--profile" => options.profile = true,
            "--dir" => dir = Some(rest.next().ok_or("--dir needs a directory")?.clone()),
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
    let summary = run(&config, &options).map_err(|e| e.to_string())?;
    if !options.quiet {
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
