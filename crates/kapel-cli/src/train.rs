//! `kapel train`: a training run described by a TOML file.

use kapel_train::config::TrainConfig;
use kapel_train::trainer::{RunOptions, run};

pub(crate) const USAGE: &str = "\
usage: kapel train CONFIG.toml [options]

Trains a model as CONFIG.toml describes it (see The Kapel Guide, Training): the
model, the token files, the batch, the optimizer, the schedule of learning
rates, the evaluations and the checkpoints. Each step goes to log.jsonl in the
run's directory, with its loss, learning rate, gradient norm, speed and model
FLOP utilization; each checkpoint is a model directory that transformers loads.

options:
  --resume                 go on from the last checkpoint of the run's directory
                           (the run computes what it would have computed without
                           stopping)
  --steps N                stop after step N, as if killed there
  --quiet                  print nothing but errors
  --profile                time each operation (the device waited for around
                           each, so slower) and print the times at the end
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let mut path = None;
    let mut options = RunOptions::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--resume" => options.resume = true,
            "--quiet" => options.quiet = true,
            "--profile" => options.profile = true,
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
    let config = TrainConfig::load(&path).map_err(|e| e.to_string())?;
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
