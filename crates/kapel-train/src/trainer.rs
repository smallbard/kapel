//! The training loop: a configuration in, a trained model out.
//!
//! Each step reads `accumulate` batches in order, computes their loss and
//! gradients, adds the gradients up, clips their global norm, and moves the
//! parameters with the optimizer at the learning rate of the schedule. Every
//! step is written to `log.jsonl` in the run's directory; the validation loss
//! is computed every `eval.every` steps, and a checkpoint taken every
//! `checkpoint.every`.
//!
//! The batches are read ahead by a thread ([`crate::data::Prefetched`]);
//! `data_ms` in the log is the time a step waited for them.
//!
//! A run resumed from a checkpoint goes on with the parameters, the
//! optimizer's moments and steps, and the position in the data it had: on
//! the same machine, it computes the bits an uninterrupted run computes.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kapel_core::{DType, Tensor};
use kapel_models::Config;
use kapel_ops::{Binary, BinaryKind, Concat};
use serde_json::{Value, json};

use crate::checkpoint::{self, Checkpoint, State, Writer};
use crate::config::{TrainConfig, matches};
use crate::data::{self, Loader, Prefetched, Source};
use crate::flops;
use crate::model::{Batch, Decoder, Options, Parameter, Parameters, init};
use crate::optim::{Optimizer, OptimizerState, global_norm};
use crate::schedule::Schedule;
use crate::tape::{Compute, Grad, Tape};

/// What can stop a run.
#[derive(Debug)]
pub struct TrainError(pub String);

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrainError {}

macro_rules! from {
    ($($t:ty),*) => {$(
        impl From<$t> for TrainError {
            fn from(e: $t) -> Self {
                TrainError(e.to_string())
            }
        }
    )*};
}

from!(
    kapel_core::Error,
    crate::config::ConfigError,
    crate::data::DataError,
    kapel_models::ModelError,
    kapel_hub::HubError,
    std::io::Error,
    String
);

type Result<T> = std::result::Result<T, TrainError>;

/// How [`run`] is driven, beyond the configuration.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Go on from the last checkpoint of the run's directory, if any.
    pub resume: bool,
    /// Stop after this step, as if the run were killed there: no final
    /// evaluation, no final checkpoint but the regular ones.
    pub stop_after: Option<usize>,
    /// Print nothing on the terminal.
    pub quiet: bool,
    /// Time each operation, the device waited for around each, and print
    /// the times of the steps after the first, which compiles the kernels,
    /// at the end ([`crate::profile::Profiled`]): slower.
    pub profile: bool,
}

/// What a run gives back.
#[derive(Debug)]
pub struct Summary {
    /// The steps done.
    pub steps: usize,
    /// The last training loss.
    pub loss: f64,
    /// The last validation loss, if any was computed.
    pub val_loss: Option<f64>,
    /// The parameters at the end, in host memory, `f32`.
    pub params: BTreeMap<String, Tensor>,
}

/// The `config.json` of the model: the file named, or the one of the
/// directory named.
fn model_json(path: &Path) -> Result<Value> {
    let file = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read(&file).map_err(|e| TrainError(format!("{}: {e}", file.display())))?;
    serde_json::from_slice(&text).map_err(|e| TrainError(format!("{}: {e}", file.display())))
}

/// The parameters the configuration starts from, with its groups applied.
fn start(config: &TrainConfig, model_config: &Config) -> Result<Parameters> {
    let mut params = if config.model.init == "weights" {
        let checkpoint = kapel_hub::Checkpoint::open(&config.model.config)?;
        Parameters::from_model(&kapel_models::Model::load(&checkpoint)?)
    } else {
        init(model_config, config.model.std, config.model.seed)?
    };
    for (name, p) in params.iter_mut() {
        if let Some(g) = config
            .optimizer
            .groups
            .iter()
            .find(|g| matches(&g.matches, name))
        {
            let Parameter {
                lr_scale,
                decay,
                trainable,
                sparse,
                ..
            } = p;
            *lr_scale = g.lr_scale.unwrap_or(*lr_scale);
            *decay = g.decay.unwrap_or(*decay);
            *trainable = g.trainable.unwrap_or(*trainable);
            *sparse = g.sparse.unwrap_or(*sparse);
        }
    }
    Ok(params)
}

/// Runs the configuration's training on the device it names.
///
/// # Errors
///
/// For a configuration, a file or a device that fails.
#[allow(clippy::needless_return)]
pub fn run(config: &TrainConfig, options: &RunOptions) -> Result<Summary> {
    let device = config.training.device.as_str();
    if device == "cpu" {
        if options.profile {
            let cpu = crate::profile::Profiled::new(kapel_cpu::Cpu::new(), || {});
            let mut run = Run::new(config, options, cpu.clone(), Box::new(|| {}), None)?;
            let reset = cpu.clone();
            run.after_first = Some(Box::new(move || reset.reset()));
            let summary = run.go();
            eprint!("{}", cpu.report());
            return summary;
        }
        return Run::new(
            config,
            options,
            kapel_cpu::Cpu::new(),
            Box::new(|| {}),
            None,
        )?
        .go();
    }
    #[cfg(feature = "cuda")]
    {
        let ordinal = device
            .strip_prefix("cuda")
            .map(|n| n.trim_start_matches(':'))
            .filter(|n| !n.is_empty())
            .map_or(Ok(0), str::parse)
            .map_err(|_| TrainError(format!("device {device:?}")))?;
        let cuda = kapel_cuda::Cuda::new(ordinal)?;
        // Every attention's backward pass comes: its forward pass's output
        // and statistics are kept rather than computed again.
        cuda.keep_attention_for_backward(true);
        cuda.reset_peak_memory();
        let sync = cuda.clone();
        let name = cuda.name().ok();
        let counter = cuda.clone();
        let peak: Box<dyn Fn() -> usize> = Box::new(move || counter.peak_memory());
        if options.profile {
            let s = cuda.clone();
            let profiled = crate::profile::Profiled::new(cuda, move || {
                s.synchronize().expect("the card");
            });
            let mut run = Run::new(config, options, profiled.clone(), Box::new(|| {}), name)?;
            let reset = profiled.clone();
            run.after_first = Some(Box::new(move || reset.reset()));
            run.peak_memory = Some(peak);
            let summary = run.go();
            eprint!("{}", profiled.report());
            return summary;
        }
        let mut run = Run::new(
            config,
            options,
            cuda,
            Box::new(move || sync.synchronize().expect("the card")),
            name,
        )?;
        run.peak_memory = Some(peak);
        return run.go();
    }
    #[cfg(not(feature = "cuda"))]
    Err(TrainError(format!(
        "device {device}: Kapel was built without the feature `cuda`"
    )))
}

/// A run in progress, on one backend.
struct Run<'a, B: Compute> {
    config: &'a TrainConfig,
    options: &'a RunOptions,
    backend: B,
    synchronize: Box<dyn Fn()>,
    model_json: Value,
    decoder: Decoder,
    params: Parameters,
    optimizer: Optimizer,
    schedule: Schedule,
    train: Prefetched,
    val: Option<Vec<data::Tokens>>,
    step: usize,
    tokens: u64,
    dir: PathBuf,
    log: std::fs::File,
    writer: Writer,
    peak: Option<f64>,
    step_flops: u64,
    /// Called once the first step is done: a profile forgets the
    /// compilation of the kernels.
    after_first: Option<Box<dyn Fn()>>,
    /// The peak memory of the card since the run started, if it counts it.
    peak_memory: Option<Box<dyn Fn() -> usize>>,
}

impl<'a, B: Compute> Run<'a, B> {
    fn new(
        config: &'a TrainConfig,
        options: &'a RunOptions,
        backend: B,
        synchronize: Box<dyn Fn()>,
        device_name: Option<String>,
    ) -> Result<Self> {
        let model_json = model_json(&config.model.config)?;
        let model_config = Config::from_json(&model_json)?;
        let decoder = Decoder::new(model_config.clone())?;
        let mut params = start(config, &model_config)?;
        let mut optimizer = Optimizer::new(&config.optimizer);
        let b = &config.batch;
        let reading = config.data.reading(b.rows, b.length);
        let open = |path: &str, split: &str| -> Result<Source> {
            let source = Source::open(path, split, &reading.cache)?;
            if let Some(v) = source.vocab()
                && v > model_config.vocab
            {
                return Err(TrainError(format!(
                    "{path}: tokens of a vocabulary of {v}, the model has {}",
                    model_config.vocab
                )));
            }
            Ok(source)
        };
        let sources = config
            .data
            .training_sources()
            .iter()
            .map(|(path, weight)| Ok((open(path, "train")?, *weight)))
            .collect::<Result<Vec<_>>>()?;
        let mut train = Loader::new(sources, reading.clone())?;
        let val = match &config.data.val {
            Some(path) if config.eval.tokens > 0 => {
                // In order, whatever the training's order: the same rows at
                // every evaluation.
                let mut r = reading.clone();
                r.shuffle = false;
                let mut loader = Loader::new(vec![(open(path, "val")?, 1.0)], r)?;
                let passes = config.eval.tokens / b.pass();
                Some(
                    (0..passes)
                        .map(|_| loader.next_batch())
                        .collect::<std::result::Result<_, _>>()?,
                )
            }
            _ => None,
        };
        let dir = config.checkpoint.dir.clone();
        std::fs::create_dir_all(&dir)?;
        let (mut step, mut tokens) = (0, 0);
        if options.resume
            && let Some((_, path)) = checkpoint::list(&dir).pop()
        {
            let saved = Checkpoint::read(&path)?;
            if saved.state.optimizer != config.optimizer.kind {
                return Err(TrainError(format!(
                    "{}: a checkpoint of {}, the configuration says {}",
                    path.display(),
                    saved.state.optimizer,
                    config.optimizer.kind
                )));
            }
            for (name, value) in saved.params {
                match params.get_mut(&name) {
                    Some(p) => p.value = value,
                    None => {
                        return Err(TrainError(format!(
                            "{}: an unknown parameter {name}",
                            path.display()
                        )));
                    }
                }
            }
            let tensors = saved
                .optimizer
                .iter()
                .map(|(n, t)| Ok((n.clone(), backend.upload(t)?)))
                .collect::<Result<_>>()?;
            optimizer.restore(OptimizerState {
                steps: saved.state.optimizer_steps,
                tensors,
            })?;
            train.seek(&saved.state.position)?;
            step = saved.state.step;
            tokens = saved.state.tokens;
            if !options.quiet {
                eprintln!("train: resumed from {} at step {step}", path.display());
            }
        }
        let params = params.to(&backend, DType::F32)?;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("log.jsonl"))?;
        let dtype = config.dtype();
        let peak = config.log.peak_tflops.or_else(|| {
            device_name
                .as_deref()
                .and_then(|n| flops::peak_tflops(n, dtype))
        });
        let step_flops = flops::step_flops(&model_config, b.pass() * b.accumulate(), b.length);
        Ok(Self {
            config,
            options,
            backend,
            synchronize,
            model_json,
            decoder,
            params,
            optimizer,
            schedule: Schedule::new(&config.schedule),
            train: train.prefetch(config.data.prefetch),
            val,
            step,
            tokens,
            dir,
            log,
            writer: Writer::new(config.checkpoint.keep),
            peak,
            step_flops,
            after_first: None,
            peak_memory: None,
        })
    }

    fn options(&self) -> Options {
        Options {
            dtype: self.config.dtype(),
            checkpoint: self.config.training.checkpoint_activations,
            chunk: self.config.training.chunk,
        }
    }

    /// One pass: its loss, left on the device, and its gradients.
    fn pass(&self, tokens: &data::Tokens) -> Result<(Tensor, Vec<(String, Grad)>)> {
        let tape = Tape::new(self.backend.clone());
        let leaves = self.params.leaves(&tape);
        let batch = Batch {
            ids: &tokens.ids,
            targets: &tokens.targets,
            rows: self.config.batch.rows,
            segments: tokens.segments.as_deref(),
        };
        let loss = self.decoder.loss(&leaves, &batch, &self.options())?;
        let value = loss.value();
        let mut grads = tape.backward(&loss)?;
        drop(loss);
        Ok((value, leaves.gradients(&mut grads)))
    }

    /// The gradients of two passes added.
    fn add(&self, a: Grad, b: Grad, shape: &[usize]) -> Result<Grad> {
        Ok(match (a, b) {
            (
                Grad::Rows {
                    indices: ia,
                    values: va,
                },
                Grad::Rows {
                    indices: ib,
                    values: vb,
                },
            ) => {
                let flat = |i: &Tensor, v: &Tensor| -> Result<(Tensor, Tensor)> {
                    let n = i.numel();
                    let mut s = vec![n];
                    s.extend_from_slice(&shape[1..]);
                    let row_major = |t: &Tensor| -> Result<Tensor> {
                        Ok(if t.is_contiguous() {
                            t.clone()
                        } else {
                            self.backend.forward(&kapel_ops::Contiguous, &[t])?
                        })
                    };
                    Ok((row_major(i)?.reshape(&[n])?, row_major(v)?.reshape(&s)?))
                };
                let ((ia, va), (ib, vb)) = (flat(&ia, &va)?, flat(&ib, &vb)?);
                Grad::Rows {
                    indices: self.backend.forward(&Concat { dim: 0 }, &[&ia, &ib])?,
                    values: self.backend.forward(&Concat { dim: 0 }, &[&va, &vb])?,
                }
            }
            (a, b) => {
                let (a, b) = (
                    a.dense(&self.backend, shape)?,
                    b.dense(&self.backend, shape)?,
                );
                Grad::Dense(self.backend.forward(
                    &Binary {
                        kind: BinaryKind::Add,
                    },
                    &[&a, &b],
                )?)
            }
        })
    }

    /// The mean loss of the validation batches, without gradients.
    fn evaluate(&self) -> Result<Option<f64>> {
        let Some(val) = &self.val else {
            return Ok(None);
        };
        let mut total = 0.0;
        for tokens in val {
            let tape = Tape::new(self.backend.clone());
            let mut frozen = self.params.clone();
            for (_, p) in frozen.iter_mut() {
                p.trainable = false;
            }
            let leaves = frozen.leaves(&tape);
            let batch = Batch {
                ids: &tokens.ids,
                targets: &tokens.targets,
                rows: self.config.batch.rows,
                segments: tokens.segments.as_deref(),
            };
            let loss = self.decoder.loss(&leaves, &batch, &self.options())?;
            total += self.backend.download(&loss.value())?.to_f64_vec()?[0];
        }
        Ok(Some(total / val.len().max(1) as f64))
    }

    /// A copy of `t` in host memory, of its own: the checkpoint is written
    /// while the next steps update the tensors in place, and a host
    /// backend's download is the tensor itself.
    fn host_copy(&self, t: &Tensor) -> Result<Tensor> {
        let host = self.backend.download(t)?;
        Ok(if std::sync::Arc::ptr_eq(host.storage(), t.storage()) {
            crate::model::owned(&host)?
        } else {
            host
        })
    }

    /// The checkpoint of the current step, in host memory.
    fn checkpoint(&self) -> Result<Checkpoint> {
        let params = self
            .params
            .iter()
            .map(|(n, p)| Ok((n.to_string(), self.host_copy(&p.value)?)))
            .collect::<Result<_>>()?;
        let state = self.optimizer.state();
        let optimizer = state
            .tensors
            .iter()
            .map(|(n, t)| Ok((n.clone(), self.host_copy(t)?)))
            .collect::<Result<_>>()?;
        // The optimizer's own handles only, during the next steps.
        let steps = state.steps;
        drop(state);
        Ok(Checkpoint {
            state: State {
                step: self.step,
                optimizer_steps: steps,
                optimizer: self.config.optimizer.kind.clone(),
                position: self.train.position().clone(),
                tokens: self.tokens,
            },
            params,
            optimizer,
            model_config: self.model_json.clone(),
            run_config: self.config.to_toml(),
        })
    }

    fn record(&mut self, line: Value) -> Result<()> {
        writeln!(self.log, "{line}")?;
        Ok(())
    }

    fn go(mut self) -> Result<Summary> {
        let c = self.config;
        let steps = c.schedule.steps;
        let accumulate = c.batch.accumulate();
        let step_tokens = (c.batch.pass() * accumulate) as u64;
        let every = c.log.every.unwrap_or(1).max(1);
        let mut loss = f64::NAN;
        let mut val_loss = None;
        if !self.options.quiet && self.step == 0 {
            eprintln!(
                "train: {} parameters, {} tokens a step ({accumulate} × {} × {}), {steps} steps",
                self.params.count(),
                step_tokens,
                c.batch.rows,
                c.batch.length
            );
        }
        while self.step < steps {
            (self.synchronize)();
            let start = Instant::now();
            let lr = c.optimizer.lr * self.schedule.factor(self.step);
            self.optimizer.set_lr(lr);
            let mut sum: Option<BTreeMap<String, Grad>> = None;
            let mut losses = Vec::with_capacity(accumulate);
            for _ in 0..accumulate {
                let tokens = self.train.next_batch()?;
                let (l, grads) = self.pass(&tokens)?;
                losses.push(l);
                sum = Some(match sum {
                    None => grads.into_iter().collect(),
                    Some(mut total) => {
                        for (name, g) in grads {
                            let shape = self
                                .params
                                .get(&name)
                                .expect("a parameter")
                                .value
                                .shape()
                                .to_vec();
                            let merged = match total.remove(&name) {
                                Some(t) => self.add(t, g, &shape)?,
                                None => g,
                            };
                            total.insert(name, merged);
                        }
                        total
                    }
                });
            }
            let grads: Vec<(String, Grad)> = sum.unwrap_or_default().into_iter().collect();
            let scale = 1.0 / accumulate as f64;
            let norm = global_norm(&self.backend, &self.params, &grads)? * scale;
            let clip = c.training.clip;
            let factor = if clip > 0.0 && norm > clip {
                clip / norm
            } else {
                1.0
            };
            self.optimizer
                .step(&self.backend, &mut self.params, grads, scale * factor)?;
            let mut values = Vec::with_capacity(losses.len());
            for l in &losses {
                values.push(self.backend.download(l)?.to_f64_vec()?[0]);
            }
            loss = values.iter().sum::<f64>() / values.len() as f64;
            (self.synchronize)();
            let seconds = start.elapsed().as_secs_f64();
            let data_ms = self.train.take_waited().as_secs_f64() * 1e3;
            self.step += 1;
            self.tokens += step_tokens;
            let mfu = self
                .peak
                .map(|p| self.step_flops as f64 / seconds / (p * 1e12));
            let line = json!({
                "step": self.step,
                "loss": loss,
                "lr": lr,
                "norm": norm,
                "tokens": self.tokens,
                "ms": seconds * 1e3,
                "data_ms": data_ms,
                "tokens_per_s": step_tokens as f64 / seconds,
                "mfu": mfu,
                "peak_gib": self.peak_memory.as_ref().map(|p| p() as f64 / (1u64 << 30) as f64),
            });
            self.record(line)?;
            if !self.options.quiet && (self.step.is_multiple_of(every) || self.step == 1) {
                eprintln!(
                    "step {:>6}/{steps} | loss {loss:.6} | norm {norm:.4} | lr {lr:.2e} | {:.1} ms | {:.0} tok/s{}",
                    self.step,
                    seconds * 1e3,
                    step_tokens as f64 / seconds,
                    mfu.map(|m| format!(" | MFU {:.1} %", 100.0 * m))
                        .unwrap_or_default()
                        + &self
                            .peak_memory
                            .as_ref()
                            .map(|p| format!(" | {:.2} GiB", p() as f64 / (1u64 << 30) as f64))
                            .unwrap_or_default()
                );
            }
            let last = self.step == steps;
            let stopping = self.options.stop_after == Some(self.step);
            if (c.eval.every > 0 && self.step.is_multiple_of(c.eval.every)) || (last && !stopping) {
                val_loss = self.evaluate()?;
                if let Some(v) = val_loss {
                    self.record(json!({"step": self.step, "val_loss": v, "tokens": self.tokens}))?;
                    if !self.options.quiet {
                        eprintln!("step {:>6}/{steps} | val loss {v:.6}", self.step);
                    }
                }
            }
            if (c.checkpoint.every > 0 && self.step.is_multiple_of(c.checkpoint.every)) || last {
                let checkpoint = self.checkpoint()?;
                self.writer.write(checkpoint, &self.dir)?;
            }
            if let Some(first) = self.after_first.take() {
                first();
            }
            if stopping {
                break;
            }
        }
        self.writer.finish()?;
        let params = self
            .params
            .iter()
            .map(|(n, p)| Ok((n.to_string(), self.host_copy(&p.value)?)))
            .collect::<Result<_>>()?;
        Ok(Summary {
            steps: self.step,
            loss,
            val_loss,
            params,
        })
    }
}
