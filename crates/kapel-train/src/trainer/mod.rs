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
//!
//! What a step learns from is the run's objective: tokens (pre-training,
//! supervised fine-tuning), pairs of answers (DPO), or answers the model
//! writes and their rewards (GRPO), in `trainer/post.rs`. Every `eval.every` steps,
//! the losses of validation and of the text of control, and the accuracy of
//! the tasks, are measured.

mod offload;
mod post;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use kapel_core::{DType, Tensor};
use kapel_models::Config;
use kapel_ops::{Binary, BinaryKind, Concat};
use serde_json::{Value, json};

use crate::checkpoint::{self, Checkpoint, State, Writer};
use crate::config::{TrainConfig, matches};
use crate::data::{self, Loader, Prefetched, Source};
use crate::flops;
use crate::lora::{self, Projection};
use crate::model::{Batch, Decoder, Options, Parameter, Parameters, init};
use crate::optim::{Optimizer, OptimizerState, global_norm};
use crate::schedule::Schedule;
use crate::tape::{Compute, Grad, Tape};
use offload::Offload;

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
    crate::rewards::RewardError,
    kapel_infer::InferError,
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
/// directory named, with the fields of `model.set` replaced.
///
/// # Errors
///
/// If the file cannot be read or is not JSON.
pub fn model_json(config: &TrainConfig) -> Result<Value> {
    let path = &config.model.config;
    let file = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read(&file).map_err(|e| TrainError(format!("{}: {e}", file.display())))?;
    let mut json: Value = serde_json::from_slice(&text)
        .map_err(|e| TrainError(format!("{}: {e}", file.display())))?;
    if let Some(fields) = json.as_object_mut() {
        for (k, v) in &config.model.set {
            let v =
                serde_json::to_value(v).map_err(|e| TrainError(format!("model.set.{k}: {e}")))?;
            fields.insert(k.clone(), v);
        }
    }
    Ok(json)
}

/// The operations of a step of the run's tokens: frozen weights with LoRA
/// ([`flops::frozen_step_flops`]), [`flops::step_flops`] otherwise.
pub fn step_flops(config: &TrainConfig, model: &Config) -> u64 {
    let b = &config.batch;
    let tokens = b.pass() * b.accumulate();
    if config.lora.is_some() {
        flops::frozen_step_flops(model, tokens, b.length)
    } else {
        flops::step_flops(model, tokens, b.length)
    }
}

/// The configuration with the steps of its budget, when it has one: the
/// budget over the operations of a step, rounded, at least one.
///
/// # Errors
///
/// If the model's configuration cannot be read, or the warmup is longer
/// than the steps paid for.
pub fn with_budget(config: &TrainConfig) -> Result<TrainConfig> {
    let mut c = config.clone();
    if let Some(budget) = c.training.budget {
        let model = Config::from_json(&model_json(&c)?)?;
        let per_step = step_flops(&c, &model) as f64;
        c.schedule.steps = ((budget / per_step).round() as usize).max(1);
        if c.schedule.warmup > c.schedule.steps {
            return Err(TrainError(format!(
                "training.budget {budget:e} pays for {} steps of {per_step:.3e} operations, \
                 fewer than the {} of warmup",
                c.schedule.steps, c.schedule.warmup
            )));
        }
    }
    Ok(c)
}

/// The parameters the configuration starts from: its adapters added, its
/// groups applied, then the frozen projections quantized for QLoRA. With
/// adapters, also the projections adapted.
fn start(config: &TrainConfig, model_config: &Config) -> Result<(Parameters, Vec<Projection>)> {
    let mut params = if config.model.init == "weights" {
        let checkpoint = kapel_hub::Checkpoint::open(&config.model.config)?;
        Parameters::from_model(&kapel_models::Model::load(&checkpoint)?)
    } else {
        init(model_config, config.model.std, config.model.seed)?
    };
    let adapted = match &config.lora {
        Some(l) => lora::add_adapters(&mut params, model_config, l, config.model.seed)?,
        None => Vec::new(),
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
    if let Some(dtype) = config
        .lora
        .as_ref()
        .map(lora::LoraConfig::quantized_dtype)
        .transpose()?
        .flatten()
    {
        lora::quantize_frozen(&mut params, model_config, dtype)?;
    }
    Ok((params, adapted))
}

/// Runs the configuration's training on the device it names.
///
/// # Errors
///
/// For a configuration, a file or a device that fails.
#[allow(clippy::needless_return)]
pub fn run(config: &TrainConfig, options: &RunOptions) -> Result<Summary> {
    let mut resolved = config.clone();
    // A model of the Hub, downloaded once.
    if !config.model.config.exists() && crate::config::is_hub_name(&config.model.config) {
        let name = config.model.config.to_string_lossy().into_owned();
        resolved.model.config =
            kapel_hub::download::snapshot(&name, "main", &kapel_hub::download::default_cache())?;
        resolved.model.hub = Some(name);
    }
    let resolved = with_budget(&resolved)?;
    let config = &resolved;
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

/// What the steps of a run learn from.
enum Objective {
    /// Tokens, read ahead.
    Tokens(Prefetched),
    /// Pairs of answers, for DPO.
    Pairs(Box<post::Pairs>),
    /// Answers written by the model, for GRPO.
    Answers(Box<post::Answers>),
}

impl Objective {
    fn position(&self) -> data::Position {
        match self {
            Self::Tokens(t) => t.position().clone(),
            Self::Pairs(p) => p.pairs.position(),
            Self::Answers(a) => a.prompts.position(),
        }
    }
}

/// The gradients of the passes of a step, added up.
#[derive(Default)]
struct Summed(BTreeMap<String, Grad>);

/// What a step computed, before the optimizer moves.
struct StepOutput {
    /// The gradients, summed over the passes.
    grads: Vec<(String, Grad)>,
    /// The factor of the sum: `1 / passes` for a mean.
    scale: f64,
    /// The loss of the step.
    loss: f64,
    /// The tokens it learnt from (their rows, padding included).
    tokens: u64,
    /// Its operations, for the MFU.
    flops: u64,
    /// What else it shows, for the log.
    metrics: serde_json::Map<String, Value>,
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
    /// The projections with adapters, for a run with LoRA.
    adapted: Vec<Projection>,
    optimizer: Optimizer,
    schedule: Schedule,
    objective: Objective,
    /// The model the run starts from, frozen, when the objective needs it
    /// and the adapters are not the difference.
    reference: Option<Parameters>,
    /// What writes answers, for GRPO and the tasks.
    generator: Option<kapel_infer::Pipeline<B>>,
    /// The tasks evaluated, their prompts read.
    tasks: Vec<post::Task>,
    val: Option<Vec<data::Tokens>>,
    control: Option<Vec<data::Tokens>>,
    step: usize,
    tokens: u64,
    dir: PathBuf,
    log: std::fs::File,
    writer: Writer,
    peak: Option<f64>,
    step_flops: u64,
    /// The model, for the operations of joined batches.
    model_config: Config,
    /// The trained parameters on the processor, with `training.offload`.
    offload: Option<Offload>,
    /// Called once the first step is done: a profile forgets the
    /// compilation of the kernels.
    after_first: Option<Box<dyn Fn()>>,
    /// The peak memory of the card since the run started, if it counts it.
    peak_memory: Option<Box<dyn Fn() -> usize>>,
    /// The time of the steps done by this process, and their tokens.
    seconds: f64,
    tokens_here: u64,
    /// The log for TensorBoard, if asked.
    events: Option<crate::tensorboard::EventWriter>,
}

impl<'a, B: Compute> Run<'a, B> {
    fn new(
        config: &'a TrainConfig,
        options: &'a RunOptions,
        backend: B,
        synchronize: Box<dyn Fn()>,
        device_name: Option<String>,
    ) -> Result<Self> {
        let model_json = model_json(config)?;
        let model_config = Config::from_json(&model_json)?;
        let mut decoder = Decoder::new(model_config.clone())?;
        if let Some(l) = &config.lora {
            decoder = decoder.with_lora(l.scale());
        }
        let (mut params, adapted) = start(config, &model_config)?;
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
        let encoder = || -> Result<crate::chat::Encoder> {
            Ok(crate::chat::Encoder::from_dir(&config.model.config)?)
        };
        let mut loader = None;
        let mut objective = if let Some(dpo) = &config.dpo {
            Some(Objective::Pairs(Box::new(post::Pairs::new(
                dpo,
                &encoder()?,
                b.length,
                config.data.seed,
                options.quiet,
            )?)))
        } else if let Some(grpo) = &config.grpo {
            Some(Objective::Answers(Box::new(post::Answers::new(
                grpo,
                &encoder()?,
                b.length,
                config.data.seed,
                options.quiet,
            )?)))
        } else {
            let sources = config
                .data
                .training_sources()
                .iter()
                .map(|(path, weight)| Ok((open(path, "train")?, *weight)))
                .collect::<Result<Vec<_>>>()?;
            loader = Some(Loader::new(sources, reading.clone())?);
            None
        };
        // In order, whatever the training's order: the same rows at every
        // evaluation.
        let held_out = |path: &Option<String>| -> Result<Option<Vec<data::Tokens>>> {
            match path {
                Some(path) if config.eval.tokens > 0 => {
                    let mut r = reading.clone();
                    r.shuffle = false;
                    let mut loader = Loader::new(vec![(open(path, "val")?, 1.0)], r)?;
                    let passes = config.eval.tokens / b.pass();
                    Ok(Some(
                        (0..passes)
                            .map(|_| loader.next_batch())
                            .collect::<std::result::Result<_, _>>()?,
                    ))
                }
                _ => Ok(None),
            }
        };
        let val = held_out(&config.data.val)?;
        let control = held_out(&config.eval.control)?;
        let tasks = if config.eval.tasks.is_empty() {
            Vec::new()
        } else {
            let e = encoder()?;
            config
                .eval
                .tasks
                .iter()
                .map(|t| post::Task::new(t, &e))
                .collect::<Result<_>>()?
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
            // With offload, the optimizer's state stays on the processor.
            let tensors = saved
                .optimizer
                .iter()
                .map(|(n, t)| {
                    Ok((
                        n.clone(),
                        if config.training.offload {
                            crate::model::owned(t)?
                        } else {
                            backend.upload(t)?
                        },
                    ))
                })
                .collect::<Result<_>>()?;
            optimizer.restore(OptimizerState {
                steps: saved.state.optimizer_steps,
                tensors,
            })?;
            let position = &saved.state.position;
            match (&mut objective, &mut loader) {
                (Some(Objective::Pairs(p)), _) => p.pairs.seek(position)?,
                (Some(Objective::Answers(a)), _) => a.prompts.seek(position)?,
                (_, Some(l)) => l.seek(position)?,
                _ => {}
            }
            step = saved.state.step;
            tokens = saved.state.tokens;
            if !options.quiet {
                eprintln!("train: resumed from {} at step {step}", path.display());
            }
        }
        let objective = match (objective, loader) {
            (Some(o), _) => o,
            (None, Some(l)) => Objective::Tokens(l.prefetch(config.data.prefetch)),
            (None, None) => unreachable!("an objective or tokens"),
        };
        // The model the objective compares with: the starting weights,
        // frozen, in the computation's type; when only adapters train, the
        // model without them, which needs no copy.
        let needs_reference =
            config.dpo.is_some() || config.grpo.as_ref().is_some_and(|g| g.beta > 0.0);
        let adapters_only = config.lora.is_some()
            && params
                .iter()
                .all(|(n, p)| !p.trainable || lora::is_factor(n));
        let reference = if needs_reference && !adapters_only {
            let (mut start, _) = start(config, &model_config)?;
            for (_, p) in start.iter_mut() {
                p.trainable = false;
            }
            Some(start.to(&backend, config.dtype())?)
        } else {
            None
        };
        // The frozen parameters in the computation's type: half the memory
        // of a frozen model in `f32`. With offload, the trained ones too, their
        // values in `f32` kept on the processor.
        let offload = if config.training.offload {
            Some(Offload::new(&backend, &params, config.dtype())?)
        } else {
            None
        };
        let master = if offload.is_some() {
            config.dtype()
        } else {
            DType::F32
        };
        let params = params.to_mixed(&backend, master, config.dtype())?;
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
        let step_flops = step_flops(config, &model_config);
        Ok(Self {
            config,
            options,
            backend,
            synchronize,
            model_json,
            decoder,
            params,
            adapted,
            optimizer,
            schedule: Schedule::new(&config.schedule),
            objective,
            reference,
            generator: None,
            tasks,
            val,
            control,
            step,
            tokens,
            dir,
            log,
            writer: Writer::new(config.checkpoint.keep),
            peak,
            step_flops,
            model_config: model_config.clone(),
            offload,
            after_first: None,
            peak_memory: None,
            seconds: 0.0,
            tokens_here: 0,
            events: if config.log.tensorboard {
                Some(crate::tensorboard::EventWriter::create(
                    &config.checkpoint.dir.join("tensorboard"),
                )?)
            } else {
                None
            },
        })
    }

    fn options(&self) -> Options {
        Options {
            dtype: self.config.dtype(),
            checkpoint: self.config.training.checkpoint_activations,
            chunk: self.config.training.chunk,
        }
    }

    /// A step of tokens: `accumulate` passes, their gradients summed, their
    /// losses averaged.
    fn token_step(&mut self) -> Result<StepOutput> {
        let accumulate = self.config.batch.accumulate();
        let mut losses = Vec::with_capacity(accumulate);
        let mut sum = Summed::default();
        // The tokens and operations of joined batches, which vary.
        let mut joined = (0u64, 0u64);
        for _ in 0..accumulate {
            let Objective::Tokens(train) = &mut self.objective else {
                unreachable!("a step of tokens");
            };
            let tokens = train.next_batch()?;
            if self.joined() {
                joined.0 += tokens.ids.len() as u64;
                joined.1 += self.joined_flops(&tokens);
            }
            let (l, grads) = self.pass(&tokens)?;
            losses.push(l);
            self.accumulate(&mut sum, grads)?;
        }
        let mut values = Vec::with_capacity(losses.len());
        for l in &losses {
            values.push(self.backend.download(l)?.to_f64_vec()?[0]);
        }
        Ok(StepOutput {
            grads: sum.0.into_iter().collect(),
            scale: 1.0 / accumulate as f64,
            loss: values.iter().sum::<f64>() / values.len() as f64,
            tokens: if self.joined() {
                joined.0
            } else {
                (self.config.batch.pass() * accumulate) as u64
            },
            flops: if self.joined() {
                joined.1
            } else {
                self.step_flops
            },
            metrics: serde_json::Map::new(),
        })
    }

    /// Whether the batches are joined ([`data::Pack::Joined`]): one row of
    /// a varying length a pass.
    fn joined(&self) -> bool {
        self.config.data.pack == data::Pack::Joined
    }

    /// The rows of a pass of `tokens`.
    fn rows(&self) -> usize {
        if self.joined() {
            1
        } else {
            self.config.batch.rows
        }
    }

    /// The operations of a pass of a joined batch: each document's, its
    /// attention over itself.
    fn joined_flops(&self, tokens: &data::Tokens) -> u64 {
        tokens
            .documents()
            .into_iter()
            .map(|n| {
                if self.config.lora.is_some() {
                    flops::frozen_step_flops(&self.model_config, n, n)
                } else {
                    flops::step_flops(&self.model_config, n, n)
                }
            })
            .sum()
    }

    /// Adds the gradients of a pass to those of the step: on the processor,
    /// in `f32`, with offload and several passes a step.
    fn accumulate(&self, sum: &mut Summed, grads: Vec<(String, Grad)>) -> Result<()> {
        if let Some(o) = &self.offload
            && self.config.batch.accumulate() > 1
        {
            return o.accumulate(&self.backend, &mut sum.0, grads);
        }
        for (name, g) in grads {
            let merged = match sum.0.remove(&name) {
                Some(t) => {
                    let shape = self
                        .params
                        .get(&name)
                        .expect("a parameter")
                        .value
                        .shape()
                        .to_vec();
                    self.add(t, g, &shape)?
                }
                None => g,
            };
            sum.0.insert(name, merged);
        }
        Ok(())
    }

    /// One pass: its loss, left on the device, and its gradients.
    fn pass(&self, tokens: &data::Tokens) -> Result<(Tensor, Vec<(String, Grad)>)> {
        let tape = Tape::new(self.backend.clone());
        let leaves = self.params.leaves(&tape);
        let batch = Batch {
            ids: &tokens.ids,
            targets: &tokens.targets,
            rows: self.rows(),
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

    /// The mean loss of held-out batches, without gradients.
    fn evaluate(&self, batches: Option<&[data::Tokens]>) -> Result<Option<f64>> {
        let Some(val) = batches else {
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
                rows: self.rows(),
                segments: tokens.segments.as_deref(),
            };
            let loss = self.decoder.loss(&leaves, &batch, &self.options())?;
            total += self.backend.download(&loss.value())?.to_f64_vec()?[0];
        }
        Ok(Some(total / val.len().max(1) as f64))
    }

    /// The value of a parameter: its `f32` copy on the processor with
    /// offload, else its own.
    fn value<'p>(&'p self, name: &str, p: &'p Parameter) -> &'p Tensor {
        self.offload
            .as_ref()
            .and_then(|o| o.params.get(name))
            .map_or(&p.value, |m| &m.value)
    }

    /// What a checkpoint keeps of `t`: with offload, the processor's
    /// tensors themselves, the checkpoint written before the next step
    /// changes them (no copy of the state: it may be most of the memory);
    /// otherwise [`Self::host_copy`].
    fn saved(&self, t: &Tensor) -> Result<Tensor> {
        if self.offload.is_some() && t.device() == kapel_core::Device::Cpu {
            Ok(t.clone())
        } else {
            self.host_copy(t)
        }
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

    /// The checkpoint of the current step, in host memory: with adapters,
    /// the parameters trained only.
    fn checkpoint(&self) -> Result<Checkpoint> {
        let adapter = self.config.lora.as_ref().map(|l| {
            lora::peft_config(
                l,
                self.decoder.config(),
                &self.adapted,
                &self
                    .config
                    .model
                    .hub
                    .clone()
                    .unwrap_or_else(|| self.config.model.config.to_string_lossy().into_owned()),
            )
        });
        let params = self
            .params
            .iter()
            .filter(|(_, p)| adapter.is_none() || p.trainable)
            .map(|(n, p)| Ok((n.to_string(), self.saved(self.value(n, p))?)))
            .collect::<Result<_>>()?;
        let state = self.optimizer.state();
        let optimizer = state
            .tensors
            .iter()
            .map(|(n, t)| Ok((n.clone(), self.saved(t)?)))
            .collect::<Result<_>>()?;
        // The optimizer's own handles only, during the next steps.
        let steps = state.steps;
        drop(state);
        Ok(Checkpoint {
            state: State {
                step: self.step,
                optimizer_steps: steps,
                optimizer: self.config.optimizer.kind.clone(),
                position: self.objective.position(),
                tokens: self.tokens,
            },
            params,
            optimizer,
            model_config: self.model_json.clone(),
            run_config: self.config.to_toml(),
            adapter,
        })
    }

    /// `summary.json` in the run's directory, once the last step is done:
    /// what a grid of runs ([`crate::sweep`]) reads of each.
    fn write_summary(&self, loss: f64, val_loss: Option<f64>) -> Result<()> {
        let embedding: usize = self
            .params
            .iter()
            .filter(|(n, _)| crate::model::is_table(n))
            .map(|(_, p)| p.value.numel())
            .sum();
        let all = self.params.count_all();
        let tokens_objective = matches!(self.objective, Objective::Tokens(_));
        let summary = json!({
            "steps": self.step,
            "tokens": self.tokens,
            "flops": tokens_objective.then_some(self.step_flops as f64 * self.step as f64),
            "step_flops": tokens_objective.then_some(self.step_flops),
            "parameters": all,
            "non_embedding_parameters": all - embedding,
            "trainable_parameters": self.params.count(),
            "loss": loss,
            "val_loss": val_loss,
            "seconds": self.seconds,
            "tokens_per_s": self.tokens_here as f64 / self.seconds.max(1e-9),
            "peak_gib": self.peak_memory.as_ref().map(|p| p() as f64 / (1u64 << 30) as f64),
        });
        let text = serde_json::to_string_pretty(&summary).map_err(|e| TrainError(e.to_string()))?;
        std::fs::write(self.dir.join("summary.json"), text + "\n")?;
        Ok(())
    }

    fn record(&mut self, line: Value) -> Result<()> {
        writeln!(self.log, "{line}")?;
        if let Some(events) = &mut self.events {
            events.line(&line)?;
        }
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
            let out = match &self.objective {
                Objective::Tokens(_) => self.token_step()?,
                Objective::Pairs(_) => self.pair_step()?,
                Objective::Answers(_) => self.answer_step()?,
            };
            let (grads, scale) = (out.grads, out.scale);
            // Summed on the processor with offload and several passes: their
            // norm computed there, as the card would.
            let norm = match &self.offload {
                Some(o) if c.batch.accumulate() > 1 => global_norm(&o.cpu, &o.params, &grads)?,
                _ => global_norm(&self.backend, &self.params, &grads)?,
            } * scale;
            let clip = c.training.clip;
            let factor = if clip > 0.0 && norm > clip {
                clip / norm
            } else {
                1.0
            };
            if !grads.is_empty() {
                match &mut self.offload {
                    Some(o) => o.step(
                        &mut self.optimizer,
                        &self.backend,
                        &mut self.params,
                        grads,
                        scale * factor,
                    )?,
                    None => {
                        self.optimizer.step(
                            &self.backend,
                            &mut self.params,
                            grads,
                            scale * factor,
                        )?;
                    }
                }
            }
            loss = out.loss;
            (self.synchronize)();
            let seconds = start.elapsed().as_secs_f64();
            let data_ms = match &mut self.objective {
                Objective::Tokens(t) => t.take_waited().as_secs_f64() * 1e3,
                _ => 0.0,
            };
            self.step += 1;
            self.tokens += out.tokens;
            self.seconds += seconds;
            self.tokens_here += out.tokens;
            let mfu = self.peak.map(|p| out.flops as f64 / seconds / (p * 1e12));
            let mut line = json!({
                "step": self.step,
                "loss": loss,
                "lr": lr,
                "norm": norm,
                "tokens": self.tokens,
                "ms": seconds * 1e3,
                "data_ms": data_ms,
                "tokens_per_s": out.tokens as f64 / seconds,
                "mfu": mfu,
                "peak_gib": self.peak_memory.as_ref().map(|p| p() as f64 / (1u64 << 30) as f64),
            });
            let shown: Vec<String> = out
                .metrics
                .iter()
                .filter_map(|(k, v)| Some(format!(" | {k} {:.4}", v.as_f64()?)))
                .collect();
            if let Some(o) = line.as_object_mut() {
                o.extend(out.metrics);
            }
            self.record(line)?;
            let step_tokens = out.tokens;
            if !self.options.quiet && (self.step.is_multiple_of(every) || self.step == 1) {
                eprintln!(
                    "step {:>6}/{steps} | loss {loss:.6} | norm {norm:.4} | lr {lr:.2e} | {:.1} ms | {:.0} tok/s{}{}",
                    self.step,
                    seconds * 1e3,
                    step_tokens as f64 / seconds,
                    mfu.map(|m| format!(" | MFU {:.1} %", 100.0 * m))
                        .unwrap_or_default()
                        + &self
                            .peak_memory
                            .as_ref()
                            .map(|p| format!(" | {:.2} GiB", p() as f64 / (1u64 << 30) as f64))
                            .unwrap_or_default(),
                    shown.concat()
                );
            }
            let last = self.step == steps;
            let stopping = self.options.stop_after == Some(self.step);
            if (c.eval.every > 0 && self.step.is_multiple_of(c.eval.every)) || (last && !stopping) {
                val_loss = self.evaluate(self.val.as_deref())?;
                if let Some(mut held) = self.evaluate_pairs()? {
                    val_loss = held.get("val_loss").and_then(Value::as_f64);
                    if !self.options.quiet {
                        let get = |k: &str| held.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                        eprintln!(
                            "step {:>6}/{steps} | val loss {:.6} | accuracy {:.4} | margin {:.4}",
                            self.step,
                            get("val_loss"),
                            get("val_accuracy"),
                            get("val_margin")
                        );
                    }
                    held.insert("step".into(), json!(self.step));
                    held.insert("tokens".into(), json!(self.tokens));
                    self.record(Value::Object(held))?;
                } else if let Some(v) = val_loss {
                    self.record(json!({"step": self.step, "val_loss": v, "tokens": self.tokens}))?;
                    if !self.options.quiet {
                        eprintln!("step {:>6}/{steps} | val loss {v:.6}", self.step);
                    }
                }
                if let Some(v) = self.evaluate(self.control.as_deref())? {
                    self.record(
                        json!({"step": self.step, "control_loss": v, "tokens": self.tokens}),
                    )?;
                    if !self.options.quiet {
                        eprintln!("step {:>6}/{steps} | control loss {v:.6}", self.step);
                    }
                }
                self.evaluate_tasks()?;
            }
            if (c.checkpoint.every > 0 && self.step.is_multiple_of(c.checkpoint.every)) || last {
                let checkpoint = self.checkpoint()?;
                self.writer.write(checkpoint, &self.dir)?;
                if self.offload.is_some() {
                    // It holds the processor's tensors, which the next step
                    // writes in place.
                    self.writer.finish()?;
                }
            }
            if let Some(first) = self.after_first.take() {
                first();
            }
            if stopping {
                break;
            }
        }
        self.writer.finish()?;
        // A run resumed at its end did nothing: its summary stays.
        if self.step == steps && self.tokens_here > 0 {
            self.write_summary(loss, val_loss)?;
        }
        // With offload, the trained values are already in host memory: moved,
        // not copied.
        let mut master = self.offload.take().map(|o| o.params);
        let params = self
            .params
            .iter()
            .map(|(n, p)| {
                let value = match master.as_mut().and_then(|m| m.remove(n)) {
                    Some(m) => m.value,
                    None => self.host_copy(&p.value)?,
                };
                Ok((n.to_string(), value))
            })
            .collect::<Result<_>>()?;
        Ok(Summary {
            steps: self.step,
            loss,
            val_loss,
            params,
        })
    }
}
