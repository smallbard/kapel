//! The configuration of a training run: one TOML file per experiment,
//! checked before anything starts, and saved with each checkpoint.
//!
//! ```toml
//! [model]
//! config = "configs/gpt2-124m.json"   # a config.json, or a model directory
//! init = "scratch"                    # or "weights": the directory's weights
//! seed = 1
//!
//! [data]
//! train = "fineweb/fineweb_train_*.bin"
//! val = "fineweb/fineweb_val_*.bin"
//!
//! [batch]
//! rows = 16
//! length = 1024
//! tokens = 524288                     # per step: 32 batches accumulated
//!
//! [optimizer]
//! kind = "adamw"
//! lr = 6e-4
//! weight_decay = 0.1
//!
//! [schedule]
//! kind = "cosine"
//! warmup = 700
//! steps = 19073
//! final = 0.0
//! ```
//!
//! Every field but `model.config`, `data.train` (or `data.sources`) and
//! `schedule.steps` has a
//! default; an unknown field is an error, so that a misspelt setting does not
//! go unnoticed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A configuration that cannot be used.
#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// A training run.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    /// The model.
    pub model: ModelConfig,
    /// The tokens, for pre-training and supervised fine-tuning.
    #[serde(default)]
    pub data: DataConfig,
    /// The size of a step.
    #[serde(default)]
    pub batch: BatchConfig,
    /// The optimizer.
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    /// The learning rate along the run.
    pub schedule: ScheduleConfig,
    /// How each step is computed.
    #[serde(default)]
    pub training: TrainingConfig,
    /// The loss on the validation tokens.
    #[serde(default)]
    pub eval: EvalConfig,
    /// The checkpoints.
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
    /// The log.
    #[serde(default)]
    pub log: LogConfig,
    /// LoRA adapters: the model frozen, only they trained
    /// ([`crate::lora`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<crate::lora::LoraConfig>,
    /// DPO: the run learns from pairs of answers instead of tokens
    /// ([`crate::preference`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpo: Option<crate::preference::DpoConfig>,
    /// GRPO: the run learns from answers it writes and their rewards
    /// ([`crate::grpo`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grpo: Option<crate::grpo::GrpoConfig>,
}

/// Where the model comes from.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// A `config.json`, a model directory holding one, or a model of the
    /// Hugging Face Hub (`owner/name`), downloaded once into Kapel's cache
    /// as `kapel run` does.
    pub config: PathBuf,
    /// `"scratch"`: new weights ([`crate::model::init`]); `"weights"`: those
    /// of the model directory.
    #[serde(default = "scratch")]
    pub init: String,
    /// The standard deviation of the new weights.
    #[serde(default = "std_default")]
    pub std: f64,
    /// The seed of the new weights.
    #[serde(default = "one")]
    pub seed: u64,
    /// Fields of the `config.json` replaced, for a model trained from
    /// scratch: one file describes a family of sizes
    /// (`set = { hidden_size = 384, num_hidden_layers = 6 }`).
    #[serde(default, skip_serializing_if = "toml::Table::is_empty")]
    pub set: toml::Table,
    /// The name of the model on the Hub, when `config` was one: set by the
    /// run, for the adapters' `base_model_name_or_path`.
    #[serde(skip)]
    pub hub: Option<String>,
}

fn scratch() -> String {
    "scratch".into()
}

fn std_default() -> f64 {
    0.02
}

fn one() -> u64 {
    1
}

/// The tokens: where they are, and how they are read ([`crate::data`]).
///
/// `train` names one source; `sources`, several, each with its weight. A
/// source is a directory of `kapel prepare` or its `dataset.json`, either
/// local or at an `http(s)://` URL, or a pattern of llm.c files.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// The training source: a path, a URL, or a pattern with `*` in its
    /// file name, read in the sorted order of their names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<String>,
    /// Several training sources, each row drawn from one of them with the
    /// probability of its weight.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceConfig>,
    /// The validation source: a dataset's `val` split, or files.
    #[serde(default)]
    pub val: Option<String>,
    /// The seed of the mixing and of the orders drawn.
    #[serde(default)]
    pub seed: u64,
    /// Read the shards, and the rows of each, in an order drawn anew for
    /// each epoch.
    #[serde(default)]
    pub shuffle: bool,
    /// `"stream"`: the tokens cut every `batch.length`; `"documents"`:
    /// whole documents in each row, padded; `"padded"`: one document a row,
    /// cut or padded (TRL's batches without packing).
    #[serde(default)]
    pub pack: crate::data::Pack,
    /// Each token attends to its own document only, and positions start
    /// again at each document.
    #[serde(default)]
    pub mask: bool,
    /// The token that starts each document, for files that have neither
    /// an index nor a manifest (50256 for llm.c's GPT-2 shards).
    #[serde(default)]
    pub separator: Option<u32>,
    /// The batches read ahead by a thread of their own.
    #[serde(default = "prefetch_default")]
    pub prefetch: usize,
    /// Where remote shards are kept while they are read; absent,
    /// `$KAPEL_DATA_CACHE` or `~/.cache/kapel/data`.
    #[serde(default)]
    pub cache: Option<PathBuf>,
}

fn prefetch_default() -> usize {
    4
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train: None,
            sources: Vec::new(),
            val: None,
            seed: 0,
            shuffle: false,
            pack: crate::data::Pack::default(),
            mask: false,
            separator: None,
            prefetch: prefetch_default(),
            cache: None,
        }
    }
}

/// One of several training sources.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    /// As [`DataConfig::train`].
    pub path: String,
    /// Its weight: its share of the rows is its weight over their sum.
    #[serde(default = "weight_default")]
    pub weight: f64,
}

fn weight_default() -> f64 {
    1.0
}

impl DataConfig {
    /// The training sources with their weights.
    pub fn training_sources(&self) -> Vec<(String, f64)> {
        match &self.train {
            Some(t) => vec![(t.clone(), 1.0)],
            None => self
                .sources
                .iter()
                .map(|s| (s.path.clone(), s.weight))
                .collect(),
        }
    }

    /// How batches of `rows × length` are read.
    pub fn reading(&self, rows: usize, length: usize) -> crate::data::Reading {
        crate::data::Reading {
            rows,
            length,
            seed: self.seed,
            shuffle: self.shuffle,
            pack: self.pack,
            mask: self.mask,
            separator: self.separator,
            cache: self
                .cache
                .clone()
                .unwrap_or_else(crate::data::default_cache),
        }
    }
}

/// The size of a step.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchConfig {
    /// The sequences of one forward and backward pass.
    #[serde(default = "rows_default")]
    pub rows: usize,
    /// Their length.
    #[serde(default = "length_default")]
    pub length: usize,
    /// The tokens of a step, a multiple of `rows × length`: the passes are
    /// accumulated. Absent: one pass a step.
    #[serde(default)]
    pub tokens: Option<usize>,
}

fn rows_default() -> usize {
    16
}

fn length_default() -> usize {
    1024
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            rows: rows_default(),
            length: length_default(),
            tokens: None,
        }
    }
}

impl BatchConfig {
    /// The tokens of one pass.
    pub fn pass(&self) -> usize {
        self.rows * self.length
    }

    /// The passes accumulated in a step.
    pub fn accumulate(&self) -> usize {
        self.tokens.map_or(1, |t| t / self.pass())
    }
}

/// The optimizer, and the groups of parameters treated apart.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    /// `"adamw"`, `"muon"` (the matrices of the blocks; AdamW for the rest)
    /// or `"adafactor"`.
    #[serde(default = "adamw")]
    pub kind: String,
    /// The peak learning rate.
    #[serde(default = "lr_default")]
    pub lr: f64,
    /// AdamW's `β₁`.
    #[serde(default = "beta1_default")]
    pub beta1: f64,
    /// AdamW's `β₂`.
    #[serde(default = "beta2_default")]
    pub beta2: f64,
    /// AdamW's `ε`.
    #[serde(default = "eps_default")]
    pub eps: f64,
    /// The decoupled weight decay of the parameters that decay.
    #[serde(default)]
    pub weight_decay: f64,
    /// Muon's momentum.
    #[serde(default = "momentum_default")]
    pub momentum: f64,
    /// Muon's learning rate, as a factor of `lr`; the AdamW of the other
    /// parameters keeps `lr`.
    #[serde(default = "one_f64")]
    pub muon_lr_scale: f64,
    /// Muon's iterations of Newton-Schulz: `"default"`, ten (DeepSeek-V4's),
    /// or `"short"`, four (SAMuon's, Smallem's `muon:short`).
    #[serde(default = "default_name")]
    pub newton_schulz: String,
    /// Groups of parameters with settings of their own, the first that
    /// matches a parameter applying.
    #[serde(default)]
    pub groups: Vec<GroupConfig>,
}

fn default_name() -> String {
    "default".into()
}

fn adamw() -> String {
    "adamw".into()
}

fn lr_default() -> f64 {
    6e-4
}

fn beta1_default() -> f64 {
    0.9
}

fn beta2_default() -> f64 {
    0.95
}

fn eps_default() -> f64 {
    1e-8
}

fn momentum_default() -> f64 {
    0.95
}

fn one_f64() -> f64 {
    1.0
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            kind: adamw(),
            lr: lr_default(),
            beta1: beta1_default(),
            beta2: beta2_default(),
            eps: eps_default(),
            weight_decay: 0.0,
            momentum: momentum_default(),
            muon_lr_scale: 1.0,
            newton_schulz: default_name(),
            groups: Vec::new(),
        }
    }
}

/// Settings of their own for the parameters whose names match.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroupConfig {
    /// A pattern of names: `*` stands for any text (`"*.bias"`,
    /// `"transformer.wte.*"`).
    pub matches: String,
    /// A factor of the learning rate.
    #[serde(default)]
    pub lr_scale: Option<f64>,
    /// Whether weight decay applies.
    #[serde(default)]
    pub decay: Option<bool>,
    /// Whether the parameter trains; `false` freezes it.
    #[serde(default)]
    pub trainable: Option<bool>,
    /// Whether its gradient stays rows ([`crate::Tape::sparse_leaf`]).
    #[serde(default)]
    pub sparse: Option<bool>,
}

/// The learning rate along the run.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleConfig {
    /// `"constant"`, `"cosine"`, `"wsd"` (warmup, stable, decay), or
    /// `"linear"` (`transformers`' default).
    #[serde(default = "cosine")]
    pub kind: String,
    /// The steps of the linear warmup from zero.
    #[serde(default)]
    pub warmup: usize,
    /// The steps of the run; with `training.budget`, those it pays for.
    #[serde(default)]
    pub steps: usize,
    /// The learning rate at the end, as a fraction of the peak.
    #[serde(default)]
    pub r#final: f64,
    /// For `"wsd"`: the fraction of the run that decays, at its end.
    #[serde(default = "decay_default")]
    pub decay: f64,
}

fn cosine() -> String {
    "cosine".into()
}

fn decay_default() -> f64 {
    0.2
}

/// How each step is computed.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// The device: `"cpu"` or `"cuda"` (`"cuda:1"` for another card).
    #[serde(default = "cuda")]
    pub device: String,
    /// The type of the computation: `"bf16"`, `"f16"` or `"f32"`; the
    /// parameters and the optimizer stay in `f32`.
    #[serde(default = "bf16")]
    pub dtype: String,
    /// The norm the gradient is clipped to; 0: no clipping.
    #[serde(default = "one_f64")]
    pub clip: f64,
    /// Recompute each block during the backward pass.
    #[serde(default)]
    pub checkpoint_activations: bool,
    /// The rows of logits the loss forms at once.
    #[serde(default = "chunk_default")]
    pub chunk: usize,
    /// The memory of the cache of keys and values when the model writes
    /// answers (GRPO, the tasks), in GiB.
    #[serde(default = "cache_gib_default")]
    pub cache_gib: f64,
    /// The computation of the run, in operations (`1e16`): the steps are
    /// those it pays for ([`crate::flops::step_flops`]), and
    /// `schedule.steps` may be left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<f64>,
}

fn cache_gib_default() -> f64 {
    4.0
}

fn cuda() -> String {
    "cuda".into()
}

fn bf16() -> String {
    "bf16".into()
}

fn chunk_default() -> usize {
    4096
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            device: cuda(),
            dtype: bf16(),
            clip: 1.0,
            checkpoint_activations: false,
            chunk: chunk_default(),
            cache_gib: cache_gib_default(),
            budget: None,
        }
    }
}

/// The validation loss.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    /// The steps between two evaluations; 0: only at the end.
    #[serde(default = "eval_every_default")]
    pub every: usize,
    /// The validation tokens read, from the start of the files.
    #[serde(default = "eval_tokens_default")]
    pub tokens: usize,
    /// A text of control, its loss measured as the validation's: what a
    /// fine-tuning makes the model forget (the `val` split of a dataset of
    /// `kapel prepare`, or files).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<String>,
    /// Tasks whose answers a program checks, their accuracy measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tasks: Vec<crate::grpo::TaskConfig>,
}

fn eval_every_default() -> usize {
    250
}

fn eval_tokens_default() -> usize {
    10 * 1024 * 1024
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            every: eval_every_default(),
            tokens: eval_tokens_default(),
            control: None,
            tasks: Vec::new(),
        }
    }
}

/// The checkpoints.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    /// The directory of the run: its checkpoints and its log.
    #[serde(default = "dir_default")]
    pub dir: PathBuf,
    /// The steps between two checkpoints; 0: only at the end.
    #[serde(default = "checkpoint_every_default")]
    pub every: usize,
    /// The checkpoints kept, the oldest removed first; 0: all.
    #[serde(default = "keep_default")]
    pub keep: usize,
}

fn dir_default() -> PathBuf {
    PathBuf::from("run")
}

fn checkpoint_every_default() -> usize {
    1000
}

fn keep_default() -> usize {
    3
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            dir: dir_default(),
            every: checkpoint_every_default(),
            keep: keep_default(),
        }
    }
}

/// The log.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogConfig {
    /// The steps between two lines on the terminal; the JSONL log has every
    /// step.
    #[serde(default)]
    pub every: Option<usize>,
    /// The peak of the card, in Top/s of the computation's type, for the
    /// model's FLOP utilization (MFU); absent: from the card's name when
    /// Kapel knows it.
    #[serde(default)]
    pub peak_tflops: Option<f64>,
    /// Also write the log as TensorBoard's events, in `tensorboard/` of the
    /// run's directory ([`crate::tensorboard`]).
    #[serde(default)]
    pub tensorboard: bool,
}

impl TrainConfig {
    /// Reads and checks a configuration; relative paths are taken from the
    /// file's directory.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, is not valid TOML, has an unknown field,
    /// or a value [`TrainConfig::validate`] refuses.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError(format!("{}: {e}", path.display())))?;
        let mut config = Self::parse(&text)?;
        let base = path.parent().unwrap_or(Path::new("."));
        config.resolve(base);
        Ok(config)
    }

    /// A configuration from its text, checked; paths are left as written.
    ///
    /// # Errors
    ///
    /// As [`TrainConfig::load`].
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// The configuration as TOML, as it is saved with each checkpoint.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a configuration serializes")
    }

    /// Relative paths taken from `base`.
    pub fn resolve(&mut self, base: &Path) {
        let join = |p: &Path| {
            if p.is_relative() {
                base.join(p)
            } else {
                p.to_path_buf()
            }
        };
        // A name of the Hub stays as written, when no such file is there.
        let joined = join(&self.model.config);
        if joined.exists() || !is_hub_name(&self.model.config) {
            self.model.config = joined;
        }
        self.checkpoint.dir = join(&self.checkpoint.dir);
        // A URL stays as written.
        let place = |p: &str| {
            if p.starts_with("http://") || p.starts_with("https://") {
                p.to_string()
            } else {
                join(Path::new(p)).to_string_lossy().into_owned()
            }
        };
        self.data.train = self.data.train.as_deref().map(place);
        self.data.val = self.data.val.as_deref().map(place);
        self.eval.control = self.eval.control.as_deref().map(place);
        let local =
            |p: &mut String| *p = join(Path::new(p.as_str())).to_string_lossy().into_owned();
        for t in &mut self.eval.tasks {
            local(&mut t.prompts);
        }
        if let Some(d) = &mut self.dpo {
            local(&mut d.train);
        }
        if let Some(g) = &mut self.grpo {
            local(&mut g.prompts);
        }
        for source in &mut self.data.sources {
            source.path = place(&source.path);
        }
        self.data.cache = self.data.cache.as_deref().map(join);
    }

    /// Checks the values.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] naming the first value refused.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let fail = |message: String| Err(ConfigError(message));
        if !matches!(self.model.init.as_str(), "scratch" | "weights") {
            return fail(format!(
                "model.init: \"scratch\" or \"weights\", got {:?}",
                self.model.init
            ));
        }
        if !(self.model.std > 0.0 && self.model.std.is_finite()) {
            return fail(format!("model.std: positive, got {}", self.model.std));
        }
        let b = &self.batch;
        if b.rows == 0 || b.length == 0 {
            return fail("batch.rows and batch.length must be positive".into());
        }
        if let Some(t) = b.tokens
            && (t == 0 || !t.is_multiple_of(b.pass()))
        {
            return fail(format!(
                "batch.tokens: a positive multiple of rows × length = {}, got {t}",
                b.pass()
            ));
        }
        let d = &self.data;
        let tokens = self.dpo.is_none() && self.grpo.is_none();
        if self.dpo.is_some() && self.grpo.is_some() {
            return fail("dpo and grpo: one method a run".into());
        }
        if tokens && d.train.is_some() == !d.sources.is_empty() {
            return fail("data: either train or sources, not both".into());
        }
        if !tokens && (d.train.is_some() || !d.sources.is_empty()) {
            return fail("data.train and data.sources: not with dpo or grpo".into());
        }
        if let Some(dpo) = &self.dpo {
            dpo.validate().map_err(ConfigError)?;
            if !b.rows.is_multiple_of(2) {
                return fail(format!(
                    "batch.rows: even for dpo, a pair a two rows, got {}",
                    b.rows
                ));
            }
        }
        if let Some(g) = &self.grpo {
            g.validate().map_err(ConfigError)?;
            if g.max_tokens >= b.length {
                return fail(format!(
                    "grpo.max_tokens: below batch.length ({}), the prompt before",
                    b.length
                ));
            }
        }
        for t in &self.eval.tasks {
            t.validate().map_err(ConfigError)?;
        }
        if let Some(s) = d
            .sources
            .iter()
            .find(|s| !(s.weight > 0.0 && s.weight.is_finite()))
        {
            return fail(format!("data.sources {:?}: weight {}", s.path, s.weight));
        }
        if d.prefetch == 0 {
            return fail("data.prefetch: at least 1".into());
        }
        let o = &self.optimizer;
        if !matches!(o.kind.as_str(), "adamw" | "muon" | "adafactor") {
            return fail(format!(
                "optimizer.kind: \"adamw\", \"muon\" or \"adafactor\", got {:?}",
                o.kind
            ));
        }
        if !matches!(o.newton_schulz.as_str(), "default" | "short") {
            return fail(format!(
                "optimizer.newton_schulz: \"default\" or \"short\", got {:?}",
                o.newton_schulz
            ));
        }
        for (name, value) in [("lr", o.lr), ("muon_lr_scale", o.muon_lr_scale)] {
            if !(value > 0.0 && value.is_finite()) {
                return fail(format!("optimizer.{name}: positive, got {value}"));
            }
        }
        for (name, value) in [
            ("beta1", o.beta1),
            ("beta2", o.beta2),
            ("momentum", o.momentum),
        ] {
            if !(0.0..1.0).contains(&value) {
                return fail(format!("optimizer.{name}: in [0, 1), got {value}"));
            }
        }
        if o.eps.is_nan() || o.eps <= 0.0 || o.weight_decay.is_nan() || o.weight_decay < 0.0 {
            return fail("optimizer.eps must be positive and weight_decay not negative".into());
        }
        for g in &o.groups {
            if g.matches.is_empty() {
                return fail("optimizer.groups: an empty pattern".into());
            }
            if let Some(s) = g.lr_scale
                && !(s >= 0.0 && s.is_finite())
            {
                return fail(format!("optimizer.groups {:?}: lr_scale {s}", g.matches));
            }
        }
        let s = &self.schedule;
        if !matches!(s.kind.as_str(), "constant" | "cosine" | "wsd" | "linear") {
            return fail(format!(
                "schedule.kind: \"constant\", \"cosine\", \"wsd\" or \"linear\", got {:?}",
                s.kind
            ));
        }
        if let Some(budget) = self.training.budget {
            if !(budget > 0.0 && budget.is_finite()) {
                return fail(format!("training.budget: positive, got {budget}"));
            }
            if !tokens {
                return fail("training.budget: for a run of tokens, not dpo or grpo".into());
            }
        } else if s.steps == 0 {
            return fail("schedule.steps: positive, or training.budget".into());
        }
        if !self.model.set.is_empty() && self.model.init != "scratch" {
            return fail("model.set: for a model trained from scratch only".into());
        }
        if s.steps > 0 && s.warmup > s.steps {
            return fail(format!(
                "schedule: {} steps with {} of warmup",
                s.steps, s.warmup
            ));
        }
        if !(0.0..=1.0).contains(&s.r#final) || !(0.0..=1.0).contains(&s.decay) {
            return fail("schedule.final and schedule.decay: in [0, 1]".into());
        }
        let t = &self.training;
        if !matches!(t.dtype.as_str(), "bf16" | "f16" | "f32") {
            return fail(format!(
                "training.dtype: \"bf16\", \"f16\" or \"f32\", got {:?}",
                t.dtype
            ));
        }
        if !(t.device == "cpu" || t.device == "cuda" || t.device.starts_with("cuda:")) {
            return fail(format!(
                "training.device: \"cpu\" or \"cuda\", got {:?}",
                t.device
            ));
        }
        if !(t.clip >= 0.0 && t.clip.is_finite()) || t.chunk == 0 {
            return fail("training.clip must not be negative, and training.chunk positive".into());
        }
        if let Some(l) = &self.lora {
            l.validate().map_err(ConfigError)?;
        }
        if self.eval.tokens > 0 && self.eval.tokens < b.pass() {
            return fail(format!(
                "eval.tokens: at least one pass ({}), got {}",
                b.pass(),
                self.eval.tokens
            ));
        }
        Ok(())
    }

    /// The type of the computation.
    pub fn dtype(&self) -> kapel_core::DType {
        match self.training.dtype.as_str() {
            "f32" => kapel_core::DType::F32,
            "f16" => kapel_core::DType::F16,
            _ => kapel_core::DType::BF16,
        }
    }
}

/// Whether `path` reads as a model of the Hub, `owner/name`.
pub fn is_hub_name(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let parts: Vec<&str> = text.split('/').collect();
    path.is_relative()
        && parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && !p.starts_with('.') && !p.ends_with(".json"))
}

/// Whether `name` matches `pattern`, where `*` stands for any text.
pub fn matches(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || name.len() < first.len() + last.len() || !name.ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        [model]
        config = "gpt2.json"
        [data]
        train = "train_*.bin"
        [schedule]
        steps = 100
    "#;

    #[test]
    fn a_minimal_configuration_takes_the_defaults() {
        let c = TrainConfig::parse(MINIMAL).unwrap();
        assert_eq!(c.batch.pass(), 16 * 1024);
        assert_eq!(c.batch.accumulate(), 1);
        assert_eq!(c.optimizer.kind, "adamw");
        assert_eq!(c.schedule.kind, "cosine");
        assert_eq!(c.training.dtype, "bf16");
        // The text saved with a checkpoint reads back the same.
        assert_eq!(TrainConfig::parse(&c.to_toml()).unwrap(), c);
    }

    #[test]
    fn mistakes_are_refused_by_name() {
        let refused = |extra: &str| {
            TrainConfig::parse(&format!("{MINIMAL}\n{extra}"))
                .unwrap_err()
                .to_string()
        };
        let e = TrainConfig::parse(&MINIMAL.replace("steps = 100", "steps = 100\nstep = 3"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("step"), "{e}");
        assert!(refused("[batch]\ntokens = 1000").contains("batch.tokens"));
        assert!(refused("[optimizer]\nkind = \"sgd\"").contains("optimizer.kind"));
        assert!(refused("[optimizer]\nbeta2 = 1.0").contains("beta2"));
        assert!(refused("[training]\ndtype = \"f8\"").contains("training.dtype"));
    }

    #[test]
    fn the_recipes_are_valid_configurations() {
        let recipes = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
        let mut read = 0;
        for entry in std::fs::read_dir(&recipes).unwrap() {
            let path = entry.unwrap().path().join("train.toml");
            if path.exists() {
                let text = std::fs::read_to_string(&path).unwrap();
                TrainConfig::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                read += 1;
            }
        }
        assert!(read >= 3, "{read} recipes");
    }

    #[test]
    fn patterns_match_names() {
        assert!(matches("*.bias", "transformer.h.0.attn.c_attn.bias"));
        assert!(matches("transformer.wte.*", "transformer.wte.weight"));
        assert!(matches("*.h.*.mlp.*", "transformer.h.3.mlp.c_fc.weight"));
        assert!(!matches("*.bias", "transformer.h.0.ln_1.weight"));
        assert!(matches("lm_head.weight", "lm_head.weight"));
        assert!(!matches("a*a", "a"));
    }
}
