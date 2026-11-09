//! The forward pass of a decoder, on any backend.

use std::collections::BTreeMap;

use kapel_core::{DType, Tensor};
use kapel_hub::Checkpoint;
use kapel_ops::{
    Affine, Attention, Backend, Binary, BinaryKind, Cast, Concat, Contiguous, GatherRows, Glu,
    LayerNorm, Matmul, PagedAttention, QuantizedMatmul, Recording, RmsNorm, Rope, Transfer, Unary,
    UnaryKind,
};

use crate::ModelError;
use crate::cache::{KvCache, PAGE, SequenceId, Step};
use crate::config::{Block, Config, Mixer, Norm};

/// A model: its configuration and its weights, by their names in the checkpoint.
/// A clone shares the weights.
#[derive(Clone)]
pub struct Model {
    pub(crate) config: Config,
    /// `config.json` as read, written back unchanged by [`Model::save`].
    pub(crate) raw_config: serde_json::Value,
    pub(crate) weights: BTreeMap<String, Tensor>,
}

type Result<T> = std::result::Result<T, ModelError>;

/// The keys and values of the positions a model has already seen, for one
/// sequence: what lets [`Model::step`] compute only the new positions. A
/// [`KvCache`] that holds one sequence, and grows with it.
#[derive(Debug)]
pub struct Cache {
    inner: KvCache,
    id: SequenceId,
}

impl Cache {
    /// An empty cache.
    pub fn new() -> Self {
        let mut inner = KvCache::new();
        let id = inner.add_sequence();
        Self { inner, id }
    }

    /// An empty cache whose tables are of type `dtype` (see
    /// [`KvCache::with_dtype`]).
    pub fn with_dtype(dtype: DType) -> Self {
        let mut inner = KvCache::new().with_dtype(dtype);
        let id = inner.add_sequence();
        Self { inner, id }
    }

    /// The number of positions it holds.
    pub fn len(&self) -> usize {
        self.inner.len(self.id)
    }

    /// Whether it holds no position.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The next token after `token`, the last one given, through a recorded
    /// pass ([`Model::record_step`], [`Model::replay_step`]): `recorded`
    /// holds the recording, made or made again when needed. The logits `[1,
    /// vocab]`, or `None` for a backend that does not record; nothing is
    /// computed then.
    ///
    /// # Errors
    ///
    /// As [`Model::step`].
    pub fn recorded_step<B: Backend + Transfer>(
        &mut self,
        model: &Model,
        backend: &B,
        recorded: &mut Option<RecordedStep>,
        token: u32,
    ) -> Result<Option<Tensor>> {
        let batch = [(self.id, token)];
        if let Some(step) = recorded
            && let Some(logits) = model.replay_step(backend, &mut self.inner, step, &batch)?
        {
            return Ok(Some(logits));
        }
        let pages = (self.len() + ROOM + 1).div_ceil(PAGE);
        let made = model.record_step(backend, &mut self.inner, &batch, 1, pages)?;
        let (step, logits) = match made {
            Recorded::Made(step, logits) => (Some(*step), Some(logits)),
            Recorded::Unrecorded(logits, _) => (None, Some(logits)),
            Recorded::Nothing => (None, None),
        };
        *recorded = step;
        Ok(logits)
    }

    /// Brings it back to its first `len` positions (see
    /// [`KvCache::truncate`]).
    ///
    /// # Panics
    ///
    /// If it holds fewer than `len` positions.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(self.id, len);
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

/// The positions a recorded step of a [`Cache`] has room for, beyond the one
/// it computes, before it is recorded again.
pub const ROOM: usize = 256;

/// The inputs of a recorded step, rewritten before each replay.
#[derive(Debug)]
struct Inputs {
    ids: Tensor,
    positions: Tensor,
    lines: Tensor,
    pages: Tensor,
    rows: Tensor,
    slots: Tensor,
}

/// A decoding pass recorded once ([`Model::record_step`]) and replayed for
/// the next tokens of any sequences of the same cache, as many as its rows
/// or fewer ([`Model::replay_step`]).
#[derive(Debug)]
pub struct RecordedStep {
    /// The cache, and its tables, the recording writes into.
    identity: (u64, u64),
    /// Its rows: one token of one sequence each.
    rows: usize,
    /// Pages per sequence of the table of pages: the room of the recording.
    widest: usize,
    inputs: Inputs,
    recording: Recording,
}

impl RecordedStep {
    /// The most sequences it computes.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The pages per sequence it reads: a sequence that needs more is
    /// recorded again.
    pub fn pages(&self) -> usize {
        self.widest
    }
}

/// What [`Model::record_step`] did.
#[derive(Debug)]
pub enum Recorded {
    /// Nothing: the backend does not record, or the cache lacks the room.
    /// Nothing was computed.
    Nothing,
    /// The step was computed (its logits `[S, vocab]`) and recorded.
    Made(Box<RecordedStep>, Tensor),
    /// The step was computed (its logits), but the backend failed to record
    /// it: out of memory, say. The error says why.
    Unrecorded(Tensor, ModelError),
}

/// Which positions get logits.
#[derive(Clone, PartialEq, Eq)]
enum Logits {
    All,
    /// The last of each sequence of a batch.
    Last,
    /// These rows.
    Rows(Vec<usize>),
}

/// The input of linear projections of a block, as [`Model::block`] shows
/// them: what the projections that read it multiply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    /// The normed hidden states, read by the query, key and value
    /// projections.
    Attention,
    /// The attention's output, read by the output projection.
    Output,
    /// The normed hidden states after the attention, read by the gate and up
    /// projections.
    Mlp,
    /// The gated unit's output, read by the down projection.
    Down,
}

impl Site {
    /// The four, in the order a block reads them.
    pub const ALL: [Site; 4] = [Site::Attention, Site::Output, Site::Mlp, Site::Down];

    /// The projections that read it, by their names in a block of Llama's
    /// kind (after `model.layers.{i}.`).
    pub fn projections(self) -> &'static [&'static str] {
        match self {
            Site::Attention => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            Site::Output => &["self_attn.o_proj"],
            Site::Mlp => &["mlp.gate_proj", "mlp.up_proj"],
            Site::Down => &["mlp.down_proj"],
        }
    }
}

/// What [`Model::block`] calls with each input of projections.
pub type Observer<'a> = dyn FnMut(Site, &Tensor) -> Result<()> + 'a;

/// The operations of the forward pass, on one backend, and who watches the
/// inputs of the projections.
struct Ops<'a, B>(&'a B, Option<std::cell::RefCell<&'a mut Observer<'a>>>);

impl<B: Backend> Ops<'_, B> {
    /// Shows `x`, the input of the projections at `site`, to the observer.
    fn see(&self, site: Site, x: &Tensor) -> Result<()> {
        match &self.1 {
            Some(observer) => (observer.borrow_mut())(site, x),
            None => Ok(()),
        }
    }

    fn linear(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        // `transformers` stores `[out, in]`: `y = x · wᵀ`, a view. Quantized
        // weights keep that layout, their blocks along `in`.
        if w.dtype().is_quantized() {
            let y = self.0.forward(&QuantizedMatmul, &[x, w])?;
            return self.bias(y, bias);
        }
        self.product(x, &w.transpose(0, 1)?, bias)
    }

    /// `x · w (+ bias)`, the bias added in the product when it has the type
    /// of `x`.
    fn product(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        match bias {
            Some(b) if b.dtype() == x.dtype() => Ok(self.0.forward(&Matmul, &[x, w, b])?),
            _ => self.bias(self.0.forward(&Matmul, &[x, w])?, bias),
        }
    }

    fn bias(&self, y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        Ok(match bias {
            Some(b) => self.0.forward(
                &Binary {
                    kind: BinaryKind::Add,
                },
                &[&y, b],
            )?,
            None => y,
        })
    }

    fn binary(&self, kind: BinaryKind, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Binary { kind }, &[a, b])?)
    }

    fn unary(&self, kind: UnaryKind, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Unary { kind }, &[x])?)
    }

    fn affine(&self, x: &Tensor, scale: f64) -> Result<Tensor> {
        Ok(self.0.forward(&Affine { scale, bias: 0.0 }, &[x])?)
    }

    /// `x` with its elements in row-major order, on the backend.
    fn contiguous(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Contiguous, &[x])?)
    }

    /// `cap · tanh(x / cap)`.
    fn softcap(&self, x: &Tensor, cap: f64) -> Result<Tensor> {
        let t = self.unary(UnaryKind::Tanh, &self.affine(x, 1.0 / cap)?)?;
        self.affine(&t, cap)
    }
}

impl Model {
    /// Reads the configuration and maps the weights of a model directory.
    ///
    /// # Errors
    ///
    /// For an unsupported architecture, or missing or unreadable weights.
    pub fn load(checkpoint: &Checkpoint) -> Result<Self> {
        let raw_config = checkpoint.config()?;
        let config = Config::from_json(&raw_config)?;
        let mut weights: BTreeMap<String, Tensor> = checkpoint
            .names()
            .map(|name| Ok((name.to_string(), checkpoint.tensor(name)?)))
            .collect::<std::result::Result<_, kapel_hub::HubError>>()?;
        if config.block == Block::Gpt2 {
            weights = gpt2_names(weights);
        }
        weights = decoder_names(weights);
        Ok(Self {
            config,
            raw_config,
            weights,
        })
    }

    /// A model of its parsed and raw configurations and its weights.
    pub(crate) fn from_parts(
        config: Config,
        raw_config: serde_json::Value,
        weights: BTreeMap<String, Tensor>,
    ) -> Self {
        Self {
            config,
            raw_config,
            weights,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// A model of the configuration `config.json` holds, `raw_config`, and
    /// of `weights`, by their names in the checkpoint: a model made or
    /// trained in memory. The weights are not checked against the
    /// configuration.
    ///
    /// # Errors
    ///
    /// For a configuration [`Config::from_json`] refuses.
    pub fn new(raw_config: serde_json::Value, weights: BTreeMap<String, Tensor>) -> Result<Self> {
        let config = Config::from_json(&raw_config)?;
        Ok(Self::from_parts(config, raw_config, weights))
    }

    /// The same configuration with other weights: a model trained from
    /// this one.
    ///
    /// # Errors
    ///
    /// Never for now; kept for the checks of the weights to come.
    pub fn with_weights(&self, weights: BTreeMap<String, Tensor>) -> Result<Self> {
        Ok(Self::from_parts(
            self.config.clone(),
            self.raw_config.clone(),
            weights,
        ))
    }

    /// `config.json` as it was read.
    pub fn raw_config(&self) -> &serde_json::Value {
        &self.raw_config
    }

    /// The model with its weights on `backend`'s device, ready for
    /// [`Model::forward`] on it.
    ///
    /// # Errors
    ///
    /// If a weight cannot be copied to the device.
    pub fn upload<B: Transfer>(&self, backend: &B) -> Result<Self> {
        self.map(|w| Ok(backend.upload(w)?))
    }

    /// The model with its float weights converted to `dtype` on `backend`, to
    /// compute in that type (`bf16` on a card, for instance). Quantized and
    /// integer weights are kept.
    ///
    /// # Errors
    ///
    /// If the backend cannot convert a weight.
    pub fn cast<B: Backend>(&self, backend: &B, dtype: DType) -> Result<Self> {
        self.map(|w| {
            Ok(if w.dtype().is_float() && w.dtype() != dtype {
                backend.forward(&Cast { dtype }, &[w])?
            } else {
                w.clone()
            })
        })
    }

    /// The model with the query, key and value projections of each layer
    /// stacked into one weight (`qkv_proj`), and the gate and up projections
    /// into another (`gate_up_proj`), on `backend`: one product instead of
    /// three, and one instead of two followed by a gated unit. The rows of
    /// the products are the same; a card reads larger weights faster, and
    /// launches fewer kernels.
    ///
    /// Only the blocks that have them (Llama, Mistral, Qwen, Gemma), and only
    /// float weights of one type: quantized ones keep their projections.
    ///
    /// # Errors
    ///
    /// If the backend cannot stack the weights.
    pub fn fused<B: Backend>(&self, backend: &B) -> Result<Self> {
        self.map(|w| Ok(w.clone()))?.into_fused(backend)
    }

    /// [`Model::fused`], taking the model: each projection's weights are
    /// let go once stacked, so that the model never takes twice its memory.
    ///
    /// # Errors
    ///
    /// As [`Model::fused`].
    pub fn into_fused<B: Backend>(mut self, backend: &B) -> Result<Self> {
        if !matches!(self.config.block, Block::Sequential | Block::Sandwich) {
            return Ok(self);
        }
        // Whether the weights `names` exist, float, of one type and width.
        let stackable = |weights: &BTreeMap<String, Tensor>, names: &[String]| {
            let parts: Option<Vec<&Tensor>> = names.iter().map(|n| weights.get(n)).collect();
            parts.is_some_and(|parts| {
                let (dtype, width) = (parts[0].dtype(), parts[0].shape().get(1));
                dtype.is_float()
                    && parts
                        .iter()
                        .all(|w| w.dtype() == dtype && w.shape().get(1) == width)
            })
        };
        let stack = |weights: &mut BTreeMap<String, Tensor>,
                     names: &[String],
                     into: String|
         -> Result<()> {
            let parts: Vec<Tensor> = names.iter().filter_map(|n| weights.remove(n)).collect();
            let refs: Vec<&Tensor> = parts.iter().collect();
            let stacked = backend.forward(&Concat { dim: 0 }, &refs)?;
            weights.insert(into, stacked);
            Ok(())
        };
        let blocks: Vec<String> = (0..self.config.layers)
            .map(|layer| format!("model.layers.{layer}"))
            .chain((0..self.config.multi_token_layers).map(|i| format!("model.mtp_layers.{i}")))
            .collect();
        let with_biases = self.config.qkv_bias;
        for p in blocks {
            let attn = |m: &str| format!("{p}.self_attn.{m}");
            let names: Vec<String> = ["q_proj", "k_proj", "v_proj"]
                .iter()
                .map(|m| attn(&format!("{m}.weight")))
                .collect();
            let biases: Vec<String> = ["q_proj", "k_proj", "v_proj"]
                .iter()
                .map(|m| attn(&format!("{m}.bias")))
                .collect();
            // With biases, the weights and the biases are stacked together.
            let w = &mut self.weights;
            if stackable(w, &names) && (!with_biases || stackable(w, &biases)) {
                stack(w, &names, attn("qkv_proj.weight"))?;
                if with_biases {
                    stack(w, &biases, attn("qkv_proj.bias"))?;
                }
            }
            let mlp: Vec<String> = ["gate_proj", "up_proj"]
                .iter()
                .map(|m| format!("{p}.mlp.{m}.weight"))
                .collect();
            let mlp_biases: Vec<String> = ["gate_proj", "up_proj"]
                .iter()
                .map(|m| format!("{p}.mlp.{m}.bias"))
                .collect();
            let mlp_bias = self.config.mlp_bias;
            if stackable(w, &mlp) && (!mlp_bias || stackable(w, &mlp_biases)) {
                stack(w, &mlp, format!("{p}.mlp.gate_up_proj.weight"))?;
                if mlp_bias {
                    stack(w, &mlp_biases, format!("{p}.mlp.gate_up_proj.bias"))?;
                }
            }
        }
        Ok(self)
    }

    /// The model with `f` applied to each weight.
    fn map(&self, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Self> {
        self.map_named(|_, w| f(w))
    }

    fn map_named(&self, f: impl Fn(&str, &Tensor) -> Result<Tensor>) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            raw_config: self.raw_config.clone(),
            weights: self
                .weights
                .iter()
                .map(|(name, w)| Ok((name.clone(), f(name, w)?)))
                .collect::<Result<_>>()?,
        })
    }

    /// The model with the weights of its linear projections (attention and
    /// MLP) quantized to `dtype` by rounding to the nearest
    /// ([`kapel_core::quantize`]: Kapel's int8 and int4 by groups, GGUF's
    /// Q8_0 and Q4_0), and with `output` its output table too (the
    /// embeddings, when the model ties them to it). The products then read
    /// the codes in place: a fraction of the bytes of the float weights.
    ///
    /// Weights whose rows are not a whole number of blocks, the norms and
    /// biases, and GPT-2's projections (stored transposed) keep their type.
    /// The weights must be in host memory: a model as loaded.
    ///
    /// # Errors
    ///
    /// If `dtype` is not one of those formats, or a weight is not in host
    /// memory.
    pub fn quantize(&self, dtype: DType, output: bool) -> Result<Self> {
        const PROJECTIONS: [&str; 10] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "dense",
            "fc1",
            "fc2",
        ];
        let head = if self.config.tie_embeddings || !self.weights.contains_key("lm_head.weight") {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let chosen = |name: &str| {
            let module = name
                .strip_suffix(".weight")
                .and_then(|n| n.rsplit('.').next());
            module.is_some_and(|m| PROJECTIONS.contains(&m)) || (output && name == head)
        };
        self.map_named(|name, w| {
            let fits = w.rank() == 2 && w.shape()[1].is_multiple_of(dtype.block_len());
            if chosen(name) && fits && w.dtype().is_float() {
                Ok(quantize_matrix(w, dtype)?)
            } else {
                Ok(w.clone())
            }
        })
    }

    /// Writes the model to `dir` in the layout of the Hub: `config.json` and
    /// `model.safetensors`, which `transformers` loads back. The weights must
    /// be in host memory: a model as loaded, not [uploaded](Model::upload).
    ///
    /// `dir` may be the directory the model was loaded from: each file is
    /// written under a temporary name and renamed into place
    /// ([`kapel_hub::write_atomic`]), so the weights mapped from the old file
    /// stay valid. A `model.safetensors.index.json` in `dir` would take
    /// precedence over the new `model.safetensors` when loading: it is removed
    /// once the weights are written (the shards it listed are left).
    ///
    /// # Errors
    ///
    /// If the directory or the files cannot be written.
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<()> {
        let dir = dir.as_ref();
        let io = |path: std::path::PathBuf, source: std::io::Error| {
            ModelError::Hub(kapel_hub::HubError::Io { path, source })
        };
        std::fs::create_dir_all(dir).map_err(|e| io(dir.to_path_buf(), e))?;
        let json = serde_json::to_vec_pretty(&self.raw_config).expect("a JSON value serializes");
        kapel_hub::write_atomic(dir.join("config.json"), |w| w.write_all(&json))?;
        let tensors: Vec<(&str, &Tensor)> =
            self.weights.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        kapel_hub::safetensors::save(dir.join("model.safetensors"), &tensors, &metadata)?;
        let index = dir.join("model.safetensors.index.json");
        match std::fs::remove_file(&index) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io(index, e)),
            _ => Ok(()),
        }
    }

    fn weight(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }

    fn optional(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    fn norm<B: Backend>(&self, ops: &Ops<'_, B>, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.weight(&format!("{prefix}.weight"))?;
        Ok(match self.config.norm {
            Norm::Rms { offset } => ops.0.forward(
                &RmsNorm {
                    eps: self.config.eps,
                    weight_offset: offset,
                },
                &[x, w],
            )?,
            Norm::Layer => {
                let b = self.weight(&format!("{prefix}.bias"))?;
                ops.0.forward(
                    &LayerNorm {
                        eps: self.config.eps,
                    },
                    &[x, w, b],
                )?
            }
        })
    }

    /// The name of the embedding table.
    fn embed_name(&self) -> &'static str {
        match self.config.block {
            Block::Gpt2 => "transformer.wte.weight",
            _ => "model.embed_tokens.weight",
        }
    }

    /// The hidden states of `ids_tensor` before the first block: their rows
    /// of the embedding table, scaled or with their positions added as the
    /// architecture says.
    fn embedded<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids_tensor: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.config;
        let ops = Ops(backend, None);
        let embed = self.weight(self.embed_name())?;
        let mut x = backend.forward(&GatherRows, &[embed, ids_tensor])?;
        // The rows of a quantized table come in `f32`: the hidden states take
        // the type of the float weights (the norms'), `bf16` on a card.
        let final_norm = match c.block {
            Block::Gpt2 => "transformer.ln_f.weight",
            Block::Parallel => "model.final_layernorm.weight",
            _ => "model.norm.weight",
        };
        if embed.dtype().is_quantized()
            && let Some(norm) = self.optional(final_norm)
            && norm.dtype() != x.dtype()
            && norm.dtype().is_float()
        {
            x = backend.forward(
                &Cast {
                    dtype: norm.dtype(),
                },
                &[&x],
            )?;
        }
        if let Some(scale) = c.embed_scale {
            // Gemma rounds the factor to the type of the hidden states.
            x = ops.affine(&x, rounded(x.dtype(), scale)?)?;
        }
        if c.block == Block::Gpt2 {
            let wpe = backend.forward(
                &GatherRows,
                &[self.weight("transformer.wpe.weight")?, positions],
            )?;
            x = ops.binary(BinaryKind::Add, &x, &wpe)?;
        }

        Ok(x)
    }

    /// The hidden states `[T, hidden]` of the token ids `ids`, one sequence,
    /// before the first block: what [`Model::block`] takes.
    ///
    /// # Errors
    ///
    /// If a weight is missing, or an operation fails on the backend.
    pub fn embed<B: Backend + Transfer>(&self, backend: &B, ids: &[u32]) -> Result<Tensor> {
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let positions = backend.upload(&Tensor::from_slice(&positions, &[ids.len()])?)?;
        let ids = backend.upload(&Tensor::from_slice(ids, &[ids.len()])?)?;
        self.embedded(backend, &ids, &positions)
    }

    /// The hidden states `x` `[T, hidden]` of one sequence through block
    /// `layer`, without a cache, every position seeing the ones before it:
    /// the forward pass one block at a time. `observe` sees the input of
    /// each group of projections ([`Site`]) as it is computed; an error it
    /// returns stops the pass.
    ///
    /// Blocks of Llama's kind only (Llama, Mistral, Qwen, Gemma), not
    /// [fused](Model::fused): neither experts nor linear attention.
    ///
    /// # Errors
    ///
    /// For another kind of block, a missing weight, an operation that fails
    /// on the backend, or an error of `observe`.
    pub fn block<B: Backend + Transfer>(
        &self,
        backend: &B,
        layer: usize,
        x: &Tensor,
        observe: &mut Observer<'_>,
    ) -> Result<Tensor> {
        let c = &self.config;
        if !matches!(c.block, Block::Sequential | Block::Sandwich)
            || c.experts.is_some()
            || c.linear.is_some()
        {
            return Err(ModelError::Config(
                "Model::block computes blocks of Llama's kind only: an attention and an MLP".into(),
            ));
        }
        if layer >= c.layers {
            return Err(ModelError::Config(format!(
                "block {layer} of a model of {} layers",
                c.layers
            )));
        }
        let t = x.shape()[0];
        let positions: Vec<u32> = (0..t as u32).collect();
        let positions = backend.upload(&Tensor::from_slice(&positions, &[t])?)?;
        // The observer's lifetime is shortened to this call.
        let observe: &mut Observer<'_> = observe;
        let ops = Ops(backend, Some(std::cell::RefCell::new(observe)));
        self.sequential(
            &ops,
            x,
            &positions,
            &mut None,
            layer,
            &format!("model.layers.{layer}"),
            c.windows[layer],
        )
    }

    /// A weight, by its name in the checkpoint.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// The names of the weights, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(String::as_str)
    }

    /// Puts `tensor` in place of the weight `name`, or adds it: the weight
    /// it replaces, if any. Its shape is not checked against the
    /// configuration.
    pub fn replace(&mut self, name: &str, tensor: Tensor) -> Option<Tensor> {
        self.weights.insert(name.to_string(), tensor)
    }

    /// The logits `[T, vocab]` of the token ids `ids`, one sequence, without a
    /// cache: every position sees the ones before it. Every operation has a
    /// gradient: the pass of training.
    ///
    /// # Errors
    ///
    /// If a weight is missing, or an operation fails on the backend.
    pub fn forward<B: Backend + Transfer>(&self, backend: &B, ids: &[u32]) -> Result<Tensor> {
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let positions = backend.upload(&Tensor::from_slice(&positions, &[ids.len()])?)?;
        let ids = backend.upload(&Tensor::from_slice(ids, &[ids.len()])?)?;
        self.run(backend, &ids, &positions, None, Logits::All)
    }

    /// The logits `[1, vocab]` of the last of `ids`, which follow the positions
    /// of `cache`; their keys and values are added to it. Each new position
    /// sees the cached ones and the new ones before it: a prompt is given at
    /// once (the prefill), then the generated tokens one at a time.
    ///
    /// # Errors
    ///
    /// If `ids` is empty, a weight is missing, or an operation fails on the
    /// backend. On an error, the cache may hold some layers of the failed step:
    /// start again from a new one.
    pub fn step<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.step_batch(backend, &mut cache.inner, &[(cache.id, ids)])
    }

    /// Like [`Model::step`], with the logits `[T, vocab]` of every one of
    /// `ids`: to score a text by pieces, or to check proposed tokens.
    ///
    /// # Errors
    ///
    /// As [`Model::step`].
    pub fn extend<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.extend_batch(backend, &mut cache.inner, &[(cache.id, ids)])
    }

    /// Like [`Model::extend`], with the logits `[last, vocab]` of the last
    /// `last` of `ids` only: to check proposed tokens after a prompt, without
    /// the logits of the prompt.
    ///
    /// # Errors
    ///
    /// As [`Model::step`], and if `last` is 0 or more than `ids` has.
    pub fn extend_last<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
        last: usize,
    ) -> Result<Tensor> {
        if last == 0 || last > ids.len() {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "extend_last",
                message: format!("the last {last} of {} tokens", ids.len()),
            }));
        }
        let rows = (ids.len() - last..ids.len()).collect();
        self.run_batch(
            backend,
            &mut cache.inner,
            &[(cache.id, ids)],
            Logits::Rows(rows),
        )
    }

    /// Like [`Model::extend_last`], with the final hidden states `[T,
    /// hidden]` of every one of `ids` too, before the final norm: what the
    /// model's multi-token layer reads ([`Model::multi_token`]).
    ///
    /// # Errors
    ///
    /// As [`Model::extend_last`].
    pub fn extend_hidden<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
        last: usize,
    ) -> Result<(Tensor, Tensor)> {
        if last == 0 || last > ids.len() {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "extend_hidden",
                message: format!("the last {last} of {} tokens", ids.len()),
            }));
        }
        let rows = (ids.len() - last..ids.len()).collect();
        let (logits, hidden) = self.run_batch_with(
            backend,
            &mut cache.inner,
            &[(cache.id, ids)],
            Logits::Rows(rows),
            true,
        )?;
        Ok((logits, hidden.expect("asked for")))
    }

    /// The multi-token prediction layer (MiMo, after DeepSeek-V3), over the
    /// positions that follow those of `cache`, its own: at position `i`, the
    /// layer reads the model's final hidden state `hidden[i]` (from
    /// [`Model::extend_hidden`]) and the token `tokens[i]` that follows the
    /// position, and predicts the token after that. The logits `[1, vocab]`
    /// of the last position: a proposal for the token after `tokens`.
    ///
    /// # Errors
    ///
    /// If the model has no multi-token layer, `hidden` and `tokens` differ
    /// in length, or an operation fails.
    pub fn multi_token<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut Cache,
        hidden: &Tensor,
        tokens: &[u32],
    ) -> Result<Tensor> {
        let c = &self.config;
        let ops = Ops(backend, None);
        let n = tokens.len();
        if c.multi_token_layers == 0 || n == 0 || hidden.shape()[0] != n {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "multi_token",
                message: format!(
                    "{} multi-token layers, {n} tokens, {} hidden states",
                    c.multi_token_layers,
                    hidden.shape()[0]
                ),
            }));
        }
        let p = "model.mtp_layers.0";
        let first = cache.len();
        let step = cache.inner.begin(backend, &[(cache.id, tokens)])?;
        let ids = backend.upload(&Tensor::from_slice(tokens, &[n])?)?;
        let mut e = backend.forward(
            &GatherRows,
            &[self.weight("model.embed_tokens.weight")?, &ids],
        )?;
        if e.dtype() != hidden.dtype() {
            e = backend.forward(
                &Cast {
                    dtype: hidden.dtype(),
                },
                &[&e],
            )?;
        }
        if first == 0 {
            // The first position has no token before it to predict from:
            // its embedding is left out, as vLLM leaves it.
            let zero = backend.zeros(&[1, c.hidden], e.dtype())?;
            e = if n == 1 {
                zero
            } else {
                backend.forward(&Concat { dim: 0 }, &[&zero, &e.narrow(0, 1, n - 1)?])?
            };
        }
        let e = self.norm(&ops, &e, &format!("{p}.token_layernorm"))?;
        let h = self.norm(&ops, hidden, &format!("{p}.hidden_layernorm"))?;
        let joined = backend.forward(&Concat { dim: 1 }, &[&h, &e])?;
        let x = ops.linear(
            &joined,
            self.weight(&format!("{p}.input_proj.weight"))?,
            None,
        )?;
        let mut paged = Some((&mut cache.inner, &step));
        let x = self.sequential(&ops, &x, &step.positions, &mut paged, 0, p, None)?;
        let x = self.norm(
            &ops,
            &x.narrow(0, n - 1, 1)?,
            &format!("{p}.final_layernorm"),
        )?;
        let head = match self.optional("lm_head.weight") {
            Some(w) if !c.tie_embeddings => w,
            _ => self.weight("model.embed_tokens.weight")?,
        };
        ops.linear(&x, head, None)
    }

    /// One pass over several sequences of `cache` at once: each entry of
    /// `batch` gives a sequence and its new tokens, which follow its cached
    /// positions and whose keys and values are added to it. The logits `[S,
    /// vocab]` of each entry's last token, in the order of `batch`.
    ///
    /// The products then have one row per new token of every sequence: the
    /// weights are read once for the whole batch, which is what makes a batch
    /// cheaper than its sequences one at a time.
    ///
    /// # Errors
    ///
    /// If an entry has no token, a sequence appears twice, `cache` is full
    /// ([`ModelError::CacheFull`]), a weight is missing, or an operation
    /// fails on the backend. On an error, the sequences of the batch may hold
    /// part of the failed pass: remove them.
    pub fn step_batch<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        batch: &[(SequenceId, &[u32])],
    ) -> Result<Tensor> {
        self.run_batch(backend, cache, batch, Logits::Last)
    }

    /// Like [`Model::step_batch`], with the logits `[R, vocab]` of every new
    /// token, the entries one after the other.
    ///
    /// # Errors
    ///
    /// As [`Model::step_batch`].
    pub fn extend_batch<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        batch: &[(SequenceId, &[u32])],
    ) -> Result<Tensor> {
        self.run_batch(backend, cache, batch, Logits::All)
    }

    fn run_batch<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        batch: &[(SequenceId, &[u32])],
        logits: Logits,
    ) -> Result<Tensor> {
        Ok(self.run_batch_with(backend, cache, batch, logits, false)?.0)
    }

    fn run_batch_with<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        batch: &[(SequenceId, &[u32])],
        logits: Logits,
        hidden: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if batch.is_empty() || batch.iter().any(|(_, ids)| ids.is_empty()) {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "step",
                message: "no token to compute".into(),
            }));
        }
        let step = cache.begin(backend, batch)?;
        let ids: Vec<u32> = batch
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        let logits = match logits {
            Logits::All => Logits::All,
            Logits::Rows(rows) => Logits::Rows(rows),
            // One token each: every row is a last one.
            _ if batch.iter().all(|(_, ids)| ids.len() == 1) => Logits::All,
            _ => {
                let mut end = 0;
                Logits::Rows(
                    batch
                        .iter()
                        .map(|(_, ids)| {
                            end += ids.len();
                            end - 1
                        })
                        .collect(),
                )
            }
        };
        let positions = step.positions.clone();
        let ids = backend.upload(&Tensor::from_slice(&ids, &[ids.len()])?)?;
        self.run_with(
            backend,
            &ids,
            &positions,
            Some((cache, &step)),
            logits,
            hidden,
        )
    }

    /// One token of each sequence of `batch`, computed as
    /// [`Model::step_batch`] computes it (the logits `[S, vocab]`), and its
    /// pass recorded ([`kapel_ops::Record`]: a CUDA graph on a card) for
    /// [`Model::replay_step`] to run again for the next tokens: the pass is
    /// then launched in one call instead of one per kernel. The recording has
    /// `rows` rows, `batch` and padding after it, and reads `pages` pages per
    /// sequence; it serves any batch of this cache as large or smaller whose
    /// sequences fit in those pages. A cache that grows makes room for them
    /// first. `None` for a backend that does not record, and for a cache of
    /// fixed size without the room: [`Recorded::Nothing`], nothing computed.
    /// If the backend fails to record, the step is computed all the same
    /// ([`Recorded::Unrecorded`]).
    ///
    /// The recording reads the tables of the cache as they are: it is
    /// recorded again once the cache grows them ([`Model::replay_step`] says
    /// so).
    ///
    /// # Errors
    ///
    /// As [`Model::step_batch`].
    pub fn record_step<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        batch: &[(SequenceId, u32)],
        rows: usize,
        pages: usize,
    ) -> Result<Recorded> {
        // The tables are made by a first pass, as it comes: never while
        // recording.
        // The experts are chosen on the host, which a recording cannot wait
        // for.
        if !backend.records()
            || self.config.experts.is_some()
            || batch.is_empty()
            || batch.len() > rows
            || !cache.has_tables(self.config.layers)
        {
            return Ok(Recorded::Nothing);
        }
        let widest = batch
            .iter()
            .map(|&(id, _)| (cache.len(id) + 1).div_ceil(PAGE))
            .max()
            .unwrap_or(1)
            .max(pages);
        let ahead: Vec<(SequenceId, usize)> = batch
            .iter()
            .map(|&(id, _)| {
                let room = if cache.is_fixed() {
                    1
                } else {
                    widest * PAGE - cache.len(id)
                };
                (id, room)
            })
            .collect();
        // Lines for the padding of the smallest batch it may serve, taken
        // now: replays then never need more.
        let scratch = match cache
            .reserve(backend, &ahead)
            .and_then(|()| cache.scratch_lines(rows - 1))
        {
            Err(ModelError::CacheFull { .. }) => return Ok(Recorded::Nothing),
            other => other?,
        };
        // The scratch pages may have grown the cache: its tables too.
        cache.reserve(backend, &[])?;
        let tokens: Vec<u32> = batch.iter().map(|&(_, token)| token).collect();
        let entries: Vec<(SequenceId, &[u32])> = batch
            .iter()
            .zip(&tokens)
            .map(|(&(id, _), token)| (id, std::slice::from_ref(token)))
            .collect();
        let mut plan = cache.plan(&entries, Some(widest))?;
        plan.pad(rows, &scratch[..rows - batch.len()]);
        let upload = |values: &[u32], shape: &[usize]| -> Result<Tensor> {
            Ok(backend.upload(&Tensor::from_slice(values, shape)?)?)
        };
        let mut ids = tokens.clone();
        ids.resize(rows, 0);
        let inputs = Inputs {
            ids: upload(&ids, &[rows])?,
            positions: upload(&plan.positions, &[rows])?,
            lines: upload(&plan.lines, &[rows])?,
            pages: upload(&plan.pages, &[rows, widest])?,
            rows: upload(&plan.rows, &[rows, 2])?,
            slots: upload(&plan.slots, &[rows])?,
        };
        let lines_needed = cache.lines();
        // The sequences the first pass starts have their states zeroed; the
        // recording, of the next tokens, starts none.
        let fresh = plan.fresh.clone();
        let step = |fresh: Vec<u32>| Step {
            positions: inputs.positions.clone(),
            lines: inputs.lines.clone(),
            pages: inputs.pages.clone(),
            rows: inputs.rows.clone(),
            slots: inputs.slots.clone(),
            fresh,
            lines_needed,
        };
        // This token as it comes: what the recording needs is then made (the
        // backend's tables for these shapes, its compiled kernels), and the
        // recording, which runs nothing, need not be replayed for it.
        let logits = {
            let step = step(fresh);
            self.run(
                backend,
                &inputs.ids,
                &step.positions,
                Some((&mut *cache, &step)),
                Logits::All,
            )?
        };
        let logits = logits.narrow(0, 0, batch.len())?;
        let identity = cache.identity();
        let recording = backend.record(&mut || {
            let step = step(Vec::new());
            let logits = self.run(
                backend,
                &inputs.ids,
                &step.positions,
                Some((&mut *cache, &step)),
                Logits::All,
            );
            logits.map_err(|e| match e {
                ModelError::Tensor(e) => e,
                other => kapel_core::Error::InvalidArgument {
                    op: "record_step",
                    message: other.to_string(),
                },
            })
        });
        // The step is computed already: a recording that fails (the device
        // out of memory, say) loses only the recording.
        let recording = match recording {
            Ok(Some(recording)) => recording,
            Ok(None) => {
                let error = kapel_core::Error::InvalidArgument {
                    op: "record_step",
                    message: "the backend recorded nothing".into(),
                };
                return Ok(Recorded::Unrecorded(logits, ModelError::Tensor(error)));
            }
            Err(e) => return Ok(Recorded::Unrecorded(logits, ModelError::Tensor(e))),
        };
        if cache.identity() != identity {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "record_step",
                message: "the tables of the cache changed while recording".into(),
            }));
        }
        let recorded = RecordedStep {
            identity,
            rows,
            widest,
            inputs,
            recording,
        };
        Ok(Recorded::Made(Box::new(recorded), logits))
    }

    /// The next token of each sequence of `batch` through the recording
    /// `recorded`: the logits `[S, vocab]`, or `None` if the recording does
    /// not fit (more sequences than its rows, the cache grew its tables, a
    /// sequence needs more pages than the recording reads, or the cache
    /// has pages to copy first); nothing is computed then, and
    /// [`Model::record_step`] records again.
    ///
    /// # Errors
    ///
    /// If a sequence appears twice, or the backend fails.
    pub fn replay_step<B: Backend + Transfer>(
        &self,
        backend: &B,
        cache: &mut KvCache,
        recorded: &mut RecordedStep,
        batch: &[(SequenceId, u32)],
    ) -> Result<Option<Tensor>> {
        let counts: Vec<(SequenceId, usize)> = batch.iter().map(|&(id, _)| (id, 1)).collect();
        // A sequence that starts here would need its states zeroed, which a
        // recording does not do.
        if batch.is_empty()
            || batch.len() > recorded.rows
            || cache.identity() != recorded.identity
            || !cache.fits(&counts, recorded.widest)
            || batch.iter().any(|&(id, _)| cache.is_empty(id))
        {
            return Ok(None);
        }
        let scratch = cache.scratch_lines(recorded.rows - batch.len())?;
        let tokens: Vec<u32> = batch.iter().map(|&(_, token)| token).collect();
        let entries: Vec<(SequenceId, &[u32])> = batch
            .iter()
            .zip(&tokens)
            .map(|(&(id, _), token)| (id, std::slice::from_ref(token)))
            .collect();
        let mut plan = cache.plan(&entries, Some(recorded.widest))?;
        let rows = recorded.rows;
        plan.pad(rows, &scratch);
        let mut ids = tokens;
        ids.resize(rows, 0);
        let host = |values: &[u32], shape: &[usize]| Tensor::from_slice(values, shape);
        let inputs = &mut recorded.inputs;
        backend.copy_from_host(&mut inputs.ids, &host(&ids, &[rows])?)?;
        backend.copy_from_host(&mut inputs.positions, &host(&plan.positions, &[rows])?)?;
        backend.copy_from_host(&mut inputs.lines, &host(&plan.lines, &[rows])?)?;
        backend.copy_from_host(
            &mut inputs.pages,
            &host(&plan.pages, &[rows, recorded.widest])?,
        )?;
        backend.copy_from_host(&mut inputs.rows, &host(&plan.rows, &[rows, 2])?)?;
        backend.copy_from_host(&mut inputs.slots, &host(&plan.slots, &[rows])?)?;
        recorded.recording.replay()?;
        Ok(Some(recorded.recording.output().narrow(
            0,
            0,
            batch.len(),
        )?))
    }

    fn run<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids_tensor: &Tensor,
        positions: &Tensor,
        paged: Option<(&mut KvCache, &Step)>,
        logits: Logits,
    ) -> Result<Tensor> {
        Ok(self
            .run_with(backend, ids_tensor, positions, paged, logits, false)?
            .0)
    }

    /// [`Model::run`], and with `hidden`, the final hidden states of every
    /// position, before the final norm: what a multi-token layer reads.
    fn run_with<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids_tensor: &Tensor,
        positions: &Tensor,
        mut paged: Option<(&mut KvCache, &Step)>,
        logits: Logits,
        hidden: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let c = &self.config;
        let ops = Ops(backend, None);
        let prefix = match c.block {
            Block::Gpt2 => "transformer.h",
            _ => "model.layers",
        };
        let embed = self.weight(self.embed_name())?;
        let mut x = self.embedded(backend, ids_tensor, positions)?;

        for layer in 0..c.layers {
            let p = format!("{prefix}.{layer}");
            x = match c.block {
                Block::Sequential | Block::Sandwich => {
                    self.sequential(&ops, &x, positions, &mut paged, layer, &p, c.windows[layer])?
                }
                Block::Parallel => {
                    let h = self.norm(&ops, &x, &format!("{p}.input_layernorm"))?;
                    let a = self.attention(
                        &ops,
                        &h,
                        positions,
                        &mut paged,
                        layer,
                        &p,
                        c.windows[layer],
                    )?;
                    let fc1 = ops.linear(
                        &h,
                        self.weight(&format!("{p}.mlp.fc1.weight"))?,
                        self.optional(&format!("{p}.mlp.fc1.bias")),
                    )?;
                    let fc2 = ops.linear(
                        &ops.unary(c.activation, &fc1)?,
                        self.weight(&format!("{p}.mlp.fc2.weight"))?,
                        self.optional(&format!("{p}.mlp.fc2.bias")),
                    )?;
                    ops.binary(BinaryKind::Add, &ops.binary(BinaryKind::Add, &x, &a)?, &fc2)?
                }
                Block::Gpt2 => {
                    let h = self.norm(&ops, &x, &format!("{p}.ln_1"))?;
                    let a = self.attention(
                        &ops,
                        &h,
                        positions,
                        &mut paged,
                        layer,
                        &p,
                        c.windows[layer],
                    )?;
                    let x = ops.binary(BinaryKind::Add, &x, &a)?;
                    let h = self.norm(&ops, &x, &format!("{p}.ln_2"))?;
                    // GPT-2's `Conv1D` stores `[in, out]`: `y = x · w`.
                    let fc = ops.product(
                        &h,
                        self.weight(&format!("{p}.mlp.c_fc.weight"))?,
                        self.optional(&format!("{p}.mlp.c_fc.bias")),
                    )?;
                    let out = ops.product(
                        &ops.unary(c.activation, &fc)?,
                        self.weight(&format!("{p}.mlp.c_proj.weight"))?,
                        self.optional(&format!("{p}.mlp.c_proj.bias")),
                    )?;
                    ops.binary(BinaryKind::Add, &x, &out)?
                }
            };
        }

        let final_norm = match c.block {
            Block::Gpt2 => "transformer.ln_f",
            Block::Parallel => "model.final_layernorm",
            _ => "model.norm",
        };
        // The hidden states of every position, before the final norm.
        let all = hidden.then(|| x.clone());
        // The norm and the head act row by row: only the rows wanted.
        match logits {
            Logits::All => {}
            // Consecutive rows: a view.
            Logits::Rows(rows) if rows.windows(2).all(|w| w[1] == w[0] + 1) => {
                x = x.narrow(0, rows[0], rows.len())?;
            }
            Logits::Rows(rows) => {
                let rows: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
                let index = backend.upload(&Tensor::from_slice(&rows, &[rows.len()])?)?;
                x = backend.forward(&GatherRows, &[&x, &index])?;
            }
            Logits::Last => unreachable!("turned into rows by run_batch"),
        }
        let h = self.norm(&ops, &x, final_norm)?;
        let head = match self.optional("lm_head.weight") {
            Some(w) if !c.tie_embeddings => w,
            _ => embed,
        };
        let mut logits = ops.linear(
            &h,
            head,
            self.optional("lm_head.bias").filter(|_| c.lm_head_bias),
        )?;
        if let Some(cap) = c.final_softcap {
            logits = ops.softcap(&logits, cap)?;
        }
        Ok((logits, all))
    }

    /// A block of Llama's kind (and Gemma 2's, its norms around each part):
    /// attention, then the gated MLP, each added to `x`. `layer` names its
    /// tables in the cache.
    #[allow(clippy::too_many_arguments)]
    fn sequential<B: Backend + Transfer>(
        &self,
        ops: &Ops<'_, B>,
        x: &Tensor,
        positions: &Tensor,
        paged: &mut Option<(&mut KvCache, &Step)>,
        layer: usize,
        p: &str,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let c = &self.config;
        let backend = ops.0;
        let h = self.norm(ops, x, &format!("{p}.input_layernorm"))?;
        ops.see(Site::Attention, &h)?;
        let mut a = match c.mixers.get(layer) {
            Some(Mixer::GatedDeltaNet) => self.delta_net(ops, &h, paged, layer, p)?,
            _ => self.attention(ops, &h, positions, paged, layer, p, window)?,
        };
        if c.block == Block::Sandwich {
            a = self.norm(ops, &a, &format!("{p}.post_attention_layernorm"))?;
        }
        let x = ops.binary(BinaryKind::Add, x, &a)?;
        let before = if c.block == Block::Sandwich {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };
        let h = self.norm(ops, &x, &format!("{p}.{before}"))?;
        if c.experts.is_some() {
            let m = self.experts(ops, &h, p)?;
            return ops.binary(BinaryKind::Add, &x, &m);
        }
        ops.see(Site::Mlp, &h)?;
        let bias = |name: &str| {
            if c.mlp_bias {
                self.optional(&format!("{p}.mlp.{name}.bias"))
            } else {
                None
            }
        };
        let inner = match self.optional(&format!("{p}.mlp.gate_up_proj.weight")) {
            // Stacked by `Model::fused`: one product, one gated unit.
            Some(w) => backend.forward(
                &Glu {
                    activation: c.activation,
                },
                &[&ops.linear(&h, w, bias("gate_up_proj"))?],
            )?,
            None => {
                let gate = ops.linear(
                    &h,
                    self.weight(&format!("{p}.mlp.gate_proj.weight"))?,
                    bias("gate_proj"),
                )?;
                let up = ops.linear(
                    &h,
                    self.weight(&format!("{p}.mlp.up_proj.weight"))?,
                    bias("up_proj"),
                )?;
                ops.binary(BinaryKind::Mul, &ops.unary(c.activation, &gate)?, &up)?
            }
        };
        ops.see(Site::Down, &inner)?;
        let mut m = ops.linear(
            &inner,
            self.weight(&format!("{p}.mlp.down_proj.weight"))?,
            bias("down_proj"),
        )?;
        if c.block == Block::Sandwich {
            m = self.norm(ops, &m, &format!("{p}.post_feedforward_layernorm"))?;
        }
        ops.binary(BinaryKind::Add, &x, &m)
    }

    /// The attention of one block: `[T, hidden]` to `[T, hidden]`, after the
    /// keys and values of `cached`, to which the new ones are added.
    #[allow(clippy::too_many_arguments)]
    fn attention<B: Backend + Transfer>(
        &self,
        ops: &Ops<'_, B>,
        h: &Tensor,
        positions: &Tensor,
        paged: &mut Option<(&mut KvCache, &Step)>,
        layer: usize,
        p: &str,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let c = &self.config;
        let t = h.shape()[0];
        let (hq, hkv, d) = (c.heads, c.kv_heads, c.head_dim);
        let bias = |name: &str| {
            if c.qkv_bias {
                self.optional(name)
            } else {
                None
            }
        };
        let (q, k, v) = if c.block == Block::Gpt2 {
            let w = self.weight(&format!("{p}.attn.c_attn.weight"))?;
            let qkv = ops.product(h, w, self.optional(&format!("{p}.attn.c_attn.bias")))?;
            let width = hq * d;
            (
                qkv.narrow(1, 0, width)?,
                qkv.narrow(1, width, width)?,
                qkv.narrow(1, 2 * width, width)?,
            )
        } else {
            let proj = |name: &str| -> Result<(&Tensor, Option<&Tensor>)> {
                Ok((
                    self.weight(&format!("{p}.self_attn.{name}.weight"))?,
                    bias(&format!("{p}.self_attn.{name}.bias")),
                ))
            };
            let linear = |name: &str| -> Result<Tensor> {
                let (w, b) = proj(name)?;
                ops.linear(h, w, b)
            };
            match self.optional(&format!("{p}.self_attn.qkv_proj.weight")) {
                // Stacked by `Model::fused`: one product, its columns cut.
                Some(w) => {
                    let qkv = ops.linear(h, w, bias(&format!("{p}.self_attn.qkv_proj.bias")))?;
                    let gated = if c.attention_gate { 2 } else { 1 };
                    let (wq, wk) = (gated * hq * d, hkv * d);
                    (
                        qkv.narrow(1, 0, wq)?,
                        qkv.narrow(1, wq, wk)?,
                        qkv.narrow(1, wq + wk, wk)?,
                    )
                }
                None => (linear("q_proj")?, linear("k_proj")?, linear("v_proj")?),
            }
        };
        // With a gate, each query head is followed by its gate: [T, H·2D] →
        // [T, H, 2D], the queries and the gates cut apart.
        let (q, gate) = if c.attention_gate {
            let both = ops.contiguous(&q)?.reshape(&[t, hq, 2 * d])?;
            let gate = ops
                .contiguous(&both.narrow(2, d, d)?)?
                .reshape(&[t, hq * d])?;
            (
                ops.contiguous(&both.narrow(2, 0, d)?)?
                    .reshape(&[t, hq * d])?,
                Some(gate),
            )
        } else {
            (q, None)
        };
        let offset = match c.norm {
            Norm::Rms { offset } => offset,
            Norm::Layer => 0.0,
        };
        // [T, H·D] → [T, H, D], a view when it can be one.
        let heads = |x: Tensor, n: usize, norm: Option<&str>| -> Result<Tensor> {
            let mut x = match x.reshape(&[t, n, d]) {
                Ok(view) => view,
                Err(_) => ops.contiguous(&x)?.reshape(&[t, n, d])?,
            };
            if let Some(name) = norm {
                let w = self.weight(&format!("{p}.self_attn.{name}.weight"))?;
                x = ops.0.forward(
                    &RmsNorm {
                        eps: c.eps,
                        weight_offset: offset,
                    },
                    &[&x, w],
                )?;
            }
            Ok(x)
        };
        let mut q = heads(q, hq, c.qk_norm.then_some("q_norm"))?;
        let mut k = heads(k, hkv, c.qk_norm.then_some("k_norm"))?;
        let v = heads(v, hkv, None)?;
        if let Some((theta, scaling, rotary_dim)) = c.rope {
            let rope = Rope {
                theta,
                layout: c.rope_layout,
                rotary_dim: Some(rotary_dim),
                scaling,
            };
            // One position per token, the same for all its heads.
            let positions = positions.reshape(&[t, 1])?;
            q = ops.0.forward(&rope, &[&q, &positions])?;
            k = ops.0.forward(&rope, &[&k, &positions])?;
        }
        let y = match paged {
            Some((cache, step)) => {
                // The new keys and values go into the tables in place, then
                // each query reads its sequence's positions through its pages.
                let tables = cache.tables(ops.0, layer, step.lines_needed, &k, &v)?;
                ops.0.write_rows(&mut tables.0, &step.lines, &k)?;
                ops.0.write_rows(&mut tables.1, &step.lines, &v)?;
                let attention = PagedAttention {
                    page: PAGE,
                    window,
                    scale: c.query_scale,
                    softcap: c.attn_softcap,
                };
                let (keys, values) = (&tables.0, &tables.1);
                let y = ops
                    .0
                    .forward(&attention, &[&q, keys, values, &step.pages, &step.rows])?;
                y.reshape(&[t, hq * d])?
            }
            None => {
                // [T, H, D] → [1, H, T, D], as views.
                let batch = |x: &Tensor, n: usize| x.reshape(&[1, t, n, d])?.transpose(1, 2);
                let attention = Attention {
                    causal: true,
                    window,
                    scale: c.query_scale,
                    softcap: c.attn_softcap,
                };
                let y = ops.0.forward(
                    &attention,
                    &[&batch(&q, hq)?, &batch(&k, hkv)?, &batch(&v, hkv)?],
                )?;
                // [1, H, T, D] → [T, H·D].
                ops.contiguous(&y.transpose(1, 2)?)?.reshape(&[t, hq * d])?
            }
        };
        Ok(if c.block == Block::Gpt2 {
            ops.product(
                &y,
                self.weight(&format!("{p}.attn.c_proj.weight"))?,
                self.optional(&format!("{p}.attn.c_proj.bias")),
            )?
        } else {
            let out = if c.block == Block::Parallel {
                "dense"
            } else {
                "o_proj"
            };
            let y = match &gate {
                Some(gate) => {
                    ops.binary(BinaryKind::Mul, &y, &ops.unary(UnaryKind::Sigmoid, gate)?)?
                }
                None => y,
            };
            ops.see(Site::Output, &y)?;
            let bias = if c.out_bias {
                self.optional(&format!("{p}.self_attn.{out}.bias"))
            } else {
                None
            };
            ops.linear(
                &y,
                self.weight(&format!("{p}.self_attn.{out}.weight"))?,
                bias,
            )?
        })
    }

    /// A layer of Gated DeltaNet: `[T, hidden]` to `[T, hidden]`, after the
    /// states of `paged` (zero states without a cache), which it carries to
    /// its tokens.
    fn delta_net<B: Backend + Transfer>(
        &self,
        ops: &Ops<'_, B>,
        h: &Tensor,
        paged: &mut Option<(&mut KvCache, &Step)>,
        layer: usize,
        p: &str,
    ) -> Result<Tensor> {
        let c = &self.config;
        let backend = ops.0;
        let lin = c.linear.ok_or_else(|| {
            ModelError::Config(format!("layer {layer} is linear, without its sizes"))
        })?;
        let t = h.shape()[0];
        let (hk, hv, dk, dv) = (lin.key_heads, lin.value_heads, lin.key_dim, lin.value_dim);
        let channels = lin.channels();
        let w = |name: &str| self.weight(&format!("{p}.linear_attn.{name}"));
        let qkv = ops.linear(h, w("in_proj_qkv.weight")?, None)?;
        let z = ops.linear(h, w("in_proj_z.weight")?, None)?;
        let b = ops.linear(h, w("in_proj_b.weight")?, None)?;
        let a = ops.linear(h, w("in_proj_a.weight")?, None)?;
        // `[C, 1, K]` in the checkpoint.
        let conv_w = w("conv1d.weight")?.reshape(&[channels, lin.conv])?;
        let conv_shape = [lin.conv - 1, channels];
        let delta_shape = [hv, dk, dv];
        let mut own;
        let (states, slots) = match paged {
            Some((cache, step)) => (
                cache.states(
                    backend,
                    layer,
                    (&conv_shape, qkv.dtype()),
                    &delta_shape,
                    &step.fresh,
                )?,
                step.slots.clone(),
            ),
            None => {
                own = (
                    backend.zeros(&[1, conv_shape[0], conv_shape[1]], qkv.dtype())?,
                    backend.zeros(&[1, hv, dk, dv], DType::F32)?,
                );
                let zeros = vec![0u32; t];
                (
                    &mut own,
                    backend.upload(&Tensor::from_slice(&zeros, &[t])?)?,
                )
            }
        };
        let mixed = backend.causal_conv(&mut states.0, &qkv, &conv_w, &slots)?;
        let mixed = ops.unary(UnaryKind::Silu, &mixed)?;
        let f32 = |x: &Tensor| -> Result<Tensor> {
            Ok(if x.dtype() == DType::F32 {
                x.clone()
            } else {
                backend.forward(&Cast { dtype: DType::F32 }, &[x])?
            })
        };
        let part = |start: usize, heads: usize, width: usize| -> Result<Tensor> {
            Ok(
                f32(&ops.contiguous(&mixed.narrow(1, start, heads * width)?)?)?
                    .reshape(&[t, heads, width])?,
            )
        };
        // Queries and keys of unit norm (as the FLA library computes it:
        // `x / √(Σx² + 10⁻⁶)`, an RMS norm of `ε / D` over `√D`), the queries
        // scaled by `1 / √D` too.
        let unit = |x: &Tensor, scale: f64| -> Result<Tensor> {
            let n = backend.forward(
                &RmsNorm {
                    eps: 1e-6 / dk as f64,
                    weight_offset: 0.0,
                },
                &[x],
            )?;
            ops.affine(&n, scale)
        };
        let q = unit(&part(0, hk, dk)?, 1.0 / dk as f64)?;
        let k = unit(&part(hk * dk, hk, dk)?, 1.0 / (dk as f64).sqrt())?;
        let v = part(2 * hk * dk, hv, dv)?;
        let beta = f32(&ops.unary(UnaryKind::Sigmoid, &b)?)?;
        // g = −exp(A_log) · softplus(a + dt_bias), in `f32`; softplus(x) =
        // max(x, 0) + log(1 + exp(−|x|)).
        let x = ops.binary(BinaryKind::Add, &f32(&a)?, &f32(w("dt_bias")?)?)?;
        let tail = ops.unary(
            UnaryKind::Log,
            &backend.forward(
                &Affine {
                    scale: 1.0,
                    bias: 1.0,
                },
                &[&ops.unary(
                    UnaryKind::Exp,
                    &ops.unary(UnaryKind::Neg, &ops.unary(UnaryKind::Abs, &x)?)?,
                )?],
            )?,
        )?;
        let softplus = ops.binary(BinaryKind::Add, &ops.unary(UnaryKind::Relu, &x)?, &tail)?;
        let decay = ops.unary(
            UnaryKind::Neg,
            &ops.unary(UnaryKind::Exp, &f32(w("A_log")?)?)?,
        )?;
        let g = ops.binary(BinaryKind::Mul, &decay, &softplus)?;
        let o = backend.delta_rule(&mut states.1, &q, &k, &v, &g, &beta, &slots)?;
        let o = if o.dtype() == h.dtype() {
            o
        } else {
            backend.forward(&Cast { dtype: h.dtype() }, &[&o])?
        };
        // The gated norm, per value head: `norm(o) · w · silu(z)`.
        let o = backend.forward(
            &RmsNorm {
                eps: c.eps,
                weight_offset: 0.0,
            },
            &[&o, w("norm.weight")?],
        )?;
        let z = ops.contiguous(&z)?.reshape(&[t, hv, dv])?;
        let o = ops.binary(BinaryKind::Mul, &o, &ops.unary(UnaryKind::Silu, &z)?)?;
        ops.linear(&o.reshape(&[t, hv * dv])?, w("out_proj.weight")?, None)
    }

    /// A mixture of experts: `[T, hidden]` to `[T, hidden]`. The router's
    /// softmax over the experts, in `f32`, chooses each token's experts, the
    /// largest probabilities (the first expert on a tie), their weights
    /// renormalized to sum to 1; each expert computes its tokens, weighted;
    /// the shared expert, gated by a sigmoid, is added.
    fn experts<B: Backend + Transfer>(
        &self,
        ops: &Ops<'_, B>,
        h: &Tensor,
        p: &str,
    ) -> Result<Tensor> {
        let c = &self.config;
        let backend = ops.0;
        let e = c.experts.expect("a model with experts");
        let t = h.shape()[0];
        let w = |name: &str| self.weight(&format!("{p}.mlp.{name}"));
        let logits = ops.linear(h, w("gate.weight")?, None)?;
        let logits = if logits.dtype() == DType::F32 {
            logits
        } else {
            backend.forward(&Cast { dtype: DType::F32 }, &[&logits])?
        };
        let probs = backend.forward(&kapel_ops::Softmax { dim: 1 }, &[&logits])?;
        let probs = backend.download(&probs)?.to_vec::<f32>()?;
        // Each token's experts, then each expert's tokens.
        let mut tokens: Vec<Vec<(u32, f32)>> = vec![Vec::new(); e.count];
        for (row, p) in probs.chunks_exact(e.count).enumerate() {
            let mut order: Vec<usize> = (0..e.count).collect();
            order.sort_by(|&i, &j| p[j].total_cmp(&p[i]).then(i.cmp(&j)));
            let chosen = &order[..e.per_token];
            let total: f32 = chosen.iter().map(|&i| p[i]).sum();
            for &i in chosen {
                // The weight in the type of the hidden states, as
                // `transformers` gives it.
                let weight = rounded(h.dtype(), f64::from(p[i] / total))? as f32;
                tokens[i].push((row as u32, weight));
            }
        }
        let gate_up = w("experts.gate_up_proj")?;
        let down = w("experts.down_proj")?;
        let (two_i, width) = (gate_up.shape()[1], gate_up.shape()[2]);
        let glu = Glu {
            activation: c.activation,
        };
        let (mut outs, mut index) = (Vec::new(), Vec::new());
        for (expert, routed) in tokens.iter().enumerate() {
            if routed.is_empty() {
                continue;
            }
            let n = routed.len();
            let rows: Vec<u32> = routed.iter().map(|&(r, _)| r).collect();
            let weights: Vec<f32> = routed.iter().map(|&(_, w)| w).collect();
            let rows_t = backend.upload(&Tensor::from_slice(&rows, &[n])?)?;
            let x = backend.forward(&GatherRows, &[h, &rows_t])?;
            let wg = gate_up.narrow(0, expert, 1)?.reshape(&[two_i, width])?;
            let inner = backend.forward(&glu, &[&ops.linear(&x, &wg, None)?])?;
            let wd = down
                .narrow(0, expert, 1)?
                .reshape(&[down.shape()[1], down.shape()[2]])?;
            let y = ops.linear(&inner, &wd, None)?;
            let mut scale = backend.upload(&Tensor::from_slice(&weights, &[n, 1])?)?;
            if scale.dtype() != y.dtype() {
                scale = backend.forward(&Cast { dtype: y.dtype() }, &[&scale])?;
            }
            outs.push(ops.binary(BinaryKind::Mul, &y, &scale)?);
            index.extend(rows);
        }
        let parts: Vec<&Tensor> = outs.iter().collect();
        let all = backend.forward(&Concat { dim: 0 }, &parts)?;
        let index = backend.upload(&Tensor::from_slice(&index, &[index.len()])?)?;
        let routed = backend.forward(&kapel_ops::ScatterAddRows { rows: t }, &[&all, &index])?;
        if e.shared_width.is_none() {
            return Ok(routed);
        }
        let s = |name: &str| w(&format!("shared_expert.{name}.weight"));
        let gate = ops.linear(h, s("gate_proj")?, None)?;
        let up = ops.linear(h, s("up_proj")?, None)?;
        let inner = ops.binary(BinaryKind::Mul, &ops.unary(c.activation, &gate)?, &up)?;
        let shared = ops.linear(&inner, s("down_proj")?, None)?;
        let open = ops.unary(
            UnaryKind::Sigmoid,
            &ops.linear(h, w("shared_expert_gate.weight")?, None)?,
        )?;
        let shared = ops.binary(BinaryKind::Mul, &shared, &open)?;
        ops.binary(BinaryKind::Add, &routed, &shared)
    }
}

/// GPT-2's weights under the names of `GPT2LMHeadModel`: the first
/// checkpoints (`openai-community/gpt2`) name them after `GPT2Model`,
/// without `transformer.`, and keep the causal masks of the attention
/// (`attn.bias`, `attn.masked_bias`), which are not weights.
fn gpt2_names(weights: BTreeMap<String, Tensor>) -> BTreeMap<String, Tensor> {
    weights
        .into_iter()
        .filter(|(name, _)| !name.ends_with(".attn.bias") && !name.ends_with(".attn.masked_bias"))
        .map(|(name, w)| {
            if name.starts_with("transformer.") || name.starts_with("lm_head.") {
                (name, w)
            } else {
                (format!("transformer.{name}"), w)
            }
        })
        .collect()
}

/// The weights of a model that also sees images under the names of its
/// decoder alone: `model.language_model.X` as `model.X` (Qwen 3.5's
/// checkpoints).
fn decoder_names(weights: BTreeMap<String, Tensor>) -> BTreeMap<String, Tensor> {
    if !weights
        .keys()
        .any(|n| n.starts_with("model.language_model."))
    {
        return weights;
    }
    weights
        .into_iter()
        .map(
            |(name, w)| match name.strip_prefix("model.language_model.") {
                Some(rest) => (format!("model.{rest}"), w),
                None => (name, w),
            },
        )
        .collect()
}

/// A weight `[N, K]` in host memory quantized to `dtype` by rounding to the
/// nearest code, by pieces of rows on every core.
///
/// # Errors
///
/// If `dtype` is not a quantized type, or `K` is not a multiple of its
/// blocks.
pub fn quantize_matrix(w: &Tensor, dtype: DType) -> kapel_core::Result<Tensor> {
    let (n, k) = (w.shape()[0], w.shape()[1]);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    // Pieces of about a million values: a few megabytes at a time.
    let rows = (1 << 20) / k.max(1) + 1;
    let pieces: Vec<usize> = (0..n).step_by(rows).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut parts: Vec<kapel_core::Result<Vec<u8>>> = Vec::new();
    parts.resize_with(pieces.len(), || Ok(Vec::new()));
    let results = std::sync::Mutex::new(parts);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(pieces.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&first) = pieces.get(i) else { break };
                    let part = w
                        .narrow(0, first, rows.min(n - first))
                        .and_then(|part| part.to_f64_vec())
                        .and_then(|values| {
                            let values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                            kapel_core::quantize(dtype, &values)
                        });
                    results.lock().unwrap_or_else(|e| e.into_inner())[i] = part;
                }
            });
        }
    });
    let mut bytes = Vec::with_capacity(dtype.storage_bytes(n * k)?);
    for part in results.into_inner().unwrap_or_else(|e| e.into_inner()) {
        bytes.extend(part?);
    }
    Tensor::from_bytes(dtype, &[n, k], &bytes)
}

fn rounded(dtype: DType, value: f64) -> Result<f64> {
    let mut bytes = vec![0; dtype.block_bytes()];
    dtype.encode(value, &mut bytes)?;
    Ok(dtype.decode(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_gpt2_checkpoints_get_the_names_of_gpt2_lm_head_model() {
        let w = || Tensor::zeros(DType::F32, &[1]).unwrap();
        let weights: BTreeMap<String, Tensor> = [
            "wte.weight",
            "h.0.attn.bias",
            "h.0.attn.masked_bias",
            "h.0.attn.c_attn.bias",
            "h.0.attn.c_proj.weight",
            "transformer.ln_f.weight",
            "lm_head.weight",
        ]
        .into_iter()
        .map(|n| (n.to_string(), w()))
        .collect();
        let names: Vec<String> = gpt2_names(weights).into_keys().collect();
        assert_eq!(
            names,
            [
                "lm_head.weight",
                "transformer.h.0.attn.c_attn.bias",
                "transformer.h.0.attn.c_proj.weight",
                "transformer.ln_f.weight",
                "transformer.wte.weight",
            ]
        );
    }

    /// `resave d d`: a model saved over the directory it was loaded from.
    #[test]
    fn a_model_saves_over_its_own_directory() {
        let dir = std::env::temp_dir().join(format!("kapel-model-resave-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "model_type": "llama", "hidden_size": 8, "num_attention_heads": 2,
            "num_hidden_layers": 1, "vocab_size": 10, "intermediate_size": 16,
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let embed: Vec<f32> = (0..80).map(|i| i as f32 / 8.0).collect();
        let norm = vec![1.0f32; 8];
        let tensors = [
            Tensor::from_slice(&embed, &[10, 8]).unwrap(),
            Tensor::from_slice(&norm, &[8]).unwrap(),
        ];
        kapel_hub::safetensors::save(
            dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &tensors[0]),
                ("model.norm.weight", &tensors[1]),
            ],
            &BTreeMap::new(),
        )
        .unwrap();
        let model = Model::load(&Checkpoint::open(&dir).unwrap()).unwrap();
        // A stale index of shards would hide the saved weights: it goes.
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();
        for _ in 0..2 {
            model.save(&dir).unwrap();
            // The weights mapped from the replaced file still read.
            let embed_again = model.weight("model.embed_tokens.weight").unwrap();
            assert_eq!(embed_again.to_vec::<f32>().unwrap(), embed);
        }
        let again = Model::load(&Checkpoint::open(&dir).unwrap()).unwrap();
        assert_eq!(again.config(), model.config());
        assert_eq!(again.weights.len(), 2);
        for (name, tensor) in &model.weights {
            assert_eq!(
                again.weights[name].to_vec::<f32>().unwrap(),
                tensor.to_vec::<f32>().unwrap(),
                "{name}"
            );
        }
        // Only the two files are left: no temporary.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
