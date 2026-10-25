//! The decoders of `kapel-models`, on a tape.
//!
//! The parameters keep the names and the layout of `transformers`
//! ([`Parameters`]): a model loaded with [`kapel_models::Model`] trains
//! here, and a trained one is a [`kapel_models::Model`] again, for
//! inference or to be saved for the Hub. [`Decoder`] computes the forward
//! pass of a batch of sequences on a [`Tape`], with the same operations as
//! the inference pass, batched: the same logits.
//!
//! A projection with a LoRA adapter beside it ([`crate::lora`]) adds its
//! product, and a frozen projection may be quantized (QLoRA): its product
//! reads the codes.

use std::collections::BTreeMap;
use std::rc::Rc;

use kapel_core::{DType, Error, Result, Tensor};
use kapel_models::{Block, Config, Model, Norm};
use kapel_ops::{Attention, Rope};

use crate::loss::linear_cross_entropy;
use crate::tape::{Compute, Grad, Gradients, Tape, Var};

/// A parameter: its value and how the optimizer treats it.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// The value, `f32` for training (the master weights).
    pub value: Tensor,
    /// Whether weight decay applies: matrices and tables, not norms or
    /// biases (nanoGPT's rule).
    pub decay: bool,
    /// Whether it is a leaf whose gradient may stay rows
    /// ([`Tape::sparse_leaf`]).
    pub sparse: bool,
    /// Whether it receives a gradient; a frozen one is a constant.
    pub trainable: bool,
    /// The factor of the learning rate for this parameter.
    pub lr_scale: f64,
}

impl Parameter {
    /// A trainable parameter, decayed if it has two dimensions or more.
    pub fn new(value: Tensor) -> Self {
        Self {
            decay: value.rank() >= 2,
            value,
            sparse: false,
            trainable: true,
            lr_scale: 1.0,
        }
    }
}

/// The parameters of a model, by their names in the checkpoint.
#[derive(Clone, Debug, Default)]
pub struct Parameters {
    map: BTreeMap<String, Parameter>,
}

impl Parameters {
    /// The weights of `model`, each a trainable [`Parameter::new`].
    pub fn from_model(model: &Model) -> Self {
        Self {
            map: model
                .names()
                .filter_map(|n| {
                    let t = model.tensor(n)?;
                    Some((n.to_string(), Parameter::new(t.clone())))
                })
                .collect(),
        }
    }

    /// The parameter `name`.
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.map.get(name)
    }

    /// The parameter `name`, to change it.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.map.get_mut(name)
    }

    /// Adds or replaces a parameter.
    pub fn insert(&mut self, name: impl Into<String>, parameter: Parameter) {
        self.map.insert(name.into(), parameter);
    }

    /// The parameters, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Parameter)> {
        self.map.iter().map(|(n, p)| (n.as_str(), p))
    }

    /// The parameters, to change them.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Parameter)> {
        self.map.iter_mut().map(|(n, p)| (n.as_str(), p))
    }

    /// The number of trainable values.
    pub fn count(&self) -> usize {
        self.map
            .values()
            .filter(|p| p.trainable)
            .map(|p| p.value.numel())
            .sum()
    }

    /// The number of values, frozen ones included.
    pub fn count_all(&self) -> usize {
        self.map.values().map(|p| p.value.numel()).sum()
    }

    /// Each value in `dtype` and on `backend`'s device: `f32` on the card
    /// for training.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    pub fn to<B: Compute>(&self, backend: &B, dtype: DType) -> Result<Self> {
        self.to_mixed(backend, dtype, dtype)
    }

    /// Each value on `backend`'s device, the float values of the trainable
    /// parameters in `dtype` (the master weights), of the frozen ones in
    /// `frozen`: a frozen model in the computation's type takes half the
    /// memory of `f32`. Quantized values stay as they are.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    pub fn to_mixed<B: Compute>(&self, backend: &B, dtype: DType, frozen: DType) -> Result<Self> {
        let map = self
            .map
            .iter()
            .map(|(n, p)| {
                let dtype = if p.trainable { dtype } else { frozen };
                let mut value = backend.upload(&p.value)?;
                if value.dtype().is_float() && value.dtype() != dtype {
                    value = backend.forward(&kapel_ops::Cast { dtype }, &[&value])?;
                }
                // A storage of its own, that the optimizer may write in place:
                // a host backend keeps the tensor it is given, which may be a
                // file mapped read-only.
                if std::sync::Arc::ptr_eq(value.storage(), p.value.storage()) {
                    value = owned(&value)?;
                }
                Ok((n.clone(), Parameter { value, ..p.clone() }))
            })
            .collect::<Result<_>>()?;
        Ok(Self { map })
    }

    /// The parameters on `tape`: leaves, sparse leaves or constants.
    pub fn leaves<B: Compute>(&self, tape: &Tape<B>) -> Leaves<B> {
        Leaves {
            vars: self
                .map
                .iter()
                .map(|(n, p)| {
                    let var = match (p.trainable, p.sparse) {
                        (false, _) => tape.constant(p.value.clone()),
                        (true, true) => tape.sparse_leaf(p.value.clone()),
                        (true, false) => tape.leaf(p.value.clone()),
                    };
                    (n.clone(), var)
                })
                .collect(),
        }
    }

    /// The model of configuration `like`'s, with these weights, in host
    /// memory and in `dtype`: what [`Model::save`] writes for the Hub.
    ///
    /// # Errors
    ///
    /// Those of the backend, or of [`Model::with_weights`].
    pub fn to_model<B: Compute>(&self, backend: &B, like: &Model, dtype: DType) -> Result<Model> {
        let mut weights = BTreeMap::new();
        for (n, p) in &self.map {
            let mut value = p.value.clone();
            if value.dtype().is_float() && value.dtype() != dtype {
                value = backend.forward(&kapel_ops::Cast { dtype }, &[&value])?;
            }
            weights.insert(n.clone(), backend.download(&value)?);
        }
        like.with_weights(weights)
            .map_err(|e| Error::InvalidArgument {
                op: "Parameters::to_model",
                message: e.to_string(),
            })
    }
}

/// A copy of a host tensor in a buffer of its own; a device tensor as it is.
pub(crate) fn owned(t: &Tensor) -> Result<Tensor> {
    let t = t.contiguous()?;
    match t.storage().host_bytes() {
        Some(bytes) => {
            let size = t.dtype().block_bytes();
            let start = t.layout().offset() * size;
            let len = t.dtype().storage_bytes(t.numel())?;
            Tensor::from_bytes(t.dtype(), t.shape(), &bytes[start..start + len])
        }
        None => Ok(t),
    }
}

/// The parameters of one step, on its tape.
pub struct Leaves<B: Compute> {
    vars: BTreeMap<String, Var<B>>,
}

impl<B: Compute> Leaves<B> {
    /// The variable of parameter `name`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if there is none.
    pub fn get(&self, name: &str) -> Result<&Var<B>> {
        self.vars.get(name).ok_or_else(|| Error::InvalidArgument {
            op: "Leaves::get",
            message: format!("no parameter {name}"),
        })
    }

    /// The variable of parameter `name`, if any.
    pub fn optional(&self, name: &str) -> Option<&Var<B>> {
        self.vars.get(name)
    }

    /// The gradients of the parameters, by name, taken out of `grads`; the
    /// leaves are let go.
    pub fn gradients(self, grads: &mut Gradients) -> Vec<(String, Grad)> {
        self.vars
            .into_iter()
            .filter_map(|(n, v)| grads.take(&v).map(|g| (n, g)))
            .collect()
    }
}

/// How the forward pass is computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// The type of the computation: the parameters are converted to it
    /// for the pass, and their gradients come back in theirs.
    pub dtype: DType,
    /// Recompute each block during the backward pass instead of keeping
    /// its activations ([`Tape::checkpoint`]).
    pub checkpoint: bool,
    /// The rows of logits formed at once by the loss
    /// ([`linear_cross_entropy`]): 4096 by default, the fastest for GPT-2 on
    /// an RTX 3090 (the products along the vocabulary fill the card from
    /// there).
    pub chunk: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            checkpoint: false,
            chunk: 4096,
        }
    }
}

/// A batch: `rows` sequences of the same length, row after row.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    /// The token ids, `rows × length`.
    pub ids: &'a [u32],
    /// The token each position should predict, as many, with
    /// [`crate::IGNORE`] for those that count in no loss.
    pub targets: &'a [u32],
    /// The number of sequences.
    pub rows: usize,
    /// The document of each token, if attention stays within documents:
    /// a token attends to the earlier tokens of its row with its number,
    /// and its position counts from the first of them. `None`: the whole
    /// row is one sequence.
    pub segments: Option<&'a [u32]>,
}

/// The position of each of the `n` tokens of rows of `length`: from the
/// row's start, or from its document's first token.
fn positions(n: usize, length: usize, segments: Option<&[u32]>) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    let mut first = 0;
    for i in 0..n {
        let new_row = i % length == 0;
        let new_document = segments.is_some_and(|s| i > 0 && s[i] != s[i - 1]);
        if new_row || new_document {
            first = i;
        }
        out.push((i - first) as u32);
    }
    out
}

/// The forward pass of a decoder on a tape.
#[derive(Clone, Debug)]
pub struct Decoder {
    config: Rc<Config>,
    /// The scale of the LoRA adapters' products, if any.
    lora: Option<f64>,
}

/// What the blocks share in one pass.
struct Pass {
    rows: usize,
    length: usize,
    /// The position of each token, `[rows · length, 1]`.
    positions: Tensor,
    /// With segments, where the document of each token starts in its row,
    /// `[rows, length]` in `i32`: attention stays within documents
    /// ([`Attention`]'s documents).
    mask: Option<Tensor>,
    /// The scale of the adapters' products.
    lora: Option<f64>,
    /// The type of the computation.
    dtype: DType,
}

impl Decoder {
    /// The decoder of `config`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for the architectures not trained yet:
    /// Phi's parallel blocks, final logits capped (Gemma 2), multi-token
    /// layers.
    pub fn new(config: Config) -> Result<Self> {
        let unsupported = |what: &str| {
            Err(Error::InvalidArgument {
                op: "Decoder::new",
                message: format!("training {what} is not supported yet"),
            })
        };
        if config.block == Block::Parallel {
            return unsupported("parallel blocks (Phi)");
        }
        if config.final_softcap.is_some() {
            return unsupported("capped logits (Gemma 2)");
        }
        if config.multi_token_layers > 0 {
            return unsupported("multi-token layers");
        }
        Ok(Self {
            config: Rc::new(config),
            lora: None,
        })
    }

    /// The decoder that adds the products of the LoRA adapters it finds
    /// among the parameters, scaled by `scale` ([`crate::lora`]).
    pub fn with_lora(mut self, scale: f64) -> Self {
        self.lora = Some(scale);
        self
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The mean cross-entropy of `batch` (`f32`, `[1]`).
    ///
    /// # Errors
    ///
    /// For a missing parameter, a batch of the wrong size, or an operation
    /// that fails on the backend.
    pub fn loss<B: Compute>(
        &self,
        leaves: &Leaves<B>,
        batch: &Batch<'_>,
        options: &Options,
    ) -> Result<Var<B>> {
        let (h, head) = self.hidden(leaves, batch, options)?;
        linear_cross_entropy(&h, &head, batch.targets, options.chunk)
    }

    /// The cross-entropy of each position of `batch`, `−ln p(target)`, 0
    /// where the target is [`crate::IGNORE`] (`f32`, `[rows · length]`):
    /// the log-probabilities of sequences and tokens for DPO and GRPO
    /// ([`crate::loss::linear_token_losses`]).
    ///
    /// # Errors
    ///
    /// As [`Decoder::loss`].
    pub fn token_losses<B: Compute>(
        &self,
        leaves: &Leaves<B>,
        batch: &Batch<'_>,
        options: &Options,
    ) -> Result<Var<B>> {
        let (h, head) = self.hidden(leaves, batch, options)?;
        crate::loss::linear_token_losses(&h, &head, batch.targets, options.chunk)
    }

    /// The logits `[rows · length, vocab]` of `batch` (its targets unread),
    /// in the computation's type.
    ///
    /// # Errors
    ///
    /// As [`Decoder::loss`].
    pub fn logits<B: Compute>(
        &self,
        leaves: &Leaves<B>,
        batch: &Batch<'_>,
        options: &Options,
    ) -> Result<Var<B>> {
        let (h, head) = self.hidden(leaves, batch, options)?;
        let mut logits = h.linear(&head, None)?;
        if self.config.lm_head_bias
            && let Some(b) = leaves.optional("lm_head.bias")
        {
            logits = logits.add(&b.cast(options.dtype)?)?;
        }
        Ok(logits)
    }

    /// The final hidden states, normed, and the output table, in the
    /// computation's type.
    fn hidden<B: Compute>(
        &self,
        leaves: &Leaves<B>,
        batch: &Batch<'_>,
        options: &Options,
    ) -> Result<(Var<B>, Var<B>)> {
        let c = &*self.config;
        let n = batch.ids.len();
        if batch.rows == 0
            || !n.is_multiple_of(batch.rows)
            || batch.targets.len() != n
            || batch.segments.is_some_and(|s| s.len() != n)
        {
            return Err(Error::InvalidArgument {
                op: "Decoder",
                message: format!(
                    "{n} tokens and {} targets in {} rows",
                    batch.targets.len(),
                    batch.rows
                ),
            });
        }
        let length = n / batch.rows;
        let tape = leaves.get(self.embed_name())?.tape();
        let backend = tape.backend().clone();
        let upload = |data: &[u32], shape: &[usize]| -> Result<Tensor> {
            backend.upload(&Tensor::from_slice(data, shape)?)
        };
        let ids = upload(batch.ids, &[n])?;
        let positions = positions(n, length, batch.segments);
        let dtype = options.dtype;
        let embed = leaves.get(self.embed_name())?;
        // The rows of the table, gathered from the parameter itself: its
        // gradient stays rows.
        let mut x = embed.gather_rows(&ids)?.cast(dtype)?;
        if let Some(scale) = self.lora {
            let (a, b) = crate::lora::embedding_factor_names(c);
            if let (Some(a), Some(b)) = (leaves.optional(&a), leaves.optional(&b)) {
                let delta = a
                    .gather_rows(&ids)?
                    .cast(dtype)?
                    .scale(rounded(dtype, scale))?
                    .linear(&b.cast(dtype)?, None)?;
                x = x.add(&delta)?;
            }
        }
        if let Some(scale) = c.embed_scale {
            x = x.scale(rounded(dtype, scale))?;
        }
        if c.block == Block::Gpt2 {
            let wpe = leaves.get("transformer.wpe.weight")?;
            x = x.add(&wpe.gather_rows(&upload(&positions, &[n])?)?.cast(dtype)?)?;
        }
        let pass = Rc::new(Pass {
            rows: batch.rows,
            length,
            positions: upload(&positions, &[n, 1])?,
            mask: batch
                .segments
                .map(|_| {
                    let starts: Vec<i32> = positions
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (i % length) as i32 - p as i32)
                        .collect();
                    backend.upload(&Tensor::from_slice(&starts, &[batch.rows, length])?)
                })
                .transpose()?,
            lora: self.lora,
            dtype,
        });
        let prefix = match c.block {
            Block::Gpt2 => "transformer.h",
            _ => "model.layers",
        };
        for layer in 0..c.layers {
            let p = format!("{prefix}.{layer}.");
            // The block's parameters, in the computation's type.
            let names: Vec<String> = leaves
                .vars
                .keys()
                .filter_map(|k| k.strip_prefix(&p).map(str::to_string))
                .collect();
            let weights: Vec<Var<B>> = names
                .iter()
                .map(|k| computed(leaves.get(&format!("{p}{k}"))?, dtype))
                .collect::<Result<_>>()?;
            let config = Rc::clone(&self.config);
            let pass = Rc::clone(&pass);
            let run = move |_: &Tape<B>, xs: &[Var<B>]| -> Result<Var<B>> {
                let found = |name: &str| names.iter().position(|k| k == name).map(|i| &xs[i + 1]);
                block(&config, layer, &xs[0], &found, &pass)
            };
            let mut inputs = vec![&x];
            inputs.extend(weights.iter());
            x = if options.checkpoint {
                tape.checkpoint(&inputs, run)?
            } else {
                let xs: Vec<Var<B>> = inputs.into_iter().cloned().collect();
                run(&tape, &xs)?
            };
        }
        let final_norm = match c.block {
            Block::Gpt2 => "transformer.ln_f",
            _ => "model.norm",
        };
        let weight = |name: &str| -> Result<Option<Var<B>>> {
            leaves.optional(name).map(|w| w.cast(dtype)).transpose()
        };
        let h = norm(
            c,
            &x,
            weight(&format!("{final_norm}.weight"))?.as_ref(),
            weight(&format!("{final_norm}.bias"))?.as_ref(),
        )?;
        let head = match leaves.optional("lm_head.weight") {
            Some(w) if !c.tie_embeddings => w,
            _ => embed,
        };
        Ok((h, head.cast(dtype)?))
    }

    fn embed_name(&self) -> &'static str {
        match self.config.block {
            Block::Gpt2 => "transformer.wte.weight",
            _ => "model.embed_tokens.weight",
        }
    }
}

/// A parameter in the computation's type; a quantized one as it is.
fn computed<B: Compute>(v: &Var<B>, dtype: DType) -> Result<Var<B>> {
    if v.dtype().is_float() {
        v.cast(dtype)
    } else {
        Ok(v.clone())
    }
}

/// `value` as a float of `dtype` holds it.
fn rounded(dtype: DType, value: f64) -> f64 {
    let mut bytes = vec![0; dtype.block_bytes()];
    if dtype.encode(value, &mut bytes).is_ok() {
        dtype.decode(&bytes).unwrap_or(value)
    } else {
        value
    }
}

/// The normalization of the configuration.
fn norm<B: Compute>(
    c: &Config,
    x: &Var<B>,
    weight: Option<&Var<B>>,
    bias: Option<&Var<B>>,
) -> Result<Var<B>> {
    match c.norm {
        Norm::Rms { offset } => x.rms_norm(weight, c.eps, offset),
        Norm::Layer => x.layer_norm(weight, bias, c.eps),
    }
}

/// One block: `x [N, hidden]` to `[N, hidden]`. `w` finds a parameter of
/// the block by its name after the block's prefix.
fn block<'a, B: Compute + 'a>(
    c: &Config,
    layer: usize,
    x: &Var<B>,
    w: &dyn Fn(&str) -> Option<&'a Var<B>>,
    pass: &Pass,
) -> Result<Var<B>> {
    let need = |name: &str| {
        w(name).ok_or_else(|| Error::InvalidArgument {
            op: "Decoder",
            message: format!("block {layer} has no parameter {name}"),
        })
    };
    let n = pass.rows * pass.length;
    let (hq, hkv, d) = (c.heads, c.kv_heads, c.head_dim);
    let attention = Attention {
        causal: true,
        window: c.windows[layer],
        scale: c.query_scale,
        softcap: c.attn_softcap,
    };
    // [N, H·D] → [B, H, T, D], as views.
    let heads = |x: &Var<B>, h: usize| -> Result<Var<B>> {
        x.reshape(&[pass.rows, pass.length, h, d])?.transpose(1, 2)
    };
    // [B, H, T, D] → [N, H·D].
    let merged =
        |y: Var<B>| -> Result<Var<B>> { y.transpose(1, 2)?.contiguous()?.reshape(&[n, hq * d]) };
    // The products of the adapters of `modules` with `x`, side by side as
    // the projections' outputs are (`widths`), zeros for those without;
    // `None` if none has one.
    let adapters = |x: &Var<B>, modules: &[&str], widths: &[usize]| -> Result<Option<Var<B>>> {
        let Some(scale) = pass.lora else {
            return Ok(None);
        };
        let mut parts = Vec::with_capacity(modules.len());
        for m in modules {
            let (a, b) = crate::lora::factor_names(m);
            parts.push(match (w(&a), w(&b)) {
                // The scale on `x · Aᵀ`, `r` wide, not on the output.
                (Some(a), Some(b)) => Some(
                    x.linear(a, None)?
                        .scale(rounded(pass.dtype, scale))?
                        .linear(b, None)?,
                ),
                _ => None,
            });
        }
        if parts.iter().all(Option::is_none) {
            return Ok(None);
        }
        if parts.len() == 1 {
            return Ok(parts.pop().flatten());
        }
        let tape = x.tape();
        let parts: Vec<Var<B>> = parts
            .into_iter()
            .zip(widths)
            .map(|(p, &width)| match p {
                Some(p) => Ok(p),
                None => Ok(tape.constant(tape.backend().zeros(&[n, width], pass.dtype)?)),
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&Var<B>> = parts.iter().collect();
        Ok(Some(tape.concat(&refs, 1)?))
    };
    // `x · Wᵀ (+ bias)` for the weights of `modules` side by side (`[out,
    // in]` each), quantized or not, plus their adapters.
    let project = |x: &Var<B>, modules: &[&str], bias: Option<&Var<B>>| -> Result<Var<B>> {
        let weights: Vec<&Var<B>> = modules
            .iter()
            .map(|m| need(&format!("{m}.weight")))
            .collect::<Result<_>>()?;
        let widths: Vec<usize> = weights.iter().map(|w| w.shape()[0]).collect();
        let quantized = weights.iter().filter(|w| w.dtype().is_quantized()).count();
        let base = if quantized == weights.len() {
            let values: Vec<Tensor> = weights.iter().map(|w| w.value()).collect();
            let refs: Vec<&Tensor> = values.iter().collect();
            let y = x.quantized_matmul(&refs)?;
            match bias {
                Some(b) => y.add(b)?,
                None => y,
            }
        } else if quantized > 0 {
            return Err(Error::InvalidArgument {
                op: "Decoder",
                message: format!("block {layer}: {modules:?} are quantized in part only"),
            });
        } else if weights.len() == 1 {
            x.linear(weights[0], bias)?
        } else {
            x.linear(&x.tape().concat(&weights, 0)?, bias)?
        };
        match adapters(x, modules, &widths)? {
            Some(delta) => base.add(&delta),
            None => Ok(base),
        }
    };
    match c.block {
        Block::Gpt2 => {
            let h = norm(c, x, w("ln_1.weight"), w("ln_1.bias"))?;
            // `Conv1D` stores `[in, out]`: `y = x · w + b`.
            let qkv = h.matmul_bias(need("attn.c_attn.weight")?, w("attn.c_attn.bias"))?;
            let width = qkv.shape()[1];
            let qkv = match adapters(&h, &["attn.c_attn"], &[width])? {
                Some(delta) => qkv.add(&delta)?,
                None => qkv,
            };
            // `[N, 3C]` seen as `[B, T, 3, H, D]`: each of q, k, v a view of
            // it, whose gradients come back as slices along the third
            // dimension, joined by one copy.
            let qkv = qkv.reshape(&[pass.rows, pass.length, 3, hq, d])?;
            let part = |i: usize| -> Result<Var<B>> {
                qkv.narrow(2, i, 1)?
                    .reshape(&[pass.rows, pass.length, hq, d])?
                    .transpose(1, 2)
            };
            let y = part(0)?.attention(&part(1)?, &part(2)?, pass.mask.as_ref(), attention)?;
            let y = merged(y)?;
            let a = y.matmul_bias(need("attn.c_proj.weight")?, w("attn.c_proj.bias"))?;
            let a = match adapters(&y, &["attn.c_proj"], &[c.hidden])? {
                Some(delta) => a.add(&delta)?,
                None => a,
            };
            let x = x.add(&a)?;
            let h = norm(c, &x, w("ln_2.weight"), w("ln_2.bias"))?;
            let fc = h.matmul_bias(need("mlp.c_fc.weight")?, w("mlp.c_fc.bias"))?;
            let fc = match adapters(&h, &["mlp.c_fc"], &[c.intermediate])? {
                Some(delta) => fc.add(&delta)?,
                None => fc,
            };
            let act = fc.unary(c.activation)?;
            let out = act.matmul_bias(need("mlp.c_proj.weight")?, w("mlp.c_proj.bias"))?;
            let out = match adapters(&act, &["mlp.c_proj"], &[c.hidden])? {
                Some(delta) => out.add(&delta)?,
                None => out,
            };
            x.add(&out)
        }
        Block::Sequential | Block::Sandwich => {
            let sandwich = c.block == Block::Sandwich;
            let h = norm(c, x, w("input_layernorm.weight"), None)?;
            // The query, key and value projections as one product.
            let names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"];
            let tape = x.tape();
            let bias = if c.qkv_bias {
                let biases: Vec<&Var<B>> = names
                    .iter()
                    .map(|m| need(&format!("{m}.bias")))
                    .collect::<Result<_>>()?;
                Some(tape.concat(&biases, 0)?)
            } else {
                None
            };
            let qkv = project(&h, &names, bias.as_ref())?;
            let (wq, wk) = (hq * d, hkv * d);
            let split = |start: usize, h: usize, norm_name: Option<&str>| -> Result<Var<B>> {
                let mut p = qkv.narrow(1, start, h * d)?.reshape(&[n, h, d])?;
                if let Some(name) = norm_name {
                    p = p.rms_norm(Some(need(name)?), c.eps, 0.0)?;
                }
                if let Some((theta, scaling, rotary_dim)) = c.rope {
                    let rope = Rope {
                        theta,
                        layout: c.rope_layout,
                        rotary_dim: Some(rotary_dim),
                        scaling,
                    };
                    p = p.rope(&pass.positions, rope)?;
                }
                Ok(p)
            };
            let q = split(0, hq, c.qk_norm.then_some("self_attn.q_norm.weight"))?;
            let k = split(wq, hkv, c.qk_norm.then_some("self_attn.k_norm.weight"))?;
            let v = qkv.narrow(1, wq + wk, wk)?;
            let as_heads = |p: &Var<B>, h: usize| -> Result<Var<B>> {
                p.reshape(&[pass.rows, pass.length, h, d])?.transpose(1, 2)
            };
            let y = as_heads(&q, hq)?.attention(
                &as_heads(&k, hkv)?,
                &heads(&v, hkv)?,
                pass.mask.as_ref(),
                attention,
            )?;
            let bias = if c.out_bias {
                w("self_attn.o_proj.bias")
            } else {
                None
            };
            let mut a = project(&merged(y)?, &["self_attn.o_proj"], bias)?;
            if sandwich {
                a = norm(c, &a, w("post_attention_layernorm.weight"), None)?;
            }
            let x = x.add(&a)?;
            let before = if sandwich {
                "pre_feedforward_layernorm.weight"
            } else {
                "post_attention_layernorm.weight"
            };
            let h = norm(c, &x, w(before), None)?;
            // The gate and up projections as one product, then the gated
            // unit: its derivative keeps one tensor instead of three.
            let inner = project(&h, &["mlp.gate_proj", "mlp.up_proj"], None)?.glu(c.activation)?;
            let mut m = project(&inner, &["mlp.down_proj"], None)?;
            if sandwich {
                m = norm(c, &m, w("post_feedforward_layernorm.weight"), None)?;
            }
            x.add(&m)
        }
        Block::Parallel => unreachable!("refused by Decoder::new"),
    }
}

/// A generator of reproducible numbers: SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// The generator of `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by the Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// A seed of its own for each parameter, from the run's seed and the
/// parameter's name: the initialization does not depend on their order.
pub(crate) fn seed_of(seed: u64, name: &str) -> u64 {
    // FNV-1a.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// How a parameter starts.
#[derive(Clone, Copy, Debug)]
enum Init {
    Normal(f64),
    Zeros,
    Ones,
}

/// The parameters of a new model of configuration `config`, in `f32`:
/// GPT-2's initialization (nanoGPT's) for every architecture: normal of
/// standard deviation `std` (0.02 in GPT-2) for the matrices and tables,
/// `std / √(2 · layers)` for the projections that add to the residual
/// stream, zeros for biases, ones for norms (zeros when the norm's weight
/// is stored minus one, Gemma).
///
/// # Errors
///
/// [`Error::InvalidArgument`] for an architecture [`Decoder::new`] refuses.
pub fn init(config: &Config, std: f64, seed: u64) -> Result<Parameters> {
    Decoder::new(config.clone())?;
    let c = config;
    let (h, v, i) = (c.hidden, c.vocab, c.intermediate);
    let (q, kv) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
    let residual = std / (2.0 * c.layers as f64).sqrt();
    let one = match c.norm {
        Norm::Rms { offset } if offset != 0.0 => Init::Zeros,
        _ => Init::Ones,
    };
    let mut shapes: Vec<(String, Vec<usize>, Init)> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, init: Init| shapes.push((name, shape, init));
    match c.block {
        Block::Gpt2 => {
            add(
                "transformer.wte.weight".into(),
                vec![v, h],
                Init::Normal(std),
            );
            let positions = c.max_positions.unwrap_or(1024);
            add(
                "transformer.wpe.weight".into(),
                vec![positions, h],
                Init::Normal(std),
            );
            for l in 0..c.layers {
                let p = |n: &str| format!("transformer.h.{l}.{n}");
                add(p("ln_1.weight"), vec![h], Init::Ones);
                add(p("ln_1.bias"), vec![h], Init::Zeros);
                add(p("attn.c_attn.weight"), vec![h, 3 * q], Init::Normal(std));
                add(p("attn.c_attn.bias"), vec![3 * q], Init::Zeros);
                add(p("attn.c_proj.weight"), vec![q, h], Init::Normal(residual));
                add(p("attn.c_proj.bias"), vec![h], Init::Zeros);
                add(p("ln_2.weight"), vec![h], Init::Ones);
                add(p("ln_2.bias"), vec![h], Init::Zeros);
                add(p("mlp.c_fc.weight"), vec![h, i], Init::Normal(std));
                add(p("mlp.c_fc.bias"), vec![i], Init::Zeros);
                add(p("mlp.c_proj.weight"), vec![i, h], Init::Normal(residual));
                add(p("mlp.c_proj.bias"), vec![h], Init::Zeros);
            }
            add("transformer.ln_f.weight".into(), vec![h], Init::Ones);
            add("transformer.ln_f.bias".into(), vec![h], Init::Zeros);
        }
        _ => {
            add(
                "model.embed_tokens.weight".into(),
                vec![v, h],
                Init::Normal(std),
            );
            for l in 0..c.layers {
                let p = |n: &str| format!("model.layers.{l}.{n}");
                add(p("input_layernorm.weight"), vec![h], one);
                add(p("self_attn.q_proj.weight"), vec![q, h], Init::Normal(std));
                add(p("self_attn.k_proj.weight"), vec![kv, h], Init::Normal(std));
                add(p("self_attn.v_proj.weight"), vec![kv, h], Init::Normal(std));
                if c.qkv_bias {
                    add(p("self_attn.q_proj.bias"), vec![q], Init::Zeros);
                    add(p("self_attn.k_proj.bias"), vec![kv], Init::Zeros);
                    add(p("self_attn.v_proj.bias"), vec![kv], Init::Zeros);
                }
                if c.qk_norm {
                    add(p("self_attn.q_norm.weight"), vec![c.head_dim], Init::Ones);
                    add(p("self_attn.k_norm.weight"), vec![c.head_dim], Init::Ones);
                }
                add(
                    p("self_attn.o_proj.weight"),
                    vec![h, q],
                    Init::Normal(residual),
                );
                if c.out_bias {
                    add(p("self_attn.o_proj.bias"), vec![h], Init::Zeros);
                }
                add(p("post_attention_layernorm.weight"), vec![h], one);
                if c.block == Block::Sandwich {
                    add(p("pre_feedforward_layernorm.weight"), vec![h], one);
                    add(p("post_feedforward_layernorm.weight"), vec![h], one);
                }
                add(p("mlp.gate_proj.weight"), vec![i, h], Init::Normal(std));
                add(p("mlp.up_proj.weight"), vec![i, h], Init::Normal(std));
                add(
                    p("mlp.down_proj.weight"),
                    vec![h, i],
                    Init::Normal(residual),
                );
            }
            add("model.norm.weight".into(), vec![h], one);
            if !c.tie_embeddings {
                add("lm_head.weight".into(), vec![v, h], Init::Normal(std));
            }
        }
    }
    let mut params = Parameters::default();
    for (name, shape, init) in shapes {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = match init {
            Init::Zeros => vec![0.0; n],
            Init::Ones => vec![1.0; n],
            Init::Normal(s) => {
                let mut rng = Rng::new(seed_of(seed, &name));
                (0..n).map(|_| (rng.normal() * s) as f32).collect()
            }
        };
        let value = Tensor::from_slice(&data, &shape)?;
        params.insert(name, Parameter::new(value));
    }
    Ok(params)
}
