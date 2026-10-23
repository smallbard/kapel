//! Quantizing models: rounding to the nearest, GPTQ and AWQ, from
//! calibration texts.
//!
//! [`quantize`] turns the linear projections of a model (attention and MLP)
//! into one of the formats Kapel writes: Kapel's int8 and int4 by groups,
//! GGUF's Q8_0, Q4_0 and Q4_1. Four [`Method`]s:
//!
//! - **Rounding to the nearest** ([`Method::Rtn`]): each block of weights
//!   by the reference rule of its format, without data: what
//!   [`Model::quantize`] does, and llama.cpp's `llama-quantize` without an
//!   importance matrix.
//! - **GPTQ** ([`Method::Gptq`], Frantar et al., 2022): the columns of each
//!   weight rounded one at a time, the error of each spread over the
//!   columns not yet rounded, as far as the inputs of the calibration texts
//!   correlate them.
//! - **AWQ** ([`Method::Awq`], Lin et al., 2023): the input channels that
//!   carry large activations made larger in the weights (and smaller in the
//!   operation before, which computes the same model) so that rounding
//!   spares them, by the factor that changes the outputs least; then the
//!   largest weights of each block clipped, by the ratio that changes the
//!   outputs least; then rounding to the nearest.
//! - **AWQ then GPTQ** ([`Method::AwqGptq`]): the scaled and clipped weights
//!   rounded by GPTQ.
//!
//! The model is computed one block at a time on a backend (the card, or the
//! processor), each block reading the outputs of the blocks before it, once
//! these are quantized: the error of a block is seen by the next. For each
//! group of projections that read one input ([`Site`]), the inputs `x` of
//! the calibration texts give the Hessian `H = Σ x xᵀ`, accumulated in
//! `f32` on the backend, and the output error of a change `δ` of a row of
//! weights is `δ H δᵀ`: GPTQ minimizes it column by column, and AWQ chooses
//! its factors and its clipping ratios by it. (AWQ's paper measures the
//! outputs of the whole attention or MLP for its factors; the outputs of the
//! projections themselves are measured here, from the same Hessians as
//! GPTQ.)

mod awq;
mod gptq;
mod linalg;

use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

use kapel_core::{DType, Tensor};
use kapel_models::{Block, Model, ModelError, Norm, Site};
use kapel_ops::{Backend, Binary, BinaryKind, Cast, Matmul, Sum, Transfer, Unary, UnaryKind};

/// How the weights are quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Rounding to the nearest, block by block, without data.
    Rtn,
    /// GPTQ.
    Gptq,
    /// AWQ: scales and clipping, then rounding to the nearest.
    Awq,
    /// AWQ's scales and clipping, then GPTQ.
    AwqGptq,
}

impl Method {
    /// The method named `name`: `rtn`, `gptq`, `awq` or `awq+gptq`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "rtn" => Self::Rtn,
            "gptq" => Self::Gptq,
            "awq" => Self::Awq,
            "awq+gptq" => Self::AwqGptq,
            _ => return None,
        })
    }

    /// Its name, as [`Method::from_name`] reads it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rtn => "rtn",
            Self::Gptq => "gptq",
            Self::Awq => "awq",
            Self::AwqGptq => "awq+gptq",
        }
    }

    /// Whether it reads calibration texts.
    pub fn calibrates(self) -> bool {
        self != Self::Rtn
    }

    fn awq(self) -> bool {
        matches!(self, Self::Awq | Self::AwqGptq)
    }

    fn gptq(self) -> bool {
        matches!(self, Self::Gptq | Self::AwqGptq)
    }
}

/// What to quantize, and how.
#[derive(Clone, Debug)]
pub struct Options {
    /// The format of the projections: Kapel's int8 or int4, or GGUF's Q8_0,
    /// Q4_0 or Q4_1.
    pub format: DType,
    /// The method.
    pub method: Method,
    /// Whether the output table (the embeddings, when the model ties them to
    /// it) is quantized too, by rounding to the nearest.
    pub output: bool,
    /// GPTQ's damping: the fraction of the mean of the Hessian's diagonal
    /// added to it (0.01, as in the paper).
    pub damping: f64,
    /// AWQ's grid: the exponents `i / grid` of its factors tried, and the
    /// clipping ratios `1 − i / grid` down to one half (20, as in the paper).
    pub grid: usize,
}

impl Options {
    /// `method` to `format`, with the papers' settings.
    pub fn new(format: DType, method: Method) -> Self {
        Self {
            format,
            method,
            output: false,
            damping: 0.01,
            grid: 20,
        }
    }
}

/// What can go wrong when quantizing.
#[derive(Debug)]
#[non_exhaustive]
pub enum QuantError {
    /// The model could not be read or run.
    Model(ModelError),
    /// An operation failed.
    Tensor(kapel_core::Error),
    /// Something this crate does not quantize.
    Unsupported(String),
    /// A Hessian GPTQ cannot use, even damped.
    Hessian(String),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Tensor(e) => e.fmt(f),
            Self::Unsupported(m) => f.write_str(m),
            Self::Hessian(m) => write!(f, "GPTQ: {m}"),
        }
    }
}

impl std::error::Error for QuantError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ModelError> for QuantError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<kapel_core::Error> for QuantError {
    fn from(e: kapel_core::Error) -> Self {
        Self::Tensor(e)
    }
}

/// A `Result` of this crate.
pub type Result<T> = std::result::Result<T, QuantError>;

/// What the calibration of a site gives: its Hessian `Σ x xᵀ` `[K, K]` and,
/// for AWQ, `Σ |x|` `[K]`, in `f32` on the backend, over `tokens` inputs.
struct Statistics {
    hessian: Tensor,
    magnitude: Option<Tensor>,
    tokens: usize,
}

/// A float weight in `f32`, in host memory, with its shape: a vector is one
/// row.
pub(crate) struct Weight {
    pub(crate) values: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

/// The model being quantized: in host memory, as it will be written, and
/// on the backend, in the type of the computations, as the next blocks see
/// it.
struct Work<'b, B> {
    backend: &'b B,
    compute: DType,
    host: Model,
    device: Model,
}

impl<B: Backend + Transfer> Work<'_, B> {
    fn weight(&self, name: &str) -> Result<Weight> {
        let w = self
            .host
            .tensor(name)
            .ok_or_else(|| QuantError::Model(ModelError::MissingWeight(name.into())))?;
        let (rows, cols) = match *w.shape() {
            [rows, cols] => (rows, cols),
            [cols] => (1, cols),
            _ => return Err(QuantError::Unsupported(format!("{name}: not a matrix"))),
        };
        let values = w.to_f64_vec()?.into_iter().map(|v| v as f32).collect();
        Ok(Weight { values, rows, cols })
    }

    /// Puts float `values` in place of the weight `name`, in its type.
    fn set_float(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let old = self
            .host
            .tensor(name)
            .ok_or_else(|| QuantError::Model(ModelError::MissingWeight(name.into())))?;
        let (dtype, shape) = (old.dtype(), old.shape().to_vec());
        let f32 = self.backend.upload(&Tensor::from_slice(values, &shape)?)?;
        let stored = if dtype == DType::F32 {
            f32
        } else {
            self.backend.forward(&Cast { dtype }, &[&f32])?
        };
        let device = if stored.dtype() == self.compute {
            stored.clone()
        } else {
            self.backend.forward(
                &Cast {
                    dtype: self.compute,
                },
                &[&stored],
            )?
        };
        self.host.replace(name, self.backend.download(&stored)?);
        self.device.replace(name, device);
        Ok(())
    }

    fn set_quantized(&mut self, name: &str, w: &Weight, bytes: &[u8], dtype: DType) -> Result<()> {
        let tensor = Tensor::from_bytes(dtype, &[w.rows, w.cols], bytes)?;
        self.device.replace(name, self.backend.upload(&tensor)?);
        self.host.replace(name, tensor);
        Ok(())
    }

    /// The statistics of `sites` of block `layer` over the hidden states
    /// `xs`, with the weights as they are now.
    fn statistics(
        &self,
        layer: usize,
        xs: &[Tensor],
        sites: &[Site],
        magnitude: bool,
    ) -> Result<BTreeMap<Site, Statistics>> {
        let backend = self.backend;
        let mut found: BTreeMap<Site, Statistics> = BTreeMap::new();
        let add = |a: &Tensor, b: &Tensor| {
            backend.forward(
                &Binary {
                    kind: BinaryKind::Add,
                },
                &[a, b],
            )
        };
        for x in xs {
            self.device.block(backend, layer, x, &mut |site, input| {
                if !sites.contains(&site) {
                    return Ok(());
                }
                let input = if input.dtype() == DType::F32 {
                    input.clone()
                } else {
                    backend.forward(&Cast { dtype: DType::F32 }, &[input])?
                };
                let h = backend.forward(&Matmul, &[&input.transpose(0, 1)?, &input])?;
                let m = if magnitude {
                    let abs = backend.forward(
                        &Unary {
                            kind: UnaryKind::Abs,
                        },
                        &[&input],
                    )?;
                    Some(backend.forward(
                        &Sum {
                            dims: vec![0],
                            keepdim: false,
                        },
                        &[&abs],
                    )?)
                } else {
                    None
                };
                let tokens = input.shape()[0];
                match found.get_mut(&site) {
                    Some(s) => {
                        s.hessian = add(&s.hessian, &h)?;
                        if let (Some(old), Some(new)) = (&s.magnitude, &m) {
                            s.magnitude = Some(add(old, new)?);
                        }
                        s.tokens += tokens;
                    }
                    None => {
                        found.insert(
                            site,
                            Statistics {
                                hessian: h,
                                magnitude: m,
                                tokens,
                            },
                        );
                    }
                }
                Ok(())
            })?;
        }
        Ok(found)
    }

    /// A tensor of the backend in host memory, in `f64`.
    fn host_values(&self, t: &Tensor) -> Result<Vec<f64>> {
        Ok(self.backend.download(t)?.to_f64_vec()?)
    }

    /// AWQ on block `layer`: for each site, the factors of its input
    /// channels and the clipping of its weights, from the statistics of the
    /// float block. A note per site.
    fn awq(
        &mut self,
        layer: usize,
        xs: &[Tensor],
        options: &Options,
        notes: &mut Vec<String>,
    ) -> Result<()> {
        let format = options.format;
        let c = self.host.config().clone();
        let name = |projection: &str| format!("model.layers.{layer}.{projection}.weight");
        let stats = self.statistics(layer, xs, &Site::ALL, true)?;
        let mut applied: BTreeMap<Site, Vec<f32>> = BTreeMap::new();
        for site in Site::ALL {
            let Some(s) = stats.get(&site) else { continue };
            let names: Vec<String> = site.projections().iter().map(|p| name(p)).collect();
            let weights = names
                .iter()
                .map(|n| self.weight(n))
                .collect::<Result<Vec<_>>>()?;
            if !weights.iter().all(|w| fits(w, format)) {
                continue;
            }
            let k = weights[0].cols;
            // Where the inverse factors go: a norm, or the rows of the
            // projection before (values into the output projection, up into
            // down), when their widths agree.
            let before = match site {
                Site::Attention | Site::Mlp => norm_before(&self.host, layer, site),
                Site::Output => {
                    let v = self.weight(&name("self_attn.v_proj"))?;
                    (v.rows == k).then(|| name("self_attn.v_proj"))
                }
                Site::Down => Some(name("mlp.up_proj")),
            };
            let Some(before) = before else {
                notes.push(format!("{site:?} unscaled"));
                continue;
            };
            let magnitude: Vec<f64> = self
                .host_values(s.magnitude.as_ref().expect("asked for"))?
                .into_iter()
                .map(|m| m / s.tokens as f64)
                .collect();
            let found = awq::scales(
                self.backend,
                &s.hessian,
                &magnitude,
                &weights,
                format,
                options.grid,
            )?;
            notes.push(format!(
                "{site:?} α {:.2} ({:.1} %)",
                found.exponent,
                100.0 * (1.0 - found.loss / found.plain)
            ));
            if found.exponent == 0.0 {
                continue;
            }
            let factors = found.factors;
            for (n, w) in names.iter().zip(&weights) {
                let scaled: Vec<f32> = w
                    .values
                    .iter()
                    .enumerate()
                    .map(|(e, &v)| v * factors[e % k])
                    .collect();
                self.set_float(n, &scaled)?;
            }
            // The operation before divides by the factors: a norm's weight
            // (Gemma's stores `weight − 1`), or rows, and their bias.
            let previous = self.weight(&before)?;
            let offset = match (c.norm, site) {
                (Norm::Rms { offset }, Site::Attention | Site::Mlp) => offset as f32,
                _ => 0.0,
            };
            let cols = previous.cols;
            let divided: Vec<f32> = previous
                .values
                .iter()
                .enumerate()
                .map(|(e, &v)| {
                    let f = if previous.rows == 1 {
                        factors[e]
                    } else {
                        factors[e / cols]
                    };
                    (v + offset) / f - offset
                })
                .collect();
            self.set_float(&before, &divided)?;
            let bias = before.replace(".weight", ".bias");
            if previous.rows > 1 && self.host.tensor(&bias).is_some() {
                let b = self.weight(&bias)?;
                let divided: Vec<f32> = b
                    .values
                    .iter()
                    .zip(&factors)
                    .map(|(&v, &f)| v / f)
                    .collect();
                self.set_float(&bias, &divided)?;
            }
            applied.insert(site, factors);
        }
        // Clipping, on the scaled weights, by the Hessians of the scaled
        // inputs; not the queries and keys, whose products the outputs of
        // their projections do not show (as in the paper).
        for site in Site::ALL {
            let Some(s) = stats.get(&site) else { continue };
            let mut h: Option<Vec<f64>> = None;
            for projection in site.projections() {
                if matches!(*projection, "self_attn.q_proj" | "self_attn.k_proj") {
                    continue;
                }
                let n = name(projection);
                let mut w = self.weight(&n)?;
                if !fits(&w, format) {
                    continue;
                }
                let h = match &mut h {
                    Some(h) => h,
                    None => {
                        let mut values = self.host_values(&s.hessian)?;
                        if let Some(f) = applied.get(&site) {
                            let k = f.len();
                            for (e, v) in values.iter_mut().enumerate() {
                                *v /= f64::from(f[e / k]) * f64::from(f[e % k]);
                            }
                        }
                        h.insert(values)
                    }
                };
                awq::clip(&mut w, h, format, options.grid);
                self.set_float(&n, &w.values)?;
            }
        }
        Ok(())
    }
}

/// Whether the rows of `w` are whole blocks of `format`.
fn fits(w: &Weight, format: DType) -> bool {
    w.cols.is_multiple_of(format.block_len())
}

/// The norm before a site, if a norm is: its weight's name.
fn norm_before(model: &Model, layer: usize, site: Site) -> Option<String> {
    let p = format!("model.layers.{layer}");
    match site {
        Site::Attention => Some(format!("{p}.input_layernorm.weight")),
        Site::Mlp => Some(match model.config().block {
            Block::Sandwich => format!("{p}.pre_feedforward_layernorm.weight"),
            _ => format!("{p}.post_attention_layernorm.weight"),
        }),
        _ => None,
    }
}

/// Quantizes the linear projections of `model` (a model as loaded, in host
/// memory) as `options` say, computing on `backend` in the float type
/// `compute` (`bf16` on a card, `f32` on the processor). GPTQ and AWQ read
/// the token ids of `calibration`, one sequence each; rounding to the
/// nearest reads nothing. `log` is told of the progress, a line per block.
///
/// The model returned is in host memory: its projections quantized, AWQ's
/// factors folded into the norms (and the projections) before them, the
/// rest as it was. Projections whose rows are not a whole number of blocks
/// of the format keep their type.
///
/// Blocks of Llama's kind only (Llama, Mistral, Qwen, Gemma) for GPTQ and
/// AWQ.
///
/// # Errors
///
/// For a format that is not written, another kind of block, no calibration
/// text for a method that needs one, a Hessian GPTQ cannot use, or an
/// operation that fails on the backend.
pub fn quantize<B: Backend + Transfer>(
    model: &Model,
    backend: &B,
    compute: DType,
    calibration: &[Vec<u32>],
    options: &Options,
    log: &mut dyn FnMut(&str),
) -> Result<Model> {
    let format = options.format;
    let probe = vec![0.0f32; format.block_len()];
    if !format.is_quantized() || kapel_core::block_scale(format, &probe).is_err() {
        return Err(QuantError::Unsupported(format!(
            "{format} is not written: int8, int4, q8_0, q4_0 and q4_1 are"
        )));
    }
    if options.method == Method::Rtn {
        return Ok(model.quantize(format, options.output)?);
    }
    let c = model.config().clone();
    if !matches!(c.block, Block::Sequential | Block::Sandwich) {
        return Err(QuantError::Unsupported(format!(
            "{} is quantized by rounding only: GPTQ and AWQ compute blocks of Llama's kind",
            c.model_type
        )));
    }
    if calibration.iter().all(Vec::is_empty) {
        return Err(QuantError::Unsupported(format!(
            "{} needs calibration texts",
            options.method.name()
        )));
    }
    let device = model.upload(backend)?.cast(backend, compute)?;
    let mut work = Work {
        backend,
        compute,
        host: model.clone(),
        device,
    };
    let mut xs = calibration
        .iter()
        .filter(|ids| !ids.is_empty())
        .map(|ids| work.device.embed(backend, ids))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let tokens: usize = xs.iter().map(|x| x.shape()[0]).sum();
    log(&format!(
        "{} to {format}: {} sequences, {tokens} tokens of calibration, {} blocks",
        options.method.name(),
        xs.len(),
        c.layers
    ));
    for layer in 0..c.layers {
        let start = Instant::now();
        let name = |projection: &str| format!("model.layers.{layer}.{projection}.weight");
        let mut notes = Vec::new();
        if options.method.awq() {
            work.awq(layer, &xs, options, &mut notes)?;
        }
        for site in Site::ALL {
            let names: Vec<String> = site.projections().iter().map(|p| name(p)).collect();
            if options.method.gptq() {
                let stats = work.statistics(layer, &xs, &[site], false)?;
                let Some(s) = stats.get(&site) else { continue };
                let k = s.hessian.shape()[0];
                let mut h = work.host_values(&s.hessian)?;
                let (u, damping, dead) =
                    gptq::prepare(&mut h, k, options.damping).map_err(QuantError::Hessian)?;
                if damping != options.damping {
                    notes.push(format!("{site:?} damped {damping}"));
                }
                if !dead.is_empty() {
                    notes.push(format!("{site:?} {} inputs never reached", dead.len()));
                }
                for n in &names {
                    let w = work.weight(n)?;
                    if fits(&w, format) {
                        let bytes = gptq::quantize(&w.values, w.rows, w.cols, format, &u, &dead);
                        work.set_quantized(n, &w, &bytes, format)?;
                    }
                }
            } else {
                for n in &names {
                    let w = work.weight(n)?;
                    if fits(&w, format) {
                        let bytes = kapel_core::quantize(format, &w.values)?;
                        work.set_quantized(n, &w, &bytes, format)?;
                    }
                }
            }
        }
        // The inputs of the next block: the outputs of this one, quantized.
        xs = xs
            .iter()
            .map(|x| work.device.block(backend, layer, x, &mut |_, _| Ok(())))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        log(&format!(
            "block {}/{}: {:.1} s{}{}",
            layer + 1,
            c.layers,
            start.elapsed().as_secs_f64(),
            if notes.is_empty() { "" } else { ", " },
            notes.join(", ")
        ));
    }
    let mut host = work.host;
    if options.output {
        let head = if c.tie_embeddings || host.tensor("lm_head.weight").is_none() {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let w = host
            .tensor(head)
            .ok_or_else(|| QuantError::Model(ModelError::MissingWeight(head.into())))?
            .clone();
        if w.rank() == 2 && w.shape()[1].is_multiple_of(format.block_len()) && w.dtype().is_float()
        {
            let values: Vec<f32> = w.to_f64_vec()?.into_iter().map(|v| v as f32).collect();
            let bytes = kapel_core::quantize(format, &values)?;
            host.replace(head, Tensor::from_bytes(format, w.shape(), &bytes)?);
        }
    }
    Ok(host)
}
