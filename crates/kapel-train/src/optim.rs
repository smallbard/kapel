//! Optimizers.

use std::collections::{BTreeMap, BTreeSet};

use kapel_core::{DType, Error, Result, Tensor};
use kapel_ops::{Affine, Binary, BinaryKind, Cast, GatherRows, ScatterAddRows, Unary, UnaryKind};

use crate::model::Parameters;
use crate::tape::{Compute, Grad};

/// Adam with decoupled weight decay (Loshchilov and Hutter, 2019).
///
/// For a gradient `g` at step `t` (from 1), with the moments `m` and `v`
/// starting at zero:
///
/// - `m ← β₁ m + (1 − β₁) g`, `v ← β₂ v + (1 − β₂) g²`;
/// - `p ← p − lr · (λ p + m̂ / (√v̂ + ε))`, with `m̂ = m / (1 − β₁ᵗ)` and
///   `v̂ = v / (1 − β₂ᵗ)`, and `λ` the weight decay of parameters that
///   decay ([`crate::model::Parameter::decay`]).
///
/// A gradient by rows ([`Grad::Rows`]) updates only those rows, their
/// moments included: the AdamW by rows of Smallem's memory tables. Rows
/// that appear several times are summed first. The other rows keep their
/// values and moments; the bias correction is that of the global step.
///
/// Parameters, moments and the update are in `f32`.
///
/// With [`AdamW::eight_bit`], the moments of a parameter in `f32` of at
/// least [`MIN_8BIT`] elements with a dense gradient are kept in 8 bits, as
/// bitsandbytes' `AdamW8bit` keeps them ([`kapel_ops::Update::adamw8bit`]):
/// a code a moment and a scale every 256 elements, 2 bytes a parameter
/// instead of 8. The step is computed in `f32` from the moments read back.
/// Smaller parameters, the tables (embeddings, positions, the output, as
/// bitsandbytes advises with its `StableEmbedding`) and parameters in other
/// types keep their moments in `f32`.
#[derive(Debug)]
pub struct AdamW {
    /// The learning rate.
    pub lr: f64,
    /// The decay of the first moment.
    pub beta1: f64,
    /// The decay of the second moment.
    pub beta2: f64,
    /// Added to the root of the second moment.
    pub eps: f64,
    /// The weight decay `λ`.
    pub weight_decay: f64,
    /// Whether the moments are kept in 8 bits where they can be.
    pub eight_bit: bool,
    steps: u64,
    moments: BTreeMap<String, (Tensor, Tensor)>,
    quantized: BTreeMap<String, kapel_ops::Moments8>,
    /// The maps of the codes ([`kapel_ops::dynamic_map`]) on the device.
    maps: Option<Tensor>,
}

/// The fewest elements of a parameter whose moments [`AdamW::eight_bit`]
/// keeps in 8 bits: bitsandbytes' `min_8bit_size`.
pub const MIN_8BIT: usize = 4096;

impl AdamW {
    /// With nanoGPT's settings: `β₁ = 0.9`, `β₂ = 0.95`, `ε = 10⁻⁸`.
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
            eight_bit: false,
            steps: 0,
            moments: BTreeMap::new(),
            quantized: BTreeMap::new(),
            maps: None,
        }
    }

    /// The steps made.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The moments `(m, v)` of each parameter that has had a gradient.
    pub fn moments(&self) -> &BTreeMap<String, (Tensor, Tensor)> {
        &self.moments
    }

    /// The moments kept in 8 bits, by parameter.
    pub fn quantized(&self) -> &BTreeMap<String, kapel_ops::Moments8> {
        &self.quantized
    }

    /// Puts the optimizer back in a saved state: steps and moments.
    pub fn restore(&mut self, steps: u64, moments: BTreeMap<String, (Tensor, Tensor)>) {
        self.steps = steps;
        self.moments = moments;
        self.quantized.clear();
    }

    /// The maps of the codes on the backend's device, uploaded once.
    fn maps<B: Compute>(&mut self, backend: &B, device: kapel_core::Device) -> Result<Tensor> {
        if let Some(maps) = &self.maps
            && maps.device() == device
        {
            return Ok(maps.clone());
        }
        let mut table = kapel_ops::dynamic_map(true).to_vec();
        table.extend(kapel_ops::dynamic_map(false));
        let maps = backend.upload(&Tensor::from_slice(&table, &[512])?)?;
        self.maps = Some(maps.clone());
        Ok(maps)
    }

    /// One step: each parameter of `params` with a gradient in `grads` (by
    /// name) moves; the others are untouched. `grad_scale` multiplies every
    /// gradient first (`1 / accumulated steps`, a clipping factor).
    ///
    /// The parameters must be the only handles on their values: the tape
    /// and its variables dropped.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a gradient of an unknown parameter;
    /// those of the backend.
    pub fn step<B: Compute>(
        &mut self,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        grad_scale: f64,
    ) -> Result<()> {
        self.advance();
        self.update(backend, params, grads, grad_scale)
    }

    /// Counts a step begun: [`AdamW::update`] then moves parameters with
    /// its bias corrections, all at once or a few at a time.
    pub fn advance(&mut self) {
        self.steps += 1;
    }

    /// The moves of the step [`AdamW::advance`] began, for the parameters
    /// with a gradient in `grads`: [`AdamW::step`] without counting a step.
    ///
    /// # Errors
    ///
    /// As [`AdamW::step`].
    pub fn update<B: Compute>(
        &mut self,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        grad_scale: f64,
    ) -> Result<()> {
        let t = self.steps as i32;
        let c1 = 1.0 - self.beta1.powi(t);
        let c2 = 1.0 - self.beta2.powi(t);
        for (name, grad) in grads {
            let param = params
                .get_mut(&name)
                .ok_or_else(|| Error::InvalidArgument {
                    op: "AdamW::step",
                    message: format!("a gradient for {name}, which is not a parameter"),
                })?;
            let decay = if param.decay { self.weight_decay } else { 0.0 };
            let shape = param.value.shape().to_vec();
            let s = Step {
                lr: self.lr * param.lr_scale,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                decay,
                c1,
                c2,
                grad_scale,
            };
            let eight = self.eight_bit
                && param.value.dtype() == DType::F32
                && param.value.numel() >= MIN_8BIT
                && !crate::model::is_table(&name)
                && !matches!(grad, Grad::Rows { .. })
                && !self.moments.contains_key(&name);
            if eight {
                let mut moments = match self.quantized.remove(&name) {
                    Some(q) => q,
                    None => {
                        let blocks = [param.value.numel().div_ceil(kapel_ops::BLOCK8)];
                        kapel_ops::Moments8 {
                            m: backend.zeros(&shape, DType::U8)?,
                            v: backend.zeros(&shape, DType::U8)?,
                            m_scale: backend.zeros(&blocks, DType::F32)?,
                            v_scale: backend.zeros(&blocks, DType::F32)?,
                        }
                    }
                };
                let maps = self.maps(backend, param.value.device())?;
                let g = grad.dense(backend, &shape)?;
                unshare(backend, &mut param.value)?;
                backend.adamw8bit(&mut param.value, &mut moments, &g, &maps, &s.fused())?;
                self.quantized.insert(name, moments);
                continue;
            }
            let (m, v) = match self.moments.remove(&name) {
                Some(mv) => mv,
                None => (
                    backend.zeros(&shape, DType::F32)?,
                    backend.zeros(&shape, DType::F32)?,
                ),
            };
            let (m, v) = match grad {
                Grad::Rows { indices, values } => {
                    by_rows(backend, &mut param.value, m, v, &indices, &values, &s)?
                }
                grad => {
                    let g = grad.dense(backend, &shape)?;
                    let (mut m, mut v) = (m, v);
                    if param.value.dtype() == DType::F32 {
                        unshare(backend, &mut param.value)?;
                        backend.adamw(&mut param.value, &mut m, &mut v, &g, &s.fused())?;
                        (m, v)
                    } else {
                        let (p, m, v) = s.apply(backend, &param.value, m, v, &g)?;
                        param.value = p;
                        (m, v)
                    }
                }
            };
            self.moments.insert(name, (m, v));
        }
        Ok(())
    }
}

/// The constants of one step for one parameter.
struct Step {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    decay: f64,
    c1: f64,
    c2: f64,
    grad_scale: f64,
}

impl Step {
    /// The constants of the backend's fused step ([`kapel_ops::Update`]).
    fn fused(&self) -> kapel_ops::AdamWStep {
        kapel_ops::AdamWStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.decay,
            correction1: self.c1,
            correction2: self.c2,
            grad_scale: self.grad_scale,
        }
    }

    /// The step composed of operations of the catalogue, for parameters
    /// that are not in `f32`.
    /// The parameter and moments after the step, for a gradient `g` of
    /// their shape.
    fn apply<B: Compute>(
        &self,
        backend: &B,
        p: &Tensor,
        m: Tensor,
        v: Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let affine =
            |x: &Tensor, scale: f64, bias: f64| backend.forward(&Affine { scale, bias }, &[x]);
        let binary = |kind, a: &Tensor, b: &Tensor| backend.forward(&Binary { kind }, &[a, b]);
        let g = if g.dtype() == DType::F32 {
            g.clone()
        } else {
            backend.forward(&Cast { dtype: DType::F32 }, &[g])?
        };
        // m ← β₁ m + (1 − β₁) s g; v ← β₂ v + (1 − β₂) s² g².
        let m = binary(
            BinaryKind::Add,
            &affine(&m, self.beta1, 0.0)?,
            &affine(&g, (1.0 - self.beta1) * self.grad_scale, 0.0)?,
        )?;
        let g2 = backend.forward(
            &Unary {
                kind: UnaryKind::Square,
            },
            &[&g],
        )?;
        let v = binary(
            BinaryKind::Add,
            &affine(&v, self.beta2, 0.0)?,
            &affine(
                &g2,
                (1.0 - self.beta2) * self.grad_scale * self.grad_scale,
                0.0,
            )?,
        )?;
        // √v̂ + ε = √(v / c₂) + ε; the update lr · m̂ / (√v̂ + ε).
        let root = backend.forward(
            &Unary {
                kind: UnaryKind::Sqrt,
            },
            &[&affine(&v, 1.0 / self.c2, 0.0)?],
        )?;
        let denominator = affine(&root, 1.0, self.eps)?;
        let update = binary(
            BinaryKind::Div,
            &affine(&m, self.lr / self.c1, 0.0)?,
            &denominator,
        )?;
        // p ← (1 − lr λ) p − update.
        let wide = if p.dtype() == DType::F32 {
            p.clone()
        } else {
            backend.forward(&Cast { dtype: DType::F32 }, &[p])?
        };
        let p_next = binary(
            BinaryKind::Sub,
            &affine(&wide, 1.0 - self.lr * self.decay, 0.0)?,
            &update,
        )?;
        let p_next = if p.dtype() == DType::F32 {
            p_next
        } else {
            backend.forward(&Cast { dtype: p.dtype() }, &[&p_next])?
        };
        Ok((p_next, m, v))
    }
}

/// `t` with a storage of its own, that may be written in place: copied if
/// another handle shares it (a clone of the parameters, say), or if it is a
/// file mapped read-only.
fn unshare<B: Compute>(backend: &B, t: &mut Tensor) -> Result<()> {
    if t.is_contiguous() && t.layout().offset() == 0 && t.storage_mut().is_some() {
        let writable = t.storage_mut().and_then(|s| s.as_any_mut()).is_some();
        if writable {
            return Ok(());
        }
    }
    let host = backend.download(t)?.contiguous()?;
    let bytes = host
        .storage()
        .host_bytes()
        .map(|b| {
            let start = host.layout().offset() * host.dtype().block_bytes();
            b[start..start + host.dtype().storage_bytes(host.numel()).unwrap_or(0)].to_vec()
        })
        .unwrap_or_default();
    *t = backend.upload(&Tensor::from_bytes(host.dtype(), host.shape(), &bytes)?)?;
    Ok(())
}

/// The step of the rows `indices` of `p` only, written in place, and the
/// moments with those rows updated.
fn by_rows<B: Compute>(
    backend: &B,
    p: &mut Tensor,
    mut m: Tensor,
    mut v: Tensor,
    indices: &Tensor,
    values: &Tensor,
    s: &Step,
) -> Result<(Tensor, Tensor)> {
    unshare(backend, p)?;
    let rows = p.shape()[0];
    let host = backend.download(indices)?;
    let listed: Vec<i64> = host.to_f64_vec()?.into_iter().map(|x| x as i64).collect();
    let unique: BTreeSet<i64> = listed.iter().copied().collect();
    if let Some(&bad) = unique.iter().find(|&&r| r < 0 || r as usize >= rows) {
        return Err(Error::IndexOutOfRange {
            op: "AdamW::step",
            index: bad,
            size: rows,
        });
    }
    let unique: Vec<u32> = unique.into_iter().map(|r| r as u32).collect();
    let position: BTreeMap<u32, u32> = unique
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();
    let remapped: Vec<u32> = listed.iter().map(|&r| position[&(r as u32)]).collect();
    let upload = |data: &[u32]| -> Result<Tensor> {
        backend.upload(&Tensor::from_slice(data, &[data.len()])?)
    };
    let (unique_t, remapped_t) = (upload(&unique)?, upload(&remapped)?);
    let width: Vec<usize> = p.shape()[1..].to_vec();
    let mut flat = vec![listed.len()];
    flat.extend(&width);
    let values = if values.is_contiguous() {
        values.reshape(&flat)?
    } else {
        backend
            .forward(&kapel_ops::Contiguous, &[values])?
            .reshape(&flat)?
    };
    let g = backend.forward(
        &ScatterAddRows { rows: unique.len() },
        &[&values, &remapped_t],
    )?;
    let gather = |t: &Tensor| backend.forward(&GatherRows, &[t, &unique_t]);
    let (p_rows, m_rows, v_rows) = (gather(p)?, gather(&m)?, gather(&v)?);
    let (p_rows, m_rows, v_rows) = if p_rows.dtype() == DType::F32 {
        let (mut p_rows, mut m_rows, mut v_rows) = (p_rows, m_rows, v_rows);
        backend.adamw(&mut p_rows, &mut m_rows, &mut v_rows, &g, &s.fused())?;
        (p_rows, m_rows, v_rows)
    } else {
        s.apply(backend, &p_rows, m_rows, v_rows, &g)?
    };
    backend.write_rows(p, &unique_t, &p_rows)?;
    backend.write_rows(&mut m, &unique_t, &m_rows)?;
    backend.write_rows(&mut v, &unique_t, &v_rows)?;
    Ok((m, v))
}

/// The state of an optimizer, as a checkpoint saves it: the steps made, and
/// its tensors by name (`"m.<parameter>"`, `"v.<parameter>"`...).
#[derive(Clone, Debug, Default)]
pub struct OptimizerState {
    /// The steps made.
    pub steps: u64,
    /// The tensors.
    pub tensors: BTreeMap<String, Tensor>,
}

impl AdamW {
    /// The state, for a checkpoint.
    pub fn state(&self) -> OptimizerState {
        let mut tensors = BTreeMap::new();
        for (name, (m, v)) in &self.moments {
            tensors.insert(format!("m.{name}"), m.clone());
            tensors.insert(format!("v.{name}"), v.clone());
        }
        for (name, q) in &self.quantized {
            tensors.insert(format!("m.{name}"), q.m.clone());
            tensors.insert(format!("v.{name}"), q.v.clone());
            tensors.insert(format!("m_scale.{name}"), q.m_scale.clone());
            tensors.insert(format!("v_scale.{name}"), q.v_scale.clone());
        }
        OptimizerState {
            steps: self.steps,
            tensors,
        }
    }

    /// Back to a saved state. Moments in `u8` are codes, with their scales.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if a moment lacks its pair, or codes their
    /// scales.
    pub fn restore_state(&mut self, state: OptimizerState) -> Result<()> {
        let mut moments = BTreeMap::new();
        let mut quantized = BTreeMap::new();
        let mut tensors = state.tensors;
        let names: Vec<String> = tensors
            .keys()
            .filter_map(|k| k.strip_prefix("m.").map(str::to_string))
            .collect();
        for name in names {
            let m = tensors.remove(&format!("m.{name}")).expect("listed");
            let v = tensors
                .remove(&format!("v.{name}"))
                .ok_or_else(|| Error::InvalidArgument {
                    op: "AdamW::restore_state",
                    message: format!("a first moment without a second for {name}"),
                })?;
            if m.dtype() == DType::U8 {
                let mut scale = |prefix: &str| {
                    tensors.remove(&format!("{prefix}.{name}")).ok_or_else(|| {
                        Error::InvalidArgument {
                            op: "AdamW::restore_state",
                            message: format!("moments in 8 bits without {prefix} for {name}"),
                        }
                    })
                };
                let (m_scale, v_scale) = (scale("m_scale")?, scale("v_scale")?);
                quantized.insert(
                    name,
                    kapel_ops::Moments8 {
                        m,
                        v,
                        m_scale,
                        v_scale,
                    },
                );
            } else {
                moments.insert(name, (m, v));
            }
        }
        self.restore(state.steps, moments);
        self.quantized = quantized;
        Ok(())
    }
}

/// `scale · x`.
fn scaled<B: Compute>(backend: &B, x: &Tensor, scale: f64) -> Result<Tensor> {
    backend.forward(&Affine { scale, bias: 0.0 }, &[x])
}

fn binary<B: Compute>(backend: &B, kind: BinaryKind, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    backend.forward(&Binary { kind }, &[a, b])
}

/// `x` in `f32`.
fn wide<B: Compute>(backend: &B, x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::F32 {
        Ok(x.clone())
    } else {
        backend.forward(&Cast { dtype: DType::F32 }, &[x])
    }
}

/// The coefficients `(a, b, c)` of Newton-Schulz's iterations: eight that
/// make small singular values grow fast, then two that bring them to 1;
/// those of DeepSeek-V4, which Smallem measured.
pub const NEWTON_SCHULZ: [(f64, f64, f64); 10] = {
    let fast = (3.4445, -4.7750, 2.0315);
    let exact = (2.0, -1.5, 0.5);
    [fast, fast, fast, fast, fast, fast, fast, fast, exact, exact]
};

/// Four iterations of coefficients all different, tuned so that the answer
/// is near 1 once a singular value passes 0.02 of the norm: twelve products
/// instead of thirty. Those of SAMuon (arXiv 2608.25990, table 4), Smallem's
/// `muon:short`, with which its laws of scale were measured.
pub const NEWTON_SCHULZ_SHORT: [(f64, f64, f64); 4] = [
    (5.30697775, -9.73226547, 4.52926445),
    (3.99123669, -4.20899105, 1.18235242),
    (2.66316843, -2.16650701, 0.56325209),
    (1.93040931, -1.31219244, 0.38289258),
];

/// The orthogonal matrices nearest to a stack `[K, m, n]` of matrices, by
/// Newton-Schulz's iterations `X ← a X + (b A + c A²) X`, `A = X Xᵀ` (or
/// `Xᵀ X` on the other side for a tall matrix), after dividing each by its
/// norm, with the coefficients `(a, b, c)` of each iteration
/// ([`NEWTON_SCHULZ`] or [`NEWTON_SCHULZ_SHORT`]). Computed in `f32`.
///
/// # Errors
///
/// Those of the backend.
pub fn orthogonalize<B: Compute>(
    backend: &B,
    stack: &Tensor,
    coefficients: &[(f64, f64, f64)],
) -> Result<Tensor> {
    use kapel_ops::{Matmul, Sum};
    let shape = stack.shape().to_vec();
    let (m, n) = (shape[1], shape[2]);
    let wide = m <= n;
    let squares = backend.forward(
        &Unary {
            kind: UnaryKind::Square,
        },
        &[stack],
    )?;
    let norms = backend.forward(
        &Sum {
            dims: vec![1, 2],
            keepdim: true,
        },
        &[&squares],
    )?;
    let norms = backend.forward(
        &Unary {
            kind: UnaryKind::Sqrt,
        },
        &[&norms],
    )?;
    let norms = backend.forward(
        &Affine {
            scale: 1.0,
            bias: 1e-7,
        },
        &[&norms],
    )?;
    let mut x = binary(backend, BinaryKind::Div, stack, &norms)?;
    for &(a, b, c) in coefficients {
        let xt = x.transpose(1, 2)?;
        let gram = if wide {
            backend.forward(&Matmul, &[&x, &xt])?
        } else {
            backend.forward(&Matmul, &[&xt, &x])?
        };
        let square = backend.forward(&Matmul, &[&gram, &gram])?;
        let polynomial = binary(
            backend,
            BinaryKind::Add,
            &scaled(backend, &gram, b)?,
            &scaled(backend, &square, c)?,
        )?;
        let correction = if wide {
            backend.forward(&Matmul, &[&polynomial, &x])?
        } else {
            backend.forward(&Matmul, &[&x, &polynomial])?
        };
        x = binary(
            backend,
            BinaryKind::Add,
            &scaled(backend, &x, a)?,
            &correction,
        )?;
    }
    Ok(x)
}

/// Muon for the matrices of the blocks, AdamW for the rest (Jordan et al.,
/// 2024; the variant of DeepSeek-V4 that Smallem measured).
///
/// For the gradient `G` of a matrix `[m, n]`, with the momentum `M` starting
/// at zero:
///
/// - `M ← μ M + G`, then Nesterov's look ahead `U = μ M + G`;
/// - `O` = the orthogonal matrix nearest to `U` ([`orthogonalize`]);
/// - `W ← (1 − lr λ) W − lr · γ √max(m, n) · O`.
///
/// An orthogonal matrix has elements of size `1 / √max(m, n)`: the factor
/// `γ √max(m, n)` (γ = 0.18) gives the update the size of an AdamW step, so
/// one learning rate serves both. The matrices of one shape are
/// orthogonalized together, one batched product for all of them.
///
/// Muon takes the parameters that decay with two dimensions, but the
/// tables (embeddings, positions, the output table): those stay with AdamW.
#[derive(Debug)]
pub struct Muon {
    /// The optimizer of the other parameters; its learning rate and weight
    /// decay are those of both.
    pub adamw: AdamW,
    /// `μ`.
    pub momentum: f64,
    /// `γ`.
    pub update_scale: f64,
    /// Muon's learning rate as a factor of AdamW's.
    pub lr_scale: f64,
    /// The coefficients of Newton-Schulz's iterations.
    pub coefficients: &'static [(f64, f64, f64)],
    momenta: BTreeMap<String, Tensor>,
}

/// Whether Muon takes the parameter `name`.
pub fn muon_takes(name: &str, param: &crate::model::Parameter) -> bool {
    param.decay && param.value.rank() == 2 && !crate::model::is_table(name)
}

impl Muon {
    /// Muon at `lr`, with `weight_decay`, its momentum 0.95.
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            adamw: AdamW::new(lr, weight_decay),
            momentum: 0.95,
            update_scale: 0.18,
            lr_scale: 1.0,
            coefficients: &NEWTON_SCHULZ,
            momenta: BTreeMap::new(),
        }
    }

    /// One step, as [`AdamW::step`].
    ///
    /// # Errors
    ///
    /// As [`AdamW::step`]; a gradient by rows for a matrix Muon takes is
    /// refused.
    pub fn step<B: Compute>(
        &mut self,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        grad_scale: f64,
    ) -> Result<()> {
        let (mut mine, rest): (Vec<_>, Vec<_>) = grads
            .into_iter()
            .partition(|(n, _)| params.get(n).is_some_and(|p| muon_takes(n, p)));
        // AdamW first: it counts the steps.
        self.adamw.step(backend, params, rest, grad_scale)?;
        let lr = self.adamw.lr * self.lr_scale;
        let mu = self.momentum;
        let mut by_shape: BTreeMap<Vec<usize>, Vec<(String, Tensor)>> = BTreeMap::new();
        for (name, grad) in mine.drain(..) {
            let shape = params
                .get(&name)
                .expect("partitioned")
                .value
                .shape()
                .to_vec();
            let g = match grad {
                Grad::Rows { .. } => {
                    return Err(Error::InvalidArgument {
                        op: "Muon::step",
                        message: format!("{name}: a gradient by rows for a matrix"),
                    });
                }
                g => wide(backend, &g.dense(backend, &shape)?)?,
            };
            let g = scaled(backend, &g, grad_scale)?;
            let momentum = match self.momenta.remove(&name) {
                Some(m) => binary(backend, BinaryKind::Add, &scaled(backend, &m, mu)?, &g)?,
                None => g.clone(),
            };
            let ahead = binary(
                backend,
                BinaryKind::Add,
                &scaled(backend, &momentum, mu)?,
                &g,
            )?;
            self.momenta.insert(name.clone(), momentum);
            by_shape.entry(shape).or_default().push((name, ahead));
        }
        for (shape, matrices) in by_shape {
            let parts: Vec<Tensor> = matrices
                .iter()
                .map(|(_, a)| a.reshape(&[1, shape[0], shape[1]]))
                .collect::<Result<_>>()?;
            let refs: Vec<&Tensor> = parts.iter().collect();
            let stack = if refs.len() == 1 {
                parts[0].clone()
            } else {
                backend.forward(&kapel_ops::Concat { dim: 0 }, &refs)?
            };
            let orthogonal = orthogonalize(backend, &stack, self.coefficients)?;
            let size = shape[0].max(shape[1]) as f64;
            let step = lr * self.update_scale * size.sqrt();
            for (i, (name, _)) in matrices.iter().enumerate() {
                let param = params.get_mut(name).expect("a parameter");
                let decay = if param.decay {
                    self.adamw.weight_decay
                } else {
                    0.0
                };
                let o = orthogonal.narrow(0, i, 1)?.reshape(&shape)?;
                let p = wide(backend, &param.value)?;
                let next = binary(
                    backend,
                    BinaryKind::Sub,
                    &scaled(backend, &p, 1.0 - lr * param.lr_scale * decay)?,
                    &scaled(backend, &o, step * param.lr_scale)?,
                )?;
                param.value = if param.value.dtype() == DType::F32 {
                    next
                } else {
                    backend.forward(
                        &Cast {
                            dtype: param.value.dtype(),
                        },
                        &[&next],
                    )?
                };
            }
        }
        Ok(())
    }

    /// The state, for a checkpoint: AdamW's, and the momenta.
    pub fn state(&self) -> OptimizerState {
        let mut state = self.adamw.state();
        for (name, m) in &self.momenta {
            state.tensors.insert(format!("muon.{name}"), m.clone());
        }
        state
    }

    /// Back to a saved state.
    ///
    /// # Errors
    ///
    /// As [`AdamW::restore_state`].
    pub fn restore_state(&mut self, mut state: OptimizerState) -> Result<()> {
        let names: Vec<String> = state
            .tensors
            .keys()
            .filter(|k| k.starts_with("muon."))
            .cloned()
            .collect();
        self.momenta.clear();
        for key in names {
            let m = state.tensors.remove(&key).expect("listed");
            self.momenta.insert(key["muon.".len()..].to_string(), m);
        }
        self.adamw.restore_state(state)
    }
}

/// Adafactor (Shazeer and Stern, 2018), without first moment: the second
/// moment of a matrix `[n, m]` kept as its means by row `r` and by column
/// `c`, `v̂ = r cᵀ / mean(r)`, in `n + m` values instead of `n m`.
///
/// At step `t`, `β₂ = 1 − t^−0.8`; `r` and `c` follow the means of `g² +
/// 10⁻³⁰`; the update `u = g / √v̂` is divided by `max(1, rms(u))`; then `p
/// ← (1 − lr λ) p − lr · u`. Other shapes keep their whole second moment.
#[derive(Debug)]
pub struct Adafactor {
    /// The learning rate.
    pub lr: f64,
    /// The weight decay `λ`.
    pub weight_decay: f64,
    steps: u64,
    moments: BTreeMap<String, Vec<Tensor>>,
}

impl Adafactor {
    /// Adafactor at `lr`, with `weight_decay`.
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            weight_decay,
            steps: 0,
            moments: BTreeMap::new(),
        }
    }

    /// One step, as [`AdamW::step`]. Gradients by rows are made dense.
    ///
    /// # Errors
    ///
    /// As [`AdamW::step`].
    pub fn step<B: Compute>(
        &mut self,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        grad_scale: f64,
    ) -> Result<()> {
        use kapel_ops::Mean;
        self.steps += 1;
        let beta = 1.0 - (self.steps as f64).powf(-0.8);
        let mean = |x: &Tensor, dims: Vec<usize>| {
            backend.forward(
                &Mean {
                    dims,
                    keepdim: true,
                },
                &[x],
            )
        };
        let unary = |kind, x: &Tensor| backend.forward(&Unary { kind }, &[x]);
        for (name, grad) in grads {
            let param = params
                .get_mut(&name)
                .ok_or_else(|| Error::InvalidArgument {
                    op: "Adafactor::step",
                    message: format!("a gradient for {name}, which is not a parameter"),
                })?;
            let shape = param.value.shape().to_vec();
            let g = scaled(
                backend,
                &wide(backend, &grad.dense(backend, &shape)?)?,
                grad_scale,
            )?;
            let g2 = backend.forward(
                &Affine {
                    scale: 1.0,
                    bias: 1e-30,
                },
                &[&unary(UnaryKind::Square, &g)?],
            )?;
            let follow = |old: Option<&Tensor>, new: Tensor| -> Result<Tensor> {
                match old {
                    Some(o) => binary(
                        backend,
                        BinaryKind::Add,
                        &scaled(backend, o, beta)?,
                        &scaled(backend, &new, 1.0 - beta)?,
                    ),
                    None => Ok(new),
                }
            };
            let old = self.moments.remove(&name);
            let (v, kept) = if shape.len() == 2 {
                let r = follow(old.as_ref().map(|m| &m[0]), mean(&g2, vec![1])?)?;
                let c = follow(old.as_ref().map(|m| &m[1]), mean(&g2, vec![0])?)?;
                let rr = mean(&r, vec![0, 1])?;
                let v = binary(
                    backend,
                    BinaryKind::Div,
                    &binary(backend, BinaryKind::Mul, &r, &c)?,
                    &rr,
                )?;
                (v, vec![r, c])
            } else {
                let v = follow(old.as_ref().map(|m| &m[0]), g2)?;
                (v.clone(), vec![v])
            };
            self.moments.insert(name.clone(), kept);
            let u = binary(backend, BinaryKind::Mul, &g, &unary(UnaryKind::Rsqrt, &v)?)?;
            // u / max(1, rms(u)).
            let all: Vec<usize> = (0..shape.len()).collect();
            let rms = unary(UnaryKind::Sqrt, &mean(&unary(UnaryKind::Square, &u)?, all)?)?;
            let one = backend.forward(
                &Affine {
                    scale: 0.0,
                    bias: 1.0,
                },
                &[&rms],
            )?;
            let divisor = binary(backend, BinaryKind::Max, &rms, &one)?;
            let u = binary(backend, BinaryKind::Div, &u, &divisor)?;
            let lr = self.lr * param.lr_scale;
            let decay = if param.decay { self.weight_decay } else { 0.0 };
            let p = wide(backend, &param.value)?;
            let next = binary(
                backend,
                BinaryKind::Sub,
                &scaled(backend, &p, 1.0 - lr * decay)?,
                &scaled(backend, &u, lr)?,
            )?;
            param.value = if param.value.dtype() == DType::F32 {
                next
            } else {
                backend.forward(
                    &Cast {
                        dtype: param.value.dtype(),
                    },
                    &[&next],
                )?
            };
        }
        Ok(())
    }

    /// The state, for a checkpoint.
    pub fn state(&self) -> OptimizerState {
        let mut tensors = BTreeMap::new();
        for (name, kept) in &self.moments {
            for (i, t) in kept.iter().enumerate() {
                tensors.insert(format!("f{i}.{name}"), t.clone());
            }
        }
        OptimizerState {
            steps: self.steps,
            tensors,
        }
    }

    /// Back to a saved state.
    pub fn restore_state(&mut self, state: OptimizerState) {
        self.steps = state.steps;
        self.moments.clear();
        for (key, t) in state.tensors {
            if let Some((index, name)) = key.split_once('.') {
                let i: usize = index[1..].parse().unwrap_or(0);
                let kept = self.moments.entry(name.to_string()).or_default();
                if kept.len() <= i {
                    kept.resize(i + 1, t.clone());
                }
                kept[i] = t;
            }
        }
    }
}

/// One of the optimizers, as a configuration chooses it.
#[derive(Debug)]
pub enum Optimizer {
    /// [`AdamW`].
    AdamW(AdamW),
    /// [`Muon`].
    Muon(Muon),
    /// [`Adafactor`].
    Adafactor(Adafactor),
}

impl Optimizer {
    /// The optimizer of a checked configuration.
    pub fn new(config: &crate::config::OptimizerConfig) -> Self {
        let mut adamw = AdamW::new(config.lr, config.weight_decay);
        adamw.beta1 = config.beta1;
        adamw.beta2 = config.beta2;
        adamw.eps = config.eps;
        match config.kind.as_str() {
            "muon" => {
                let mut muon = Muon::new(config.lr, config.weight_decay);
                muon.adamw = adamw;
                muon.momentum = config.momentum;
                muon.lr_scale = config.muon_lr_scale;
                if config.newton_schulz == "short" {
                    muon.coefficients = &NEWTON_SCHULZ_SHORT;
                }
                Optimizer::Muon(muon)
            }
            "adafactor" => Optimizer::Adafactor(Adafactor::new(config.lr, config.weight_decay)),
            "adamw8bit" => {
                adamw.eight_bit = true;
                Optimizer::AdamW(adamw)
            }
            _ => Optimizer::AdamW(adamw),
        }
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        match self {
            Optimizer::AdamW(o) => o.lr = lr,
            Optimizer::Muon(o) => o.adamw.lr = lr,
            Optimizer::Adafactor(o) => o.lr = lr,
        }
    }

    /// The steps made.
    pub fn steps(&self) -> u64 {
        match self {
            Optimizer::AdamW(o) => o.steps(),
            Optimizer::Muon(o) => o.adamw.steps(),
            Optimizer::Adafactor(o) => o.steps,
        }
    }

    /// One step.
    ///
    /// # Errors
    ///
    /// Those of the optimizer.
    pub fn step<B: Compute>(
        &mut self,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        grad_scale: f64,
    ) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.step(backend, params, grads, grad_scale),
            Optimizer::Muon(o) => o.step(backend, params, grads, grad_scale),
            Optimizer::Adafactor(o) => o.step(backend, params, grads, grad_scale),
        }
    }

    /// The state, for a checkpoint.
    pub fn state(&self) -> OptimizerState {
        match self {
            Optimizer::AdamW(o) => o.state(),
            Optimizer::Muon(o) => o.state(),
            Optimizer::Adafactor(o) => o.state(),
        }
    }

    /// Back to a saved state.
    ///
    /// # Errors
    ///
    /// If the state is not one of this optimizer.
    pub fn restore(&mut self, state: OptimizerState) -> Result<()> {
        match self {
            Optimizer::AdamW(o) => o.restore_state(state),
            Optimizer::Muon(o) => o.restore_state(state),
            Optimizer::Adafactor(o) => {
                o.restore_state(state);
                Ok(())
            }
        }
    }
}

/// The norm of all the gradients together, `√Σ g²`, read back from the
/// device: what clipping compares with its bound. The sums of squares of
/// the parameters ([`norm_parts`]) are added in `f32`, in order
/// ([`norm_of_parts`]).
///
/// # Errors
///
/// Those of the backend.
pub fn global_norm<B: Compute>(
    backend: &B,
    params: &Parameters,
    grads: &[(String, Grad)],
) -> Result<f64> {
    Ok(norm_of_parts(&norm_parts(backend, params, grads)?))
}

/// The sum of squares of each gradient, `Σ g²` in `f32` on the device,
/// read back together.
///
/// # Errors
///
/// Those of the backend.
pub fn norm_parts<B: Compute>(
    backend: &B,
    params: &Parameters,
    grads: &[(String, Grad)],
) -> Result<Vec<f32>> {
    use kapel_ops::{Concat, Sum};
    let mut parts = Vec::with_capacity(grads.len());
    for (name, grad) in grads {
        let values = match grad {
            // Repeated rows count as many times as they are summed: the
            // dense gradient's norm needs them summed first.
            Grad::Rows { .. } | Grad::Slice { .. } => {
                let shape = params
                    .get(name)
                    .map(|p| p.value.shape().to_vec())
                    .unwrap_or_default();
                grad.clone().dense(backend, &shape)?
            }
            Grad::Dense(g) => g.clone(),
        };
        let square = backend.forward(
            &Unary {
                kind: UnaryKind::Square,
            },
            &[&wide(backend, &values)?],
        )?;
        let dims: Vec<usize> = (0..square.rank()).collect();
        let part = backend.forward(
            &Sum {
                dims,
                keepdim: false,
            },
            &[&square],
        )?;
        parts.push(part.reshape(&[1])?);
    }
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let all = backend.forward(&Concat { dim: 0 }, &refs)?;
    Ok(backend
        .download(&all)?
        .to_f64_vec()?
        .into_iter()
        .map(|x| x as f32)
        .collect())
}

/// `√` of the sum of `parts`, added in `f32` in their order.
pub fn norm_of_parts(parts: &[f32]) -> f64 {
    let mut total: Option<f32> = None;
    for &p in parts {
        total = Some(total.map_or(p, |t| t + p));
    }
    total.map_or(0.0, |t| f64::from(t).sqrt())
}
