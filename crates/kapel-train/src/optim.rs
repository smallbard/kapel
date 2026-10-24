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
    steps: u64,
    moments: BTreeMap<String, (Tensor, Tensor)>,
}

impl AdamW {
    /// With nanoGPT's settings: `β₁ = 0.9`, `β₂ = 0.95`, `ε = 10⁻⁸`.
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
            steps: 0,
            moments: BTreeMap::new(),
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

    /// Puts the optimizer back in a saved state: steps and moments.
    pub fn restore(&mut self, steps: u64, moments: BTreeMap<String, (Tensor, Tensor)>) {
        self.steps = steps;
        self.moments = moments;
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
        self.steps += 1;
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
            let (m, v) = match self.moments.remove(&name) {
                Some(mv) => mv,
                None => (
                    backend.zeros(&shape, DType::F32)?,
                    backend.zeros(&shape, DType::F32)?,
                ),
            };
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
            let (m, v) = match grad {
                Grad::Rows { indices, values } => {
                    by_rows(backend, &mut param.value, m, v, &indices, &values, &s)?
                }
                grad => {
                    let g = grad.dense(backend, &shape)?;
                    let (p, m, v) = s.apply(backend, &param.value, m, v, &g)?;
                    param.value = p;
                    (m, v)
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
    let values = values.reshape(&flat)?;
    let g = backend.forward(
        &ScatterAddRows { rows: unique.len() },
        &[&values, &remapped_t],
    )?;
    let gather = |t: &Tensor| backend.forward(&GatherRows, &[t, &unique_t]);
    let (p_rows, m_rows, v_rows) = (gather(p)?, gather(&m)?, gather(&v)?);
    let (p_rows, m_rows, v_rows) = s.apply(backend, &p_rows, m_rows, v_rows, &g)?;
    backend.write_rows(p, &unique_t, &p_rows)?;
    backend.write_rows(&mut m, &unique_t, &m_rows)?;
    backend.write_rows(&mut v, &unique_t, &v_rows)?;
    Ok((m, v))
}
