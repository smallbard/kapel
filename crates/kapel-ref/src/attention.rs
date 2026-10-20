//! Scaled dot-product attention.

use kapel_core::{Result, Tensor};
use kapel_ops::{Attention, Operation, Run};

use crate::Reference;
use crate::array::{Array, broadcast_map};

/// The inputs, read once, with their sizes.
struct Inputs {
    q: Array,
    k: Array,
    v: Array,
    /// The additive mask, broadcast to `[B, Hq, Tq, Tk]`.
    mask: Option<Vec<f64>>,
    b: usize,
    hq: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    dv: usize,
}

impl Inputs {
    fn read(inputs: &[&Tensor]) -> Result<Self> {
        let (q, k, v) = (
            Array::of(inputs[0])?,
            Array::of(inputs[1])?,
            Array::of(inputs[2])?,
        );
        let [b, hq, tq, d] = [q.shape[0], q.shape[1], q.shape[2], q.shape[3]];
        let (hkv, tk, dv) = (k.shape[1], k.shape[2], v.shape[3]);
        let mask = match inputs.get(3) {
            Some(mask) => {
                let mask = Array::of(mask)?;
                let map = broadcast_map(&mask.shape, &[b, hq, tq, tk]);
                Some(map.into_iter().map(|at| mask.data[at]).collect())
            }
            None => None,
        };
        Ok(Self {
            q,
            k,
            v,
            mask,
            b,
            hq,
            hkv,
            tq,
            tk,
            d,
            dv,
        })
    }

    fn q(&self, b: usize, h: usize, i: usize) -> &[f64] {
        let at = ((b * self.hq + h) * self.tq + i) * self.d;
        &self.q.data[at..at + self.d]
    }

    fn k_at(&self, b: usize, h: usize, j: usize) -> usize {
        ((b * self.hkv + h) * self.tk + j) * self.d
    }

    fn v_at(&self, b: usize, h: usize, j: usize) -> usize {
        ((b * self.hkv + h) * self.tk + j) * self.dv
    }

    /// The derivative of the softcapped score by the scaled score:
    /// `1 − tanh²(scale · q·k / softcap)`, or 1 without a softcap.
    fn softcap_slope(&self, op: &Attention, b: usize, h: usize, i: usize, j: usize) -> f64 {
        let Some(cap) = op.softcap else { return 1.0 };
        let kv = h / (self.hq / self.hkv);
        let k = &self.k.data[self.k_at(b, kv, j)..][..self.d];
        let dot: f64 = self.q(b, h, i).iter().zip(k).map(|(a, b)| a * b).sum();
        1.0 - (op.scale_for(self.d) * dot / cap).tanh().powi(2)
    }

    /// The visible keys of query `i`, and their attention weights.
    fn weights(&self, op: &Attention, b: usize, h: usize, i: usize) -> Vec<(usize, f64)> {
        let kv = h / (self.hq / self.hkv);
        let position = self.tk - self.tq + i;
        let scale = op.scale_for(self.d);
        let visible = (0..self.tk).filter(|&j| {
            (!op.causal || j <= position) && op.window.is_none_or(|w| position < j + w)
        });
        let q = self.q(b, h, i);
        let scores: Vec<(usize, f64)> = visible
            .map(|j| {
                let k = &self.k.data[self.k_at(b, kv, j)..][..self.d];
                let dot: f64 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                let mask = self
                    .mask
                    .as_ref()
                    .map_or(0.0, |m| m[((b * self.hq + h) * self.tq + i) * self.tk + j]);
                let s = match op.softcap {
                    Some(cap) => cap * (scale * dot / cap).tanh(),
                    None => scale * dot,
                };
                (j, s + mask)
            })
            .collect();
        let max = scores.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = scores.iter().map(|s| (s.1 - max).exp()).sum();
        scores
            .into_iter()
            .map(|(j, s)| (j, (s - max).exp() / total))
            .collect()
    }
}

impl Run<Attention> for Reference {
    fn forward(&self, op: &Attention, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = Inputs::read(inputs)?;
        let mut out = Array::zeros(&signature.shape);
        for b in 0..x.b {
            for h in 0..x.hq {
                let kv = h / (x.hq / x.hkv);
                for i in 0..x.tq {
                    let y = &mut out.data[((b * x.hq + h) * x.tq + i) * x.dv..][..x.dv];
                    for (j, a) in x.weights(op, b, h, i) {
                        let v = &x.v.data[x.v_at(b, kv, j)..][..x.dv];
                        y.iter_mut().zip(v).for_each(|(y, v)| *y += a * v);
                    }
                }
            }
        }
        out.tensor()
    }

    /// With `a` the weights of a query and `g` the gradient of its output
    /// (and `∂s` multiplied by the slope of the softcap, if any):
    /// `∂v[j] += a[j]·g`, `∂s[j] = a[j]·(g·v[j] − Σ a·(g·v))`, then
    /// `∂q += scale·Σ ∂s[j]·k[j]` and `∂k[j] += scale·∂s[j]·q`.
    fn backward(
        &self,
        op: &Attention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer(inputs)?;
        let x = Inputs::read(inputs)?;
        let g = Array::of(grad)?;
        let scale = op.scale_for(x.d);
        let (mut gq, mut gk, mut gv) = (
            Array::zeros(&x.q.shape),
            Array::zeros(&x.k.shape),
            Array::zeros(&x.v.shape),
        );
        for b in 0..x.b {
            for h in 0..x.hq {
                let kv = h / (x.hq / x.hkv);
                for i in 0..x.tq {
                    let row = (b * x.hq + h) * x.tq + i;
                    let gy = &g.data[row * x.dv..][..x.dv];
                    let weights = x.weights(op, b, h, i);
                    let ga: Vec<f64> = weights
                        .iter()
                        .map(|&(j, _)| {
                            let v = &x.v.data[x.v_at(b, kv, j)..][..x.dv];
                            gy.iter().zip(v).map(|(a, b)| a * b).sum()
                        })
                        .collect();
                    let mean: f64 = weights.iter().zip(&ga).map(|((_, a), ga)| a * ga).sum();
                    for (&(j, a), &ga) in weights.iter().zip(&ga) {
                        let gs = a * (ga - mean) * x.softcap_slope(op, b, h, i, j);
                        let (ka, va) = (x.k_at(b, kv, j), x.v_at(b, kv, j));
                        for (gv, gy) in gv.data[va..va + x.dv].iter_mut().zip(gy) {
                            *gv += a * gy;
                        }
                        for c in 0..x.d {
                            gq.data[row * x.d + c] += scale * gs * x.k.data[ka + c];
                            gk.data[ka + c] += scale * gs * x.q(b, h, i)[c];
                        }
                    }
                }
            }
        }
        let mut grads = vec![Some(gq.tensor()?), Some(gk.tensor()?), Some(gv.tensor()?)];
        if inputs.len() == 4 {
            grads.push(None);
        }
        Ok(grads)
    }
}
