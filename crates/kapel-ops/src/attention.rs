//! Scaled dot-product attention.

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, rank, same_device, shape};

/// Scaled dot-product attention, with grouped keys and values.
///
/// Inputs: `q [B, Hq, Tq, D]`, `k [B, Hkv, Tk, D]`, `v [B, Hkv, Tk, Dv]`, and
/// optionally an additive `mask` that broadcasts to `[B, Hq, Tq, Tk]`. `Hq` is a
/// multiple of `Hkv`: query head `h` reads key and value head `h / (Hq / Hkv)`.
///
/// Query `i` sits at position `p(i) = Tk − Tq + i`: the queries are the last
/// `Tq` positions of the `Tk` keys, as when generating with a cache. For each
/// query, key `j` is visible if
/// - `causal` is off, or `j ≤ p(i)`;
/// - and `window` is off, or `p(i) − j < window`.
///
/// Then `s[j] = scale · q·k[j] + mask[i, j]` for the visible keys,
/// `a = softmax(s)` over them, and `y = Σⱼ a[j] · v[j]`. A query that sees no
/// key, or whose visible keys are all masked with `−∞`, gives zeros and passes
/// no gradient. Finite scores, however low (such as `f32::MIN`, the padding
/// mask of `transformers`), take part in the softmax. `scale` defaults to
/// `1 / √D`.
///
/// With `softcap` (Gemma 2), the scaled scores are capped smoothly before the
/// mask is added: `scale · q·k` becomes `softcap · tanh(scale · q·k / softcap)`.
///
/// The mask may be in any float type, not necessarily that of `q`; it
/// receives no gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attention {
    /// Hide the keys after each query's position.
    pub causal: bool,
    /// Only the last `window` positions, the query's included, are visible.
    pub window: Option<usize>,
    /// The factor of the scores; `None` for `1 / √D`.
    pub scale: Option<f64>,
    /// The bound of the scores, approached through `tanh`; `None` for none.
    pub softcap: Option<f64>,
}

impl Attention {
    /// The causal attention of a decoder, with the default scale.
    pub fn causal() -> Self {
        Self {
            causal: true,
            window: None,
            scale: None,
            softcap: None,
        }
    }

    /// The factor of the scores for a head width `d`.
    pub fn scale_for(&self, d: usize) -> f64 {
        self.scale.unwrap_or(1.0 / (d as f64).sqrt())
    }
}

impl Operation for Attention {
    const NAME: &'static str = "attention";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        arity(op, inputs, &[3, 4], "3 or 4")?;
        same_device(op, inputs)?;
        let (q, k, v) = (inputs[0], inputs[1], inputs[2]);
        for (tensor, what) in [(q, "q"), (k, "k"), (v, "v")] {
            float(op, tensor)?;
            rank(op, tensor, 4, what)?;
            if tensor.dtype() != q.dtype() {
                return Err(Error::DTypeMismatch {
                    op,
                    expected: q.dtype(),
                    got: tensor.dtype(),
                });
            }
        }
        let [b, hq, tq, d] = [q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]];
        let [bk, hkv, tk, dk] = [k.shape()[0], k.shape()[1], k.shape()[2], k.shape()[3]];
        let [bv, hv, tv, dv] = [v.shape()[0], v.shape()[1], v.shape()[2], v.shape()[3]];
        let invalid = |message: String| Error::InvalidArgument { op, message };
        if bk != b || bv != b || hv != hkv || tv != tk || dk != d {
            return Err(invalid(format!(
                "expected q [B, Hq, Tq, D], k [B, Hkv, Tk, D], v [B, Hkv, Tk, Dv]; got {:?}, {:?}, {:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if d == 0 {
            return Err(invalid(
                "the head width D must be positive: the default scale is 1 / √D".into(),
            ));
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(invalid(format!(
                "the {hq} query heads must be a multiple of the {hkv} key heads"
            )));
        }
        if tq > tk {
            return Err(invalid(format!(
                "{tq} queries need at least as many keys, got {tk}"
            )));
        }
        if self.window == Some(0) {
            return Err(invalid("a window of 0 positions hides every key".into()));
        }
        if let Some(mask) = inputs.get(3) {
            float(op, mask)?;
            let target = [b, hq, tq, tk];
            if shape::broadcast(op, mask.shape(), &target)? != target {
                return Err(Error::BroadcastIncompatible {
                    op,
                    from: mask.shape().to_vec(),
                    to: target.to_vec(),
                });
            }
        }
        Ok(Signature {
            shape: vec![b, hq, tq, dv],
            dtype: q.dtype(),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index < 3
    }
}
