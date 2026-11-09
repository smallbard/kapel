//! Mixtures of experts on the device: choosing each token's experts, and
//! multiplying each token by its own experts' weights, without the host
//! reading which they are.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, indices, rank, same_device};

/// The positions of the `k` largest values along the last dimension: `x [...,
/// E]` of a float type gives `y [..., k]` of `u32`, the largest first, and on
/// a tie the lower position first (NaN, by `f32::total_cmp`, above every
/// number). It chooses the experts of each token from its router's
/// probabilities. Integer output: no input receives a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopK {
    /// The positions kept, at least 1 and at most `E`.
    pub k: usize,
}

impl TopK {
    /// The `k` positions of the largest of `values`, as the operation gives
    /// them: what every backend computes, here on the host.
    pub fn choose(k: usize, values: &[f32]) -> Vec<u32> {
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_by(|&i, &j| {
            values[j as usize]
                .total_cmp(&values[i as usize])
                .then(i.cmp(&j))
        });
        order.truncate(k);
        order
    }
}

impl Operation for TopK {
    const NAME: &'static str = "top_k";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[1], "1")?;
        let x = inputs[0];
        float(Self::NAME, x)?;
        let Some(&e) = x.shape().last() else {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: "a scalar has no last dimension".into(),
            });
        };
        if self.k == 0 || self.k > e {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!("the {} largest of {e}", self.k),
            });
        }
        if e > u32::MAX as usize {
            return Err(Error::Overflow { op: Self::NAME });
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().expect("checked above") = self.k;
        Ok(Signature {
            shape,
            dtype: DType::U32,
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        false
    }
}

/// Values picked along the last dimension: `x [..., E]` of a float type and
/// integer `idx [..., k]` with the same leading dimensions give `y [..., k]`
/// with `y[..., j] = x[..., idx[..., j]]`: the probabilities of each token's
/// chosen experts. For inference: no input receives a gradient. An index of
/// `E` or more is an error ([`GatherColumns::check`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherColumns;

impl GatherColumns {
    /// Checks the values of `idx`, as a backend reads them, against `x`'s
    /// last dimension `e`.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfRange`] for the first index out of range.
    pub fn check(idx: &[usize], e: usize) -> Result<()> {
        match idx.iter().find(|&&i| i >= e) {
            Some(&i) => Err(Error::IndexOutOfRange {
                op: Self::NAME,
                index: i as i64,
                size: e,
            }),
            None => Ok(()),
        }
    }
}

impl Operation for GatherColumns {
    const NAME: &'static str = "gather_columns";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
        same_device(Self::NAME, inputs)?;
        let (x, idx) = (inputs[0], inputs[1]);
        float(Self::NAME, x)?;
        indices(Self::NAME, idx)?;
        if x.rank() == 0
            || x.rank() != idx.rank()
            || x.shape()[..x.rank() - 1] != idx.shape()[..idx.rank() - 1]
        {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "values {:?} and indices {:?} must share all but their last dimension",
                    x.shape(),
                    idx.shape()
                ),
            });
        }
        Ok(Signature {
            shape: idx.shape().to_vec(),
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        false
    }
}

/// The product of each token by its experts: `x [T, K]` (one input per
/// token) or `[T, k, K]` (one per token and expert), of a float type; the
/// experts' weights stacked, `w [E · N, K]`, expert `e` the rows `e · N ..
/// (e + 1) · N`, of a float type or quantized (blocks along `K`); integer
/// `idx [T, k]`, each token's experts. The output is `y [T, k, N]` in the
/// type of `x`, with
///
/// `y[t, j, n] = Σᵢ x[t, (j,) i] · w[idx[t, j] · N + n, i]`,
///
/// the sums in `f32` at least, `w` at its exact (dequantized) value. For
/// inference: no input receives a gradient. An expert of `experts` or more
/// is an error where the backend reads the indices
/// ([`ExpertMatmul::check`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertMatmul {
    /// The experts the weights stack, `E`.
    pub experts: usize,
}

impl ExpertMatmul {
    /// Checks the values of `idx`, as a backend reads them.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfRange`] for the first index out of range.
    pub fn check(&self, idx: &[usize]) -> Result<()> {
        match idx.iter().find(|&&i| i >= self.experts) {
            Some(&i) => Err(Error::IndexOutOfRange {
                op: Self::NAME,
                index: i as i64,
                size: self.experts,
            }),
            None => Ok(()),
        }
    }
}

impl Operation for ExpertMatmul {
    const NAME: &'static str = "expert_matmul";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[3], "3")?;
        same_device(Self::NAME, inputs)?;
        let (x, w, idx) = (inputs[0], inputs[1], inputs[2]);
        float(Self::NAME, x)?;
        indices(Self::NAME, idx)?;
        rank(Self::NAME, w, 2, "the weights")?;
        rank(Self::NAME, idx, 2, "the experts")?;
        if !(w.dtype().is_float() || w.dtype().is_quantized()) {
            return Err(Error::UnsupportedDType {
                op: Self::NAME,
                dtype: w.dtype(),
            });
        }
        let (t, k) = (idx.shape()[0], idx.shape()[1]);
        let width = w.shape()[1];
        match x.shape() {
            [tx, kx] if *tx == t && *kx == width => {}
            [tx, jx, kx] if *tx == t && *jx == k && *kx == width => {}
            _ => {
                return Err(Error::InvalidArgument {
                    op: Self::NAME,
                    message: format!(
                        "inputs {:?} for experts {:?} and weights {:?}: [T, K] or [T, k, K]",
                        x.shape(),
                        idx.shape(),
                        w.shape()
                    ),
                });
            }
        }
        if self.experts == 0 || !w.shape()[0].is_multiple_of(self.experts) {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "{} rows of weights for {} experts",
                    w.shape()[0],
                    self.experts
                ),
            });
        }
        if w.dtype().is_quantized() && !width.is_multiple_of(w.dtype().block_len()) {
            return Err(Error::BlockMisaligned {
                dtype: w.dtype(),
                elements: width,
            });
        }
        Ok(Signature {
            shape: vec![t, k, w.shape()[0] / self.experts],
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, _: usize) -> bool {
        false
    }
}
