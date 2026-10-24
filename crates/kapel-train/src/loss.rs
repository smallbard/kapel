//! Losses.

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{Binary, BinaryKind, Cast, Concat, CrossEntropy, Matmul, Sum};

use crate::tape::{Compute, Grad, Var};

/// The multiple the vocabulary is padded to in the products of
/// [`linear_cross_entropy`].
const ALIGN: usize = 64;

/// A target that is not predicted: its position counts in no loss.
pub const IGNORE: u32 = CrossEntropy::IGNORE;

/// The number of targets that count, each checked against the vocabulary.
fn counted(targets: &[u32], vocab: usize) -> Result<usize> {
    let mut count = 0;
    for &t in targets {
        if CrossEntropy::target(i64::from(t), vocab)?.is_some() {
            count += 1;
        }
    }
    Ok(count)
}

/// The whole rows of `padded` columns that `g [N, V]` is the first `V`
/// columns of, when it is laid out so (the padding written zero by the
/// backend that chose the layout).
fn padded_rows(g: &Tensor, padded: usize) -> Result<Option<Tensor>> {
    let n = g.shape()[0];
    let size = g.dtype().block_bytes();
    let fits = g.layout().offset() == 0
        && g.strides()[1] == 1
        && (n <= 1 || g.strides()[0] == padded)
        && g.storage().len() == n * padded * size;
    if !fits || n <= 1 {
        return Ok(None);
    }
    Ok(Some(Tensor::from_storage(
        std::sync::Arc::clone(g.storage()),
        kapel_core::Layout::contiguous(&[n, padded])?,
        g.dtype(),
    )?))
}

fn check_len(n: usize, targets: &[u32]) -> Result<()> {
    if targets.len() == n {
        Ok(())
    } else {
        Err(kapel_core::Error::InvalidArgument {
            op: "cross_entropy",
            message: format!("{} targets for {n} positions", targets.len()),
        })
    }
}

/// The mean cross-entropy of `logits [N, V]` against `targets` (`N` token
/// ids, [`IGNORE`] for the positions that do not count), by the fused
/// [`CrossEntropy`]: in `f32` (`f64` for logits in `f64`). The logits are
/// kept whole for the backward pass: for a large vocabulary,
/// [`linear_cross_entropy`] is lighter.
///
/// # Errors
///
/// [`kapel_core::Error::IndexOutOfRange`] for a target beyond the vocabulary;
/// those of the backend.
pub fn cross_entropy<B: Compute>(logits: &Var<B>, targets: &[u32]) -> Result<Var<B>> {
    let [n, vocab] = logits.shape()[..] else {
        return Err(kapel_core::Error::InvalidArgument {
            op: "cross_entropy",
            message: format!("logits [N, V], got {:?}", logits.shape()),
        });
    };
    check_len(n, targets)?;
    let count = counted(targets, vocab)?;
    let tape = logits.tape();
    let t = tape.backend().upload(&Tensor::from_slice(targets, &[n])?)?;
    let losses = Var::apply(CrossEntropy, &[logits, &tape.constant(t)])?;
    losses.sum_all()?.scale(1.0 / count.max(1) as f64)
}

/// The mean cross-entropy of the logits `h · wᵀ` against `targets`, for
/// hidden states `h [N, C]` and an output table `w [V, C]`, without forming
/// the logits whole: they are computed `chunk` rows at a time, and each
/// piece's gradients at once. The memory of a large vocabulary is that of
/// `chunk` rows of logits instead of `N`.
///
/// The logits are computed in the type of `h` and `w`, and normalized by
/// [`CrossEntropy`]; the loss is `f32` (`f64` for inputs in `f64`). The
/// gradients are formed during the forward pass, when `h` or `w` needs one,
/// and scaled by the loss's own during the backward pass.
///
/// # Errors
///
/// As [`cross_entropy`].
pub fn linear_cross_entropy<B: Compute>(
    h: &Var<B>,
    w: &Var<B>,
    targets: &[u32],
    chunk: usize,
) -> Result<Var<B>> {
    let tape = h.tape();
    let backend = tape.backend().clone();
    let (hv, wv) = (h.value(), w.value());
    let invalid = |message: String| kapel_core::Error::InvalidArgument {
        op: "linear_cross_entropy",
        message,
    };
    let ([n, c], [vocab, c2]) = (hv.shape(), wv.shape()) else {
        return Err(invalid(format!(
            "h [N, C] and w [V, C], got {:?} and {:?}",
            hv.shape(),
            wv.shape()
        )));
    };
    let (n, c, vocab) = (*n, *c, *vocab);
    if c != *c2 {
        return Err(invalid(format!("h [N, {c}] and w [V, {c2}]")));
    }
    check_len(n, targets)?;
    let count = counted(targets, vocab)?;
    let (needs_h, needs_w) = (h.requires_grad(), w.requires_grad());
    let dtype = hv.dtype();
    let wide = CrossEntropy::output_dtype(dtype);
    let add = Binary {
        kind: BinaryKind::Add,
    };
    let scale = 1.0 / count.max(1) as f64;
    // The table with rows of zeros up to a multiple of `ALIGN`: GPT-2's
    // 50,257 columns of logits, rows of an odd length, slow the products down
    // eightfold on a card. The logits of the padding are not read.
    let padded = vocab.next_multiple_of(ALIGN);
    let wp = if padded == vocab {
        wv.clone()
    } else {
        let zeros = backend.zeros(&[padded - vocab, c], wv.dtype())?;
        backend.forward(&Concat { dim: 0 }, &[&wv, &zeros])?
    };
    let wt = wp.transpose(0, 1)?;
    let chunk = chunk.max(1);
    let mut total: Option<Tensor> = None;
    let mut gh_parts = Vec::new();
    let mut gw: Option<Tensor> = None;
    for start in (0..n).step_by(chunk) {
        let len = chunk.min(n - start);
        let rows = &targets[start..start + len];
        if rows.iter().all(|&t| t == IGNORE) {
            if needs_h {
                gh_parts.push(backend.zeros(&[len, c], dtype)?);
            }
            continue;
        }
        let x = hv.narrow(0, start, len)?;
        let logits = backend.forward(&Matmul, &[&x, &wt])?.narrow(1, 0, vocab)?;
        let t = backend.upload(&Tensor::from_slice(rows, &[len])?)?;
        let losses = backend.forward(&CrossEntropy, &[&logits, &t])?;
        let part = backend.forward(
            &Sum {
                dims: vec![0],
                keepdim: true,
            },
            &[&losses],
        )?;
        total = Some(match total {
            None => part,
            Some(t) => backend.forward(&add, &[&t, &part])?,
        });
        if needs_h || needs_w {
            // d(mean loss) / d(loss of a row) = 1 / count, for every row:
            // an ignored row's logits receive zeros.
            let g = if wide == DType::F64 {
                Tensor::from_slice(&vec![scale; len], &[len])?
            } else {
                Tensor::from_slice(&vec![scale as f32; len], &[len])?
            };
            let g = backend.upload(&g)?;
            let g = backend
                .backward(&CrossEntropy, &[&logits, &t], &g)?
                .swap_remove(0)
                .expect("the logits receive a gradient");
            drop(logits);
            let g = if padded == vocab {
                g
            } else if let Some(wide) = padded_rows(&g, padded)? {
                // The card gives the gradient the layout of the logits, the
                // first columns of rows of `padded`, the others zeros.
                wide
            } else {
                let zeros = backend.zeros(&[len, padded - vocab], g.dtype())?;
                backend.forward(&Concat { dim: 1 }, &[&g, &zeros])?
            };
            if needs_h {
                gh_parts.push(backend.forward(&Matmul, &[&g, &wp])?);
            }
            if needs_w {
                let part = backend.forward(&Matmul, &[&g.transpose(0, 1)?, &x])?;
                let part = backend.forward(&Cast { dtype: wide }, &[&part])?;
                gw = Some(match gw {
                    None => part,
                    Some(acc) => backend.forward(&add, &[&acc, &part])?,
                });
            }
        }
    }
    let total = match total {
        Some(t) => t,
        None => backend.zeros(&[1], wide)?,
    };
    let loss = backend.forward(&kapel_ops::Affine { scale, bias: 0.0 }, &[&total])?;
    let gh = match gh_parts.len() {
        0 => None,
        1 => gh_parts.pop(),
        _ => {
            let refs: Vec<&Tensor> = gh_parts.iter().collect();
            Some(backend.forward(&Concat { dim: 0 }, &refs)?)
        }
    };
    let wdtype = wv.dtype();
    let gw = gw.map(|g| g.narrow(0, 0, vocab)).transpose()?;
    drop((hv, wv, wp, wt));
    let mul = Binary {
        kind: BinaryKind::Mul,
    };
    Ok(tape.custom(&[h, w], loss, move || {
        move |backend: &B, g: Tensor, needed: &[bool]| -> Result<Vec<Option<Grad>>> {
            // `g` is the loss's gradient, `[1]`.
            let scaled = |t: Tensor, to: DType| -> Result<Tensor> {
                let s = backend.forward(&Cast { dtype: t.dtype() }, &[&g])?;
                let s = s.reshape(&vec![1; t.rank()])?;
                let y = backend.forward(&mul, &[&t, &s])?;
                if y.dtype() == to {
                    Ok(y)
                } else {
                    backend.forward(&Cast { dtype: to }, &[&y])
                }
            };
            let gh = match gh.filter(|_| needed[0]) {
                Some(gh) => Some(Grad::Dense(scaled(gh, dtype)?)),
                None => None,
            };
            let gw = match gw.filter(|_| needed[1]) {
                Some(gw) => Some(Grad::Dense(scaled(gw, wdtype)?)),
                None => None,
            };
            Ok(vec![gh, gw])
        }
    }))
}
