//! AWQ (Lin et al., 2023): factors for the input channels, and clipping.
//!
//! A projection's output `W x` does not change when an input channel `c` is
//! multiplied by `s_c` in the weights and divided by it in the inputs (in
//! the operation that computes them). Rounding then treats the channel's
//! weights `s_c` times larger: the channels with large activations, the
//! ones that count, lose less. The factors are `s = m^α`, `m` the mean
//! magnitude of each channel's inputs, normalized so that
//! `max(s) · min(s) = 1`; `α` is chosen on a grid from 0 (no factor) to
//! almost 1, by the output error of the rounded weights, `Σ δ H δᵀ` over
//! the rows, computed on the backend.
//!
//! Then each block of each row may be clipped: its weights brought within
//! `r` times the largest magnitude, `r` from 1 down to one half, the one of
//! smallest output error on the block (`δ H_bb δᵀ`, `H_bb` the part of the
//! Hessian of its columns).

use kapel_core::{DType, Tensor, block_code, block_scale, block_value};
use kapel_ops::{Backend, Binary, BinaryKind, Matmul, Sum, Transfer};

use crate::Weight;
use crate::linalg::parallel;

/// The factors AWQ chose for a site.
pub(crate) struct Found {
    /// The exponent of the magnitudes.
    pub(crate) exponent: f64,
    /// The factors of the input channels.
    pub(crate) factors: Vec<f32>,
    /// The output error with them.
    pub(crate) loss: f64,
    /// The output error without.
    pub(crate) plain: f64,
}

/// `values` (rows of `cols`) rounded to `format` and read back, on every
/// core.
fn rounded(values: &[f32], cols: usize, format: DType) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    let rows_per = (1 << 16) / cols.max(1) + 1;
    let pieces: Vec<(usize, &mut [f32])> = out
        .chunks_mut(rows_per * cols)
        .enumerate()
        .map(|(p, o)| (p * rows_per * cols, o))
        .collect();
    parallel(pieces, |(at, piece)| {
        let source = &values[at..at + piece.len()];
        let bytes = kapel_core::quantize(format, source).expect("whole blocks of a written format");
        let back = kapel_core::dequantize(format, &bytes).expect("its own blocks");
        for (o, v) in piece.iter_mut().zip(back) {
            *o = v as f32;
        }
    });
    out
}

/// `Σ δ H δᵀ` over the rows of `delta` `[rows, k]`, on the backend.
fn output_error<B: Backend + Transfer>(
    backend: &B,
    hessian: &Tensor,
    delta: &[f32],
    rows: usize,
    k: usize,
) -> crate::Result<f64> {
    let d = backend.upload(&Tensor::from_slice(delta, &[rows, k])?)?;
    let dh = backend.forward(&Matmul, &[&d, hessian])?;
    let product = backend.forward(
        &Binary {
            kind: BinaryKind::Mul,
        },
        &[&dh, &d],
    )?;
    let total = backend.forward(
        &Sum {
            dims: vec![0, 1],
            keepdim: false,
        },
        &[&product],
    )?;
    Ok(backend.download(&total)?.to_f64_vec()?[0])
}

/// The factors of the input channels of `weights` (projections that read
/// the same input, whose Hessian `hessian` `[K, K]` is on the backend and
/// whose mean magnitudes are `magnitude`) that make the smallest output
/// error once rounded to `format`, among the exponents `i / grid`.
pub(crate) fn scales<B: Backend + Transfer>(
    backend: &B,
    hessian: &Tensor,
    magnitude: &[f64],
    weights: &[Weight],
    format: DType,
    grid: usize,
) -> crate::Result<Found> {
    let k = magnitude.len();
    let mut best: Option<Found> = None;
    let mut plain = 0.0;
    for i in 0..grid.max(1) {
        let exponent = i as f64 / grid.max(1) as f64;
        let raw: Vec<f64> = magnitude
            .iter()
            .map(|&m| m.max(1e-4).powf(exponent))
            .collect();
        let (low, high) = raw
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
        let norm = (low * high).sqrt();
        let factors: Vec<f32> = raw.iter().map(|&v| (v / norm) as f32).collect();
        let mut loss = 0.0;
        for w in weights {
            let scaled: Vec<f32> = w
                .values
                .iter()
                .enumerate()
                .map(|(e, &v)| v * factors[e % k])
                .collect();
            let back = rounded(&scaled, k, format);
            let delta: Vec<f32> = back
                .iter()
                .enumerate()
                .map(|(e, &q)| q / factors[e % k] - w.values[e])
                .collect();
            loss += output_error(backend, hessian, &delta, w.rows, k)?;
        }
        if i == 0 {
            plain = loss;
        }
        if best.as_ref().is_none_or(|b| loss < b.loss) {
            best = Some(Found {
                exponent,
                factors,
                loss,
                plain,
            });
        }
    }
    let mut found = best.expect("one exponent at least");
    found.plain = plain;
    Ok(found)
}

/// The Hessian `h` `[K, K]` reduced to the columns `from..from + len`.
fn block_of(h: &[f64], k: usize, from: usize, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len * len);
    for i in from..from + len {
        out.extend_from_slice(&h[i * k + from..i * k + from + len]);
    }
    out
}

/// Clips each block of each row of `w` to the ratio of its largest
/// magnitude, from 1 down to one half by steps of `1 / grid`, that rounds
/// to `format` with the smallest output error, by the Hessian `h` of its
/// inputs.
pub(crate) fn clip(w: &mut Weight, h: &[f64], format: DType, grid: usize) {
    let (k, len) = (w.cols, format.block_len());
    let blocks: Vec<Vec<f64>> = (0..k / len).map(|b| block_of(h, k, b * len, len)).collect();
    let steps = grid.max(1) / 2;
    let rows: Vec<&mut [f32]> = w.values.chunks_mut(k).collect();
    parallel(rows, |row| {
        let mut clipped = vec![0.0f32; len];
        let mut delta = vec![0.0f64; len];
        for (b, hb) in blocks.iter().enumerate() {
            let block = &mut row[b * len..(b + 1) * len];
            let top = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if top == 0.0 {
                continue;
            }
            let mut best = (1.0f32, f64::INFINITY);
            for i in 0..steps.max(1) {
                let ratio = 1.0 - i as f32 / grid.max(1) as f32;
                let limit = top * ratio;
                for (c, &v) in clipped.iter_mut().zip(block.iter()) {
                    *c = v.clamp(-limit, limit);
                }
                let scale = block_scale(format, &clipped).expect("a written format");
                for ((d, &c), &v) in delta.iter_mut().zip(&clipped).zip(block.iter()) {
                    let q = block_value(format, &scale, block_code(format, &scale, c));
                    *d = q - f64::from(v);
                }
                let mut error = 0.0;
                for (i, &di) in delta.iter().enumerate() {
                    let row = &hb[i * len..(i + 1) * len];
                    let dot: f64 = row.iter().zip(&delta).map(|(&hv, &dj)| hv * dj).sum();
                    error += di * dot;
                }
                if error < best.1 {
                    best = (ratio, error);
                }
            }
            let limit = top * best.0;
            for v in block.iter_mut() {
                *v = v.clamp(-limit, limit);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapel_core::QuantFormat;

    fn noise(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// Inputs where a few channels are much larger than the others, as in
    /// language models: AWQ's factors lower the output error of rounding,
    /// and so does clipping.
    #[test]
    fn factors_and_clipping_lower_the_output_error() {
        let cpu = kapel_cpu::Cpu::new();
        let (n, k, t) = (48, 128, 300);
        let mut x = noise(t * k, 11);
        for i in 0..t {
            for c in [3, 40, 77] {
                x[i * k + c] *= 30.0;
            }
        }
        let mut h = vec![0.0f64; k * k];
        let mut magnitude = vec![0.0f64; k];
        for i in 0..t {
            for a in 0..k {
                magnitude[a] += x[i * k + a].abs() / t as f64;
                for b in 0..k {
                    h[a * k + b] += x[i * k + a] * x[i * k + b];
                }
            }
        }
        let hessian =
            Tensor::from_slice(&h.iter().map(|&v| v as f32).collect::<Vec<_>>(), &[k, k]).unwrap();
        let w = Weight {
            values: noise(n * k, 5).iter().map(|&v| v as f32).collect(),
            rows: n,
            cols: k,
        };
        let format = DType::Quantized(QuantFormat::Q4_0);
        let found = scales(
            &cpu,
            &hessian,
            &magnitude,
            std::slice::from_ref(&w),
            format,
            20,
        )
        .unwrap();
        assert!(found.exponent > 0.0);
        assert!(
            found.loss < 0.9 * found.plain,
            "{} vs {}",
            found.loss,
            found.plain
        );
        // Clipping never makes a block worse: the ratio 1 is among those tried.
        // Against the weights before clipping.
        let error = |c: &Weight| {
            let back = rounded(&c.values, k, format);
            let delta: Vec<f32> = back.iter().zip(&w.values).map(|(q, v)| q - v).collect();
            let d: Vec<f64> = delta.iter().map(|&v| f64::from(v)).collect();
            let mut total = 0.0;
            for r in 0..n {
                for a in 0..k {
                    for b in 0..k {
                        total += d[r * k + a] * h[a * k + b] * d[r * k + b];
                    }
                }
            }
            total
        };
        let before = error(&w);
        let mut c = Weight {
            values: w.values.clone(),
            rows: n,
            cols: k,
        };
        clip(&mut c, &h, format, 20);
        assert!(c.values != w.values, "some block is clipped");
        assert!(error(&c) < before * 1.0001, "{} vs {before}", error(&c));
    }
}
