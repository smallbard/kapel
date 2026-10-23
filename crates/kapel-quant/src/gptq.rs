//! GPTQ (Frantar et al., 2022): the columns of a weight quantized one at a
//! time, the error of each spread over the columns not yet quantized, as
//! the inputs they meet are correlated with its own.
//!
//! A projection `y = W x` with `W` `[N, K]` and inputs `x` whose Hessian is
//! `H = Σ x xᵀ` `[K, K]`: the output error of a change `δ` of a row of `W`
//! is `δ H δᵀ`. Column `j` is rounded, and the error of each row is moved
//! onto the columns after it through the factor `U` of `H⁻¹ = Uᵀ U`
//! ([`crate::linalg`]):
//!
//! ```text
//! e      = (w_j − q_j) / U[j, j]
//! w_k   −= e · U[j, k]            for every column k > j
//! ```
//!
//! As in the paper, the columns go by batches of 128: the errors of a batch
//! reach the columns after it in one product. The rows are independent:
//! pieces of [`ROWS`] rows run on every core. A block of the format takes
//! its scale (and Q4_1 its minimum) from its weights as they are when its
//! first column comes, by the format's own rule ([`kapel_core::block_scale`]),
//! and its codes are those of that scale ([`kapel_core::block_code`]).
//!
//! The Hessian is damped by a fraction of the mean of its diagonal (1 % by
//! default), and ten times more until it is positive definite. A column no
//! input ever reached (a zero on the diagonal) weighs nothing: its weights
//! become zeros.

use kapel_core::{BlockScale, DType, block_code, block_scale, block_value, encode_block};

use crate::linalg::{TILE, inverse_factor, parallel, product};

/// The columns of a batch.
const BATCH: usize = 128;
/// The rows of a piece of work.
pub(crate) const ROWS: usize = 64;

/// The factor `U` of the damped inverse of `h` `[k, k]`, the damping used,
/// and the columns no input reached. The diagonal of `h` is changed.
pub(crate) fn prepare(
    h: &mut [f64],
    k: usize,
    damping: f64,
) -> Result<(Vec<f64>, f64, Vec<usize>), String> {
    let dead: Vec<usize> = (0..k).filter(|&i| h[i * k + i] <= 0.0).collect();
    for &i in &dead {
        h[i * k + i] = 1.0;
    }
    let mean = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    if !mean.is_finite() {
        return Err("the Hessian is not finite".into());
    }
    let mut damping = damping;
    loop {
        let mut damped = h.to_vec();
        for i in 0..k {
            damped[i * k + i] += damping * mean;
        }
        match inverse_factor(&damped, k) {
            Ok(u) => return Ok((u, damping, dead)),
            Err(e) if damping < 1.0 => {
                let _ = e;
                damping = (damping * 10.0).min(1.0);
            }
            Err(e) => {
                return Err(format!(
                    "the Hessian is not positive definite, even damped by {damping}: pivot {} at {}",
                    e.pivot, e.at
                ));
            }
        }
    }
}

/// The bytes of `w` `[n, k]` (row-major) quantized to `dtype` by GPTQ with
/// the factor `u` of [`prepare`]; the columns of `dead` read as zeros.
pub(crate) fn quantize(
    w: &[f32],
    n: usize,
    k: usize,
    dtype: DType,
    u: &[f64],
    dead: &[usize],
) -> Vec<u8> {
    assert_eq!(w.len(), n * k, "gptq: a weight [n, k]");
    let group = dtype.block_len();
    assert!(k.is_multiple_of(group), "gptq: rows of whole blocks");
    let batch = BATCH.max(group).next_multiple_of(group);
    let row_bytes = dtype.storage_bytes(k).expect("whole blocks");
    let mut out = vec![0u8; n * row_bytes];
    let pieces: Vec<(usize, &mut [u8])> = out
        .chunks_mut(ROWS * row_bytes)
        .enumerate()
        .map(|(p, bytes)| (p * ROWS, bytes))
        .collect();
    parallel(pieces, |(r0, bytes)| {
        let rows = bytes.len() / row_bytes;
        let mut w: Vec<f64> = w[r0 * k..(r0 + rows) * k]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        for &c in dead {
            for r in 0..rows {
                w[r * k + c] = 0.0;
            }
        }
        let mut codes = vec![0u8; rows * k];
        let mut scales: Vec<BlockScale> = Vec::with_capacity(rows * (k / group));
        // The errors of a batch, `[rows, batch]`.
        let mut errors = vec![0.0; rows * batch];
        let mut values = vec![0.0f32; group];
        let mut b0 = 0;
        while b0 < k {
            let b1 = (b0 + batch).min(k);
            errors.fill(0.0);
            for j in b0..b1 {
                if (j - b0).is_multiple_of(group) {
                    for r in 0..rows {
                        for (v, &x) in values.iter_mut().zip(&w[r * k + j..r * k + j + group]) {
                            *v = x as f32;
                        }
                        let scale = block_scale(dtype, &values).expect("a written format");
                        scales.push(scale);
                    }
                }
                let g = j / group;
                let diagonal = u[j * k + j];
                for r in 0..rows {
                    let scale = &scales[g * rows + r];
                    let x = w[r * k + j];
                    let code = block_code(dtype, scale, x as f32);
                    codes[r * k + j] = code;
                    let e = (x - block_value(dtype, scale, code)) / diagonal;
                    errors[r * batch + j - b0] = e;
                    // The rest of the batch at once.
                    let row = &mut w[r * k + j + 1..r * k + b1];
                    for (v, &f) in row.iter_mut().zip(&u[j * k + j + 1..j * k + b1]) {
                        *v -= e * f;
                    }
                }
            }
            // The columns after the batch: `W −= E · U[batch, after]`, by
            // tiles of columns.
            let mut c0 = b1;
            while c0 < k {
                let width = (4 * TILE).min(k - c0);
                product(
                    &mut w[c0..],
                    k,
                    &errors,
                    batch,
                    &u[b0 * k + c0..],
                    k,
                    (rows, b1 - b0, width),
                    -1.0,
                );
                c0 += width;
            }
            b0 = b1;
        }
        // The scales were pushed group by group, `rows` at a time.
        let groups = k / group;
        let mut row = Vec::with_capacity(row_bytes);
        for r in 0..rows {
            row.clear();
            for g in 0..groups {
                let scale = &scales[g * rows + r];
                encode_block(
                    dtype,
                    scale,
                    &codes[r * k + g * group..r * k + (g + 1) * group],
                    &mut row,
                );
            }
            bytes[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(&row);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapel_core::{QuantFormat, dequantize};

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

    /// Inputs `[t, k]` whose components are correlated, and their Hessian.
    fn inputs(t: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
        let base = noise(t * 8, 7);
        let own = noise(t * k, 9);
        let x: Vec<f64> = (0..t * k)
            .map(|e| {
                let (i, c) = (e / k, e % k);
                base[i * 8 + c % 8] + 0.3 * own[e]
            })
            .collect();
        let mut h = vec![0.0; k * k];
        for i in 0..t {
            for a in 0..k {
                for b in 0..k {
                    h[a * k + b] += x[i * k + a] * x[i * k + b];
                }
            }
        }
        (x, h)
    }

    /// `Σ ‖(W − Q) x‖²` over the inputs.
    fn output_error(w: &[f32], q: &[f64], x: &[f64], n: usize, k: usize) -> f64 {
        let t = x.len() / k;
        let mut total = 0.0;
        for i in 0..t {
            for r in 0..n {
                let e: f64 = (0..k)
                    .map(|c| (f64::from(w[r * k + c]) - q[r * k + c]) * x[i * k + c])
                    .sum();
                total += e * e;
            }
        }
        total
    }

    #[test]
    fn gptq_changes_the_outputs_less_than_rounding() {
        let (n, k) = (70, 192);
        let (x, mut h) = inputs(400, k);
        let w: Vec<f32> = noise(n * k, 3).iter().map(|&v| v as f32).collect();
        for dtype in [
            DType::int4(64).unwrap(),
            DType::int8(32).unwrap(),
            DType::Quantized(QuantFormat::Q4_0),
            DType::Quantized(QuantFormat::Q4_1),
            DType::Quantized(QuantFormat::Q8_0),
        ] {
            let (u, _, dead) = prepare(&mut h.clone(), k, 0.01).unwrap();
            let bytes = quantize(&w, n, k, dtype, &u, &dead);
            let gptq = dequantize(dtype, &bytes).unwrap();
            let rounded = dequantize(dtype, &kapel_core::quantize(dtype, &w).unwrap()).unwrap();
            let (a, b) = (
                output_error(&w, &gptq, &x, n, k),
                output_error(&w, &rounded, &x, n, k),
            );
            assert!(a < 0.8 * b, "{dtype}: GPTQ {a:e}, rounding {b:e}");
        }
        // A column no input reaches becomes zeros.
        for i in 0..k {
            h[5 * k + i] = 0.0;
            h[i * k + 5] = 0.0;
        }
        let dtype = DType::Quantized(QuantFormat::Q8_0);
        let (u, _, dead) = prepare(&mut h, k, 0.01).unwrap();
        assert_eq!(dead, [5]);
        let q = dequantize(dtype, &quantize(&w, n, k, dtype, &u, &dead)).unwrap();
        assert!((0..n).all(|r| q[r * k + 5] == 0.0));
    }

    /// Independent inputs of the same scale: nothing to spread, GPTQ is
    /// rounding to the nearest.
    #[test]
    fn independent_inputs_give_rounding() {
        let (n, k) = (9, 64);
        let mut h = vec![0.0; k * k];
        for i in 0..k {
            h[i * k + i] = 2.0;
        }
        let w: Vec<f32> = noise(n * k, 5).iter().map(|&v| v as f32).collect();
        for dtype in [
            DType::int8(32).unwrap(),
            DType::Quantized(QuantFormat::Q4_0),
            DType::Quantized(QuantFormat::Q4_1),
        ] {
            let (u, _, dead) = prepare(&mut h.clone(), k, 0.01).unwrap();
            assert_eq!(
                quantize(&w, n, k, dtype, &u, &dead),
                kapel_core::quantize(dtype, &w).unwrap(),
                "{dtype}"
            );
        }
    }

    #[test]
    fn a_hessian_that_is_not_one_is_an_error() {
        let mut h = vec![f64::NAN; 16];
        assert!(prepare(&mut h, 4, 0.01).is_err());
    }
}
