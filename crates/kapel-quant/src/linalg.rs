//! The linear algebra of GPTQ, in `f64` on every core: the factor `U` of
//! the inverse of a Hessian, `H⁻¹ = Uᵀ · U` with `U` upper triangular.
//!
//! `H` of a projection is `K × K`, `K` its width in (up to some 15 000), and
//! the factor costs a few `K³` operations: the matrices are cut in tiles of
//! [`TILE`] × [`TILE`], and the products of tiles, where the work is, run
//! on every core.
//!
//! With `P` the matrix that reverses the order of the indices, `P H P =
//! L Lᵀ` (Cholesky) gives `H = R Rᵀ` with `R = P L P` upper triangular, so
//! that `H⁻¹ = R⁻ᵀ R⁻¹`: `U = R⁻¹ = P L⁻¹ P`. A Cholesky factorization and
//! the inverse of a triangular matrix, instead of the three steps of the
//! reference implementation (Cholesky, inverse, Cholesky again).

use std::sync::Mutex;

/// The side of a tile.
pub(crate) const TILE: usize = 64;

/// A matrix that is not positive definite: the pivot where it showed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NotDefinite {
    pub(crate) at: usize,
    pub(crate) pivot: f64,
}

/// Runs `work` on every item, on every core: items are taken in order, so
/// that the largest ones should come first.
pub(crate) fn parallel<T: Send>(items: Vec<T>, work: impl Fn(T) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let count = items.len();
    let queue = Mutex::new(items.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..threads.min(count) {
            scope.spawn(|| {
                loop {
                    let item = queue.lock().unwrap_or_else(|e| e.into_inner()).next();
                    let Some(item) = item else { break };
                    work(item);
                }
            });
        }
    });
}

/// `s[r][c] += sign · Σₘ a[r][m] · b[m][c]` over `rows × inner × cols`, the
/// three matrices row-major with strides `ls`, `la`, `lb`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product(
    s: &mut [f64],
    ls: usize,
    a: &[f64],
    la: usize,
    b: &[f64],
    lb: usize,
    (rows, inner, cols): (usize, usize, usize),
    sign: f64,
) {
    for r in 0..rows {
        let out = &mut s[r * ls..r * ls + cols];
        for m in 0..inner {
            let factor = sign * a[r * la + m];
            if factor == 0.0 {
                continue;
            }
            let row = &b[m * lb..m * lb + cols];
            for (o, &v) in out.iter_mut().zip(row) {
                *o += factor * v;
            }
        }
    }
}

/// The Cholesky factor `L` of the symmetric positive definite `a` `[n, n]`
/// (`a = L Lᵀ`), in place: the lower triangle, the upper one zeroed.
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> Result<(), NotDefinite> {
    assert_eq!(a.len(), n * n, "cholesky: an n × n matrix");
    let mut k0 = 0;
    while k0 < n {
        let kn = TILE.min(n - k0);
        // The diagonal tile, unblocked.
        for i in k0..k0 + kn {
            for j in k0..=i {
                let mut sum = a[i * n + j];
                for m in k0..j {
                    sum -= a[i * n + m] * a[j * n + m];
                }
                if i == j {
                    if sum.is_nan() || sum <= 0.0 {
                        return Err(NotDefinite { at: i, pivot: sum });
                    }
                    a[i * n + i] = sum.sqrt();
                } else {
                    a[i * n + j] = sum / a[j * n + j];
                }
            }
        }
        let next = k0 + kn;
        if next == n {
            break;
        }
        let diagonal: Vec<f64> = (0..kn * kn)
            .map(|e| a[(k0 + e / kn) * n + k0 + e % kn])
            .collect();
        // The panel below it, row by row: `L_ik = A_ik · L_kk⁻ᵀ`.
        let rows: Vec<&mut [f64]> = a[next * n..].chunks_mut(n * TILE).collect();
        parallel(rows, |block| {
            for row in block.chunks_mut(n) {
                let x = &mut row[k0..k0 + kn];
                for m in 0..kn {
                    let mut sum = x[m];
                    for p in 0..m {
                        sum -= x[p] * diagonal[m * kn + p];
                    }
                    x[m] = sum / diagonal[m * kn + m];
                }
            }
        });
        // The panel transposed, `[kn, n]`, for the trailing update.
        let mut panel = vec![0.0; kn * n];
        for r in next..n {
            for m in 0..kn {
                panel[m * n + r] = a[r * n + k0 + m];
            }
        }
        // The trailing matrix: `A_ij −= L_ik L_jkᵀ`, its lower triangle,
        // block row by block row (the longest first).
        let mut blocks: Vec<(usize, &mut [f64])> = a[next * n..]
            .chunks_mut(n * TILE)
            .enumerate()
            .map(|(b, rows)| (next + b * TILE, rows))
            .collect();
        blocks.reverse();
        parallel(blocks, |(r0, rows)| {
            let height = rows.len() / n;
            let mut local = vec![0.0; height * kn];
            for r in 0..height {
                local[r * kn..(r + 1) * kn].copy_from_slice(&rows[r * n + k0..r * n + k0 + kn]);
            }
            let mut c0 = next;
            while c0 < r0 + height {
                let width = TILE.min(r0 + height - c0);
                product(
                    &mut rows[c0..],
                    n,
                    &local,
                    kn,
                    &panel[c0..],
                    n,
                    (height, kn, width),
                    -1.0,
                );
                c0 += width;
            }
        });
        k0 = next;
    }
    for i in 0..n {
        a[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

/// The inverse of the lower triangular `l` `[n, n]`, lower triangular.
pub(crate) fn lower_inverse(l: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(l.len(), n * n, "lower_inverse: an n × n matrix");
    let blocks = n.div_ceil(TILE);
    let span = |b: usize| (b * TILE, TILE.min(n - b * TILE));
    // The inverse of each diagonal tile.
    let mut diagonals: Vec<Vec<f64>> = vec![Vec::new(); blocks];
    {
        let slots: Vec<(usize, &mut Vec<f64>)> = diagonals.iter_mut().enumerate().collect();
        parallel(slots, |(b, slot)| {
            let (o, w) = span(b);
            let mut x = vec![0.0; w * w];
            for j in 0..w {
                x[j * w + j] = 1.0 / l[(o + j) * n + o + j];
                for i in j + 1..w {
                    let mut sum = 0.0;
                    for m in j..i {
                        sum += l[(o + i) * n + o + m] * x[m * w + j];
                    }
                    x[i * w + j] = -sum / l[(o + i) * n + o + i];
                }
            }
            *slot = x;
        });
    }
    // Block column by block column, the longest first: `X_jj = L_jj⁻¹`, then
    // `X_ij = −L_ii⁻¹ · Σₖ L_ik X_kj` for `i > j`, `k` from `j` to `i − 1`.
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); blocks];
    {
        let slots: Vec<(usize, &mut Vec<f64>)> = columns.iter_mut().enumerate().collect();
        parallel(slots, |(j, slot)| {
            let (oj, wj) = span(j);
            // The rows from `oj` down, `wj` wide.
            let mut column = vec![0.0; (n - oj) * wj];
            column[..wj * wj].copy_from_slice(&diagonals[j]);
            for (i, diagonal) in diagonals.iter().enumerate().skip(j + 1) {
                let (oi, wi) = span(i);
                let mut sum = vec![0.0; wi * wj];
                product(
                    &mut sum,
                    wj,
                    &l[oi * n + oj..],
                    n,
                    &column,
                    wj,
                    (wi, oi - oj, wj),
                    1.0,
                );
                let at = (oi - oj) * wj;
                product(
                    &mut column[at..],
                    wj,
                    diagonal,
                    wi,
                    &sum,
                    wj,
                    (wi, wi, wj),
                    -1.0,
                );
            }
            *slot = column;
        });
    }
    let mut x = vec![0.0; n * n];
    for (j, column) in columns.iter().enumerate() {
        let (oj, wj) = span(j);
        for r in oj..n {
            x[r * n + oj..r * n + oj + wj]
                .copy_from_slice(&column[(r - oj) * wj..(r - oj + 1) * wj]);
        }
    }
    x
}

/// The upper triangular `U` of `H⁻¹ = Uᵀ · U`, `h` `[n, n]` symmetric
/// positive definite.
pub(crate) fn inverse_factor(h: &[f64], n: usize) -> Result<Vec<f64>, NotDefinite> {
    assert_eq!(h.len(), n * n, "inverse_factor: an n × n matrix");
    let reversed = |m: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = m[(n - 1 - i) * n + (n - 1 - j)];
            }
        }
        out
    };
    let mut l = reversed(h);
    cholesky(&mut l, n).map_err(|e| NotDefinite {
        at: n - 1 - e.at,
        pivot: e.pivot,
    })?;
    Ok(reversed(&lower_inverse(&l, n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric positive definite matrix: `X Xᵀ / k + I / 10`.
    fn spd(n: usize, seed: u64) -> Vec<f64> {
        let k = n + 7;
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let x: Vec<f64> = (0..n * k).map(|_| next()).collect();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..k).map(|m| x[i * k + m] * x[j * k + m]).sum();
                h[i * n + j] = dot / k as f64 + if i == j { 0.1 } else { 0.0 };
            }
        }
        h
    }

    fn multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for m in 0..n {
                for j in 0..n {
                    out[i * n + j] += a[i * n + m] * b[m * n + j];
                }
            }
        }
        out
    }

    fn transposed(a: &[f64], n: usize) -> Vec<f64> {
        (0..n * n).map(|e| a[(e % n) * n + e / n]).collect()
    }

    #[test]
    fn the_factor_inverts_the_matrix() {
        // Sizes below, at and across tiles.
        for n in [1, 5, TILE, TILE + 3, 2 * TILE + 17] {
            let h = spd(n, n as u64);
            let u = inverse_factor(&h, n).unwrap();
            for i in 0..n {
                for j in 0..i {
                    assert_eq!(u[i * n + j], 0.0, "{n}: U is upper triangular");
                }
            }
            // `H · Uᵀ U = I`.
            let inverse = multiply(&transposed(&u, n), &u, n);
            let identity = multiply(&h, &inverse, n);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    let e = (identity[i * n + j] - want).abs();
                    assert!(e < 1e-9, "{n}: ({i}, {j}) off by {e:e}");
                }
            }
        }
    }

    #[test]
    fn cholesky_matches_the_unblocked_one() {
        let n = 3 * TILE + 5;
        let h = spd(n, 3);
        let mut blocked = h.clone();
        cholesky(&mut blocked, n).unwrap();
        let mut plain = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let dot: f64 = (0..j).map(|m| plain[i * n + m] * plain[j * n + m]).sum();
                plain[i * n + j] = if i == j {
                    (h[i * n + i] - dot).sqrt()
                } else {
                    (h[i * n + j] - dot) / plain[j * n + j]
                };
            }
        }
        for (a, b) in blocked.iter().zip(&plain) {
            assert!((a - b).abs() < 1e-10, "{a} ≠ {b}");
        }
    }

    #[test]
    fn an_indefinite_matrix_is_an_error() {
        let mut h = spd(TILE + 9, 5);
        let n = TILE + 9;
        h[(n - 3) * n + (n - 3)] = -1.0;
        let error = inverse_factor(&h, n).unwrap_err();
        assert!(error.pivot <= 0.0);
        let nan = vec![f64::NAN; 4];
        assert!(inverse_factor(&nan, 2).is_err());
    }
}
