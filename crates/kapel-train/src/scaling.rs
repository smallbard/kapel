//! Laws of scale fitted to a grid of runs: the curves at constant
//! computation (Hoffmann et al., 2022, their second method) and the formula
//! of the loss (their third), Smallem's `scaling.rs`.
//!
//! - For each budget `C`, the losses of the sizes tried lie on a parabola in
//!   `ln N`: its lowest point is the best size for that budget, `N_opt(C)`,
//!   and the tokens it sees, `D_opt(C)`. Power laws `N_opt ∝ C^a`,
//!   `D_opt ∝ C^b` and `L_opt ∝ C^l` through these optima predict larger
//!   budgets.
//! - `L(N, D) = E + A / N^α + B / D^β` fitted to every run: a Huber loss of
//!   threshold 10⁻³ on `ln L`, written with a log-sum-exp as the paper does,
//!   **summed** over the runs (Besiroglu et al., 2024, found Chinchilla
//!   averaged it, which stopped L-BFGS early), minimized by BFGS from the
//!   paper's grid of 4,500 starts. A bootstrap gives each constant, and each
//!   prediction, an interval.

use std::collections::BTreeMap;

use crate::model::Rng;

/// The parabola `y = a x² + b x + c` of least squares through `(x, y)`.
///
/// # Panics
///
/// With fewer than three points.
pub fn fit_parabola(points: &[(f64, f64)]) -> (f64, f64, f64) {
    assert!(points.len() >= 3, "three points at least for a parabola");
    // The normal equations, `[Σx⁴ Σx³ Σx²; Σx³ Σx² Σx; Σx² Σx n]`.
    let mut m = [[0.0; 3]; 3];
    let mut v = [0.0; 3];
    for &(x, y) in points {
        let powers = [x * x, x, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += powers[i] * powers[j];
            }
            v[i] += powers[i] * y;
        }
    }
    let s = solve3(m, v);
    (s[0], s[1], s[2])
}

/// A 3 × 3 system by Gauss's elimination, with pivoting.
fn solve3(mut m: [[f64; 3]; 3], mut v: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .expect("a row");
        m.swap(col, pivot);
        v.swap(col, pivot);
        for row in col + 1..3 {
            let factor = m[row][col] / m[col][col];
            let pivot_row = m[col];
            for (value, pivot) in m[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            v[row] -= factor * v[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let rest: f64 = (row + 1..3).map(|k| m[row][k] * x[k]).sum();
        x[row] = (v[row] - rest) / m[row][row];
    }
    x
}

/// The line of least squares `y = slope · x + intercept`.
pub fn fit_line(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let covariance: f64 = points.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    let variance: f64 = points.iter().map(|&(x, _)| (x - mx).powi(2)).sum();
    let slope = covariance / variance;
    (slope, my - slope * mx)
}

/// A power law `y = k · x^exponent` fitted in logarithms: `(exponent, k)`.
pub fn fit_power_law(points: &[(f64, f64)]) -> (f64, f64) {
    let logs: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x.ln(), y.ln())).collect();
    let (slope, intercept) = fit_line(&logs);
    (slope, intercept.exp())
}

/// One run of a grid: its size, its tokens, its computation and its loss.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// Its name.
    pub name: String,
    /// The parameters counted (all of them, or but the tables).
    pub parameters: f64,
    /// The tokens it learnt from.
    pub tokens: f64,
    /// Its operations.
    pub flops: f64,
    /// The budget it was given, if any: the runs of one budget make a curve.
    pub budget: Option<f64>,
    /// Its final loss.
    pub loss: f64,
}

/// The best size for one budget: the lowest point of the parabola of its
/// runs in `ln N`.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimum {
    /// The budget.
    pub budget: f64,
    /// The best size.
    pub parameters: f64,
    /// The tokens of the best size: the budget over its operations by
    /// token, interpolated between the runs.
    pub tokens: f64,
    /// The loss at the lowest point.
    pub loss: f64,
    /// The curvature `a` of the parabola: a minimum only when positive.
    pub curvature: f64,
    /// The runs of the curve, by size.
    pub runs: Vec<Point>,
}

/// The curves at constant computation: the runs grouped by budget, three at
/// least to a curve, each with its optimum.
pub fn isoflops(points: &[Point]) -> Vec<Optimum> {
    let mut budgets: BTreeMap<u64, Vec<&Point>> = BTreeMap::new();
    for p in points {
        if let Some(b) = p.budget {
            budgets.entry(b.to_bits()).or_default().push(p);
        }
    }
    let mut curves = Vec::new();
    for (bits, mut runs) in budgets {
        if runs.len() < 3 {
            continue;
        }
        runs.sort_by(|a, b| a.parameters.total_cmp(&b.parameters));
        let budget = f64::from_bits(bits);
        let curve: Vec<(f64, f64)> = runs.iter().map(|r| (r.parameters.ln(), r.loss)).collect();
        let (a, b, c) = fit_parabola(&curve);
        let best = -b / (2.0 * a);
        let per_token: Vec<(f64, f64)> = runs
            .iter()
            .map(|r| (r.parameters.ln(), (r.flops / r.tokens).ln()))
            .collect();
        let (slope, intercept) = fit_line(&per_token);
        curves.push(Optimum {
            budget,
            parameters: best.exp(),
            tokens: budget / (intercept + slope * best).exp(),
            loss: a * best * best + b * best + c,
            curvature: a,
            runs: runs.into_iter().cloned().collect(),
        });
    }
    curves
}

/// The power laws through the optima of the curves: `(exponent, k)` of
/// `N_opt`, `D_opt` and `L_opt` against the budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimaLaws {
    /// `N_opt = k · C^exponent`.
    pub parameters: (f64, f64),
    /// `D_opt = k · C^exponent`.
    pub tokens: (f64, f64),
    /// `L_opt = k · C^exponent`.
    pub loss: (f64, f64),
}

impl OptimaLaws {
    /// The laws through the optima that are minima, two at least.
    pub fn fit(optima: &[Optimum]) -> Option<Self> {
        let minima: Vec<&Optimum> = optima.iter().filter(|o| o.curvature > 0.0).collect();
        if minima.len() < 2 {
            return None;
        }
        let law = |f: &dyn Fn(&Optimum) -> f64| {
            fit_power_law(&minima.iter().map(|o| (o.budget, f(o))).collect::<Vec<_>>())
        };
        Some(Self {
            parameters: law(&|o| o.parameters),
            tokens: law(&|o| o.tokens),
            loss: law(&|o| o.loss),
        })
    }

    /// The size, tokens and loss they predict for a budget.
    pub fn predict(&self, budget: f64) -> (f64, f64, f64) {
        let at = |(e, k): (f64, f64)| k * budget.powf(e);
        (at(self.parameters), at(self.tokens), at(self.loss))
    }
}

/// Chinchilla's formula, `L(N, D) = E + A / N^α + B / D^β`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossLaw {
    /// The loss no size or data removes.
    pub e: f64,
    /// `A`.
    pub a: f64,
    /// `α`.
    pub alpha: f64,
    /// `B`.
    pub b: f64,
    /// `β`.
    pub beta: f64,
}

impl LossLaw {
    /// The loss of `parameters` trained on `tokens`.
    pub fn predict(&self, parameters: f64, tokens: f64) -> f64 {
        self.e + self.a / parameters.powf(self.alpha) + self.b / tokens.powf(self.beta)
    }

    /// The size and tokens of least loss for a computation `c = k N D`:
    /// under `N D = c / k`, `N = G (c/k)^(β/(α+β))` with
    /// `G = (α A / (β B))^(1/(α+β))`, as in the paper. `(N, D)`.
    pub fn optimum(&self, compute: f64, k: f64) -> (f64, f64) {
        let budget = compute / k;
        let g = (self.alpha * self.a / (self.beta * self.b)).powf(1.0 / (self.alpha + self.beta));
        let parameters = g * budget.powf(self.beta / (self.alpha + self.beta));
        (parameters, budget / parameters)
    }
}

/// Huber's loss of threshold `delta`, and its derivative.
fn huber(r: f64, delta: f64) -> (f64, f64) {
    if r.abs() <= delta {
        (0.5 * r * r, r)
    } else {
        (delta * (r.abs() - 0.5 * delta), delta * r.signum())
    }
}

/// The objective of the paper and its gradient, for `θ = (a, b, e, α, β)`
/// with `A = eᵃ`, `B = eᵇ`, `E = eᵉ`:
/// `Σ Huber(LSE(a − α ln N, b − β ln D, e) − ln L)`.
fn objective(theta: &[f64; 5], points: &[(f64, f64, f64)], delta: f64) -> (f64, [f64; 5]) {
    let [a, b, e, alpha, beta] = *theta;
    let mut value = 0.0;
    let mut grad = [0.0; 5];
    for &(ln_n, ln_d, ln_l) in points {
        let terms = [a - alpha * ln_n, b - beta * ln_d, e];
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = terms.map(|t| (t - max).exp());
        let total: f64 = weights.iter().sum();
        let (h, dh) = huber(max + total.ln() - ln_l, delta);
        value += h;
        let soft = weights.map(|w| w / total);
        grad[0] += dh * soft[0];
        grad[1] += dh * soft[1];
        grad[2] += dh * soft[2];
        grad[3] -= dh * soft[0] * ln_n;
        grad[4] -= dh * soft[1] * ln_d;
    }
    (value, grad)
}

/// The objective minimized by BFGS from `start`, with Armijo's line
/// search: the point found and its value.
fn bfgs(start: [f64; 5], points: &[(f64, f64, f64)], delta: f64) -> ([f64; 5], f64) {
    let identity = || {
        std::array::from_fn::<[f64; 5], 5, _>(|i| {
            std::array::from_fn(|j| f64::from(u8::from(i == j)))
        })
    };
    let mut x = start;
    let (mut f, mut g) = objective(&x, points, delta);
    let mut h = identity();
    for _ in 0..500 {
        let direction: [f64; 5] =
            std::array::from_fn(|i| -(0..5).map(|j| h[i][j] * g[j]).sum::<f64>());
        let slope: f64 = (0..5).map(|i| direction[i] * g[i]).sum();
        if slope >= 0.0 {
            // Not a way down: start the curvature again.
            h = identity();
            if g.iter().all(|v| *v == 0.0) {
                break;
            }
            continue;
        }
        let mut step = 1.0;
        let (mut next, mut f_next, mut g_next);
        loop {
            next = std::array::from_fn(|i| x[i] + step * direction[i]);
            (f_next, g_next) = objective(&next, points, delta);
            if f_next <= f + 1e-4 * step * slope || step < 1e-12 {
                break;
            }
            step *= 0.5;
        }
        let s: [f64; 5] = std::array::from_fn(|i| next[i] - x[i]);
        let y: [f64; 5] = std::array::from_fn(|i| g_next[i] - g[i]);
        let sy: f64 = (0..5).map(|i| s[i] * y[i]).sum();
        let converged = (f - f_next).abs() < 1e-14 * (1.0 + f.abs());
        (x, f, g) = (next, f_next, g_next);
        if converged {
            break;
        }
        if sy > 1e-12 {
            let hy: [f64; 5] = std::array::from_fn(|i| (0..5).map(|j| h[i][j] * y[j]).sum());
            let yhy: f64 = (0..5).map(|i| y[i] * hy[i]).sum();
            for i in 0..5 {
                for j in 0..5 {
                    h[i][j] +=
                        (sy + yhy) * s[i] * s[j] / (sy * sy) - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
        }
    }
    (x, f)
}

/// The starts of the paper: `α, β ∈ {0, 0.5, …, 2}`, `e ∈ {−1, −0.5, …, 1}`,
/// `a, b ∈ {0, 5, …, 25}`.
fn starts() -> Vec<[f64; 5]> {
    let mut out = Vec::with_capacity(4500);
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0] {
        for beta in [0.0, 0.5, 1.0, 1.5, 2.0] {
            for e in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                for a in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
                    for b in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
                        out.push([a, b, e, alpha, beta]);
                    }
                }
            }
        }
    }
    out
}

/// Chinchilla's formula fitted to `(N, D, L)` triples, as the paper does
/// it, the starts shared among the cores; `None` with fewer than five.
pub fn fit_loss_law(points: &[(f64, f64, f64)]) -> Option<LossLaw> {
    if points.len() < 5 {
        return None;
    }
    let logs: Vec<(f64, f64, f64)> = points
        .iter()
        .map(|&(n, d, l)| (n.ln(), d.ln(), l.ln()))
        .collect();
    let starts = starts();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = starts.len().div_ceil(threads);
    let best = std::thread::scope(|scope| {
        let handles: Vec<_> = starts
            .chunks(chunk)
            .map(|part| {
                let logs = &logs;
                scope.spawn(move || {
                    part.iter()
                        .map(|&s| bfgs(s, logs, 1e-3))
                        .filter(|(_, f)| f.is_finite())
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().expect("a fit"))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    })?;
    let [a, b, e, alpha, beta] = best.0;
    Some(LossLaw {
        e: e.exp(),
        a: a.exp(),
        alpha,
        b: b.exp(),
        beta,
    })
}

/// The formula fitted again to `rounds` draws of the points with
/// replacement, from `seed`: an interval for each constant and prediction.
pub fn bootstrap(points: &[(f64, f64, f64)], rounds: usize, seed: u64) -> Vec<LossLaw> {
    let mut rng = Rng::new(seed);
    (0..rounds)
        .filter_map(|_| {
            let sample: Vec<(f64, f64, f64)> = (0..points.len())
                .map(|_| points[(rng.next_u64() % points.len() as u64) as usize])
                .collect();
            fit_loss_law(&sample)
        })
        .collect()
}

/// The quantile `q` of `values`, the nearest rank.
pub fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parabola_through_three_points_is_exact() {
        let (a, b, c) = fit_parabola(&[(0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]);
        assert!((a - 1.0).abs() < 1e-12 && (b + 2.0).abs() < 1e-12 && (c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_power_law_is_a_line_in_logarithms() {
        let points: Vec<(f64, f64)> = [1e15f64, 1e16, 1e17]
            .iter()
            .map(|&c| (c, 3.0 * c.powf(0.5)))
            .collect();
        let (exponent, k) = fit_power_law(&points);
        assert!((exponent - 0.5).abs() < 1e-12 && (k / 3.0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_loss_law_is_found_again_from_noisy_runs() {
        let truth = LossLaw {
            e: 0.7,
            a: 40.0,
            alpha: 0.35,
            b: 300.0,
            beta: 0.3,
        };
        let mut rng = Rng::new(1);
        let mut points = Vec::new();
        for n in [1e6, 3e6, 1e7, 3e7, 1e8] {
            for d in [1e7, 3e7, 1e8, 3e8, 1e9] {
                let noise = 1.0 + 0.002 * (rng.uniform() - 0.5);
                points.push((n, d, truth.predict(n, d) * noise));
            }
        }
        let fitted = fit_loss_law(&points).unwrap();
        for &(n, d, _) in &points {
            let error = (fitted.predict(n, d) / truth.predict(n, d) - 1.0).abs();
            assert!(error < 0.003, "{fitted:?}");
        }
        assert!((fitted.alpha - truth.alpha).abs() < 0.05, "{fitted:?}");
        assert!((fitted.beta - truth.beta).abs() < 0.05, "{fitted:?}");
        // The closed form is the least loss at its computation.
        let (n, d) = truth.optimum(1e18, 6.0);
        let best = truth.predict(n, d);
        for factor in [0.8, 1.25] {
            assert!(truth.predict(n * factor, d / factor) > best);
        }
    }

    #[test]
    fn each_budget_has_its_best_size() {
        // Losses on parabolas in ln N whose lowest points follow
        // `N_opt = 0.1 C^0.5`, the tokens `C / 6N`.
        let mut points = Vec::new();
        for budget in [1e15, 1e16, 1e17] {
            let best = 0.1 * f64::sqrt(budget);
            for factor in [0.25, 0.5, 1.0, 2.0, 4.0] {
                let n: f64 = best * factor;
                let x = (n / best).ln();
                points.push(Point {
                    name: format!("{budget:e}-{factor}"),
                    parameters: n,
                    tokens: budget / (6.0 * n),
                    flops: budget,
                    budget: Some(budget),
                    loss: 3.0 - 0.1 * budget.log10() + 0.05 * x * x,
                });
            }
        }
        let curves = isoflops(&points);
        assert_eq!(curves.len(), 3);
        let laws = OptimaLaws::fit(&curves).unwrap();
        assert!((laws.parameters.0 - 0.5).abs() < 1e-9, "{laws:?}");
        assert!((laws.tokens.0 - 0.5).abs() < 1e-9, "{laws:?}");
        let (n, d, _) = laws.predict(1e18);
        assert!((n / 1e8 - 1.0).abs() < 1e-6 && (d * 6.0 * n / 1e18 - 1.0).abs() < 1e-6);
    }
}
