//! Special functions shared by the operations' definitions.

use std::f64::consts::PI;

/// The error function, `erf(x) = 2/√π ∫₀ˣ e^(−t²) dt`, to about 1e−15.
///
/// Below 2.5 in magnitude, its Taylor series, whose terms alternate and stay
/// small; above, `1 − erfc(x)` with the continued fraction of `erfc`, which
/// converges quickly there.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let a = x.abs();
    let value = if a < 2.5 {
        // Σ (−1)ⁿ x^(2n+1) / (n! (2n+1))
        let x2 = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0);
        loop {
            n += 1.0;
            term *= -x2 / n;
            let next = term / (2.0 * n + 1.0);
            sum += next;
            if next.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        2.0 / PI.sqrt() * sum
    } else {
        1.0 - erfc_tail(a)
    };
    value.copysign(x)
}

/// `erfc(x)` for `x ≥ 2.5`, by the continued fraction
/// `erfc(x) = e^(−x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))`,
/// evaluated with the modified Lentz method.
fn erfc_tail(x: f64) -> f64 {
    let tiny = 1e-300;
    let mut f = x;
    let (mut c, mut d) = (x, 0.0);
    for k in 1..200 {
        let a = f64::from(k) / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = x + a / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / PI.sqrt() / f
}

/// The standard normal cumulative distribution, `Φ(x) = (1 + erf(x/√2)) / 2`.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_tabulated_values() {
        // Values from Abramowitz and Stegun, table 7.1, and from mpmath.
        let table = [
            (0.0, 0.0),
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (2.5, 0.999_593_047_982_555),
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
        ];
        for (x, want) in table {
            assert!(
                (erf(x) - want).abs() < 2e-15,
                "erf({x}) = {}, want {want}",
                erf(x)
            );
            assert!((erf(-x) + want).abs() < 2e-15);
        }
        assert_eq!(erf(30.0), 1.0);
    }

    #[test]
    fn erf_is_continuous_where_the_methods_meet() {
        let below = erf(2.5 - 1e-12);
        let above = erf(2.5 + 1e-12);
        assert!((above - below).abs() < 1e-13);
    }
}
