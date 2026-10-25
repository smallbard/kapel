//! AdamW against its formula, and by rows against dense.

use kapel_core::Tensor;
use kapel_train::Grad;
use kapel_train::model::{Parameter, Parameters};
use kapel_train::optim::AdamW;

fn params(values: &[f32], shape: &[usize], decay: bool) -> Parameters {
    let mut p = Parameters::default();
    let mut parameter = Parameter::new(Tensor::from_slice(values, shape).unwrap());
    parameter.decay = decay;
    p.insert("w", parameter);
    p
}

#[test]
fn adamw_follows_its_formula() {
    let cpu = kapel_cpu::Cpu::new();
    let mut p = params(&[1.0, -2.0, 0.5], &[3], true);
    let mut opt = AdamW::new(0.1, 0.01);
    let grads = [[0.5f32, -1.0, 2.0], [-0.25, 0.75, 1.0]];
    let (mut m, mut v, mut x) = ([0.0f64; 3], [0.0f64; 3], [1.0f64, -2.0, 0.5]);
    for (t, g) in grads.iter().enumerate() {
        let tensor = Tensor::from_slice(g, &[3]).unwrap();
        opt.step(&cpu, &mut p, vec![("w".into(), Grad::Dense(tensor))], 0.5)
            .unwrap();
        let t = t as i32 + 1;
        for i in 0..3 {
            let g = f64::from(g[i]) * 0.5;
            m[i] = 0.9 * m[i] + 0.1 * g;
            v[i] = 0.95 * v[i] + 0.05 * g * g;
            let (mh, vh) = (
                m[i] / (1.0 - 0.9f64.powi(t)),
                v[i] / (1.0 - 0.95f64.powi(t)),
            );
            x[i] -= 0.1 * (0.01 * x[i] + mh / (vh.sqrt() + 1e-8));
        }
    }
    let got = p.get("w").unwrap().value.to_vec::<f32>().unwrap();
    for (a, b) in got.iter().zip(x) {
        assert!((f64::from(*a) - b).abs() < 1e-6, "{a} and {b}");
    }
}

#[test]
fn adamw_by_rows_moves_only_the_rows_read() {
    let cpu = kapel_cpu::Cpu::new();
    let table: Vec<f32> = (0..12).map(|i| i as f32 / 4.0).collect();
    // Without decay, the first step leaves rows without gradient as they
    // are: dense and by rows agree.
    let mut dense = params(&table, &[4, 3], false);
    let mut rows = dense.clone();
    let indices = Tensor::from_slice(&[2u32, 0, 2], &[3]).unwrap();
    let values = Tensor::from_slice(
        &[1.0f32, -1.0, 0.5, 0.25, 0.25, 0.25, -2.0, 1.0, 1.0],
        &[3, 3],
    )
    .unwrap();
    let grad = Grad::Rows {
        indices: indices.clone(),
        values: values.clone(),
    };
    let full = grad.clone().dense(&cpu, &[4, 3]).unwrap();
    let (mut a, mut b) = (AdamW::new(0.1, 0.0), AdamW::new(0.1, 0.0));
    a.step(&cpu, &mut dense, vec![("w".into(), Grad::Dense(full))], 1.0)
        .unwrap();
    b.step(&cpu, &mut rows, vec![("w".into(), grad)], 1.0)
        .unwrap();
    let (x, y) = (
        dense.get("w").unwrap().value.to_vec::<f32>().unwrap(),
        rows.get("w").unwrap().value.to_vec::<f32>().unwrap(),
    );
    assert_eq!(x, y);
    assert_eq!(&y[3..6], &table[3..6]);
    assert_eq!(&y[9..12], &table[9..12]);
    assert_ne!(&y[0..3], &table[0..3]);
    assert_eq!(
        a.moments()["w"].0.to_vec::<f32>().unwrap(),
        b.moments()["w"].0.to_vec::<f32>().unwrap()
    );
}

/// The eigenvalues of `X Xᵀ` for `X` `[m, n]`, `m ≤ n`, by Jacobi's rotations.
fn gram_eigenvalues(x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut a: Vec<f64> = (0..m * m)
        .map(|ij| (0..n).map(|k| x[ij / m * n + k] * x[ij % m * n + k]).sum())
        .collect();
    for _ in 0..100 {
        for p in 0..m {
            for q in p + 1..m {
                if a[p * m + q].abs() < 1e-15 {
                    continue;
                }
                let theta = (a[q * m + q] - a[p * m + p]) / (2.0 * a[p * m + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..m {
                    let (akp, akq) = (a[k * m + p], a[k * m + q]);
                    a[k * m + p] = c * akp - s * akq;
                    a[k * m + q] = s * akp + c * akq;
                }
                for k in 0..m {
                    let (apk, aqk) = (a[p * m + k], a[q * m + k]);
                    a[p * m + k] = c * apk - s * aqk;
                    a[q * m + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..m).map(|i| a[i * m + i]).collect()
}

#[test]
fn both_iterations_of_newton_schulz_give_nearly_orthogonal_matrices() {
    use kapel_train::optim::{NEWTON_SCHULZ, NEWTON_SCHULZ_SHORT, orthogonalize};
    let (m, n) = (8, 16);
    // Singular values spread over two orders of magnitude.
    let values: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (((r * 31 + c * 17) % 13) as f32 - 6.0) * 0.1 * (1.0 + r as f32).powi(2)
        })
        .collect();
    let x = Tensor::from_slice(&values, &[1, m, n]).unwrap();
    let cpu = kapel_cpu::Cpu::new();
    for (coefficients, low, high) in [
        (&NEWTON_SCHULZ[..], 0.9, 1.1),
        (&NEWTON_SCHULZ_SHORT[..], 0.6, 1.3),
    ] {
        let o = orthogonalize(&cpu, &x, coefficients)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        for e in gram_eigenvalues(&o, m, n) {
            let s = e.sqrt();
            assert!(
                (low..high).contains(&s),
                "{} iterations: {s}",
                coefficients.len()
            );
        }
    }
}
