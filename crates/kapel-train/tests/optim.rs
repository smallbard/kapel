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

/// The target of [`bowl`].
fn target() -> Vec<f32> {
    (0..64 * 128)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

/// `steps` steps of `opt` on the bowl `½ |w − target|²` from `p`, a
/// parameter `[64, 128]` (32 blocks of 8-bit moments); the distance to the
/// target after each.
fn bowl(opt: &mut AdamW, p: &mut Parameters, steps: usize) -> Vec<f64> {
    let cpu = kapel_cpu::Cpu::new();
    let target = target();
    let mut distances = Vec::new();
    for _ in 0..steps {
        let w = p.get("w").unwrap().value.to_vec::<f32>().unwrap();
        let g: Vec<f32> = w.iter().zip(&target).map(|(w, t)| w - t).collect();
        let grad = Tensor::from_slice(&g, &[64, 128]).unwrap();
        opt.step(&cpu, p, vec![("w".into(), Grad::Dense(grad))], 1.0)
            .unwrap();
        let w = p.get("w").unwrap().value.to_vec::<f32>().unwrap();
        let d: f64 = w
            .iter()
            .zip(&target)
            .map(|(w, t)| f64::from(w - t).powi(2))
            .sum();
        distances.push(d.sqrt());
    }
    distances
}

fn zeros() -> Parameters {
    params(&vec![0.0; 64 * 128], &[64, 128], false)
}

fn eight_bit(lr: f64) -> AdamW {
    let mut opt = AdamW::new(lr, 0.0);
    opt.eight_bit = true;
    opt
}

#[test]
fn adamw_in_8_bits_follows_adamw() {
    let a = bowl(&mut AdamW::new(0.01, 0.0), &mut zeros(), 200);
    let mut eight = eight_bit(0.01);
    let b = bowl(&mut eight, &mut zeros(), 200);
    assert!(eight.moments().is_empty());
    assert_eq!(eight.quantized()["w"].m_scale.shape(), &[32]);
    // Both from 51.5 to near the target, never apart by more than 1 % of
    // where they started.
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "from {:.3} to {:.4} and {:.4}, at most {:.4} apart",
        a[0], a[199], b[199], worst
    );
    assert!(
        a[199] < 0.001 * a[0] && b[199] < 0.001 * a[0],
        "{} and {}",
        a[199],
        b[199]
    );
    assert!(worst < 0.01 * a[0], "{worst}");
}

#[test]
fn small_parameters_keep_their_moments_in_f32() {
    let cpu = kapel_cpu::Cpu::new();
    let mut p = params(&[1.0, -2.0, 0.5], &[3], true);
    let mut opt = eight_bit(0.1);
    let grad = Tensor::from_slice(&[0.5f32, -1.0, 2.0], &[3]).unwrap();
    opt.step(&cpu, &mut p, vec![("w".into(), Grad::Dense(grad))], 1.0)
        .unwrap();
    assert!(opt.quantized().is_empty());
    assert_eq!(opt.moments()["w"].0.dtype(), kapel_core::DType::F32);
}

#[test]
fn moments_in_8_bits_resume_bit_for_bit() {
    let straight = bowl(&mut eight_bit(0.01), &mut zeros(), 10);
    let mut p = zeros();
    let mut half = eight_bit(0.01);
    bowl(&mut half, &mut p, 5);
    let state = half.state();
    drop(half);
    assert_eq!(state.tensors["m.w"].dtype(), kapel_core::DType::U8);
    assert!(state.tensors.contains_key("v_scale.w"));
    let mut resumed = eight_bit(0.01);
    resumed.restore_state(state).unwrap();
    let rest = bowl(&mut resumed, &mut p, 5);
    assert_eq!(rest[4].to_bits(), straight[9].to_bits());
}
