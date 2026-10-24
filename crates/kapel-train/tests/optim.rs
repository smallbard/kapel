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
