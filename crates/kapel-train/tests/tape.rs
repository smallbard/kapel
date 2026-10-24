//! The tape against finite differences, on the reference backend in `f64`,
//! and the processor's gradients against the reference's.

use kapel_core::{DType, Tensor};
use kapel_ops::Transfer;
use kapel_ops::{Attention, Rope, RopeLayout, RopeScaling, UnaryKind};
use kapel_ref::Reference;
use kapel_train::{Compute, Grad, IGNORE, Tape, Var, cross_entropy, linear_cross_entropy};

/// Reproducible values in `[-1, 1)`.
fn values(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = kapel_train::model::Rng::new(seed);
    (0..n).map(|_| 2.0 * rng.uniform() - 1.0).collect()
}

fn tensor(seed: u64, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_slice(&values(seed, n), shape).unwrap()
}

/// A scalar of `out`: `Σ out · r` for fixed `r`, so that every element of
/// the output matters.
fn weighted<B: Compute>(out: &Var<B>, seed: u64) -> Var<B> {
    let tape = out.tape();
    let shape = out.shape();
    let r = tape.constant(
        tape.backend()
            .upload(&cast(&tensor(seed, &shape), out.dtype()))
            .unwrap(),
    );
    out.mul(&r).unwrap().sum_all().unwrap()
}

fn cast(t: &Tensor, dtype: DType) -> Tensor {
    use kapel_ops::Run;
    let y = Reference.forward(&kapel_ops::Cast { dtype }, &[t]).unwrap();
    assert_eq!(y.dtype(), dtype);
    y
}

type Build<'a, B> = dyn Fn(&Tape<B>, &[Var<B>]) -> Var<B> + 'a;

/// The gradients of `f` at `inputs` on `backend`, dense, in `f64`.
fn gradients<B: Compute>(backend: B, inputs: &[Tensor], f: &Build<'_, B>) -> (f64, Vec<Vec<f64>>) {
    let tape = Tape::new(backend.clone());
    let leaves: Vec<Var<B>> = inputs
        .iter()
        .map(|t| tape.leaf(backend.upload(t).unwrap()))
        .collect();
    let out = f(&tape, &leaves);
    let loss = backend
        .download(&out.value())
        .unwrap()
        .to_f64_vec()
        .unwrap()[0];
    let mut grads = tape.backward(&out).unwrap();
    let dense = leaves
        .iter()
        .map(|l| match grads.dense(l).unwrap() {
            Some(g) => backend.download(&g).unwrap().to_f64_vec().unwrap(),
            None => vec![0.0; l.numel()],
        })
        .collect();
    (loss, dense)
}

/// Checks the gradients of `f` against central differences, in `f64`.
fn check(name: &str, shapes: &[&[usize]], f: &Build<'_, Reference>) {
    let inputs: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| tensor(100 + i as u64, s))
        .collect();
    let (_, grads) = gradients(Reference, &inputs, f);
    let h = 1e-6;
    for (i, input) in inputs.iter().enumerate() {
        let base = input.to_f64_vec().unwrap();
        for j in 0..base.len() {
            let at = |delta: f64| {
                let mut moved = base.clone();
                moved[j] += delta;
                let mut xs = inputs.clone();
                xs[i] = Tensor::from_slice(&moved, input.shape()).unwrap();
                gradients(Reference, &xs, f).0
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            let analytic = grads[i][j];
            assert!(
                (numeric - analytic).abs() <= 1e-6 * (1.0 + numeric.abs()),
                "{name}: input {i}, element {j}: {analytic} by the tape, {numeric} by differences"
            );
        }
    }
}

#[test]
fn elementwise_and_broadcast() {
    check("add, sub, mul, div", &[&[2, 3], &[3], &[2, 1]], &|_, x| {
        let a = x[0].add(&x[1]).unwrap();
        let b = a.mul(&x[2]).unwrap().sub(&x[1]).unwrap();
        let positive = x[2]
            .unary(UnaryKind::Square)
            .unwrap()
            .affine(1.0, 0.5)
            .unwrap();
        weighted(&b.div(&positive).unwrap(), 1)
    });
    check("unary and affine", &[&[4, 3]], &|_, x| {
        let y = x[0].unary(UnaryKind::Tanh).unwrap().scale(3.0).unwrap();
        let z = x[0].unary(UnaryKind::GeluTanh).unwrap();
        weighted(&y.add(&z).unwrap().unary(UnaryKind::Silu).unwrap(), 2)
    });
    check("gated unit", &[&[3, 8]], &|_, x| {
        weighted(&x[0].glu(UnaryKind::Silu).unwrap(), 3)
    });
}

#[test]
fn views_and_slices() {
    check("narrow pieces joined", &[&[4, 6]], &|_, x| {
        let a = x[0].narrow(1, 0, 2).unwrap();
        let b = x[0].narrow(1, 4, 2).unwrap();
        weighted(&a.mul(&b).unwrap(), 4)
    });
    check("overlapping slices", &[&[4, 6]], &|_, x| {
        let a = x[0].narrow(1, 0, 4).unwrap();
        let b = x[0].narrow(1, 2, 4).unwrap();
        weighted(&a.mul(&b).unwrap(), 5)
    });
    check("transpose, permute, reshape", &[&[2, 3, 4]], &|_, x| {
        let y = x[0].transpose(0, 2).unwrap().permute(&[1, 0, 2]).unwrap();
        let y = y.reshape(&[3, 8]).unwrap().contiguous().unwrap();
        weighted(&y, 6)
    });
    check("concat and broadcast", &[&[2, 3], &[1, 3]], &|tape, x| {
        let b = x[1].broadcast_to(&[2, 3]).unwrap();
        weighted(&tape.concat(&[&x[0], &b, &x[0]], 0).unwrap(), 7)
    });
}

#[test]
fn reductions_and_normalizations() {
    check("sum and mean", &[&[2, 3, 4]], &|_, x| {
        let s = x[0].sum(&[1], false).unwrap();
        let m = x[0].mean(&[0, 2], true).unwrap();
        let y = s.mul(&s).unwrap().sum_all().unwrap();
        weighted(&m, 8).add(&y).unwrap()
    });
    check("softmax and log-softmax", &[&[3, 5]], &|_, x| {
        let a = x[0].softmax(1).unwrap();
        let b = x[0].log_softmax(0).unwrap();
        weighted(&a.add(&b).unwrap(), 9)
    });
    check("layer norm", &[&[3, 6], &[6], &[6]], &|_, x| {
        weighted(
            &x[0].layer_norm(Some(&x[1]), Some(&x[2]), 1e-5).unwrap(),
            10,
        )
    });
    check("rms norm", &[&[3, 6], &[6]], &|_, x| {
        weighted(&x[0].rms_norm(Some(&x[1]), 1e-6, 1.0).unwrap(), 11)
    });
    check("max", &[&[3, 5]], &|_, x| {
        weighted(&x[0].max(1, false).unwrap(), 12)
    });
}

#[test]
fn products_attention_and_rotations() {
    check(
        "matmul and linear",
        &[&[2, 3, 4], &[5, 4], &[5]],
        &|_, x| {
            let y = x[0].linear(&x[1], Some(&x[2])).unwrap();
            weighted(&y, 13)
        },
    );
    check(
        "attention",
        &[&[1, 4, 5, 3], &[1, 2, 5, 3], &[1, 2, 5, 2]],
        &|_, x| {
            let y = x[0]
                .attention(&x[1], &x[2], None, Attention::causal())
                .unwrap();
            weighted(&y, 14)
        },
    );
    check("rope", &[&[5, 2, 4]], &|tape, x| {
        let positions = tape
            .backend()
            .upload(&Tensor::from_slice(&[0u32, 3, 1, 7, 2], &[5, 1]).unwrap())
            .unwrap();
        let rope = Rope {
            theta: 10_000.0,
            layout: RopeLayout::Half,
            rotary_dim: None,
            scaling: RopeScaling::None,
        };
        weighted(&x[0].rope(&positions, rope).unwrap(), 15)
    });
    check("rows gathered and scattered", &[&[5, 3]], &|_, x| {
        let idx = Tensor::from_slice(&[4u32, 1, 4, 0], &[2, 2]).unwrap();
        let rows = x[0].gather_rows(&idx).unwrap();
        let back = rows
            .unary(UnaryKind::Tanh)
            .unwrap()
            .scatter_add_rows(&idx, 6)
            .unwrap();
        weighted(&back, 16)
    });
}

#[test]
fn losses() {
    check("cross-entropy", &[&[4, 7]], &|_, x| {
        cross_entropy(&x[0], &[3, IGNORE, 0, 6]).unwrap()
    });
    check("fused cross-entropy", &[&[5, 4], &[7, 4]], &|_, x| {
        linear_cross_entropy(&x[0], &x[1], &[3, IGNORE, 0, 6, 6], 2).unwrap()
    });
}

#[test]
fn the_fused_loss_is_the_loss_of_the_logits() {
    let inputs = [tensor(1, &[9, 6]), tensor(2, &[11, 6])];
    let targets = [3, IGNORE, 0, 10, 10, 2, IGNORE, 5, 1];
    let plain: &Build<'_, Reference> =
        &|_, x| cross_entropy(&x[0].linear(&x[1], None).unwrap(), &targets).unwrap();
    let fused: &Build<'_, Reference> =
        &|_, x| linear_cross_entropy(&x[0], &x[1], &targets, 4).unwrap();
    let (a, ga) = gradients(Reference, &inputs, plain);
    let (b, gb) = gradients(Reference, &inputs, fused);
    assert!((a - b).abs() < 1e-12, "{a} and {b}");
    for (x, y) in ga.iter().flatten().zip(gb.iter().flatten()) {
        assert!((x - y).abs() < 1e-12, "{x} and {y}");
    }
}

#[test]
fn checkpointed_regions_give_the_same_gradients() {
    let inputs = [tensor(3, &[4, 6]), tensor(4, &[6, 6]), tensor(5, &[6])];
    let region = |_: &Tape<Reference>, x: &[Var<Reference>]| {
        let h = x[0].layer_norm(Some(&x[2]), None, 1e-5)?;
        let h = h.matmul(&x[1])?.unary(UnaryKind::GeluTanh)?;
        x[0].add(&h)
    };
    let plain: &Build<'_, Reference> = &move |tape, x| {
        let y = region(tape, x).unwrap();
        weighted(&region(tape, &[y, x[1].clone(), x[2].clone()]).unwrap(), 17)
    };
    let saved: &Build<'_, Reference> = &move |tape, x| {
        let y = tape.checkpoint(&[&x[0], &x[1], &x[2]], region).unwrap();
        let y = tape.checkpoint(&[&y, &x[1], &x[2]], region).unwrap();
        weighted(&y, 17)
    };
    let (a, ga) = gradients(Reference, &inputs, plain);
    let (b, gb) = gradients(Reference, &inputs, saved);
    assert_eq!(a, b);
    assert_eq!(ga, gb);
}

#[test]
fn a_sparse_leaf_receives_rows() {
    let tape = Tape::new(Reference);
    let table = tape.sparse_leaf(tensor(6, &[10, 3]));
    let dense = tape.leaf(tensor(6, &[10, 3]));
    let idx = Tensor::from_slice(&[7u32, 2, 7], &[3]).unwrap();
    let y = table
        .gather_rows(&idx)
        .unwrap()
        .add(&dense.gather_rows(&idx).unwrap())
        .unwrap();
    let mut grads = tape.backward(&weighted(&y, 18)).unwrap();
    let Some(Grad::Rows { indices, values }) = grads.take(&table) else {
        panic!("the sparse table's gradient is not rows");
    };
    assert_eq!(indices.shape(), [3]);
    assert_eq!(values.shape(), [3, 3]);
    let rows = Grad::Rows { indices, values }
        .dense(&Reference, &[10, 3])
        .unwrap();
    let Some(Grad::Dense(full)) = grads.take(&dense) else {
        panic!("the dense table's gradient is not dense");
    };
    assert_eq!(rows.to_f64_vec().unwrap(), full.to_f64_vec().unwrap());
}

#[test]
fn values_are_let_go_when_no_gradient_needs_them() {
    let tape = Tape::new(Reference);
    let x = tape.leaf(tensor(7, &[3, 4]));
    let y = x.affine(2.0, 1.0).unwrap();
    let z = y.unary(UnaryKind::Exp).unwrap();
    let w = z.sum_all().unwrap();
    assert_eq!(tape.held(), 4);
    // `y` is saved by the exponential's derivative, not by the tape: its
    // node lets it go with its handle. A sum saves nothing of `z`.
    drop((y, z));
    assert_eq!(tape.held(), 2);
    assert!(tape.backward(&w).unwrap().get(&x).is_some());
}

#[test]
fn constants_receive_no_gradient_and_record_nothing() {
    let tape = Tape::new(Reference);
    let c = tape.constant(tensor(8, &[2, 2]));
    let y = c.matmul(&c).unwrap();
    assert!(!y.requires_grad());
    let x = tape.leaf(tensor(9, &[2, 2]));
    let loss = weighted(&x.mul(&y).unwrap(), 19);
    let mut grads = tape.backward(&loss).unwrap();
    assert!(grads.take(&c).is_none());
    assert!(grads.take(&x).is_some());
}

/// The processor's gradients, in `f32`, against the reference's.
#[test]
fn the_processor_agrees_with_the_reference() {
    let cpu = kapel_cpu::Cpu::new();
    let shapes: [&[usize]; 4] = [&[2, 5, 8], &[12, 8], &[8], &[16, 4]];
    let f64s: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| tensor(20 + i as u64, s))
        .collect();
    let f32s: Vec<Tensor> = f64s.iter().map(|t| cast(t, DType::F32)).collect();
    fn net<B: Compute>(_: &Tape<B>, x: &[Var<B>]) -> Var<B> {
        let h = x[0].rms_norm(Some(&x[2]), 1e-5, 0.0).unwrap();
        let qkv = h.linear(&x[1], None).unwrap();
        let heads = |i: usize| {
            qkv.narrow(2, i * 4, 4)
                .unwrap()
                .reshape(&[2, 5, 1, 4])
                .unwrap()
                .transpose(1, 2)
                .unwrap()
        };
        let y = heads(0)
            .attention(&heads(1), &heads(2), None, Attention::causal())
            .unwrap();
        let y = y
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape(&[10, 4])
            .unwrap();
        let targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, IGNORE];
        linear_cross_entropy(&y, &x[3], &targets, 3).unwrap()
    }
    let (a, ga) = gradients(Reference, &f64s, &net::<Reference>);
    let (b, gb) = gradients(cpu, &f32s, &net::<kapel_cpu::Cpu>);
    assert!((a - b).abs() < 1e-5, "{a} and {b}");
    for (i, (x, y)) in ga.iter().zip(&gb).enumerate() {
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (p, q) in x.iter().zip(y) {
            assert!(
                (p - q).abs() <= 1e-4 * scale.max(1e-3),
                "input {i}: {p} and {q}"
            );
        }
    }
}
