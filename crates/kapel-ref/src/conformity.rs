//! The conformity suite: every operation of the catalogue, on fixed random
//! cases, compared to the reference.
//!
//! A new backend checks itself in one line, in its tests:
//!
//! ```
//! # use kapel_core::DType;
//! # let backend = kapel_ref::Reference;
//! kapel_ref::conformity::assert_conforms(&backend, DType::F32);
//! ```
//!
//! For each case, the inputs are drawn in `f64`, rounded to the data type under
//! test, and given to both the reference and the backend: the comparison
//! measures the backend's own computation, not the rounding of its inputs. The
//! forward outputs are compared, then the gradients of the backward pass for a
//! random output gradient. A value passes if it is within
//! [`tolerance`]`(dtype) × max(1, max |reference|)` of the reference.
//!
//! [`gradient_failures`] checks the reference's own gradients against finite
//! differences.

use std::fmt;

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{
    Affine, Attention, Backend, Binary, BinaryKind, Cast, GatherRows, LayerNorm, LogSoftmax,
    Matmul, Max, Mean, Operation, QuantizedMatmul, RmsNorm, Rope, RopeLayout, RopeScaling, Run,
    ScatterAddRows, Softmax, Sum, Transfer, Unary, UnaryKind,
};

use crate::Reference;
use crate::rng::Rng;

/// A case that did not pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The operation.
    pub op: &'static str,
    /// The case, with its parameters and shapes.
    pub case: String,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]: {}", self.op, self.case, self.message)
    }
}

/// The relative tolerance of a computation in `dtype`.
pub fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => 1e-10,
        DType::F32 => 1e-4,
        DType::F16 => 1e-2,
        DType::BF16 => 5e-2,
        _ => 0.25,
    }
}

/// Runs every case on `backend`, computing in `dtype`, and returns the failures.
pub fn check<B: Backend + Transfer>(backend: &B, dtype: DType) -> Vec<Failure> {
    let mut visitor = Conformity {
        backend,
        dtype,
        failures: Vec::new(),
    };
    cases(&mut visitor, dtype);
    visitor.failures
}

/// Runs every case on `backend`, computing in `dtype`, and panics with the list
/// of failures if there is any.
///
/// # Panics
///
/// If a case fails.
pub fn assert_conforms<B: Backend + Transfer>(backend: &B, dtype: DType) {
    let failures = check(backend, dtype);
    assert!(
        failures.is_empty(),
        "{} conformity failures in {dtype}:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|f| format!("  {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Checks the reference's gradients against central finite differences, in
/// `f64`, on every case, and returns the failures.
pub fn gradient_failures() -> Vec<Failure> {
    gradient_check().1
}

/// The number of input elements whose gradient was checked, and the failures.
pub(crate) fn gradient_check() -> (usize, Vec<Failure>) {
    let mut visitor = FiniteDifferences {
        failures: Vec::new(),
        rng: Rng::new(7),
        checked: 0,
    };
    cases(&mut visitor, DType::F64);
    (visitor.checked, visitor.failures)
}

/// An input of a case: a host tensor, possibly seen through a transposition of
/// its last two dimensions, applied after upload so that backends see a strided
/// view.
struct Arg {
    base: Tensor,
    transposed: bool,
}

impl Arg {
    fn view(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.transposed {
            tensor.transpose(tensor.rank() - 2, tensor.rank() - 1)
        } else {
            Ok(tensor.clone())
        }
    }
}

/// What to do with each case.
trait Visitor {
    type B: Backend;

    fn visit<O: Operation>(&mut self, op: &O, case: String, args: Vec<Arg>)
    where
        Self::B: Run<O>,
        Reference: Run<O>;
}

/// Values drawn in `[lo, hi)`, rounded to `dtype`.
fn random(rng: &mut Rng, shape: &[usize], dtype: DType, lo: f64, hi: f64) -> Tensor {
    let count: usize = shape.iter().product();
    let size = dtype.block_bytes();
    let mut bytes = vec![0; count * size];
    for out in bytes.chunks_exact_mut(size) {
        dtype
            .encode(rng.uniform(lo, hi), out)
            .expect("a scalar type");
    }
    Tensor::from_bytes(dtype, shape, &bytes).expect("sizes agree")
}

/// Random bytes in a quantized format, with valid scales.
fn random_quantized(rng: &mut Rng, shape: &[usize], dtype: DType) -> Tensor {
    let count: usize = shape.iter().product();
    let (block, block_bytes) = (dtype.block_len(), dtype.block_bytes());
    let mut bytes = Vec::with_capacity(count / block * block_bytes);
    for _ in 0..count / block {
        let scale = rng.uniform(0.005, 0.02);
        let start = bytes.len();
        match dtype {
            DType::Quantized(kapel_core::QuantFormat::Q8_0 | kapel_core::QuantFormat::Q4_0) => {
                bytes.extend(kapel_core::F16::from_f64(scale).0.to_le_bytes());
            }
            _ => bytes.extend((scale as f32).to_le_bytes()),
        }
        while bytes.len() < start + block_bytes {
            bytes.push(rng.next_u64() as u8);
        }
    }
    Tensor::from_bytes(dtype, shape, &bytes).expect("whole blocks")
}

fn indices(rng: &mut Rng, shape: &[usize], below: usize) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<u32> = (0..count).map(|_| rng.below(below) as u32).collect();
    Tensor::from_slice(&values, shape).expect("sizes agree")
}

fn plain(tensor: Tensor) -> Arg {
    Arg {
        base: tensor,
        transposed: false,
    }
}

/// Every case of the suite. The generator has a fixed seed: the cases are the
/// same on every run and every machine.
fn cases<V: Visitor>(v: &mut V, dtype: DType) {
    let mut rng = Rng::new(42);
    let rng = &mut rng;
    let x = |rng: &mut Rng, shape: &[usize]| plain(random(rng, shape, dtype, -1.0, 1.0));
    let positive = |rng: &mut Rng, shape: &[usize]| plain(random(rng, shape, dtype, 0.5, 2.0));

    use UnaryKind as U;
    for kind in [
        U::Neg,
        U::Exp,
        U::Square,
        U::Abs,
        U::Sigmoid,
        U::Tanh,
        U::Relu,
        U::Silu,
        U::GeluTanh,
        U::GeluErf,
    ] {
        v.visit(
            &Unary { kind },
            format!("{kind:?} [2, 3, 4]"),
            vec![x(rng, &[2, 3, 4])],
        );
    }
    for kind in [U::Log, U::Sqrt, U::Rsqrt] {
        v.visit(
            &Unary { kind },
            format!("{kind:?} [3, 5]"),
            vec![positive(rng, &[3, 5])],
        );
    }
    use BinaryKind as K;
    for kind in [K::Add, K::Sub, K::Mul, K::Max, K::Min] {
        let args = vec![x(rng, &[2, 3, 4]), x(rng, &[3, 1])];
        v.visit(
            &Binary { kind },
            format!("{kind:?} [2, 3, 4] · [3, 1]"),
            args,
        );
    }
    v.visit(
        &Binary { kind: K::Div },
        "Div [4] · [2, 4]".into(),
        vec![x(rng, &[4]), positive(rng, &[2, 4])],
    );
    v.visit(
        &Affine {
            scale: 1.5,
            bias: -0.25,
        },
        "[5, 3]".into(),
        vec![x(rng, &[5, 3])],
    );
    for target in [DType::F32, DType::BF16, DType::F16] {
        v.visit(
            &Cast { dtype: target },
            format!("to {target}"),
            vec![x(rng, &[3, 4])],
        );
    }
    let wide = plain(random(rng, &[3, 4], dtype, -150.0, 150.0));
    v.visit(
        &Cast { dtype: DType::I8 },
        "to i8, saturating".into(),
        vec![wide],
    );

    v.visit(
        &Matmul,
        "[2, 3, 4] · [2, 4, 5]".into(),
        vec![x(rng, &[2, 3, 4]), x(rng, &[2, 4, 5])],
    );
    v.visit(
        &Matmul,
        "[3, 4] · [2, 4, 5], broadcast".into(),
        vec![x(rng, &[3, 4]), x(rng, &[2, 4, 5])],
    );
    let transposed = Arg {
        base: random(rng, &[5, 4], dtype, -1.0, 1.0),
        transposed: true,
    };
    v.visit(
        &Matmul,
        "[2, 3, 4] · [5, 4]ᵀ, strided".into(),
        vec![x(rng, &[2, 3, 4]), transposed],
    );
    for format in [
        DType::int8(32).expect("valid"),
        DType::int4(32).expect("valid"),
    ]
    .into_iter()
    .chain([kapel_core::QuantFormat::Q8_0, kapel_core::QuantFormat::Q4_0].map(DType::Quantized))
    {
        let w = plain(random_quantized(rng, &[5, 64], format));
        v.visit(
            &QuantizedMatmul,
            format!("[2, 3, 64] · {format} [5, 64]ᵀ"),
            vec![x(rng, &[2, 3, 64]), w],
        );
    }

    v.visit(
        &Sum {
            dims: vec![1],
            keepdim: false,
        },
        "dim 1 of [2, 3, 4]".into(),
        vec![x(rng, &[2, 3, 4])],
    );
    v.visit(
        &Sum {
            dims: vec![0, 2],
            keepdim: true,
        },
        "dims 0, 2, kept".into(),
        vec![x(rng, &[2, 3, 4])],
    );
    v.visit(
        &Mean {
            dims: vec![2],
            keepdim: false,
        },
        "dim 2 of [2, 3, 4]".into(),
        vec![x(rng, &[2, 3, 4])],
    );
    v.visit(
        &Mean {
            dims: vec![0, 1],
            keepdim: true,
        },
        "dims 0, 1, kept".into(),
        vec![x(rng, &[2, 3, 4])],
    );
    v.visit(
        &Max {
            dim: 2,
            keepdim: false,
        },
        "dim 2 of [2, 3, 4]".into(),
        vec![x(rng, &[2, 3, 4])],
    );
    v.visit(
        &Max {
            dim: 0,
            keepdim: true,
        },
        "dim 0, kept".into(),
        vec![x(rng, &[3, 5])],
    );

    v.visit(
        &Softmax { dim: 2 },
        "dim 2 of [2, 3, 5]".into(),
        vec![x(rng, &[2, 3, 5])],
    );
    v.visit(
        &Softmax { dim: 0 },
        "dim 0 of [4, 3]".into(),
        vec![x(rng, &[4, 3])],
    );
    let large = plain(random(rng, &[3, 6], dtype, -10.0, 10.0));
    v.visit(
        &LogSoftmax { dim: 1 },
        "dim 1 of [3, 6], large values".into(),
        vec![large],
    );
    v.visit(
        &LayerNorm { eps: 1e-5 },
        "[4, 6], weight and bias".into(),
        vec![x(rng, &[4, 6]), x(rng, &[6]), x(rng, &[6])],
    );
    v.visit(
        &LayerNorm { eps: 1e-5 },
        "[2, 3, 6], no weight".into(),
        vec![x(rng, &[2, 3, 6])],
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 0.0,
        },
        "[4, 6], weight".into(),
        vec![x(rng, &[4, 6]), x(rng, &[6])],
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 1.0,
        },
        "[4, 6], weight + 1 (Gemma)".into(),
        vec![x(rng, &[4, 6]), x(rng, &[6])],
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 0.0,
        },
        "[3, 6], no weight".into(),
        vec![x(rng, &[3, 6])],
    );

    let attention = |rng: &mut Rng, hq: usize, hkv: usize, tq: usize, tk: usize| {
        vec![
            plain(random(rng, &[2, hq, tq, 8], dtype, -1.0, 1.0)),
            plain(random(rng, &[2, hkv, tk, 8], dtype, -1.0, 1.0)),
            plain(random(rng, &[2, hkv, tk, 6], dtype, -1.0, 1.0)),
        ]
    };
    v.visit(
        &Attention::causal(),
        "causal, 4 query heads, 2 key heads".into(),
        attention(rng, 4, 2, 5, 5),
    );
    v.visit(
        &Attention::causal(),
        "causal, 3 queries after 4 cached keys".into(),
        attention(rng, 2, 2, 3, 7),
    );
    let window = Attention {
        window: Some(2),
        ..Attention::causal()
    };
    v.visit(
        &window,
        "causal, window of 2, one key head".into(),
        attention(rng, 4, 1, 5, 5),
    );
    let mut masked = attention(rng, 2, 2, 3, 4);
    masked.push(plain(random(rng, &[1, 1, 3, 4], dtype, -2.0, 0.0)));
    let free = Attention {
        causal: false,
        window: None,
        scale: Some(0.5),
        softcap: None,
    };
    v.visit(&free, "not causal, additive mask, scale 0.5".into(), masked);
    let capped = Attention {
        softcap: Some(1.5),
        scale: Some(2.0),
        ..Attention::causal()
    };
    v.visit(
        &capped,
        "causal, scores capped at 1.5 (Gemma 2)".into(),
        attention(rng, 2, 1, 4, 4),
    );

    let rope = |layout, rotary_dim, scaling| Rope {
        theta: 10_000.0,
        layout,
        rotary_dim,
        scaling,
    };
    let scalings = [
        ("plain", RopeScaling::None),
        ("linear", RopeScaling::Linear { factor: 4.0 }),
        ("NTK", RopeScaling::Ntk { factor: 4.0 }),
        (
            "Llama 3",
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 64,
            },
        ),
        (
            "YaRN",
            RopeScaling::Yarn {
                factor: 4.0,
                original_max_position: 64,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
            },
        ),
    ];
    for (name, scaling) in scalings {
        let args = vec![x(rng, &[2, 3, 4, 16]), plain(indices(rng, &[4], 256))];
        v.visit(
            &rope(RopeLayout::Half, None, scaling),
            format!("{name}, half pairs"),
            args,
        );
    }
    let args = vec![x(rng, &[2, 5, 8]), plain(indices(rng, &[2, 5], 100))];
    v.visit(
        &rope(RopeLayout::Interleaved, None, RopeScaling::None),
        "interleaved, positions per token".into(),
        args,
    );
    let args = vec![x(rng, &[3, 2, 10]), plain(indices(rng, &[3, 1], 100))];
    v.visit(
        &rope(RopeLayout::Half, Some(4), RopeScaling::None),
        "4 of 10 components rotated".into(),
        args,
    );

    v.visit(
        &GatherRows,
        "[6, 4] by [2, 3]".into(),
        vec![x(rng, &[6, 4]), plain(indices(rng, &[2, 3], 6))],
    );
    let args = vec![x(rng, &[2, 3, 4]), plain(indices(rng, &[2, 3], 6))];
    v.visit(
        &ScatterAddRows { rows: 6 },
        "[2, 3, 4] into 6 rows".into(),
        args,
    );
}

/// Compares a backend to the reference.
struct Conformity<'a, B> {
    backend: &'a B,
    dtype: DType,
    failures: Vec<Failure>,
}

impl<B: Backend + Transfer> Conformity<'_, B> {
    fn run<O: Operation>(&self, op: &O, args: &[Arg]) -> Result<Vec<String>>
    where
        B: Run<O>,
        Reference: Run<O>,
    {
        let mut problems = Vec::new();
        let host: Vec<Tensor> = args
            .iter()
            .map(|a| a.view(&a.base))
            .collect::<Result<_>>()?;
        let host_refs: Vec<&Tensor> = host.iter().collect();
        let mut device = Vec::new();
        for arg in args {
            device.push(arg.view(&self.backend.upload(&arg.base)?)?);
        }
        let device_refs: Vec<&Tensor> = device.iter().collect();
        let signature = op.infer(&host_refs)?;

        let expected = Run::<O>::forward(&Reference, op, &host_refs)?;
        let got = self
            .backend
            .download(&Run::<O>::forward(self.backend, op, &device_refs)?)?;
        let tol = if signature.dtype.is_float() {
            tolerance(self.dtype)
        } else {
            0.0
        };
        compare("output", &expected, &got, tol, &mut problems)?;

        if !signature.dtype.is_float() {
            return Ok(problems);
        }
        let mut rng = Rng::new(signature.shape.iter().product::<usize>() as u64);
        let grad = random(&mut rng, &signature.shape, self.dtype, -1.0, 1.0);
        let expected = Run::<O>::backward(&Reference, op, &host_refs, &grad)?;
        let got = Run::<O>::backward(self.backend, op, &device_refs, &self.backend.upload(&grad)?)?;
        if expected.len() != got.len() {
            problems.push(format!(
                "{} gradients, expected {}",
                got.len(),
                expected.len()
            ));
        }
        for (i, (e, g)) in expected.iter().zip(&got).enumerate() {
            match (e, g) {
                (Some(e), Some(g)) => {
                    compare(
                        &format!("gradient {i}"),
                        e,
                        &self.backend.download(g)?,
                        tol,
                        &mut problems,
                    )?;
                }
                (None, None) => {}
                _ => problems.push(format!("gradient {i}: present for only one side")),
            }
        }
        Ok(problems)
    }
}

/// Appends a problem if `got` is not within `tol` of `expected`.
fn compare(
    what: &str,
    expected: &Tensor,
    got: &Tensor,
    tol: f64,
    problems: &mut Vec<String>,
) -> Result<()> {
    if expected.shape() != got.shape() {
        problems.push(format!(
            "{what}: shape {:?}, expected {:?}",
            got.shape(),
            expected.shape()
        ));
        return Ok(());
    }
    let (e, g) = (expected.to_f64_vec()?, got.to_f64_vec()?);
    let scale = e.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    if let Some((i, (a, b))) = e
        .iter()
        .zip(&g)
        .enumerate()
        .find(|(_, (a, b))| !((*a - *b).abs() <= tol * scale || (a.is_nan() && b.is_nan())))
    {
        problems.push(format!(
            "{what}: element {i} is {b}, expected {a} (tolerance {:.1e})",
            tol * scale
        ));
    }
    Ok(())
}

impl<B: Backend + Transfer> Visitor for Conformity<'_, B> {
    type B = B;

    fn visit<O: Operation>(&mut self, op: &O, case: String, args: Vec<Arg>)
    where
        B: Run<O>,
        Reference: Run<O>,
    {
        let problems = self
            .run(op, &args)
            .unwrap_or_else(|error| vec![error.to_string()]);
        self.failures
            .extend(problems.into_iter().map(|message| Failure {
                op: O::NAME,
                case: case.clone(),
                message,
            }));
    }
}

/// Checks the reference's gradients with central differences.
struct FiniteDifferences {
    failures: Vec<Failure>,
    rng: Rng,
    checked: usize,
}

impl FiniteDifferences {
    fn run<O: Operation>(&mut self, op: &O, args: &[Arg]) -> Result<Vec<String>>
    where
        Reference: Run<O>,
    {
        // A conversion rounds: its gradient is a convention (it passes through),
        // not a derivative that finite differences could check.
        if O::NAME == Cast::NAME {
            return Ok(Vec::new());
        }
        let inputs: Vec<Tensor> = args
            .iter()
            .map(|a| a.view(&a.base))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = inputs.iter().collect();
        let signature = op.infer(&refs)?;
        let count: usize = signature.shape.iter().product();
        let weights: Vec<f64> = (0..count).map(|_| self.rng.uniform(-1.0, 1.0)).collect();
        let grad = Tensor::from_slice(&weights, &signature.shape)?;
        let analytic = Run::<O>::backward(&Reference, op, &refs, &grad)?;
        // The loss is Σ weights · output; its derivative by one input element.
        let loss = |inputs: &[&Tensor]| -> Result<f64> {
            let y = Run::<O>::forward(&Reference, op, inputs)?.to_f64_vec()?;
            Ok(y.iter().zip(&weights).map(|(a, b)| a * b).sum())
        };
        let mut problems = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let Some(Some(analytic)) = analytic.get(index) else {
                continue;
            };
            if !op.differentiable(index) || !input.dtype().is_float() {
                continue;
            }
            let analytic = analytic.to_f64_vec()?;
            let values = input.to_f64_vec()?;
            for element in 0..values.len() {
                let h = 1e-6 * values[element].abs().max(1.0);
                let shifted = |delta: f64| -> Result<f64> {
                    let mut changed = values.clone();
                    changed[element] += delta;
                    let tensor = Tensor::from_slice(&changed, input.shape())?;
                    let mut all: Vec<&Tensor> = refs.clone();
                    all[index] = &tensor;
                    loss(&all)
                };
                let numeric = (shifted(h)? - shifted(-h)?) / (2.0 * h);
                self.checked += 1;
                let a = analytic[element];
                if (a - numeric).abs() > 1e-5 * (1.0 + numeric.abs()) {
                    problems.push(format!(
                        "input {index}, element {element}: gradient {a}, finite difference {numeric}"
                    ));
                    break;
                }
            }
        }
        Ok(problems)
    }
}

impl Visitor for FiniteDifferences {
    type B = Reference;

    fn visit<O: Operation>(&mut self, op: &O, case: String, args: Vec<Arg>)
    where
        Reference: Run<O>,
    {
        let problems = self
            .run(op, &args)
            .unwrap_or_else(|error| vec![error.to_string()]);
        self.failures
            .extend(problems.into_iter().map(|message| Failure {
                op: O::NAME,
                case: case.clone(),
                message,
            }));
    }
}
