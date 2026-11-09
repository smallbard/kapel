//! The conformity suite: every operation of the catalogue, on fixed random
//! cases, compared to the reference.
//!
//! A new backend checks itself in one line, in its tests:
//!
//! ```
//! # use kapel_core::DType;
//! # let backend = kapel_ref::Reference;
//! // The reference returns f64: it conforms to itself in f64.
//! kapel_ref::conformity::assert_conforms(&backend, DType::F64);
//! ```
//!
//! For each case, the inputs are drawn in `f64`, rounded to the data type under
//! test, and given to both the reference and the backend: the comparison
//! measures the backend's own computation, not the rounding of its inputs. The
//! forward outputs are compared, then the gradients of the backward pass for a
//! random output gradient. A value passes if it is within
//! [`tolerance`]`(dtype) × max(1, max |reference|)` of the reference (of the
//! output's type, when it is narrower than `dtype`). The
//! forward output must also have the type of the operation's signature, and a
//! backward call with an output gradient of the wrong shape must return an
//! error.
//!
//! The cases include empty tensors (a dimension of size 0), strided inputs
//! (transposed, narrowed, broadcast with zero strides), and attention with
//! padding masks that hide every key of a query, with `−∞` or with `f32::MIN`.
//!
//! [`gradient_failures`] checks the reference's own gradients against finite
//! differences.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{
    Affine, ArgMax, Attention, Backend, Binary, BinaryKind, Cast, Concat, Contiguous, CrossEntropy,
    GatherRows, Glu, LayerNorm, LogSoftmax, Matmul, Max, Mean, Operation, PagedAttention,
    QuantizedMatmul, RmsNorm, Rope, RopeLayout, RopeScaling, Run, ScatterAddRows, Softmax, Sum,
    Transfer, Unary, UnaryKind, WRITE_ROWS, Write,
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
pub fn check<B: Backend + Transfer + Write>(backend: &B, dtype: DType) -> Vec<Failure> {
    let mut visitor = Conformity {
        backend,
        dtype,
        failures: Vec::new(),
    };
    cases(&mut visitor, dtype);
    let mut failures = visitor.failures;
    failures.extend(write_failures(backend, dtype));
    failures.extend(refusal_failures(backend, dtype));
    failures.extend(update_failures(backend, dtype));
    failures.extend(recurrent_failures(backend, dtype));
    failures
}

/// A table of `shape` and `dtype` on `backend`, the only handle on it, with
/// the values of `values`: a state to write in place.
fn table<B: Write + Transfer>(backend: &B, values: &Tensor, dtype: DType) -> Result<Tensor> {
    let shape = values.shape();
    let mut t = backend.zeros(shape, dtype)?;
    let rows = shape[0];
    let index = Tensor::from_slice(&(0..rows as u32).collect::<Vec<_>>(), &[rows])?;
    backend.write_rows(&mut t, &backend.upload(&index)?, &backend.upload(values)?)?;
    Ok(t)
}

/// The causal convolution and the delta rule, in place, compared with the
/// reference: their outputs and the states they leave, rows of several
/// sequences in one call, and the calls that must be refused.
fn recurrent_failures<B: Backend + Transfer>(backend: &B, dtype: DType) -> Vec<Failure> {
    use kapel_ops::{CAUSAL_CONV, DELTA_RULE, Recurrent};
    let mut failures = Vec::new();
    let mut rng = Rng::new(19);
    let tol = tolerance(dtype);
    // Rows of three sequences, interleaved: each reads what its previous row
    // left.
    let slots = Tensor::from_slice(&[2u32, 2, 0, 2, 1, 0], &[6]).expect("sizes agree");
    let (c, taps) = (37, 4);
    let x = random(&mut rng, &[6, c + 5], dtype, -1.0, 1.0);
    let weight = random(&mut rng, &[c, taps], dtype, -1.0, 1.0);
    let conv0 = random(&mut rng, &[3, taps - 1, c], DType::F32, -1.0, 1.0);
    for state_dtype in [DType::F32, dtype] {
        let case = format!("6 rows of 3 sequences, {c} channels, a state in {state_dtype}");
        let outcome = (|| -> Result<Vec<String>> {
            let x = x.narrow(1, 3, c)?;
            let mut rs = table(&Reference, &conv0, state_dtype)?;
            let want = Reference.causal_conv(&mut rs, &x, &weight, &slots)?;
            let mut bs = table(backend, &conv0, state_dtype)?;
            let got = backend.causal_conv(
                &mut bs,
                &backend.upload(&x)?,
                &backend.upload(&weight)?,
                &backend.upload(&slots)?,
            )?;
            let mut problems = Vec::new();
            let got = backend.download(&got)?;
            if got.dtype() != dtype {
                problems.push(format!("type {}, expected {dtype}", got.dtype()));
            }
            compare("output", &want, &got, tol, &mut problems)?;
            compare("state", &rs, &backend.download(&bs)?, tol, &mut problems)?;
            Ok(problems)
        })();
        let problems = outcome.unwrap_or_else(|e| vec![e.to_string()]);
        failures.extend(problems.into_iter().map(|message| Failure {
            op: CAUSAL_CONV,
            case: case.clone(),
            message,
        }));
    }
    // Small heads, and heads wide enough for a card to share a column
    // between several warps and the columns between several blocks.
    for (dk, dv) in [(8, 6), (64, 40)] {
        let (h, hk) = (4, 2);
        let q = random(&mut rng, &[6, hk, dk], dtype, -0.5, 0.5);
        let k = random(&mut rng, &[6, hk, dk], dtype, -0.5, 0.5);
        let v = random(&mut rng, &[6, h, dv], dtype, -1.0, 1.0);
        let g = random(&mut rng, &[6, h], dtype, -1.0, 0.0);
        let beta = random(&mut rng, &[6, h], dtype, 0.0, 1.0);
        let delta0 = random(&mut rng, &[3, h, dk, dv], DType::F32, -0.5, 0.5);
        for layout in [kapel_ops::HeadLayout::Grouped, kapel_ops::HeadLayout::Tiled] {
            let case =
                format!("6 rows of 3 sequences, {h} heads reading {hk} {layout:?}, {dk} × {dv}");
            let outcome = (|| -> Result<Vec<String>> {
                let mut rs = table(&Reference, &delta0, DType::F32)?;
                let want = Reference.delta_rule(&mut rs, &q, &k, &v, &g, &beta, &slots, layout)?;
                let mut bs = table(backend, &delta0, DType::F32)?;
                let up = |t: &Tensor| backend.upload(t);
                let got = backend.delta_rule(
                    &mut bs,
                    &up(&q)?,
                    &up(&k)?,
                    &up(&v)?,
                    &up(&g)?,
                    &up(&beta)?,
                    &up(&slots)?,
                    layout,
                )?;
                let mut problems = Vec::new();
                let got = backend.download(&got)?;
                if got.dtype() != dtype {
                    problems.push(format!("type {}, expected {dtype}", got.dtype()));
                }
                compare("output", &want, &got, tol, &mut problems)?;
                compare("state", &rs, &backend.download(&bs)?, tol, &mut problems)?;
                Ok(problems)
            })();
            let problems = outcome.unwrap_or_else(|e| vec![e.to_string()]);
            failures.extend(problems.into_iter().map(|message| Failure {
                op: DELTA_RULE,
                case: case.clone(),
                message,
            }));
        }
    }
    let (h, hk, dk, dv) = (4, 2, 8, 6);
    let q = random(&mut rng, &[6, hk, dk], dtype, -0.5, 0.5);
    let k = random(&mut rng, &[6, hk, dk], dtype, -0.5, 0.5);
    let v = random(&mut rng, &[6, h, dv], dtype, -1.0, 1.0);
    let g = random(&mut rng, &[6, h], dtype, -1.0, 0.0);
    let beta = random(&mut rng, &[6, h], dtype, 0.0, 1.0);
    let delta0 = random(&mut rng, &[3, h, dk, dv], DType::F32, -0.5, 0.5);
    // Calls to refuse.
    let beyond = Tensor::from_slice(&[2u32, 3, 0, 2, 1, 0], &[6]).expect("sizes agree");
    let refused: [(&'static str, &str, Result<Tensor>); 3] = [
        (
            CAUSAL_CONV,
            "a slot beyond the table",
            (|| {
                let mut bs = table(backend, &conv0, DType::F32)?;
                let x = backend.upload(&x.narrow(1, 0, c)?)?;
                backend.causal_conv(
                    &mut bs,
                    &x,
                    &backend.upload(&weight)?,
                    &backend.upload(&beyond)?,
                )
            })(),
        ),
        (
            DELTA_RULE,
            "a slot beyond the table",
            (|| {
                let mut bs = table(backend, &delta0, DType::F32)?;
                let up = |t: &Tensor| backend.upload(t);
                backend.delta_rule(
                    &mut bs,
                    &up(&q)?,
                    &up(&k)?,
                    &up(&v)?,
                    &up(&g)?,
                    &up(&beta)?,
                    &up(&beyond)?,
                    kapel_ops::HeadLayout::Grouped,
                )
            })(),
        ),
        (
            DELTA_RULE,
            "a state that is shared",
            (|| {
                let mut bs = table(backend, &delta0, DType::F32)?;
                let _other = bs.clone();
                let up = |t: &Tensor| backend.upload(t);
                backend.delta_rule(
                    &mut bs,
                    &up(&q)?,
                    &up(&k)?,
                    &up(&v)?,
                    &up(&g)?,
                    &up(&beta)?,
                    &up(&slots)?,
                    kapel_ops::HeadLayout::Grouped,
                )
            })(),
        ),
    ];
    for (op, case, outcome) in refused {
        if outcome.is_ok() {
            failures.push(Failure {
                op,
                case: case.into(),
                message: "accepted".into(),
            });
        }
    }
    failures
}

/// AdamW in place, compared with the reference, and the calls that must be
/// refused.
fn update_failures<B: Backend + Transfer>(backend: &B, dtype: DType) -> Vec<Failure> {
    use kapel_ops::{ADAMW, AdamWStep, Update};
    let mut failures = Vec::new();
    let mut fail = |case: &str, message: String| {
        failures.push(Failure {
            op: ADAMW,
            case: case.into(),
            message,
        });
    };
    let mut rng = Rng::new(17);
    let step = AdamWStep {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.1,
        correction1: 1.0 - 0.9f64.powi(3),
        correction2: 1.0 - 0.95f64.powi(3),
        grad_scale: 0.5,
    };
    let state = |rng: &mut Rng, lo: f64| random(rng, &[5, 7], DType::F32, lo, 1.0);
    let (p0, m0, v0) = (
        state(&mut rng, -1.0),
        state(&mut rng, -1.0),
        state(&mut rng, 0.0),
    );
    let grad = random(&mut rng, &[7, 5], dtype, -1.0, 1.0);
    let case = format!("[5, 7], a gradient in {dtype} seen transposed");
    // Copies of their own: an upload to a host backend keeps the tensor.
    let copy = |t: &Tensor| -> Result<Tensor> {
        let bytes: Vec<u8> = t
            .to_f64_vec()?
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect();
        Tensor::from_bytes(t.dtype(), t.shape(), &bytes)
    };
    let fresh = |t: &Tensor| -> Result<Tensor> { backend.upload(&copy(t)?) };
    let outcome = (|| -> Result<Option<String>> {
        let (mut rp, mut rm, mut rv) = (copy(&p0)?, copy(&m0)?, copy(&v0)?);
        let g = grad.transpose(0, 1)?;
        Reference.adamw(&mut rp, &mut rm, &mut rv, &g, &step)?;
        let (mut bp, mut bm, mut bv) = (fresh(&p0)?, fresh(&m0)?, fresh(&v0)?);
        let bg = backend.upload(&grad)?.transpose(0, 1)?;
        backend.adamw(&mut bp, &mut bm, &mut bv, &bg, &step)?;
        for (name, expected, got) in [("p", &rp, &bp), ("m", &rm, &bm), ("v", &rv, &bv)] {
            let got = backend.download(got)?.to_f64_vec()?;
            let expected = expected.to_f64_vec()?;
            for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                if (a - b).abs() > 1e-6 * (1.0 + b.abs()) {
                    return Ok(Some(format!("{name}[{i}] = {a}, expected {b}")));
                }
            }
        }
        Ok(None)
    })();
    match outcome {
        Ok(None) => {}
        Ok(Some(message)) => fail(&case, message),
        Err(error) => fail(&case, error.to_string()),
    }
    let refused = (|| -> Result<()> {
        let (mut p, mut m, mut v) = (fresh(&p0)?, fresh(&m0)?, fresh(&v0)?);
        let _shared = p.clone();
        backend.adamw(
            &mut p,
            &mut m,
            &mut v,
            &backend.upload(&grad)?.transpose(0, 1)?,
            &step,
        )
    })();
    if refused.is_ok() {
        fail("a shared parameter", "accepted".into());
    }
    let refused = (|| -> Result<()> {
        let (mut p, mut m, mut v) = (fresh(&p0)?, fresh(&m0)?, fresh(&v0)?);
        backend.adamw(&mut p, &mut m, &mut v, &backend.upload(&grad)?, &step)
    })();
    if refused.is_ok() {
        fail("a gradient of another shape", "accepted".into());
    }
    failures.extend(update8_failures(backend, dtype, &step));
    failures
}

/// AdamW with moments in 8 bits, compared with the reference: three full
/// blocks and a partial one, from codes and scales drawn at random. The
/// parameters and the scales agree within `f32`'s rounding; a code may be
/// its neighbour, a moment that rounding put on the other side of a
/// midpoint.
fn update8_failures<B: Backend + Transfer>(
    backend: &B,
    dtype: DType,
    step: &kapel_ops::AdamWStep,
) -> Vec<Failure> {
    use kapel_ops::{ADAMW8, Moments8, Update, dynamic_map};
    let mut failures = Vec::new();
    let mut rng = Rng::new(19);
    let (rows, cols) = (3usize, 300usize);
    let n = rows * cols;
    let blocks = n.div_ceil(kapel_ops::BLOCK8);
    let p0 = random(&mut rng, &[rows, cols], DType::F32, -1.0, 1.0);
    let codes = |rng: &mut Rng| -> Tensor {
        let c: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
        Tensor::from_bytes(DType::U8, &[rows, cols], &c).expect("sizes agree")
    };
    let (m0, v0) = (codes(&mut rng), codes(&mut rng));
    let ms0 = random(&mut rng, &[blocks], DType::F32, 0.01, 0.1);
    let vs0 = random(&mut rng, &[blocks], DType::F32, 0.0001, 0.01);
    let grad = random(&mut rng, &[cols, rows], dtype, -1.0, 1.0);
    let mut table = dynamic_map(true).to_vec();
    table.extend(dynamic_map(false));
    let maps = Tensor::from_slice(&table, &[512]).expect("sizes agree");
    let case = format!("[3, 300], a gradient in {dtype} seen transposed");
    let copy = |t: &Tensor| -> Result<Tensor> {
        let bytes = t.storage().host_bytes().expect("on the host").to_vec();
        Tensor::from_bytes(t.dtype(), t.shape(), &bytes)
    };
    let state = |up: &dyn Fn(&Tensor) -> Result<Tensor>| -> Result<(Tensor, Moments8)> {
        Ok((
            up(&p0)?,
            Moments8 {
                m: up(&m0)?,
                v: up(&v0)?,
                m_scale: up(&ms0)?,
                v_scale: up(&vs0)?,
            },
        ))
    };
    let outcome = (|| -> Result<Option<String>> {
        let (mut rp, mut rm) = state(&copy)?;
        Reference.adamw8bit(&mut rp, &mut rm, &grad.transpose(0, 1)?, &maps, step)?;
        let (mut bp, mut bm) = state(&|t| backend.upload(&copy(t)?))?;
        let bg = backend.upload(&grad)?.transpose(0, 1)?;
        backend.adamw8bit(&mut bp, &mut bm, &bg, &backend.upload(&maps)?, step)?;
        for (name, expected, got) in [
            ("p", &rp, &bp),
            ("m_scale", &rm.m_scale, &bm.m_scale),
            ("v_scale", &rm.v_scale, &bm.v_scale),
        ] {
            let got = backend.download(got)?.to_f64_vec()?;
            for (i, (a, b)) in got.iter().zip(&expected.to_f64_vec()?).enumerate() {
                if (a - b).abs() > 1e-6 * (1.0 + b.abs()) {
                    return Ok(Some(format!("{name}[{i}] = {a}, expected {b}")));
                }
            }
        }
        for (name, expected, got) in [("m", &rm.m, &bm.m), ("v", &rm.v, &bm.v)] {
            let got = backend.download(got)?.to_f64_vec()?;
            for (i, (a, b)) in got.iter().zip(&expected.to_f64_vec()?).enumerate() {
                if (a - b).abs() > 1.0 {
                    return Ok(Some(format!("code {name}[{i}] = {a}, expected {b}")));
                }
            }
        }
        Ok(None)
    })();
    match outcome {
        Ok(None) => {}
        Ok(Some(message)) => failures.push(Failure {
            op: ADAMW8,
            case,
            message,
        }),
        Err(error) => failures.push(Failure {
            op: ADAMW8,
            case,
            message: error.to_string(),
        }),
    }
    let refused = (|| -> Result<()> {
        let (mut p, mut m) = state(&|t| backend.upload(&copy(t)?))?;
        m.m_scale = backend.upload(&random(
            &mut Rng::new(1),
            &[blocks + 1],
            DType::F32,
            0.0,
            1.0,
        ))?;
        backend.adamw8bit(
            &mut p,
            &mut m,
            &backend.upload(&grad)?.transpose(0, 1)?,
            &backend.upload(&maps)?,
            step,
        )
    })();
    if refused.is_ok() {
        failures.push(Failure {
            op: ADAMW8,
            case: "scales of another number of blocks".into(),
            message: "accepted".into(),
        });
    }
    failures
}

/// The calls of operations that must be refused for the values of their
/// inputs, which `infer` does not read.
fn refusal_failures<B: Backend + Transfer>(backend: &B, dtype: DType) -> Vec<Failure> {
    let mut failures = Vec::new();
    let mut rng = Rng::new(13);
    let logits = random(&mut rng, &[3, 5], dtype, -1.0, 1.0);
    for (case, targets) in [
        ("a target beyond the vocabulary", [1u32, 5, 0]),
        ("a target below −1", [1u32, (-2i32) as u32, 0]),
    ] {
        let refused = (|| -> Result<Tensor> {
            let t = if case.contains("−1") {
                let signed: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
                Tensor::from_slice(&signed, &[3])?
            } else {
                Tensor::from_slice(&targets, &[3])?
            };
            let y = backend.forward(
                &CrossEntropy,
                &[&backend.upload(&logits)?, &backend.upload(&t)?],
            )?;
            backend.download(&y)
        })();
        if refused.is_ok() {
            failures.push(Failure {
                op: CrossEntropy::NAME,
                case: case.into(),
                message: "accepted".into(),
            });
        }
    }
    failures
}

/// Runs every case on `backend`, computing in `dtype`, and panics with the list
/// of failures if there is any.
///
/// # Panics
///
/// If a case fails.
pub fn assert_conforms<B: Backend + Transfer + Write>(backend: &B, dtype: DType) {
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

/// How an input is seen: the view is applied after upload, so that backends
/// receive a strided tensor.
#[derive(Clone, Debug)]
enum View {
    /// The tensor itself, contiguous.
    Whole,
    /// Two dimensions swapped.
    Transpose(usize, usize),
    /// `len` elements of dimension `dim` from `start`.
    Narrow {
        dim: usize,
        start: usize,
        len: usize,
    },
    /// Repeated to a larger shape, with zero strides.
    Broadcast(Vec<usize>),
}

/// An input of a case: a host tensor and the view backends see of it.
struct Arg {
    base: Tensor,
    view: View,
}

impl Arg {
    fn view(&self, tensor: &Tensor) -> Result<Tensor> {
        match &self.view {
            View::Whole => Ok(tensor.clone()),
            &View::Transpose(a, b) => tensor.transpose(a, b),
            &View::Narrow { dim, start, len } => tensor.narrow(dim, start, len),
            View::Broadcast(shape) => tensor.broadcast_to(shape),
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

    /// A case whose output rounds away the effect of some inputs: its gradient
    /// is the exact derivative, which finite differences cannot see. It is
    /// compared to the reference, but not checked by finite differences.
    fn visit_rounded<O: Operation>(&mut self, op: &O, case: String, args: Vec<Arg>)
    where
        Self::B: Run<O>,
        Reference: Run<O>,
    {
        self.visit(op, case, args);
    }
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
    use kapel_core::QuantFormat as Q;
    // Where each format keeps its `f16` scales and minimums; Kapel's formats
    // start with an `f32` scale.
    let halves: &[usize] = match dtype {
        DType::Quantized(Q::Q8_0 | Q::Q4_0 | Q::Q5_0 | Q::Iq4Nl | Q::Iq4Xs | Q::Iq3S) => &[0],
        DType::Quantized(Q::Q4_1 | Q::Q5_1 | Q::Q4K | Q::Q5K) => &[0, 2],
        DType::Quantized(Q::Q2K) => &[80, 82],
        DType::Quantized(Q::Q3K) => &[108],
        DType::Quantized(Q::Q6K) => &[208],
        _ => &[],
    };
    for _ in 0..count / block {
        let start = bytes.len();
        while bytes.len() < start + block_bytes {
            bytes.push(rng.next_u64() as u8);
        }
        let block = &mut bytes[start..];
        if halves.is_empty() {
            block[..4].copy_from_slice(&(rng.uniform(0.005, 0.02) as f32).to_le_bytes());
        }
        for &at in halves {
            let scale = kapel_core::F16::from_f64(rng.uniform(0.005, 0.02));
            block[at..at + 2].copy_from_slice(&scale.0.to_le_bytes());
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
        view: View::Whole,
    }
}

/// `tensor` seen with its last two dimensions swapped.
fn transposed(tensor: Tensor) -> Arg {
    let rank = tensor.rank();
    Arg {
        base: tensor,
        view: View::Transpose(rank - 2, rank - 1),
    }
}

fn viewed(tensor: Tensor, view: View) -> Arg {
    Arg { base: tensor, view }
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
        U::Softplus,
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
    // Whole pieces of eight values, which a backend may compute eight at a
    // time: a bias added to rows, a product of the same shape, a function.
    for kind in [K::Add, K::Mul] {
        let args = vec![x(rng, &[4, 16]), x(rng, &[16])];
        v.visit(
            &Binary { kind },
            format!("{kind:?} [4, 16] · [16], a row"),
            args,
        );
        let args = vec![x(rng, &[2, 24]), x(rng, &[2, 24])];
        v.visit(
            &Binary { kind },
            format!("{kind:?} [2, 24] · [2, 24]"),
            args,
        );
    }
    v.visit(
        &Unary { kind: U::GeluTanh },
        "GeluTanh [4, 16]".into(),
        vec![x(rng, &[4, 16])],
    );
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
        v.visit(
            &Cast { dtype: target },
            format!("[4, 16] to {target}"),
            vec![x(rng, &[4, 16])],
        );
        // Quantized weights decoded, as for merging an adapter into them.
        use kapel_core::QuantFormat as Q;
        for format in [
            DType::int8(64).expect("valid"),
            DType::int4(32).expect("valid"),
            DType::Quantized(Q::Q8_0),
            DType::Quantized(Q::Q4_0),
            DType::Quantized(Q::Q4K),
        ] {
            v.visit(
                &Cast { dtype: target },
                format!("{format} [3, 256] to {target}"),
                vec![plain(random_quantized(rng, &[3, 256], format))],
            );
        }
    }
    let mut r = |shape: &[usize]| random(rng, shape, dtype, -1.0, 1.0);
    for (case, arg) in [
        ("[3, 4]", plain(r(&[3, 4]))),
        ("[2, 3, 4], last two swapped", transposed(r(&[2, 3, 4]))),
        (
            "[5, 4], rows 1 to 3",
            viewed(
                r(&[5, 4]),
                View::Narrow {
                    dim: 0,
                    start: 1,
                    len: 3,
                },
            ),
        ),
        (
            "[3] broadcast to [2, 3]",
            viewed(r(&[3]), View::Broadcast(vec![2, 3])),
        ),
        ("[0, 4]", plain(r(&[0, 4]))),
    ] {
        v.visit(&Contiguous, case.into(), vec![arg]);
    }
    let mut r = |shape: &[usize]| random(rng, shape, dtype, -1.0, 1.0);
    v.visit(
        &Concat { dim: 0 },
        "[2, 3] and [1, 3] along 0".into(),
        vec![plain(r(&[2, 3])), plain(r(&[1, 3]))],
    );
    v.visit(
        &Concat { dim: 1 },
        "[2, 1, 4], [3, 2, 4]ᵀ, [2, 0, 4] along 1".into(),
        vec![
            plain(r(&[2, 1, 4])),
            viewed(r(&[3, 2, 4]), View::Transpose(0, 1)),
            plain(r(&[2, 0, 4])),
        ],
    );
    // A cache of keys `[1, H, T, D]` growing by one position.
    v.visit(
        &Concat { dim: 2 },
        "[1, 2, 5, 4] and [1, 2, 1, 4] along 2".into(),
        vec![plain(r(&[1, 2, 5, 4])), plain(r(&[1, 2, 1, 4]))],
    );
    v.visit(
        &Concat { dim: 1 },
        "[3] broadcast to [2, 3], and [2, 2] along 1".into(),
        vec![
            viewed(r(&[3]), View::Broadcast(vec![2, 3])),
            plain(r(&[2, 2])),
        ],
    );
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
    let strided = transposed(random(rng, &[5, 4], dtype, -1.0, 1.0));
    v.visit(
        &Matmul,
        "[2, 3, 4] · [5, 4]ᵀ, strided".into(),
        vec![x(rng, &[2, 3, 4]), strided],
    );
    // With a bias: a linear layer, batched, its weight seen transposed, of
    // sizes that no tile divides.
    v.visit(
        &Matmul,
        "[2, 3, 4] · [4, 5] + [5]".into(),
        vec![x(rng, &[2, 3, 4]), x(rng, &[4, 5]), x(rng, &[5])],
    );
    v.visit(
        &Matmul,
        "[70, 37] · [45, 37]ᵀ + [45]".into(),
        vec![
            x(rng, &[70, 37]),
            transposed(random(rng, &[45, 37], dtype, -1.0, 1.0)),
            x(rng, &[45]),
        ],
    );
    // The gradient of a weight, `xᵀ · g`, and both factors transposed, with
    // sizes that no tile divides.
    v.visit(
        &Matmul,
        "[70, 37]ᵀ · [70, 45], the first transposed".into(),
        vec![
            transposed(random(rng, &[70, 37], dtype, -1.0, 1.0)),
            x(rng, &[70, 45]),
        ],
    );
    v.visit(
        &Matmul,
        "[2, 9, 3]ᵀ · [2, 5, 9]ᵀ, both transposed".into(),
        vec![
            transposed(random(rng, &[2, 9, 3], dtype, -1.0, 1.0)),
            transposed(random(rng, &[2, 5, 9], dtype, -1.0, 1.0)),
        ],
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
    // Every format, on one row (generation), a few, and more than a backend
    // may take in place.
    {
        use kapel_core::QuantFormat as Q;
        let formats = [
            DType::int8(64).expect("valid"),
            DType::int8(40).expect("valid"),
            DType::int4(128).expect("valid"),
            DType::int4(40).expect("valid"),
        ]
        .into_iter()
        .chain(
            [
                Q::Q8_0,
                Q::Q4_0,
                Q::Q4_1,
                Q::Q5_0,
                Q::Q5_1,
                Q::Q2K,
                Q::Q3K,
                Q::Q4K,
                Q::Q5K,
                Q::Q6K,
                Q::Iq4Nl,
                Q::Iq4Xs,
                Q::Iq3S,
            ]
            .map(DType::Quantized),
        );
        for format in formats {
            let w = random_quantized(rng, &[7, 1280], format);
            for rows in [1, 3, 17] {
                v.visit(
                    &QuantizedMatmul,
                    format!("[{rows}, 1280] · {format} [7, 1280]ᵀ"),
                    vec![x(rng, &[rows, 1280]), plain(w.clone())],
                );
            }
            v.visit(
                &GatherRows,
                format!("{format} [7, 1280] by [2, 2]"),
                vec![plain(w), plain(indices(rng, &[2, 2], 7))],
            );
        }
        // Weights of several formats, their rows stacked.
        let stack = [(Q::Q4K, 5), (Q::Q6K, 3), (Q::Q8_0, 2)]
            .map(|(f, n)| random_quantized(rng, &[n, 512], DType::Quantized(f)));
        for rows in [1, 3, 17] {
            let mut args = vec![x(rng, &[rows, 512])];
            args.extend(stack.iter().cloned().map(plain));
            v.visit(
                &QuantizedMatmul,
                format!("[{rows}, 512] · [q4_k [5, 512]; q6_k [3, 512]; q8_0 [2, 512]]ᵀ"),
                args,
            );
        }
    }

    // Mixtures of experts: the choice of experts (with ties, taken by the
    // lower position), their probabilities, and the products by them.
    {
        use kapel_core::QuantFormat as Q;
        use kapel_ops::{ExpertMatmul, GatherColumns, TopK};
        let mut probs = random(rng, &[5, 16], dtype, 0.0, 1.0)
            .to_f64_vec()
            .expect("host values");
        probs[3 * 16 + 2] = 0.5;
        probs[3 * 16 + 9] = 0.5;
        probs[3 * 16 + 11] = 0.5;
        let probs = Tensor::from_slice(&probs, &[5, 16])
            .and_then(|t| Reference.forward(&Cast { dtype }, &[&t]))
            .expect("values");
        for k in [1, 3, 16] {
            v.visit(
                &TopK { k },
                format!("{k} of [5, 16]"),
                vec![plain(probs.clone())],
            );
        }
        let chosen = indices(rng, &[5, 3], 16);
        v.visit(
            &GatherColumns,
            "[5, 16] at [5, 3]".into(),
            vec![plain(probs.clone()), plain(chosen)],
        );
        let experts = indices(rng, &[5, 3], 4);
        let op = ExpertMatmul { experts: 4 };
        for w in [
            random(rng, &[4 * 7, 64], dtype, -1.0, 1.0),
            random_quantized(rng, &[4 * 7, 256], DType::Quantized(Q::Q8_0)),
            random_quantized(rng, &[4 * 7, 256], DType::Quantized(Q::Iq4Xs)),
            random_quantized(rng, &[4 * 7, 256], DType::Quantized(Q::Iq3S)),
        ] {
            let width = w.shape()[1];
            let format = w.dtype();
            v.visit(
                &op,
                format!("[5, {width}] by 3 of 4 experts of {format} [7, {width}]"),
                vec![
                    x(rng, &[5, width]),
                    plain(w.clone()),
                    plain(experts.clone()),
                ],
            );
            v.visit(
                &op,
                format!("[5, 3, {width}] by 3 of 4 experts of {format} [7, {width}]"),
                vec![x(rng, &[5, 3, width]), plain(w), plain(experts.clone())],
            );
        }
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
    // Cross-entropy: targets of both integer types, rows that count in no
    // loss, rows wider than a warp and not a multiple of it, strided
    // logits, large logits.
    let ignore = CrossEntropy::IGNORE;
    v.visit(
        &CrossEntropy,
        "[4, 7], u32 targets, one ignored".into(),
        vec![
            x(rng, &[4, 7]),
            plain(Tensor::from_slice(&[3u32, ignore, 0, 6], &[4]).expect("4 targets")),
        ],
    );
    v.visit(
        &CrossEntropy,
        "[3, 100], i32 targets, one ignored".into(),
        vec![
            plain(random(rng, &[3, 100], dtype, -4.0, 4.0)),
            plain(Tensor::from_slice(&[99i32, -1, 31], &[3]).expect("3 targets")),
        ],
    );
    v.visit(
        &CrossEntropy,
        "[7, 5]ᵀ, strided".into(),
        vec![
            transposed(random(rng, &[7, 5], dtype, -1.0, 1.0)),
            plain(Tensor::from_slice(&[0u32, 6, 2, 2, 5], &[5]).expect("5 targets")),
        ],
    );
    v.visit(
        &CrossEntropy,
        "[2, 9], large values".into(),
        vec![
            plain(random(rng, &[2, 9], dtype, -30.0, 30.0)),
            plain(Tensor::from_slice(&[8u32, 1], &[2]).expect("2 targets")),
        ],
    );
    v.visit(
        &CrossEntropy,
        "[0, 5], empty".into(),
        vec![
            x(rng, &[0, 5]),
            plain(Tensor::from_slice::<u32>(&[], &[0]).expect("none")),
        ],
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

    // Rows wider than a warp of 32, and not a multiple of it.
    v.visit(
        &LayerNorm { eps: 1e-5 },
        "[3, 100], weight and bias".into(),
        vec![x(rng, &[3, 100]), x(rng, &[100]), x(rng, &[100])],
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 0.0,
        },
        "[3, 100], weight".into(),
        vec![x(rng, &[3, 100]), x(rng, &[100])],
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
    // Signed positions, negative and far: their phase needs `f64` angles.
    let far = Tensor::from_slice(&[-3i32, 0, 70_001, 300_007], &[4]).expect("sizes agree");
    v.visit(
        &rope(RopeLayout::Half, None, RopeScaling::None),
        "i32 positions, negative and far".into(),
        vec![x(rng, &[1, 2, 4, 16]), plain(far)],
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

    // Drawn after the cases above, so that adding them did not change those.
    empty_cases(v, rng, dtype);
    strided_cases(v, rng, dtype);
    masked_attention_cases(v, rng, dtype);
    paged_cases(v, rng, dtype);
    glu_cases(v, rng, dtype);
    argmax_cases(v, rng, dtype);
}

/// Positions of maxima: along the last dimension and another, with ties
/// (values rounded to the type), and lines with NaN.
fn argmax_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let last = ArgMax {
        dim: 1,
        keepdim: false,
    };
    let rounded = |rng: &mut Rng, shape: &[usize]| {
        let x = random(rng, shape, DType::F64, -4.0, 4.0);
        let values: Vec<f64> = x
            .to_f64_vec()
            .expect("host")
            .iter()
            .map(|v| v.round())
            .collect();
        let x = Tensor::from_slice(&values, shape).expect("sizes agree");
        Run::<Cast>::forward(&Reference, &Cast { dtype }, &[&x]).expect("a float type")
    };
    v.visit(
        &last,
        "rows of 300, ties".into(),
        vec![plain(rounded(rng, &[5, 300]))],
    );
    v.visit(
        &ArgMax {
            dim: 0,
            keepdim: true,
        },
        "columns, kept".into(),
        vec![plain(rounded(rng, &[7, 4]))],
    );
    let nan = f32::NAN;
    let x = Tensor::from_slice(
        &[
            nan,
            1.0,
            3.0,
            3.0,
            nan,
            nan,
            nan,
            nan,
            f32::NEG_INFINITY,
            nan,
            f32::NEG_INFINITY,
            -1.0,
        ],
        &[3, 4],
    )
    .expect("sizes agree");
    v.visit(&last, "NaN, −∞ and ties".into(), vec![plain(x)]);
}

/// Gated units: SiLU and GELU gates, rows of several widths, a strided input.
fn glu_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let x = |rng: &mut Rng, shape: &[usize]| random(rng, shape, dtype, -2.0, 2.0);
    let silu = Glu {
        activation: UnaryKind::Silu,
    };
    v.visit(&silu, "SiLU, [3, 10]".into(), vec![plain(x(rng, &[3, 10]))]);
    v.visit(
        &Glu {
            activation: UnaryKind::GeluTanh,
        },
        "GELU, [2, 3, 8]".into(),
        vec![plain(x(rng, &[2, 3, 8]))],
    );
    v.visit(
        &silu,
        "SiLU, transposed input".into(),
        vec![transposed(x(rng, &[6, 4]))],
    );
}

/// Cases with a dimension of size 0: the results are empty, or zeros where
/// they sum over nothing (a product with `K = 0`, the gradient of a weight
/// applied to no row).
fn empty_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let x = |rng: &mut Rng, shape: &[usize]| plain(random(rng, shape, dtype, -1.0, 1.0));
    v.visit(
        &Unary {
            kind: UnaryKind::Exp,
        },
        "Exp [0, 3], empty".into(),
        vec![x(rng, &[0, 3])],
    );
    v.visit(
        &Binary {
            kind: BinaryKind::Mul,
        },
        "Mul [2, 0, 3] · [3], empty".into(),
        vec![x(rng, &[2, 0, 3]), x(rng, &[3])],
    );
    v.visit(
        &Binary {
            kind: BinaryKind::Add,
        },
        "Add [0, 3] · [1, 3], empty".into(),
        vec![x(rng, &[0, 3]), x(rng, &[1, 3])],
    );
    v.visit(
        &Cast { dtype: DType::F32 },
        "to f32, [0, 3], empty".into(),
        vec![x(rng, &[0, 3])],
    );
    v.visit(
        &Matmul,
        "[2, 0, 4] · [2, 4, 5], no rows".into(),
        vec![x(rng, &[2, 0, 4]), x(rng, &[2, 4, 5])],
    );
    v.visit(
        &Matmul,
        "[3, 0] · [0, 5], K = 0: zeros".into(),
        vec![x(rng, &[3, 0]), x(rng, &[0, 5])],
    );
    v.visit(
        &Matmul,
        "[0, 3, 4] · [4, 2], empty batch".into(),
        vec![x(rng, &[0, 3, 4]), x(rng, &[4, 2])],
    );
    v.visit(
        &Matmul,
        "[3, 4] · [4, 0], N = 0".into(),
        vec![x(rng, &[3, 4]), x(rng, &[4, 0])],
    );
    for format in [
        DType::int8(32).expect("valid"),
        DType::Quantized(kapel_core::QuantFormat::Q8_0),
    ] {
        let w = plain(random_quantized(rng, &[5, 64], format));
        v.visit(
            &QuantizedMatmul,
            format!("[0, 64] · {format} [5, 64]ᵀ, no rows"),
            vec![x(rng, &[0, 64]), w],
        );
        let w = plain(random_quantized(rng, &[0, 64], format));
        v.visit(
            &QuantizedMatmul,
            format!("[3, 64] · {format} [0, 64]ᵀ, N = 0"),
            vec![x(rng, &[3, 64]), w],
        );
        let w = plain(random_quantized(rng, &[5, 0], format));
        v.visit(
            &QuantizedMatmul,
            format!("[3, 0] · {format} [5, 0]ᵀ, K = 0: zeros"),
            vec![x(rng, &[3, 0]), w],
        );
    }
    v.visit(
        &Sum {
            dims: vec![1],
            keepdim: false,
        },
        "dim 1 of [2, 0, 3]: zeros".into(),
        vec![x(rng, &[2, 0, 3])],
    );
    v.visit(
        &Sum {
            dims: vec![1],
            keepdim: true,
        },
        "dim 1 of [0, 4], kept, empty".into(),
        vec![x(rng, &[0, 4])],
    );
    v.visit(
        &Mean {
            dims: vec![1],
            keepdim: false,
        },
        "dim 1 of [0, 4], empty".into(),
        vec![x(rng, &[0, 4])],
    );
    v.visit(
        &Max {
            dim: 1,
            keepdim: false,
        },
        "dim 1 of [0, 4], empty".into(),
        vec![x(rng, &[0, 4])],
    );
    v.visit(
        &Softmax { dim: 1 },
        "dim 1 of [0, 5], empty".into(),
        vec![x(rng, &[0, 5])],
    );
    v.visit(
        &LogSoftmax { dim: 0 },
        "dim 0 of [0, 5], empty".into(),
        vec![x(rng, &[0, 5])],
    );
    v.visit(
        &LayerNorm { eps: 1e-5 },
        "[0, 6], weight and bias: zero gradients".into(),
        vec![x(rng, &[0, 6]), x(rng, &[6]), x(rng, &[6])],
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 0.0,
        },
        "[2, 0, 6], weight: zero gradient".into(),
        vec![x(rng, &[2, 0, 6]), x(rng, &[6])],
    );
    v.visit(
        &Attention::causal(),
        "causal, no queries: zero key and value gradients".into(),
        vec![
            x(rng, &[2, 2, 0, 8]),
            x(rng, &[2, 1, 3, 8]),
            x(rng, &[2, 1, 3, 6]),
        ],
    );
    v.visit(
        &Attention::causal(),
        "causal, empty batch".into(),
        vec![
            x(rng, &[0, 2, 3, 8]),
            x(rng, &[0, 2, 3, 8]),
            x(rng, &[0, 2, 3, 6]),
        ],
    );
    v.visit(
        &Rope {
            theta: 10_000.0,
            layout: RopeLayout::Half,
            rotary_dim: None,
            scaling: RopeScaling::None,
        },
        "[0, 4, 16], empty".into(),
        vec![x(rng, &[0, 4, 16]), plain(indices(rng, &[4], 100))],
    );
    v.visit(
        &GatherRows,
        "[6, 4] by [0]: zero gradient".into(),
        vec![x(rng, &[6, 4]), plain(indices(rng, &[0], 6))],
    );
}

/// Cases whose inputs are strided views: transposed, narrowed, or broadcast
/// with zero strides.
fn strided_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let r = |rng: &mut Rng, shape: &[usize]| random(rng, shape, dtype, -1.0, 1.0);
    v.visit(
        &Unary {
            kind: UnaryKind::Silu,
        },
        "Silu [4, 3]ᵀ, strided".into(),
        vec![transposed(r(rng, &[4, 3]))],
    );
    let narrowed = viewed(
        r(rng, &[3, 7]),
        View::Narrow {
            dim: 1,
            start: 2,
            len: 4,
        },
    );
    v.visit(
        &Binary {
            kind: BinaryKind::Mul,
        },
        "Mul [4, 3]ᵀ · [3, 7][:, 2..6], strided".into(),
        vec![transposed(r(rng, &[4, 3])), narrowed],
    );
    let repeated = viewed(r(rng, &[3, 1]), View::Broadcast(vec![2, 3, 4]));
    v.visit(
        &Binary {
            kind: BinaryKind::Sub,
        },
        "Sub [2, 3, 4] · [3, 1] broadcast to [2, 3, 4], zero strides".into(),
        vec![plain(r(rng, &[2, 3, 4])), repeated],
    );
    v.visit(
        &Sum {
            dims: vec![1],
            keepdim: false,
        },
        "dim 1 of [4, 3]ᵀ, strided".into(),
        vec![transposed(r(rng, &[4, 3]))],
    );
    v.visit(
        &Softmax { dim: 1 },
        "dim 1 of [5, 3]ᵀ, strided".into(),
        vec![transposed(r(rng, &[5, 3]))],
    );
    let narrowed = viewed(
        r(rng, &[4, 9]),
        View::Narrow {
            dim: 1,
            start: 3,
            len: 6,
        },
    );
    v.visit(
        &LogSoftmax { dim: 1 },
        "dim 1 of [4, 9][:, 3..9], strided".into(),
        vec![narrowed],
    );
    v.visit(
        &LayerNorm { eps: 1e-5 },
        "[6, 4]ᵀ, weight and bias, strided".into(),
        vec![
            transposed(r(rng, &[6, 4])),
            plain(r(rng, &[6])),
            plain(r(rng, &[6])),
        ],
    );
    let narrowed = viewed(
        r(rng, &[4, 10]),
        View::Narrow {
            dim: 1,
            start: 2,
            len: 6,
        },
    );
    v.visit(
        &RmsNorm {
            eps: 1e-6,
            weight_offset: 0.0,
        },
        "[4, 10][:, 2..8], weight, strided".into(),
        vec![narrowed, plain(r(rng, &[6]))],
    );
    // The layout of a model's projections: [B, T, H, D] seen as [B, H, T, D].
    let heads = |rng: &mut Rng, h: usize, t: usize, d: usize| {
        viewed(r(rng, &[2, t, h, d]), View::Transpose(1, 2))
    };
    v.visit(
        &Attention::causal(),
        "causal, [B, T, H, D] inputs seen as [B, H, T, D], strided".into(),
        vec![
            heads(rng, 4, 5, 8),
            heads(rng, 2, 5, 8),
            heads(rng, 2, 5, 6),
        ],
    );
}

/// Attention with padding masks. The masks are `f32` whatever the type under
/// test (a mask may be in any float type): `f32::MIN` rounds to `−∞` in `bf16`
/// and `f16`.
fn masked_attention_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let r = |rng: &mut Rng, shape: &[usize]| random(rng, shape, dtype, -1.0, 1.0);
    let free = Attention {
        causal: false,
        ..Attention::causal()
    };
    // Mask [2, 1, 3, 4]: in batch 0, query 1 has every key masked; in batch 1,
    // query 0 has its keys 1 and 3 masked.
    let mask = |rng: &mut Rng, value: f32| {
        let mut values: Vec<f32> = (0..24).map(|_| rng.uniform(-2.0, 0.0) as f32).collect();
        values[4..8].fill(value);
        values[12 + 1] = value;
        values[12 + 3] = value;
        plain(Tensor::from_slice(&values, &[2, 1, 3, 4]).expect("sizes agree"))
    };
    let mut args = vec![
        plain(r(rng, &[2, 2, 3, 8])),
        plain(r(rng, &[2, 2, 4, 8])),
        plain(r(rng, &[2, 2, 4, 6])),
    ];
    args.push(mask(rng, f32::NEG_INFINITY));
    v.visit(
        &free,
        "not causal, f32 padding mask of −∞, one query fully masked: zeros".into(),
        args,
    );
    let mut args = vec![
        plain(r(rng, &[2, 2, 3, 8])),
        plain(r(rng, &[2, 2, 4, 8])),
        plain(r(rng, &[2, 2, 4, 6])),
    ];
    args.push(mask(rng, f32::MIN));
    // A query whose keys are all at f32::MIN: the scores vanish next to the
    // mask, its weights are uniform, and finite differences see no change, but
    // the gradient is the derivative of the softmax, as for any finite mask.
    v.visit_rounded(
        &free,
        "not causal, f32 padding mask of f32::MIN, one query fully masked: uniform".into(),
        args,
    );
    // Documents: for each key, where its document starts.
    let documents = |starts: &[i32], b: usize, dtype: DType| {
        let shape = [b, starts.len() / b];
        plain(if dtype == DType::U32 {
            let unsigned: Vec<u32> = starts.iter().map(|&s| s as u32).collect();
            Tensor::from_slice(&unsigned, &shape).expect("sizes agree")
        } else {
            Tensor::from_slice(starts, &shape).expect("sizes agree")
        })
    };
    v.visit(
        &Attention::causal(),
        "causal, documents [2, 6] in i32, grouped keys".into(),
        vec![
            plain(r(rng, &[2, 2, 6, 8])),
            plain(r(rng, &[2, 1, 6, 8])),
            plain(r(rng, &[2, 1, 6, 6])),
            documents(&[0, 0, 0, 3, 3, 5, 0, 1, 1, 1, 1, 1], 2, DType::I32),
        ],
    );
    // Longer than the tiles of the card: documents that start inside a tile,
    // on a tile's edge, of one token, and one over a whole row. Too large for
    // finite differences.
    let t = 200;
    let mut starts = Vec::with_capacity(2 * t);
    for i in 0..t {
        let s = [0, 64, 70, 130, 131, 199]
            .iter()
            .rev()
            .find(|&&s| s <= i)
            .copied()
            .unwrap_or(0);
        starts.push(s as i32);
    }
    starts.extend(std::iter::repeat_n(0, t));
    v.visit_rounded(
        &Attention::causal(),
        "causal, documents [2, 200] in u32, starts in and on the edges of tiles".into(),
        vec![
            plain(r(rng, &[2, 4, t, 16])),
            plain(r(rng, &[2, 2, t, 16])),
            plain(r(rng, &[2, 2, t, 16])),
            documents(&starts, 2, DType::U32),
        ],
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
        // An output narrower than the type under test (the `f32` rows of a
        // quantized table) is compared at its own tolerance.
        let tol = if signature.dtype.is_float() {
            tolerance(self.dtype).max(tolerance(signature.dtype))
        } else {
            0.0
        };
        if got.dtype() != signature.dtype {
            problems.push(format!(
                "output: type {}, expected {}",
                got.dtype(),
                signature.dtype
            ));
        }
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

        // A gradient of the wrong shape is user input: it must be refused.
        let mut wrong = signature.shape.clone();
        match wrong.last_mut() {
            Some(last) => *last += 1,
            None => wrong.push(2),
        }
        let bad = self
            .backend
            .upload(&random(&mut rng, &wrong, self.dtype, -1.0, 1.0))?;
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            Run::<O>::backward(self.backend, op, &device_refs, &bad).is_ok()
        }));
        match outcome {
            Ok(false) => {}
            Ok(true) => problems.push(format!(
                "backward accepted a gradient of shape {wrong:?} for an output of shape {:?}",
                signature.shape
            )),
            Err(_) => problems.push(format!(
                "backward panicked on a gradient of shape {wrong:?} for an output of shape {:?}",
                signature.shape
            )),
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

    fn visit_rounded<O: Operation>(&mut self, _: &O, _: String, _: Vec<Arg>)
    where
        Reference: Run<O>,
    {
    }

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

/// Attention over pages: one sequence decoding, several sequences with
/// several queries each (a prompt after a cache), a window, a softcap, a
/// query that sees nothing, strided queries, and a cache of another type
/// than the queries.
fn paged_cases<V: Visitor>(v: &mut V, rng: &mut Rng, dtype: DType) {
    let x = |rng: &mut Rng, shape: &[usize], dtype: DType| random(rng, shape, dtype, -1.0, 1.0);
    let table = |values: &[u32], shape: &[usize]| {
        plain(Tensor::from_slice(values, shape).expect("sizes agree"))
    };
    // Pages of 4 lines, 6 pages: 24 lines, 2 key heads of 8.
    let (lines, hkv, d) = (24, 2, 8);
    let cache = |rng: &mut Rng, dtype: DType| {
        vec![
            plain(x(rng, &[lines, hkv, d], dtype)),
            plain(x(rng, &[lines, hkv, d], dtype)),
        ]
    };
    let paged = PagedAttention::new(4);
    let mut args = vec![plain(x(rng, &[1, 4, d], dtype))];
    args.extend(cache(rng, dtype));
    args.push(table(&[5, 0, 3], &[1, 3]));
    args.push(table(&[0, 10], &[1, 2]));
    v.visit(&paged, "one sequence decoding, 10 positions".into(), args);

    // Sequence 0 on pages 1, 4; sequence 1 on pages 2, 0, 5. Three queries of
    // sequence 1 (a prompt after 4 cached positions), two of sequence 0.
    let pages = [1, 4, 0, 2, 0, 5];
    let rows = [1, 5, 0, 7, 1, 6, 1, 7, 0, 8];
    let several = |rng: &mut Rng, q: Arg, dtype_cache: DType| {
        let mut args = vec![q];
        args.extend(cache(rng, dtype_cache));
        args.push(table(&pages, &[2, 3]));
        args.push(table(&rows, &[5, 2]));
        args
    };
    let q = plain(x(rng, &[5, 6, d], dtype));
    v.visit(
        &paged,
        "two sequences, five queries".into(),
        several(rng, q, dtype),
    );
    let windowed = PagedAttention {
        window: Some(3),
        softcap: Some(2.0),
        scale: Some(0.7),
        ..paged
    };
    let q = plain(x(rng, &[5, 6, d], dtype));
    v.visit(
        &windowed,
        "window of 3, scores capped at 2, scale 0.7".into(),
        several(rng, q, dtype),
    );
    // Queries narrowed out of a wider projection, as from a fused one.
    let q = viewed(
        x(rng, &[5, 10, d], dtype),
        View::Narrow {
            dim: 1,
            start: 2,
            len: 6,
        },
    );
    v.visit(&paged, "strided queries".into(), several(rng, q, dtype));
    let q = plain(x(rng, &[5, 6, d], dtype));
    let other = if dtype == DType::F16 {
        DType::BF16
    } else {
        DType::F16
    };
    v.visit(
        &paged,
        format!("a cache in {other}"),
        several(rng, q, other),
    );
    let mut args = vec![plain(x(rng, &[2, 2, d], dtype))];
    args.extend(cache(rng, dtype));
    args.push(table(&[3, 1], &[1, 2]));
    args.push(table(&[0, 0, 0, 3], &[2, 2]));
    v.visit(&paged, "a query that sees nothing".into(), args);

    // The widths of real heads, a long sequence (cut into several chunks by
    // a backend), and a short one: pages of 16, in a shuffled order.
    for (d, hq, window) in [(64, 6, None), (128, 8, Some(100)), (128, 16, None)] {
        let lines = 40 * 16;
        let pages: Vec<u32> = (0..40).map(|p| (p * 7) % 40).collect();
        let rows = [0, 300, 1, 5, 1, 6];
        let wide = PagedAttention {
            window,
            ..PagedAttention::new(16)
        };
        let mut args = vec![plain(x(rng, &[3, hq, d], dtype))];
        args.push(plain(x(rng, &[lines, 2, d], dtype)));
        args.push(plain(x(rng, &[lines, 2, d], dtype)));
        args.push(table(&pages, &[2, 20]));
        args.push(table(&rows, &[3, 2]));
        v.visit(
            &wide,
            format!("heads of {d}, {hq} queries for 2 keys, 300 positions, window {window:?}"),
            args,
        );
    }

    // Caches of 8-bit floats, with heads of 64.
    for cache_dtype in [DType::F8E4M3, DType::F8E5M2] {
        let lines = 10 * 16;
        let pages: Vec<u32> = (0..10).rev().collect();
        let rows = [0, 70, 1, 9, 0, 71];
        let mut args = vec![plain(x(rng, &[3, 6, 64], dtype))];
        args.push(plain(x(rng, &[lines, 3, 64], cache_dtype)));
        args.push(plain(x(rng, &[lines, 3, 64], cache_dtype)));
        args.push(table(&pages, &[2, 5]));
        args.push(table(&rows, &[3, 2]));
        v.visit(
            &PagedAttention::new(16),
            format!("a cache in {cache_dtype}, heads of 64"),
            args,
        );
    }

    // A prompt of 400 positions after 10 cached ones (a backend may compute
    // it apart, as a prefill), then a query of another sequence decoding.
    for window in [None, Some(170)] {
        let lines = 52 * 16;
        let pages: Vec<u32> = (0..52).map(|p| (p * 5) % 52).collect();
        let mut rows: Vec<u32> = (0..400).flat_map(|i| [0, 11 + i]).collect();
        rows.extend([1, 9]);
        let prompt = PagedAttention {
            window,
            ..PagedAttention::new(16)
        };
        let mut args = vec![plain(x(rng, &[401, 4, 64], dtype))];
        args.push(plain(x(rng, &[lines, 2, 64], dtype)));
        args.push(plain(x(rng, &[lines, 2, 64], dtype)));
        args.push(table(&pages, &[2, 26]));
        args.push(table(&rows, &[401, 2]));
        v.visit(
            &prompt,
            format!("a prompt of 400 after 10 positions, and a query decoding, window {window:?}"),
            args,
        );
    }
}

/// Rows written in place, compared to the reference, and the calls that must
/// be refused.
fn write_failures<B: Write + Transfer>(backend: &B, dtype: DType) -> Vec<Failure> {
    let mut failures = Vec::new();
    let mut fail = |case: &str, message: String| {
        failures.push(Failure {
            op: WRITE_ROWS,
            case: case.into(),
            message,
        });
    };
    let mut rng = Rng::new(11);
    let rows = random(&mut rng, &[3, 2, 4], dtype, -1.0, 1.0);
    let index = Tensor::from_slice(&[4u32, 0, 2], &[3]).expect("sizes agree");
    for table_dtype in [dtype, DType::F16, DType::F8E4M3] {
        let case = format!("3 rows of {dtype} into 5 rows of {table_dtype}");
        let outcome = (|| -> Result<Option<String>> {
            let mut expected = Write::zeros(&Reference, &[5, 2, 4], table_dtype)?;
            Reference.write_rows(&mut expected, &index, &rows)?;
            let mut table = backend.zeros(&[5, 2, 4], table_dtype)?;
            backend.write_rows(
                &mut table,
                &backend.upload(&index)?,
                &backend.upload(&rows)?,
            )?;
            let got = backend.download(&table)?;
            if got.dtype() != table_dtype {
                return Ok(Some(format!(
                    "type {}, expected {table_dtype}",
                    got.dtype()
                )));
            }
            // Conversions are exact: the same bits.
            Ok((got.to_f64_vec()? != expected.to_f64_vec()?)
                .then(|| "the table differs from the reference".to_string()))
        })();
        match outcome {
            Ok(None) => {}
            Ok(Some(message)) => fail(&case, message),
            Err(error) => fail(&case, error.to_string()),
        }
    }
    let refused = |what: &str, call: &dyn Fn() -> Result<()>| match call() {
        Ok(()) => Some(format!("{what} was accepted")),
        Err(_) => None,
    };
    let upload = |t: &Tensor| backend.upload(t).expect("an upload");
    type Check<'a> = (&'a str, Box<dyn Fn() -> Result<()> + 'a>);
    let checks: [Check<'_>; 4] = [
        (
            "a row out of range",
            Box::new(|| {
                let mut table = backend.zeros(&[5, 2, 4], dtype)?;
                let index = Tensor::from_slice(&[4u32, 5, 2], &[3])?;
                backend.write_rows(&mut table, &upload(&index), &upload(&rows))
            }),
        ),
        (
            "a row written twice",
            Box::new(|| {
                let mut table = backend.zeros(&[5, 2, 4], dtype)?;
                let index = Tensor::from_slice(&[1u32, 3, 1], &[3])?;
                backend.write_rows(&mut table, &upload(&index), &upload(&rows))
            }),
        ),
        (
            "a shared table",
            Box::new(|| {
                let mut table = backend.zeros(&[5, 2, 4], dtype)?;
                let _view = table.clone();
                backend.write_rows(&mut table, &upload(&index), &upload(&rows))
            }),
        ),
        (
            "rows of another width",
            Box::new(|| {
                let mut table = backend.zeros(&[5, 2, 3], dtype)?;
                backend.write_rows(&mut table, &upload(&index), &upload(&rows))
            }),
        ),
    ];
    for (what, call) in &checks {
        if let Some(message) = refused(what, call.as_ref()) {
            fail(what, message);
        }
    }
    failures
}
