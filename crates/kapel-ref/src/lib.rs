//! The reference backend: every operation of the catalogue in plain `f64`,
//! written to be read.
//!
//! [`Reference`] reads its inputs at their exact value, whatever their type,
//! computes in `f64`, and returns `f64` tensors, except the operations that
//! copy: [`kapel_ops::Cast`] returns its target type, [`kapel_ops::Contiguous`]
//! and [`kapel_ops::Concat`] their inputs'. It is slow on purpose: each
//! operation is the direct transcription of its definition in `kapel-ops`, so
//! that it can be checked by eye.
//!
//! Every other backend is checked against it with the [`conformity`] suite, and
//! its own gradients against finite differences.
//!
//! ```
//! use kapel_core::Tensor;
//! use kapel_ops::{Matmul, Run};
//! use kapel_ref::Reference;
//!
//! let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
//! let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;
//! let y = Reference.forward(&Matmul, &[&a, &b])?;
//! assert_eq!(y.to_vec::<f64>()?, vec![19.0, 22.0, 43.0, 50.0]);
//! # Ok::<(), kapel_core::Error>(())
//! ```

mod array;
mod attention;
pub mod conformity;
mod elementwise;
#[cfg(test)]
mod golden;
mod index;
mod matmul;
mod norm;
mod reduce;
mod rng;
mod rope;

use kapel_core::{Device, Error, Result, Tensor};
use kapel_ops::Transfer;

/// The reference backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reference;

impl Transfer for Reference {
    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor.device() {
            Device::Cpu => Ok(tensor.clone()),
            device => Err(Error::WrongDevice {
                op: "upload",
                expected: Device::Cpu,
                got: device,
            }),
        }
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        self.upload(tensor)
    }
}

#[cfg(test)]
mod tests {
    use kapel_core::DType;
    use kapel_ops::{Attention, Operation, Rope, RopeLayout, RopeScaling, Run, Softmax};

    use super::*;

    #[test]
    fn the_reference_conforms_to_itself() {
        let typed = Typed { exp_off: false };
        for dtype in [DType::F64, DType::F32, DType::BF16] {
            conformity::assert_conforms(&typed, dtype);
        }
    }

    #[test]
    fn the_suite_checks_output_types() {
        // The reference returns f64: in f32, every case but the casts to
        // another type is reported.
        let failures = conformity::check(&Reference, DType::F32);
        assert!(failures.iter().any(|f| f.message.contains("type f64")));
        assert!(
            failures
                .iter()
                .all(|f| f.message == "output: type f64, expected f32"),
            "{failures:?}"
        );
    }

    #[test]
    fn gradients_match_finite_differences() {
        let (checked, failures) = conformity::gradient_check();
        // Every differentiable input of every case is checked, element by element.
        assert!(checked > 1500, "only {checked} elements checked");
        assert!(
            failures.is_empty(),
            "{}",
            failures
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// The reference, with its outputs converted to the type of their
    /// signature, as a backend returns them; with `exp_off`, `exp` is a little
    /// off: the suite must see it.
    struct Typed {
        exp_off: bool,
    }

    /// `y` converted to the type of `op`'s output.
    fn typed<O: Operation>(op: &O, inputs: &[&Tensor], y: Tensor) -> Result<Tensor> {
        let dtype = op.infer(inputs)?.dtype;
        if y.dtype() == dtype {
            return Ok(y);
        }
        Reference.forward(&kapel_ops::Cast { dtype }, &[&y])
    }

    impl Transfer for Typed {
        fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
            Reference.upload(tensor)
        }

        fn download(&self, tensor: &Tensor) -> Result<Tensor> {
            Reference.download(tensor)
        }
    }

    macro_rules! delegate {
        ($($op:ty),*) => {$(
            impl Run<$op> for Typed {
                fn forward(&self, op: &$op, inputs: &[&Tensor]) -> Result<Tensor> {
                    typed(op, inputs, Reference.forward(op, inputs)?)
                }

                fn backward(
                    &self,
                    op: &$op,
                    inputs: &[&Tensor],
                    grad: &Tensor,
                ) -> Result<Vec<Option<Tensor>>> {
                    Reference.backward(op, inputs, grad)
                }
            }
        )*};
    }

    delegate!(
        kapel_ops::Binary,
        kapel_ops::Affine,
        kapel_ops::Cast,
        kapel_ops::Contiguous,
        kapel_ops::Concat,
        kapel_ops::Matmul,
        kapel_ops::QuantizedMatmul,
        kapel_ops::Sum,
        kapel_ops::Mean,
        kapel_ops::Max,
        Softmax,
        kapel_ops::LogSoftmax,
        kapel_ops::LayerNorm,
        kapel_ops::RmsNorm,
        Attention,
        Rope,
        kapel_ops::GatherRows,
        kapel_ops::ScatterAddRows
    );

    impl Run<kapel_ops::Unary> for Typed {
        fn forward(&self, op: &kapel_ops::Unary, inputs: &[&Tensor]) -> Result<Tensor> {
            let y = Reference.forward(op, inputs)?;
            if !self.exp_off || op.kind != kapel_ops::UnaryKind::Exp {
                return typed(op, inputs, y);
            }
            let off: Vec<f64> = y
                .to_vec::<f64>()?
                .iter()
                .map(|v| v * (1.0 + 1e-3))
                .collect();
            typed(op, inputs, Tensor::from_slice(&off, y.shape())?)
        }

        fn backward(
            &self,
            op: &kapel_ops::Unary,
            inputs: &[&Tensor],
            grad: &Tensor,
        ) -> Result<Vec<Option<Tensor>>> {
            Reference.backward(op, inputs, grad)
        }
    }

    #[test]
    fn the_suite_sees_a_small_error() {
        let failures = conformity::check(&Typed { exp_off: true }, DType::F32);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert_eq!(failures[0].op, "unary");
        assert!(failures[0].case.starts_with("Exp"));
    }

    #[test]
    fn softmax_of_equal_values_is_uniform() {
        let x = Tensor::from_slice(&[3.0f64; 4], &[1, 4]).unwrap();
        let y = Reference.forward(&Softmax { dim: 1 }, &[&x]).unwrap();
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![0.25; 4]);
    }

    #[test]
    fn rope_turns_a_pair_by_its_angle() {
        // With R = 2 the only frequency is 1: position p turns by p radians.
        let rope = Rope {
            theta: 10_000.0,
            layout: RopeLayout::Half,
            rotary_dim: None,
            scaling: RopeScaling::None,
        };
        let x = Tensor::from_slice(&[1.0f64, 0.0], &[1, 2]).unwrap();
        let p = Tensor::from_slice(&[0u32], &[1]).unwrap();
        let y = Reference
            .forward(&rope, &[&x, &p])
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_eq!(y, vec![1.0, 0.0]);
        let quarter = Rope { theta: 1.0, ..rope };
        let angle = Tensor::from_slice(&[1u32], &[1]).unwrap();
        let y = Reference
            .forward(&quarter, &[&x, &angle])
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert!((y[0] - 1f64.cos()).abs() < 1e-15 && (y[1] - 1f64.sin()).abs() < 1e-15);
    }

    #[test]
    fn causal_attention_of_the_first_query_reads_the_first_value() {
        let q = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::from_slice(&[10.0f64, 20.0], &[1, 1, 2, 1]).unwrap();
        let y = Reference
            .forward(&Attention::causal(), &[&q, &k, &v])
            .unwrap();
        let y = y.to_vec::<f64>().unwrap();
        assert_eq!(y[0], 10.0);
        // The second query sees both keys, with scores 0 and 1/√2.
        let s = 1.0 / 2f64.sqrt();
        let a = 1.0 / (1.0 + s.exp());
        assert!((y[1] - (10.0 * a + 20.0 * (1.0 - a))).abs() < 1e-12);
    }

    /// Attention of 2 queries over 3 keys, not causal, with the mask `mask`
    /// `[1, 1, 2, 3]`: the outputs and the gradients of q, k and v.
    fn masked_attention(mask: [f32; 6]) -> (Vec<f64>, [Vec<f64>; 3]) {
        let q = Tensor::from_slice(&[0.3f64, -0.2, 0.5, 0.1], &[1, 1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[0.1f64, 0.4, -0.3, 0.2, 0.6, -0.5], &[1, 1, 3, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f64, 2.0, 4.0], &[1, 1, 3, 1]).unwrap();
        let mask = Tensor::from_slice(&mask, &[1, 1, 2, 3]).unwrap();
        let op = Attention {
            causal: false,
            ..Attention::causal()
        };
        let inputs = [&q, &k, &v, &mask];
        let y = Reference.forward(&op, &inputs).unwrap();
        let g = Tensor::from_slice(&[1.0f64, 1.0], &[1, 1, 2, 1]).unwrap();
        let grads = Reference.backward(&op, &inputs, &g).unwrap();
        assert!(grads[3].is_none());
        let grad = |i: usize| grads[i].as_ref().unwrap().to_vec::<f64>().unwrap();
        (y.to_vec::<f64>().unwrap(), [grad(0), grad(1), grad(2)])
    }

    #[test]
    fn a_query_whose_keys_are_all_masked_gives_zeros_and_no_gradient() {
        let ninf = f32::NEG_INFINITY;
        let (y, [gq, gk, gv]) = masked_attention([ninf, ninf, ninf, 0.0, ninf, 0.0]);
        assert_eq!(y[0], 0.0);
        assert_eq!(&gq[..2], &[0.0, 0.0]);
        assert!(
            y.iter()
                .chain(&gq)
                .chain(&gk)
                .chain(&gv)
                .all(|v| v.is_finite())
        );
        // The second query reads keys 0 and 2 only: key 1 receives nothing.
        assert_eq!(&gk[2..4], &[0.0, 0.0]);
        assert_eq!(gv[1], 0.0);
    }

    #[test]
    fn a_query_whose_keys_are_all_at_f32_min_averages_the_values() {
        let min = f32::MIN;
        let (y, [gq, ..]) = masked_attention([min, min, min, 0.0, 0.0, 0.0]);
        assert!((y[0] - 7.0 / 3.0).abs() < 1e-15, "{}", y[0]);
        // The softmax still has a derivative there.
        assert!(gq[..2].iter().any(|&g| g != 0.0));
    }

    #[test]
    fn a_gradient_of_the_wrong_shape_is_an_error() {
        let a = Tensor::from_slice(&[1.0f64; 6], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f64; 12], &[3, 4]).unwrap();
        let g = Tensor::from_slice(&[1.0f64; 6], &[2, 3]).unwrap();
        assert!(
            Reference
                .backward(&kapel_ops::Matmul, &[&a, &b], &g)
                .is_err()
        );
        let g = Tensor::from_slice(&[1u32; 8], &[2, 4]).unwrap();
        assert!(
            Reference
                .backward(&kapel_ops::Matmul, &[&a, &b], &g)
                .is_err()
        );
    }

    #[test]
    fn a_cast_to_or_from_an_integer_passes_no_gradient() {
        let x = Tensor::from_slice(&[1i8, 2], &[2]).unwrap();
        let g = Tensor::from_slice(&[1.0f32, 1.0], &[2]).unwrap();
        let cast = kapel_ops::Cast { dtype: DType::F32 };
        let grads = Reference.backward(&cast, &[&x], &g).unwrap();
        assert!(grads.len() == 1 && grads[0].is_none());
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let cast = kapel_ops::Cast { dtype: DType::I8 };
        let grads = Reference.backward(&cast, &[&x], &g).unwrap();
        assert!(grads.len() == 1 && grads[0].is_none());
    }

    #[test]
    fn a_bad_call_is_rejected_before_any_computation() {
        let q = Tensor::from_slice(&[0.0f64; 24], &[1, 3, 1, 8]).unwrap();
        let k = Tensor::from_slice(&[0.0f64; 16], &[1, 2, 1, 8]).unwrap();
        let error = Reference
            .forward(&Attention::causal(), &[&q, &k, &k])
            .unwrap_err();
        assert!(
            error.to_string().contains("multiple of the 2 key heads"),
            "{error}"
        );
        assert!(Attention::causal().infer(&[&q]).is_err());
    }
}
