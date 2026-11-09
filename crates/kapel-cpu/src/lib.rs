//! The CPU backend.
//!
//! [`Cpu`] runs the whole catalogue of `kapel-ops` in `f32`, as fast as the
//! machine allows:
//!
//! - operations are shared between a permanent group of threads ([`pool`]);
//! - matrix products are cut into blocks that fit in the caches, around an
//!   AVX-512 or AVX2 micro-kernel ([`gemm`], [`kernels`]);
//! - row computations, the exponential included, are vectorized ([`math`]);
//! - products by quantized weights (Kapel's int8 and int4, GGUF's formats)
//!   read the codes in place and decode them in registers: in generation,
//!   they read the weights at the speed of memory. With
//!   [`Cpu::with_byte_activations`], their activations are quantized to
//!   bytes too, and the products are sums of bytes (AVX-512 VNNI).
//!
//! The instructions are detected at run time: the same build runs on a
//! processor without AVX-512, with the AVX2 kernels, or without either.
//!
//! The work is split by the shapes of the tensors, never by the number of
//! threads: with one, two or sixteen threads, a computation gives the same bits.
//!
//! Inputs may be `f32`, `bf16` or `f16`; computations are done in `f32`, and
//! outputs take the inputs' type. `f64` is left to the reference backend.

/// Compiles a function written once in three versions: AVX-512, AVX2, and
/// without explicit vector instructions. `$body` is an `#[inline(always)]`
/// function generic over `FMA`, which says whether multiplications and
/// additions may be fused.
macro_rules! multiversion {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty = $body:ident;
    ) => {
        $(#[$attr])*
        $vis fn $name(isa: $crate::Isa, $($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
            unsafe fn wide($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn narrow($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            match isa.kind() {
                // SAFETY: an `Isa` of kind `Avx512` can only be built by
                // `Isa::new` or `Isa::detect`, after checking that the
                // processor has AVX-512 F/BW/DQ/VL, AVX2 and FMA: the
                // instructions `wide` is compiled with.
                #[cfg(target_arch = "x86_64")]
                $crate::IsaKind::Avx512 => unsafe { wide($($arg),*) },
                // SAFETY: likewise, an `Isa` of kind `Avx2` proves AVX2 and
                // FMA, the instructions of `narrow`.
                #[cfg(target_arch = "x86_64")]
                $crate::IsaKind::Avx2 => unsafe { narrow($($arg),*) },
                _ => $body::<false>($($arg),*),
            }
        }
    };
}

mod attention;
mod backend;
mod bytes;
mod elementwise;
mod experts;
pub mod gemm;
mod index;
mod io;
mod isa;
pub mod kernels;
mod lines;
mod loss;
pub mod math;
mod matmul;
mod norm;
mod paged;
pub mod pool;
mod quantized;
mod recurrent;
mod reduce;
mod rope;
mod update;

pub use backend::Cpu;
pub use isa::{Isa, IsaKind, default_threads};

/// `a × b + c`, fused when the processor can.
#[inline(always)]
pub(crate) fn fma<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

#[cfg(test)]
mod tests {
    use kapel_core::{DType, Tensor};
    use kapel_ops::{Attention, Matmul, Run};
    use kapel_ref::conformity;

    use super::*;

    #[test]
    fn the_backend_conforms_in_f32_and_bf16() {
        let cpu = Cpu::new();
        conformity::assert_conforms(&cpu, DType::F32);
        conformity::assert_conforms(&cpu, DType::BF16);
    }

    #[test]
    fn every_instruction_set_conforms() {
        for kind in [IsaKind::Scalar, IsaKind::Avx2, IsaKind::Avx512] {
            match Cpu::with_threads(4).with_isa(kind) {
                Ok(cpu) => conformity::assert_conforms(&cpu, DType::F32),
                Err(_) => assert!(!kind.available()),
            }
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_number_of_threads() {
        let values = |n: usize| {
            (0..n)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let a = Tensor::from_slice(&values(300 * 200), &[300, 200]).unwrap();
        let b = Tensor::from_slice(&values(200 * 500), &[200, 500]).unwrap();
        let q = Tensor::from_slice(&values(2 * 4 * 64 * 32), &[2, 4, 64, 32]).unwrap();
        let k = Tensor::from_slice(&values(2 * 2 * 64 * 32), &[2, 2, 64, 32]).unwrap();
        let run = |cpu: &Cpu| {
            let y = cpu
                .forward(&Matmul, &[&a, &b])
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let z = cpu
                .forward(&Attention::causal(), &[&q, &k, &k])
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            (y, z)
        };
        let one = run(&Cpu::with_threads(1));
        for threads in [2, 3, 8] {
            assert_eq!(run(&Cpu::with_threads(threads)), one, "{threads} threads");
        }
    }

    #[test]
    fn attention_does_not_depend_on_the_number_of_threads() {
        // One batch element and two key heads: the queries are split in
        // blocks, forward and backward, with a mask and a softcap.
        let values = |n: usize, seed: usize| {
            (0..n)
                .map(|i| (((i + seed) * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let (hq, hkv, tq, tk, d) = (4, 2, 300, 320, 16);
        let q = Tensor::from_slice(&values(hq * tq * d, 1), &[1, hq, tq, d]).unwrap();
        let k = Tensor::from_slice(&values(hkv * tk * d, 2), &[1, hkv, tk, d]).unwrap();
        let v = Tensor::from_slice(&values(hkv * tk * d, 3), &[1, hkv, tk, d]).unwrap();
        let g = Tensor::from_slice(&values(hq * tq * d, 4), &[1, hq, tq, d]).unwrap();
        let mask = Tensor::from_slice(&values(tq * tk, 5), &[1, 1, tq, tk]).unwrap();
        let op = Attention {
            softcap: Some(5.0),
            ..Attention::causal()
        };
        let run = |cpu: &Cpu| {
            let inputs = [&q, &k, &v, &mask];
            let mut all = vec![cpu.forward(&op, &inputs).unwrap().to_vec::<f32>().unwrap()];
            for grad in cpu
                .backward(&op, &inputs, &g)
                .unwrap()
                .into_iter()
                .flatten()
            {
                all.push(grad.to_vec::<f32>().unwrap());
            }
            all
        };
        let one = run(&Cpu::with_threads(1));
        for threads in [2, 5, 16] {
            assert!(run(&Cpu::with_threads(threads)) == one, "{threads} threads");
        }
    }

    #[test]
    fn attention_keeps_nan_apart_from_masked_rows() {
        let cpu = Cpu::with_threads(2);
        let q = Tensor::from_slice(&[1.0f32; 3 * 4], &[1, 1, 3, 4]).unwrap();
        let k = Tensor::from_slice(&[0.5f32; 3 * 4], &[1, 1, 3, 4]).unwrap();
        let ninf = f32::NEG_INFINITY;
        // Row 0: every key masked, zeros. Row 1: masked but for a NaN, which
        // propagates. Row 2: ordinary.
        let mask = [ninf, ninf, ninf, ninf, f32::NAN, ninf, 0.0, 0.0, 0.0];
        let mask = Tensor::from_slice(&mask, &[1, 1, 3, 3]).unwrap();
        let y = cpu
            .forward(
                &Attention {
                    causal: false,
                    ..Attention::causal()
                },
                &[&q, &k, &k, &mask],
            )
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(y[..4].iter().all(|&x| x == 0.0), "{y:?}");
        assert!(y[4..8].iter().all(|x| x.is_nan()), "{y:?}");
        assert!(y[8..].iter().all(|&x| (x - 0.5).abs() < 1e-6), "{y:?}");
    }

    #[test]
    fn f64_is_left_to_the_reference() {
        let x = Tensor::from_slice(&[1.0f64, 2.0], &[1, 2]).unwrap();
        let error = Cpu::with_threads(1)
            .forward(&Matmul, &[&x, &x.transpose(0, 1).unwrap()])
            .unwrap_err();
        assert!(error.to_string().contains("f64"), "{error}");
    }
}
