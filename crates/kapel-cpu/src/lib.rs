//! The CPU backend.
//!
//! [`Cpu`] runs the whole catalogue of `kapel-ops` in `f32`, as fast as the
//! machine allows:
//!
//! - operations are shared between a permanent group of threads ([`pool`]);
//! - matrix products are cut into blocks that fit in the caches, around an
//!   AVX-512 or AVX2 micro-kernel ([`gemm`], [`kernels`]);
//! - row computations, the exponential included, are vectorized ([`math`]).
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
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty = $body:ident;
    ) => {
        $(#[$attr])*
        pub fn $name(isa: $crate::Isa, $($arg: $ty),*) -> $ret {
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
            match isa {
                // SAFETY: `Isa::Avx512` is only chosen when the processor has
                // these instructions (`Isa::available`).
                #[cfg(target_arch = "x86_64")]
                $crate::Isa::Avx512 => unsafe { wide($($arg),*) },
                // SAFETY: likewise for AVX2 and FMA.
                #[cfg(target_arch = "x86_64")]
                $crate::Isa::Avx2 => unsafe { narrow($($arg),*) },
                _ => $body::<false>($($arg),*),
            }
        }
    };
}

mod attention;
mod backend;
mod elementwise;
pub mod gemm;
mod index;
mod io;
mod isa;
pub mod kernels;
mod lines;
pub mod math;
mod matmul;
mod norm;
pub mod pool;
mod reduce;
mod rope;

pub use backend::Cpu;
pub use isa::{Isa, default_threads};

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
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
            if let Ok(cpu) = Cpu::with_threads(4).with_isa(isa) {
                conformity::assert_conforms(&cpu, DType::F32);
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
    fn f64_is_left_to_the_reference() {
        let x = Tensor::from_slice(&[1.0f64, 2.0], &[1, 2]).unwrap();
        let error = Cpu::with_threads(1)
            .forward(&Matmul, &[&x, &x.transpose(0, 1).unwrap()])
            .unwrap_err();
        assert!(error.to_string().contains("f64"), "{error}");
    }
}
