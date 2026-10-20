//! The CUDA backend.
//!
//! `Cuda` runs the whole catalogue of `kapel-ops` on an NVIDIA device, written
//! entirely in Rust with CUDA Rust:
//!
//! - matrix products are cutile-rs tile kernels, in `f32`, `bf16` and `f16`
//!   with an `f32` accumulator, compiled by `tileiras` at their first launch and
//!   kept in a cache on disk;
//! - the other operations, in `f32`, are cuda-oxide kernels (`kernels/`, built
//!   by `kernels/build.sh`), whose PTX is embedded in this crate.
//!
//! Each kernel gives the same bits on every run: one thread, or one tile block,
//! computes each output, adding its terms in a fixed order, without atomics.
//!
//! The backend needs the feature `cuda`, the CUDA toolkit 13.2 or later, and a
//! device; without the feature, this crate is empty, so that the workspace
//! builds anywhere. Build `cutile-compiler` without debug information (see the
//! workspace's `Cargo.toml`): with it, tile kernels are compiled unoptimized.

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod ops;
#[cfg(feature = "cuda")]
mod tile;

#[cfg(feature = "cuda")]
pub use device::Cuda;

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use kapel_core::{DType, Tensor};
    use kapel_ops::{Matmul, Run, Transfer};
    use kapel_ref::conformity;

    use super::*;

    fn cuda() -> Cuda {
        Cuda::new(0).expect("a CUDA device")
    }

    #[test]
    fn the_backend_conforms_in_f32() {
        conformity::assert_conforms(&cuda(), DType::F32);
    }

    #[test]
    fn tensors_go_to_the_device_and_back() {
        let cuda = cuda();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let back = cuda.download(&cuda.upload(&x).unwrap()).unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        let strided = cuda.upload(&x.transpose(0, 1).unwrap()).unwrap();
        assert_eq!(strided.shape(), [3, 2]);
        assert_eq!(
            cuda.download(&strided).unwrap().to_vec::<f32>().unwrap(),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn products_in_16_bits_match_the_reference() {
        let cuda = cuda();
        for dtype in [DType::BF16, DType::F16] {
            // Sizes that the tiles do not divide, a broadcast batch and a
            // transposed operand.
            let (m, k, n) = (130, 70, 90);
            let values = |len: usize, seed: usize| -> Vec<f64> {
                (0..len)
                    .map(|i| ((i * 7919 + seed) % 1000) as f64 / 500.0 - 1.0)
                    .collect()
            };
            let host = |values: Vec<f64>, shape: &[usize]| {
                let f32s: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                let x = Tensor::from_slice(&f32s, shape).unwrap();
                Run::<kapel_ops::Cast>::forward(
                    &kapel_ref::Reference,
                    &kapel_ops::Cast { dtype },
                    &[&x],
                )
                .unwrap()
            };
            let a = host(values(2 * m * k, 1), &[2, m, k]);
            let bt = host(values(n * k, 2), &[n, k]);
            let b = bt.transpose(0, 1).unwrap();
            let expected = kapel_ref::Reference.forward(&Matmul, &[&a, &b]).unwrap();
            let (da, db) = (cuda.upload(&a).unwrap(), cuda.upload(&bt).unwrap());
            let got = cuda
                .forward(&Matmul, &[&da, &db.transpose(0, 1).unwrap()])
                .unwrap();
            assert_eq!(got.dtype(), dtype);
            let got = cuda.download(&got).unwrap().to_f64_vec().unwrap();
            let expected = expected.to_f64_vec().unwrap();
            let tol =
                conformity::tolerance(dtype) * expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for (i, (e, g)) in expected.iter().zip(&got).enumerate() {
                assert!(
                    (e - g).abs() <= tol,
                    "{dtype} element {i}: {g}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn results_are_the_same_on_every_run() {
        let cuda = cuda();
        let values: Vec<f32> = (0..200 * 300)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let a = cuda
            .upload(&Tensor::from_slice(&values, &[200, 300]).unwrap())
            .unwrap();
        let run = || {
            let y = cuda
                .forward(&Matmul, &[&a, &a.transpose(0, 1).unwrap()])
                .unwrap();
            cuda.download(&y).unwrap().to_vec::<f32>().unwrap()
        };
        let first = run();
        for _ in 0..3 {
            assert_eq!(run(), first);
        }
    }
}
