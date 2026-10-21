//! The CUDA backend.
//!
//! `Cuda` runs the whole catalogue of `kapel-ops` on an NVIDIA device, written
//! entirely in Rust with CUDA Rust:
//!
//! - matrix products and attention (in the manner of FlashAttention 2) are
//!   cutile-rs tile kernels, in `f32`, `bf16` and `f16` with `f32`
//!   accumulators, compiled by `tileiras` at their first launch and kept in a
//!   cache on disk;
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
mod attention;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod ops;
mod signatures;
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

    /// Attentions over several tiles of queries and keys, with grouped heads,
    /// compared to the reference in both directions, and repeated: causal;
    /// with a window and queries after a cache; with a mask holding −∞ and a
    /// softcap, not causal; for each width of tiles. Tiles inside the visible region and on its edges
    /// take different paths.
    #[test]
    fn attention_over_many_tiles_matches_the_reference() {
        use kapel_ops::Attention;
        let cuda = cuda();
        let (b, hq, hkv) = (2, 8, 2);
        let values = |len: usize, seed: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        };
        let windowed = Attention {
            window: Some(70),
            ..Attention::causal()
        };
        let capped = Attention {
            causal: false,
            softcap: Some(2.0),
            ..Attention::causal()
        };
        for (op, tq, tk, d, masked) in [
            (Attention::causal(), 200, 200, 64, false),
            (windowed, 136, 200, 64, false),
            (capped, 200, 200, 64, true),
            (Attention::causal(), 200, 200, 128, false),
            (windowed, 136, 200, 256, false),
        ] {
            for dtype in [DType::F32, DType::BF16] {
                let host = |len: usize, seed: usize, shape: &[usize]| {
                    let x = Tensor::from_slice(&values(len, seed), shape).unwrap();
                    Run::<kapel_ops::Cast>::forward(
                        &kapel_ref::Reference,
                        &kapel_ops::Cast { dtype },
                        &[&x],
                    )
                    .unwrap()
                };
                let mut inputs = vec![
                    host(b * hq * tq * d, 1, &[b, hq, tq, d]),
                    host(b * hkv * tk * d, 2, &[b, hkv, tk, d]),
                    host(b * hkv * tk * d, 3, &[b, hkv, tk, d]),
                ];
                if masked {
                    let mask: Vec<f32> = (0..tq * tk)
                        .map(|n| match n % 7 {
                            0 => f32::NEG_INFINITY,
                            r => r as f32 / 7.0 - 0.5,
                        })
                        .collect();
                    inputs.push(Tensor::from_slice(&mask, &[1, 1, tq, tk]).unwrap());
                }
                let g = host(b * hq * tq * d, 4, &[b, hq, tq, d]);
                let reference = kapel_ref::Reference;
                let refs: Vec<&Tensor> = inputs.iter().collect();
                let mut expected = vec![reference.forward(&op, &refs).unwrap()];
                expected.extend(
                    reference
                        .backward(&op, &refs, &g)
                        .unwrap()
                        .into_iter()
                        .take(3)
                        .map(Option::unwrap),
                );
                let on_device: Vec<Tensor> =
                    inputs.iter().map(|x| cuda.upload(x).unwrap()).collect();
                let on_device: Vec<&Tensor> = on_device.iter().collect();
                let dg = cuda.upload(&g).unwrap();
                let run = || {
                    let mut got = vec![cuda.forward(&op, &on_device).unwrap()];
                    got.extend(
                        cuda.backward(&op, &on_device, &dg)
                            .unwrap()
                            .into_iter()
                            .take(3)
                            .map(Option::unwrap),
                    );
                    got.iter()
                        .map(|x| cuda.download(x).unwrap().to_f64_vec().unwrap())
                        .collect::<Vec<_>>()
                };
                let got = run();
                assert_eq!(run(), got, "{op:?} {dtype}: a second run differs");
                // Products in 16 bits round the weights and the gradients of
                // the scores to 16 bits as well.
                let tolerance = match dtype {
                    DType::F32 => conformity::tolerance(dtype),
                    _ => 4.0 * conformity::tolerance(dtype),
                };
                for (n, (e, g)) in expected.iter().zip(&got).enumerate() {
                    let e = e.to_f64_vec().unwrap();
                    let tol = tolerance * e.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                    let worst = e
                        .iter()
                        .zip(g)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        worst <= tol,
                        "{op:?} {dtype} result {n}: error {worst:.2e} above {tol:.2e}"
                    );
                }
            }
        }
    }

    #[test]
    fn one_device_has_one_backend() {
        let (a, b) = (cuda(), cuda());
        assert!(std::sync::Arc::ptr_eq(&a.inner, &b.inner));
        // A tensor of one is a tensor of the other: same stream, same pool.
        let x = a
            .upload(&Tensor::from_slice(&[1.0f32, -2.0], &[2]).unwrap())
            .unwrap();
        let y = b
            .forward(
                &kapel_ops::Unary {
                    kind: kapel_ops::UnaryKind::Abs,
                },
                &[&x],
            )
            .unwrap();
        assert_eq!(a.download(&y).unwrap().to_vec::<f32>().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn launches_beyond_32_bit_positions_are_refused() {
        use kapel_core::Error;

        use crate::device::Args;
        let cuda = cuda();
        let too_many = u32::MAX as usize + 1;
        let args = || Args::new().u32(0).slice(0, 4).slice(0, 4);
        // Nothing is launched: the checks come first.
        assert!(matches!(
            cuda.launch("unary_forward", too_many, args()),
            Err(Error::Overflow { .. })
        ));
        let long = Args::new().u32(0).slice(0, too_many).slice(0, 4);
        assert!(matches!(
            cuda.launch("unary_forward", 1, long),
            Err(Error::Overflow { .. })
        ));
        // Arguments that do not match the kernel's parameters.
        let wrong = Args::new().slice(0, 4).slice(0, 4);
        assert!(matches!(
            cuda.launch("unary_forward", 1, wrong),
            Err(Error::Device { .. })
        ));
    }

    /// Masks whose rows are entirely `−∞` (zeros, no gradient), entirely
    /// `f32::MIN` (uniform weights over the visible keys), or padding with
    /// `f32::MIN`, over several tiles, compared to the reference; with the
    /// mask in `bf16` as well, converted by the backend.
    #[test]
    fn attention_with_extreme_masks_matches_the_reference() {
        use kapel_ops::Attention;
        let cuda = cuda();
        let (b, hq, hkv, tq, tk, d) = (1, 2, 1, 70, 90, 64);
        let values = |len: usize, seed: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        };
        let mask: Vec<f32> = (0..tq * tk)
            .map(|n| {
                let (i, j) = (n / tk, n % tk);
                match i % 5 {
                    0 => f32::NEG_INFINITY,
                    1 => f32::MIN,
                    2 if j >= tk - 20 => f32::MIN,
                    3 if j % 3 == 0 => f32::NEG_INFINITY,
                    _ => (j % 7) as f32 / 7.0 - 0.5,
                }
            })
            .collect();
        let windowed = Attention {
            window: Some(40),
            ..Attention::causal()
        };
        let free = Attention {
            causal: false,
            ..Attention::causal()
        };
        for op in [free, Attention::causal(), windowed] {
            for (dtype, mask_dtype) in [
                (DType::F32, DType::F32),
                (DType::BF16, DType::F32),
                (DType::F32, DType::BF16),
            ] {
                let cast = |x: Tensor, dtype: DType| {
                    Run::<kapel_ops::Cast>::forward(
                        &kapel_ref::Reference,
                        &kapel_ops::Cast { dtype },
                        &[&x],
                    )
                    .unwrap()
                };
                let host = |len: usize, seed: usize, shape: &[usize]| {
                    cast(
                        Tensor::from_slice(&values(len, seed), shape).unwrap(),
                        dtype,
                    )
                };
                let m = cast(
                    Tensor::from_slice(&mask, &[1, 1, tq, tk]).unwrap(),
                    mask_dtype,
                );
                let inputs = [
                    host(b * hq * tq * d, 1, &[b, hq, tq, d]),
                    host(b * hkv * tk * d, 2, &[b, hkv, tk, d]),
                    host(b * hkv * tk * d, 3, &[b, hkv, tk, d]),
                    m,
                ];
                let g = host(b * hq * tq * d, 4, &[b, hq, tq, d]);
                let refs: Vec<&Tensor> = inputs.iter().collect();
                let reference = kapel_ref::Reference;
                let mut expected = vec![reference.forward(&op, &refs).unwrap()];
                expected.extend(
                    reference
                        .backward(&op, &refs, &g)
                        .unwrap()
                        .into_iter()
                        .take(3)
                        .map(Option::unwrap),
                );
                let on_device: Vec<Tensor> =
                    inputs.iter().map(|x| cuda.upload(x).unwrap()).collect();
                let on_device: Vec<&Tensor> = on_device.iter().collect();
                let dg = cuda.upload(&g).unwrap();
                let mut got = vec![cuda.forward(&op, &on_device).unwrap()];
                got.extend(
                    cuda.backward(&op, &on_device, &dg)
                        .unwrap()
                        .into_iter()
                        .take(3)
                        .map(Option::unwrap),
                );
                let tolerance = match dtype {
                    DType::F32 => conformity::tolerance(dtype),
                    _ => 4.0 * conformity::tolerance(dtype),
                };
                for (n, (e, g)) in expected.iter().zip(&got).enumerate() {
                    let e = e.to_f64_vec().unwrap();
                    let g = cuda.download(g).unwrap().to_f64_vec().unwrap();
                    let tol = tolerance * e.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                    for (i, (a, b)) in e.iter().zip(&g).enumerate() {
                        assert!(
                            (a - b).abs() <= tol,
                            "{op:?} {dtype} mask {mask_dtype} result {n} element {i}: \
                             {b}, expected {a}"
                        );
                    }
                }
            }
        }
    }

    /// Normalization weights in another float type than `x`: converted, and
    /// their gradients given back in their own type.
    #[test]
    fn norms_take_weights_of_any_float_type() {
        use kapel_ops::{LayerNorm, RmsNorm};
        let cuda = cuda();
        let reference = kapel_ref::Reference;
        let values: Vec<f32> = (0..4 * 6)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x = Tensor::from_slice(&values, &[4, 6]).unwrap();
        let w = Run::<kapel_ops::Cast>::forward(
            &reference,
            &kapel_ops::Cast { dtype: DType::BF16 },
            &[&Tensor::from_slice(&values[..6], &[6]).unwrap()],
        )
        .unwrap();
        let bias = Tensor::from_slice(&values[6..12], &[6]).unwrap();
        let g = Tensor::from_slice(&values, &[4, 6]).unwrap();
        let (dx, dw, db, dg) = (
            cuda.upload(&x).unwrap(),
            cuda.upload(&w).unwrap(),
            cuda.upload(&bias).unwrap(),
            cuda.upload(&g).unwrap(),
        );
        let check = |expected: Vec<Tensor>, got: Vec<Tensor>| {
            for (e, g) in expected.iter().zip(&got) {
                let g = cuda.download(g).unwrap();
                let (e, g) = (e.to_f64_vec().unwrap(), g.to_f64_vec().unwrap());
                for (a, b) in e.iter().zip(&g) {
                    // Gradients in `bf16` are rounded to 8 bits.
                    assert!(
                        (a - b).abs() <= 1e-2 * a.abs().max(1.0),
                        "{b}, expected {a}"
                    );
                }
            }
        };
        let layer = LayerNorm { eps: 1e-5 };
        let host = [&x, &w, &bias];
        let device = [&dx, &dw, &db];
        let mut expected = vec![reference.forward(&layer, &host).unwrap()];
        expected.extend(
            reference
                .backward(&layer, &host, &g)
                .unwrap()
                .into_iter()
                .flatten(),
        );
        let grads = cuda.backward(&layer, &device, &dg).unwrap();
        assert_eq!(grads[1].as_ref().unwrap().dtype(), DType::BF16);
        let mut got = vec![cuda.forward(&layer, &device).unwrap()];
        got.extend(grads.into_iter().flatten());
        check(expected, got);
        let rms = RmsNorm {
            eps: 1e-6,
            weight_offset: 1.0,
        };
        let mut expected = vec![reference.forward(&rms, &[&x, &w]).unwrap()];
        expected.extend(
            reference
                .backward(&rms, &[&x, &w], &g)
                .unwrap()
                .into_iter()
                .flatten(),
        );
        let mut got = vec![cuda.forward(&rms, &[&dx, &dw]).unwrap()];
        got.extend(
            cuda.backward(&rms, &[&dx, &dw], &dg)
                .unwrap()
                .into_iter()
                .flatten(),
        );
        check(expected, got);
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
