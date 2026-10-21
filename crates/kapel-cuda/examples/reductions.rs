//! Measures operations that the host prepares tables for (sums, means,
//! maxima, softmaxes, their gradients), at sizes where the time to queue them
//! counts as much as the kernels. Measure on an idle device.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example reductions`

use std::time::Instant;

use kapel_core::{Error, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{LogSoftmax, Max, Mean, Run, Softmax, Sum, Transfer};

fn main() -> Result<(), Error> {
    let cuda = Cuda::new(0)?;
    for shape in [[8, 64, 64], [16, 128, 256], [32, 256, 512]] {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x = cuda.upload(&Tensor::from_slice(&values, &shape)?)?;
        let g = x.clone();
        let sum = Sum {
            dims: vec![1],
            keepdim: false,
        };
        let mean = Mean {
            dims: vec![0, 2],
            keepdim: true,
        };
        let max = Max {
            dim: 2,
            keepdim: false,
        };
        let step = || -> Result<(), Error> {
            for dim in [1, 2] {
                cuda.forward(&Softmax { dim }, &[&x])?;
                cuda.backward(&Softmax { dim }, &[&x], &g)?;
                cuda.backward(&LogSoftmax { dim }, &[&x], &g)?;
            }
            let s = cuda.forward(&sum, &[&x])?;
            cuda.backward(&sum, &[&x], &s)?;
            let m = cuda.forward(&mean, &[&x])?;
            cuda.backward(&mean, &[&x], &m)?;
            let y = cuda.forward(&max, &[&x])?;
            cuda.backward(&max, &[&x], &y)?;
            Ok(())
        };
        step()?;
        cuda.synchronize()?;
        let runs = 50;
        let start = Instant::now();
        for _ in 0..runs {
            step()?;
        }
        cuda.synchronize()?;
        let time = start.elapsed().as_secs_f64() / f64::from(runs);
        println!(
            "f32 {shape:?}: {:.3} ms per step of 13 operations",
            time * 1e3
        );
    }
    Ok(())
}
