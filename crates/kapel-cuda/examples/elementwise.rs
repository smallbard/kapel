//! Measures the element-wise operations of a training step of GPT-2 124M
//! (16 × 1024 tokens) in bf16: the bias added to a product, a residual sum,
//! a conversion, in GB/s of what they read and write.
//!
//! `cargo run --release -p kapel-cuda --features cuda --example elementwise`

use std::time::Instant;

use kapel_core::{BF16, DType, Tensor};
use kapel_cuda::Cuda;
use kapel_ops::{Binary, BinaryKind, Cast, Run, Transfer};

fn matrix(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<BF16> = (0..n)
        .map(|i| BF16::from_f64((i % 97) as f64 / 97.0))
        .collect();
    Tensor::from_slice(&values, shape).expect("sizes agree")
}

fn time(
    cuda: &Cuda,
    bytes: f64,
    what: &str,
    f: &dyn Fn() -> kapel_core::Result<Tensor>,
) -> kapel_core::Result<()> {
    f()?;
    cuda.synchronize()?;
    let runs = 20;
    let start = Instant::now();
    for _ in 0..runs {
        f()?;
    }
    cuda.synchronize()?;
    let t = start.elapsed().as_secs_f64() / f64::from(runs);
    println!(
        "{what:>40}: {:7.3} ms, {:6.0} GB/s",
        t * 1e3,
        bytes / t / 1e9
    );
    Ok(())
}

fn main() -> kapel_core::Result<()> {
    let cuda = Cuda::new(0)?;
    let add = Binary {
        kind: BinaryKind::Add,
    };
    for width in [768, 3072] {
        let x = cuda.upload(&matrix(&[16384, width]))?;
        let y = cuda.upload(&matrix(&[16384, width]))?;
        let b = cuda.upload(&matrix(&[width]))?;
        let bytes = 16384.0 * width as f64 * 2.0;
        time(
            &cuda,
            2.0 * bytes,
            &format!("[16384, {width}] + [{width}]"),
            &|| cuda.forward(&add, &[&x, &b]),
        )?;
        time(
            &cuda,
            3.0 * bytes,
            &format!("[16384, {width}] + [16384, {width}]"),
            &|| cuda.forward(&add, &[&x, &y]),
        )?;
        time(
            &cuda,
            3.0 * bytes,
            &format!("[16384, {width}] bf16 to f32"),
            &|| cuda.forward(&Cast { dtype: DType::F32 }, &[&x]),
        )?;
    }
    Ok(())
}
