//! Where the time of a generated token goes, on the CPU: each operation of
//! the backend is timed while a model generates, then the times are listed
//! by operation (by weight type for the products by quantized weights).
//!
//! `cargo run --release -p kapel-cli --example profile -- MODEL [TOKENS]`,
//! where `MODEL` is a model directory or a GGUF file.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use kapel_core::{DType, Result, Tensor};
use kapel_hub::{Checkpoint, Gguf};
use kapel_models::{Cache, Model};
use kapel_ops::{Operation, Run, Transfer};

/// A backend whose operations are timed.
struct Timed<B> {
    inner: B,
    times: RefCell<BTreeMap<String, (Duration, usize)>>,
    /// The bytes of the weights of each product, by name.
    bytes: RefCell<BTreeMap<String, f64>>,
}

impl<B> Timed<B> {
    fn record(&self, name: String, time: Duration) {
        let mut times = self.times.borrow_mut();
        let entry = times.entry(name).or_default();
        entry.0 += time;
        entry.1 += 1;
    }
}

impl<O: Operation, B: Run<O>> Run<O> for Timed<B> {
    fn forward(&self, op: &O, inputs: &[&Tensor]) -> Result<Tensor> {
        let start = Instant::now();
        let out = self.inner.forward(op, inputs)?;
        let name = match inputs.get(1) {
            Some(w) if w.dtype().is_quantized() => {
                let (n, k) = (w.shape()[0], w.shape()[1]);
                let bytes = w.dtype().storage_bytes(n * k).unwrap_or(0) as f64;
                self.bytes
                    .borrow_mut()
                    .entry(format!("{} {} {n}×{k}", O::NAME, w.dtype()))
                    .or_insert(bytes);
                format!("{} {} {n}×{k}", O::NAME, w.dtype())
            }
            _ => O::NAME.to_string(),
        };
        self.record(name, start.elapsed());
        Ok(out)
    }

    fn backward(&self, op: &O, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        self.inner.backward(op, inputs, grad)
    }
}

impl<B: kapel_ops::Write> kapel_ops::Write for Timed<B> {
    fn zeros(&self, shape: &[usize], dtype: kapel_core::DType) -> kapel_core::Result<Tensor> {
        self.inner.zeros(shape, dtype)
    }

    fn write_rows(
        &self,
        table: &mut Tensor,
        index: &Tensor,
        rows: &Tensor,
    ) -> kapel_core::Result<()> {
        let start = Instant::now();
        let result = self.inner.write_rows(table, index, rows);
        self.record(kapel_ops::WRITE_ROWS.to_string(), start.elapsed());
        result
    }

    fn copy_from_host(&self, target: &mut Tensor, values: &Tensor) -> kapel_core::Result<()> {
        self.inner.copy_from_host(target, values)
    }
}

impl<B: kapel_ops::Record> kapel_ops::Record for Timed<B> {
    fn records(&self) -> bool {
        self.inner.records()
    }

    fn record(
        &self,
        pass: &mut dyn FnMut() -> kapel_core::Result<Tensor>,
    ) -> kapel_core::Result<Option<kapel_ops::Recording>> {
        self.inner.record(pass)
    }
}

impl<B: kapel_ops::Update> kapel_ops::Update for Timed<B> {
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &kapel_ops::AdamWStep,
    ) -> kapel_core::Result<()> {
        self.inner.adamw(param, m, v, grad, step)
    }
}

impl<B: Transfer> Transfer for Timed<B> {
    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        self.inner.upload(tensor)
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        self.inner.download(tensor)
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: profile MODEL [TOKENS]")?;
    let tokens: usize = args.next().map_or(Ok(64), |t| t.parse())?;
    let cpu = Timed {
        // `BYTES=1`: activations in bytes.
        inner: kapel_cpu::Cpu::new()
            .with_byte_activations(std::env::var("BYTES").is_ok_and(|v| v == "1")),
        times: RefCell::new(BTreeMap::new()),
        bytes: RefCell::new(BTreeMap::new()),
    };
    let model = if std::path::Path::new(&path).is_file() {
        Model::from_gguf(&Gguf::open(&path)?)?
    } else {
        Model::load(&Checkpoint::open(&path)?)?
    };
    let model = model.cast(&cpu, DType::F32)?;
    let mut cache = Cache::new();
    // A short prompt, then tokens one by one; only the steps are timed.
    model.step(&cpu, &[1, 2, 3, 4, 5, 6, 7, 8], &mut cache)?;
    cpu.times.borrow_mut().clear();
    let start = Instant::now();
    let mut next = 9;
    for _ in 0..tokens {
        let logits = model.step(&cpu, &[next], &mut cache)?;
        let values = logits.to_f64_vec()?;
        next = (0..values.len())
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))
            .unwrap_or(0) as u32;
    }
    let total = start.elapsed();
    let times = cpu.times.borrow();
    let mut rows: Vec<_> = times.iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.0));
    let per_token = |d: Duration| d.as_secs_f64() * 1e3 / tokens as f64;
    println!(
        "{tokens} tokens: {:.2} ms per token ({:.1} tokens/s)",
        per_token(total),
        tokens as f64 / total.as_secs_f64()
    );
    let mut timed = Duration::ZERO;
    for (name, (time, count)) in rows {
        timed += *time;
        let rate = cpu.bytes.borrow().get(name).map_or(String::new(), |b| {
            format!(
                ", {:5.1} GB/s",
                b * *count as f64 / time.as_secs_f64() / 1e9
            )
        });
        println!(
            "{name:>40}: {:7.3} ms per token, {:5} calls per token, {:7.1} µs per call{rate}",
            per_token(*time),
            count / tokens,
            time.as_secs_f64() * 1e6 / *count as f64
        );
    }
    println!(
        "{:>40}: {:7.3} ms per token",
        "outside operations",
        per_token(total - timed)
    );
    Ok(())
}
