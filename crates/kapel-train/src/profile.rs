//! The time of each operation: a backend that wraps another and times
//! every call, waiting for the device before and after.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{
    Affine, ArgMax, Attention, Binary, Cast, Concat, Contiguous, CrossEntropy, GatherRows, Glu,
    LayerNorm, LogSoftmax, Matmul, Max, Mean, Operation, PagedAttention, QuantizedMatmul, Record,
    Recording, RmsNorm, Rope, Run, ScatterAddRows, Softmax, Sum, Transfer, Unary, Write,
};

/// The time spent in one kind of call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    /// The calls.
    pub calls: u64,
    /// Their time, the device waited for.
    pub time: Duration,
}

type Synchronize = Arc<dyn Fn() + Send + Sync>;

/// A backend that times each call of `B`, by operation and direction
/// (`"matmul"`, `"matmul backward"`). It waits for the device around each
/// call: the times add up to the pass's, and the pass is slower.
#[derive(Clone)]
pub struct Profiled<B> {
    inner: B,
    synchronize: Synchronize,
    spent: Arc<Mutex<BTreeMap<String, Spent>>>,
}

impl<B> Profiled<B> {
    /// `inner`, timed; `synchronize` waits for its device (nothing for the
    /// processor).
    pub fn new(inner: B, synchronize: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            inner,
            synchronize: Arc::new(synchronize),
            spent: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// The backend timed.
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// The time spent so far, by kind of call.
    pub fn spent(&self) -> BTreeMap<String, Spent> {
        self.spent.lock().expect("not poisoned").clone()
    }

    /// Forgets the times.
    pub fn reset(&self) {
        self.spent.lock().expect("not poisoned").clear();
    }

    /// The times as a table, the longest first.
    pub fn report(&self) -> String {
        let spent = self.spent();
        let total: Duration = spent.values().map(|s| s.time).sum();
        let mut rows: Vec<_> = spent.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.time));
        let mut out = format!("{:<32} {:>8} {:>10} {:>6}\n", "call", "calls", "ms", "%");
        for (name, s) in rows {
            out += &format!(
                "{:<32} {:>8} {:>10.1} {:>6.1}\n",
                name,
                s.calls,
                s.time.as_secs_f64() * 1e3,
                100.0 * s.time.as_secs_f64() / total.as_secs_f64().max(1e-12)
            );
        }
        out
    }

    fn time<T>(&self, name: impl FnOnce() -> String, f: impl FnOnce() -> T) -> T {
        (self.synchronize)();
        let start = Instant::now();
        let out = f();
        (self.synchronize)();
        let time = start.elapsed();
        let mut spent = self.spent.lock().expect("not poisoned");
        let entry = spent.entry(name()).or_default();
        entry.calls += 1;
        entry.time += time;
        out
    }
}

/// The name of a call of `op`: its kind, and what tells its costs apart.
fn label<O: Operation>(op: &O) -> String {
    let debug = format!("{op:?}");
    // The kind of a unary or binary function, the dimension of a softmax.
    match O::NAME {
        "unary" | "binary" | "glu" => format!(
            "{} {}",
            O::NAME,
            debug
                .split(['{', '}', ':', ','])
                .nth(2)
                .unwrap_or("")
                .trim()
        ),
        name => name.to_string(),
    }
}

macro_rules! timed {
    ($($op:ty),* $(,)?) => {$(
        impl<B: Run<$op>> Run<$op> for Profiled<B> {
            fn forward(&self, op: &$op, inputs: &[&Tensor]) -> Result<Tensor> {
                self.time(|| label(op), || self.inner.forward(op, inputs))
            }

            fn backward(
                &self,
                op: &$op,
                inputs: &[&Tensor],
                grad: &Tensor,
            ) -> Result<Vec<Option<Tensor>>> {
                self.time(
                    || format!("{} backward", label(op)),
                    || self.inner.backward(op, inputs, grad),
                )
            }
        }
    )*};
}

timed!(
    Unary,
    Glu,
    Binary,
    Affine,
    Cast,
    Contiguous,
    Concat,
    Matmul,
    QuantizedMatmul,
    Sum,
    Mean,
    Max,
    ArgMax,
    Softmax,
    LogSoftmax,
    CrossEntropy,
    LayerNorm,
    RmsNorm,
    Attention,
    PagedAttention,
    Rope,
    GatherRows,
    ScatterAddRows,
);

impl<B: Transfer> Transfer for Profiled<B> {
    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        self.time(|| "upload".into(), || self.inner.upload(tensor))
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        self.time(|| "download".into(), || self.inner.download(tensor))
    }
}

impl<B: Write> Write for Profiled<B> {
    fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        self.time(|| "zeros".into(), || self.inner.zeros(shape, dtype))
    }

    fn write_rows(&self, table: &mut Tensor, index: &Tensor, rows: &Tensor) -> Result<()> {
        self.time(
            || "write_rows".into(),
            || self.inner.write_rows(table, index, rows),
        )
    }

    fn copy_from_host(&self, target: &mut Tensor, values: &Tensor) -> Result<()> {
        self.inner.copy_from_host(target, values)
    }
}

impl<B: kapel_ops::Update> kapel_ops::Update for Profiled<B> {
    fn adamw(
        &self,
        param: &mut Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        grad: &Tensor,
        step: &kapel_ops::AdamWStep,
    ) -> Result<()> {
        self.time(
            || "adamw".into(),
            || self.inner.adamw(param, m, v, grad, step),
        )
    }
}

impl<B: Record> Record for Profiled<B> {
    fn records(&self) -> bool {
        false
    }

    fn record(&self, _: &mut dyn FnMut() -> Result<Tensor>) -> Result<Option<Recording>> {
        Ok(None)
    }
}
