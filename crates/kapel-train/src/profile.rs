//! The time of each operation: a backend that wraps another and times
//! every call, waiting for the device before and after. It also counts the
//! operations of the products and the attentions, from the shapes of their
//! inputs: the rate of each kind of call, and the computation a step
//! really asks for, which [`crate::flops`] predicts.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{
    Affine, ArgMax, Attention, Binary, Cast, Concat, Contiguous, CrossEntropy, ExpertMatmul,
    GatherColumns, GatherRows, Glu, LayerNorm, LogSoftmax, Matmul, Max, Mean, Operation,
    PagedAttention, QuantizedMatmul, Record, Recording, RmsNorm, Rope, Run, ScatterAddRows,
    Softmax, Sum, TopK, Transfer, Unary, Write,
};

/// The time spent in one kind of call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    /// The calls.
    pub calls: u64,
    /// Their time, the device waited for.
    pub time: Duration,
    /// Their operations (two a multiply-add), for the products and the
    /// attentions ([`Cost`]); 0 for the others.
    pub operations: f64,
}

/// The operations of a call, from the shapes of its inputs: two a
/// multiply-add of a product, nothing for the operations of a few per
/// element.
pub trait Cost {
    /// The operations of the forward call.
    fn forward_cost(&self, _inputs: &[&Tensor]) -> f64 {
        0.0
    }

    /// The operations of the backward call that gave `grads`.
    fn backward_cost(&self, _inputs: &[&Tensor], _grads: &[Option<Tensor>]) -> f64 {
        0.0
    }
}

/// The multiply-adds of `a [..., M, K] · b [..., K, N]`, the batch
/// dimensions broadcast.
fn product(a: &Tensor, b: &Tensor) -> f64 {
    let (ra, rb) = (a.rank(), b.rank());
    let (m, k, n) = (a.shape()[ra - 2], a.shape()[ra - 1], b.shape()[rb - 1]);
    let batch = |s: &[usize], i: usize| if i < s.len() { s[s.len() - 1 - i] } else { 1 };
    let (ba, bb) = (&a.shape()[..ra - 2], &b.shape()[..rb - 2]);
    let batch: usize = (0..ba.len().max(bb.len()))
        .map(|i| batch(ba, i).max(batch(bb, i)))
        .product();
    (batch * m * k * n) as f64
}

impl Cost for Matmul {
    fn forward_cost(&self, inputs: &[&Tensor]) -> f64 {
        2.0 * product(inputs[0], inputs[1])
    }

    /// A product for the gradient of each input but the bias.
    fn backward_cost(&self, inputs: &[&Tensor], grads: &[Option<Tensor>]) -> f64 {
        let n = grads.iter().take(2).filter(|g| g.is_some()).count();
        2.0 * n as f64 * product(inputs[0], inputs[1])
    }
}

impl Cost for QuantizedMatmul {
    fn forward_cost(&self, inputs: &[&Tensor]) -> f64 {
        let x = inputs[0];
        let k = x.shape().last().copied().unwrap_or(0);
        let rows = x.numel() / k.max(1);
        let n: usize = inputs[1..].iter().map(|w| w.shape()[0]).sum();
        2.0 * (rows * k * n) as f64
    }

    /// The gradient of the input only: the weights are frozen.
    fn backward_cost(&self, inputs: &[&Tensor], _grads: &[Option<Tensor>]) -> f64 {
        self.forward_cost(inputs)
    }
}

/// The keys a query of an attention reads, as [`crate::flops`] counts
/// them: all of them, but those beyond its window, causal or not.
fn attention_parts(op: &Attention, inputs: &[&Tensor]) -> (f64, f64, f64) {
    let (q, k, v) = (inputs[0].shape(), inputs[1].shape(), inputs[2].shape());
    let keys = op.window.map_or(k[2], |w| w.min(k[2]));
    let pairs = (q[0] * q[1] * q[2] * keys) as f64;
    (pairs, q[3] as f64, v[3] as f64)
}

impl Cost for Attention {
    /// The scores `q · kᵀ` and the mix of the values.
    fn forward_cost(&self, inputs: &[&Tensor]) -> f64 {
        let (pairs, d, dv) = attention_parts(self, inputs);
        2.0 * pairs * (d + dv)
    }

    /// The scores again (the probabilities are not kept, only their
    /// statistics), then the gradients of the probabilities, the values,
    /// the queries and the keys: five products where the forward call has
    /// two.
    fn backward_cost(&self, inputs: &[&Tensor], _grads: &[Option<Tensor>]) -> f64 {
        let (pairs, d, dv) = attention_parts(self, inputs);
        2.0 * pairs * (3.0 * d + 2.0 * dv)
    }
}

macro_rules! costless {
    ($($op:ty),* $(,)?) => {$(impl Cost for $op {})*};
}

costless!(
    Unary,
    Glu,
    Binary,
    Affine,
    Cast,
    Contiguous,
    Concat,
    Sum,
    Mean,
    Max,
    ArgMax,
    Softmax,
    LogSoftmax,
    CrossEntropy,
    LayerNorm,
    RmsNorm,
    PagedAttention,
    Rope,
    GatherRows,
    ScatterAddRows,
    TopK,
    GatherColumns,
    ExpertMatmul,
);

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

    /// The operations counted so far, all calls together.
    pub fn operations(&self) -> f64 {
        self.spent().values().map(|s| s.operations).sum()
    }

    /// The times as a table, the longest first, with the rate of the calls
    /// whose operations are counted.
    pub fn report(&self) -> String {
        let spent = self.spent();
        let total: Duration = spent.values().map(|s| s.time).sum();
        let mut rows: Vec<_> = spent.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.time));
        let mut out = format!(
            "{:<32} {:>8} {:>10} {:>6} {:>8}\n",
            "call", "calls", "ms", "%", "Top/s"
        );
        for (name, s) in rows {
            let seconds = s.time.as_secs_f64();
            let rate = if s.operations > 0.0 {
                format!("{:.1}", s.operations / seconds.max(1e-12) / 1e12)
            } else {
                String::new()
            };
            out += &format!(
                "{:<32} {:>8} {:>10.1} {:>6.1} {:>8}\n",
                name,
                s.calls,
                seconds * 1e3,
                100.0 * seconds / total.as_secs_f64().max(1e-12),
                rate
            );
        }
        out
    }

    fn time<T>(&self, name: impl FnOnce() -> String, f: impl FnOnce() -> T) -> T {
        self.time_counted(name, f, |_| 0.0)
    }

    fn time_counted<T>(
        &self,
        name: impl FnOnce() -> String,
        f: impl FnOnce() -> T,
        cost: impl FnOnce(&T) -> f64,
    ) -> T {
        (self.synchronize)();
        let start = Instant::now();
        let out = f();
        (self.synchronize)();
        let time = start.elapsed();
        let operations = cost(&out);
        let mut spent = self.spent.lock().expect("not poisoned");
        let entry = spent.entry(name()).or_default();
        entry.calls += 1;
        entry.time += time;
        entry.operations += operations;
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
                self.time_counted(
                    || label(op),
                    || self.inner.forward(op, inputs),
                    |_| op.forward_cost(inputs),
                )
            }

            fn backward(
                &self,
                op: &$op,
                inputs: &[&Tensor],
                grad: &Tensor,
            ) -> Result<Vec<Option<Tensor>>> {
                self.time_counted(
                    || format!("{} backward", label(op)),
                    || self.inner.backward(op, inputs, grad),
                    |grads| match grads {
                        Ok(g) => op.backward_cost(inputs, g),
                        Err(_) => 0.0,
                    },
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
    TopK,
    GatherColumns,
    ExpertMatmul,
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

    fn adamw8bit(
        &self,
        param: &mut Tensor,
        moments: &mut kapel_ops::Moments8,
        grad: &Tensor,
        maps: &Tensor,
        step: &kapel_ops::AdamWStep,
    ) -> Result<()> {
        self.time(
            || "adamw8bit".into(),
            || self.inner.adamw8bit(param, moments, grad, maps, step),
        )
    }
}

impl<B: kapel_ops::Recurrent> kapel_ops::Recurrent for Profiled<B> {
    fn causal_conv(
        &self,
        state: &mut Tensor,
        x: &Tensor,
        weight: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor> {
        self.time(
            || "causal_conv".into(),
            || self.inner.causal_conv(state, x, weight, slots),
        )
    }

    fn delta_rule(
        &self,
        state: &mut Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        slots: &Tensor,
        layout: kapel_ops::HeadLayout,
    ) -> Result<Tensor> {
        self.time(
            || "delta_rule".into(),
            || {
                self.inner
                    .delta_rule(state, q, k, v, g, beta, slots, layout)
            },
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
