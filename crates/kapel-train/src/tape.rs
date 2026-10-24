//! Reverse-mode differentiation.
//!
//! A [`Tape`] records each operation as it is computed, on any backend of
//! the catalogue. [`Tape::backward`] then reads the record backwards to give
//! the gradient of a scalar, the loss, with respect to every leaf.
//!
//! What the tape keeps is what the gradients need, and no more:
//!
//! - a value lives while a [`Var`] holds it, or while the derivative of an
//!   operation that read it needs it: a [`Var`] is a counted handle, and each
//!   operation keeps only the inputs its derivative reads (a sum keeps
//!   nothing, a product keeps both factors, a rotation only its positions);
//! - nothing is kept for a computation that no leaf reaches: constants and
//!   frozen weights cost no memory beyond their own;
//! - during the backward pass, each operation's saved tensors are let go as
//!   soon as its derivative has run;
//! - [`Tape::checkpoint`] keeps only the inputs of a region, and computes it
//!   again during the backward pass (activation checkpointing).
//!
//! The gradient of a slice reaches its source as a slice, joined to the
//! others without padding; that of rows gathered from a table reaches the
//! table as rows ([`Grad::Rows`]), dense only if the table is not a
//! [sparse leaf](Tape::sparse_leaf).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use kapel_core::{DType, Error, Result, Tensor};
use kapel_ops::{
    Affine, Attention, Backend, Binary, BinaryKind, Cast, Concat, Contiguous, GatherRows, Glu,
    LayerNorm, LogSoftmax, Matmul, Max, Mean, Operation, QuantizedMatmul, RmsNorm, Rope, Run,
    ScatterAddRows, Softmax, Sum, Transfer, Unary, UnaryKind,
};

/// A backend the tape can compute on: the whole catalogue, transfers, and
/// a cheap clone (each backend of Kapel shares its device when cloned).
pub trait Compute: Backend + Transfer + Clone + 'static {}

impl<B: Backend + Transfer + Clone + 'static> Compute for B {}

/// The gradient of a value.
#[derive(Clone, Debug)]
pub enum Grad {
    /// Of the value's shape.
    Dense(Tensor),
    /// Zero but for some rows of a table `[R, ...]`: `values[i]` is to be
    /// added to row `indices[i]`; a row may come several times. `indices`
    /// is a `u32` or `i32` tensor of any shape, `values` has its shape then
    /// that of a row.
    Rows {
        /// The rows, integers.
        indices: Tensor,
        /// What is added to them.
        values: Tensor,
    },
    /// Zero but along `dim` from `start`, where it is `values`: the gradient
    /// of a slice of the value. Never that of a leaf.
    Slice {
        /// The dimension of the slice.
        dim: usize,
        /// Its first index along it.
        start: usize,
        /// The gradient of the slice.
        values: Tensor,
    },
}

impl Grad {
    /// The gradient as a tensor of the value's shape `shape`.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    pub fn dense<B: Compute>(self, backend: &B, shape: &[usize]) -> Result<Tensor> {
        match self {
            Grad::Dense(g) => Ok(g),
            Grad::Rows { indices, values } => backend.forward(
                &ScatterAddRows {
                    rows: shape.first().copied().unwrap_or(0),
                },
                &[&values, &indices],
            ),
            Grad::Slice { dim, start, values } => pad(backend, &values, dim, start, shape),
        }
    }
}

/// `values` placed along `dim` from `start` in zeros of `shape`.
fn pad<B: Compute>(
    backend: &B,
    values: &Tensor,
    dim: usize,
    start: usize,
    shape: &[usize],
) -> Result<Tensor> {
    let len = values.shape()[dim];
    let after = shape[dim] - start - len;
    let mut parts = Vec::with_capacity(3);
    let zeros = |n: usize| {
        let mut s = shape.to_vec();
        s[dim] = n;
        backend.zeros(&s, values.dtype())
    };
    if start > 0 {
        parts.push(zeros(start)?);
    }
    parts.push(values.clone());
    if after > 0 {
        parts.push(zeros(after)?);
    }
    if parts.len() == 1 {
        return Ok(parts.pop().expect("one part"));
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    backend.forward(&Concat { dim }, &refs)
}

/// The derivative of an operation recorded on the tape: given the gradient
/// of its output, those of its inputs. It holds what it saved from the
/// forward pass, and runs once: [`Tape::backward`] lets it go after.
///
/// Implement it for an operation whose derivative is written by hand, and
/// record the operation with [`Tape::custom`].
pub trait Derivative<B: Compute> {
    /// The gradients of the inputs, in their order: `None` for one that
    /// receives none. `needed[i]` says whether input `i` needs one; a
    /// derivative may skip the others.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    fn backward(
        self: Box<Self>,
        backend: &B,
        grad: Tensor,
        needed: &[bool],
    ) -> Result<Vec<Option<Grad>>>;
}

/// Any closure of the right shape is a derivative.
impl<B: Compute, F> Derivative<B> for F
where
    F: FnOnce(&B, Tensor, &[bool]) -> Result<Vec<Option<Grad>>>,
{
    fn backward(
        self: Box<Self>,
        backend: &B,
        grad: Tensor,
        needed: &[bool],
    ) -> Result<Vec<Option<Grad>>> {
        (*self)(backend, grad, needed)
    }
}

enum Kind<B: Compute> {
    /// A value whose gradient is wanted; `sparse`, its gradient may stay
    /// rows.
    Leaf { sparse: bool },
    /// A value without gradient.
    Constant,
    /// The output of an operation.
    Op {
        inputs: Vec<usize>,
        derivative: Option<Box<dyn Derivative<B>>>,
    },
}

struct Node<B: Compute> {
    /// Kept while a handle holds it.
    value: Option<Tensor>,
    handles: usize,
    shape: Vec<usize>,
    dtype: DType,
    /// Whether a leaf reaches it.
    grad: bool,
    kind: Kind<B>,
}

struct Inner<B: Compute> {
    backend: B,
    nodes: RefCell<Vec<Node<B>>>,
}

/// The record of a computation, for one backward pass.
///
/// Nodes are added in the order they are computed: a node always comes
/// after its inputs, and the backward pass reads them from the last.
///
/// A clone shares the record.
pub struct Tape<B: Compute> {
    inner: Rc<Inner<B>>,
}

impl<B: Compute> Clone for Tape<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

/// A value on a tape: a counted handle. The value is let go when the last
/// handle is dropped, unless an operation saved it for its derivative.
pub struct Var<B: Compute> {
    tape: Rc<Inner<B>>,
    index: usize,
}

impl<B: Compute> Clone for Var<B> {
    fn clone(&self) -> Self {
        self.tape.nodes.borrow_mut()[self.index].handles += 1;
        Self {
            tape: Rc::clone(&self.tape),
            index: self.index,
        }
    }
}

impl<B: Compute> Drop for Var<B> {
    fn drop(&mut self) {
        // The tape is never borrowed while a handle drops; if it were, the
        // value would only be kept longer.
        if let Ok(mut nodes) = self.tape.nodes.try_borrow_mut() {
            let node = &mut nodes[self.index];
            node.handles -= 1;
            if node.handles == 0 {
                node.value = None;
            }
        }
    }
}

impl<B: Compute> std::fmt::Debug for Var<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nodes = self.tape.nodes.borrow();
        let node = &nodes[self.index];
        write!(
            f,
            "Var({}, {:?} {}{})",
            self.index,
            node.shape,
            node.dtype,
            if node.grad { ", grad" } else { "" }
        )
    }
}

/// The gradients given by [`Tape::backward`], one per leaf reached.
#[derive(Debug, Default)]
pub struct Gradients {
    grads: HashMap<usize, Grad>,
}

impl Gradients {
    /// The gradient of the loss with respect to the leaf `var`: `None` if
    /// `var` is not a leaf, or the loss does not depend on it.
    pub fn get<B: Compute>(&self, var: &Var<B>) -> Option<&Grad> {
        self.grads.get(&var.index)
    }

    /// The gradient of `var`, taken out.
    pub fn take<B: Compute>(&mut self, var: &Var<B>) -> Option<Grad> {
        self.grads.remove(&var.index)
    }

    /// The gradient of `var` as a tensor of its shape.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    pub fn dense<B: Compute>(&mut self, var: &Var<B>) -> Result<Option<Tensor>> {
        let shape = var.shape();
        self.take(var)
            .map(|g| g.dense(&var.tape.backend, &shape))
            .transpose()
    }

    /// The number of leaves with a gradient.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether no leaf has a gradient.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

impl<B: Compute> Tape<B> {
    /// An empty tape on `backend`.
    pub fn new(backend: B) -> Self {
        Self {
            inner: Rc::new(Inner {
                backend,
                nodes: RefCell::new(Vec::new()),
            }),
        }
    }

    /// The backend.
    pub fn backend(&self) -> &B {
        &self.inner.backend
    }

    /// The number of nodes recorded.
    pub fn len(&self) -> usize {
        self.inner.nodes.borrow().len()
    }

    /// Whether nothing is recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of nodes whose value is held by a handle. The tensors
    /// saved by derivatives are not counted: they are the derivatives'.
    pub fn held(&self) -> usize {
        self.inner
            .nodes
            .borrow()
            .iter()
            .filter(|n| n.value.is_some())
            .count()
    }

    fn push(&self, value: Tensor, grad: bool, kind: Kind<B>) -> Var<B> {
        let mut nodes = self.inner.nodes.borrow_mut();
        nodes.push(Node {
            shape: value.shape().to_vec(),
            dtype: value.dtype(),
            value: Some(value),
            handles: 1,
            grad,
            kind,
        });
        Var {
            tape: Rc::clone(&self.inner),
            index: nodes.len() - 1,
        }
    }

    /// A value whose gradient is wanted: a parameter.
    pub fn leaf(&self, value: Tensor) -> Var<B> {
        self.push(value, true, Kind::Leaf { sparse: false })
    }

    /// A leaf whose gradient may stay [`Grad::Rows`]: a table of which few
    /// rows are read (embeddings, tables of n-grams).
    pub fn sparse_leaf(&self, value: Tensor) -> Var<B> {
        self.push(value, true, Kind::Leaf { sparse: true })
    }

    /// A value without gradient: an input, a frozen weight.
    pub fn constant(&self, value: Tensor) -> Var<B> {
        self.push(value, false, Kind::Constant)
    }

    /// Records an operation computed outside the tape: its output `value`,
    /// its inputs, and its `derivative`, which is kept only if an input
    /// needs a gradient. The derivative of a fused operation written by
    /// hand goes in here.
    ///
    /// # Panics
    ///
    /// If an input is on another tape.
    pub fn custom<D: Derivative<B> + 'static>(
        &self,
        inputs: &[&Var<B>],
        value: Tensor,
        derivative: impl FnOnce() -> D,
    ) -> Var<B> {
        self.record(inputs, value, || Box::new(derivative()))
    }

    fn record(
        &self,
        inputs: &[&Var<B>],
        value: Tensor,
        derivative: impl FnOnce() -> Box<dyn Derivative<B>>,
    ) -> Var<B> {
        let grad = {
            let nodes = self.inner.nodes.borrow();
            inputs.iter().any(|x| {
                assert!(
                    Rc::ptr_eq(&x.tape, &self.inner),
                    "a variable of another tape"
                );
                nodes[x.index].grad
            })
        };
        if !grad {
            return self.push(value, false, Kind::Constant);
        }
        let kind = Kind::Op {
            inputs: inputs.iter().map(|x| x.index).collect(),
            derivative: Some(derivative()),
        };
        self.push(value, true, kind)
    }

    /// The gradients of the scalar `loss` with respect to the leaves.
    ///
    /// The tape is spent: the derivatives run once, and are let go.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if `loss` has more than one element; those
    /// of the backend.
    pub fn backward(&self, loss: &Var<B>) -> Result<Gradients> {
        if loss.numel() != 1 {
            return Err(Error::InvalidArgument {
                op: "backward",
                message: format!("the loss must be a scalar, got shape {:?}", loss.shape()),
            });
        }
        let one = Tensor::from_slice(&[1.0f32], &[1])?.reshape(&loss.shape())?;
        let one = self.backend().upload(&one)?;
        let one = cast(self.backend(), &one, loss.dtype())?;
        self.backward_with(loss, one)
    }

    /// The gradients with respect to the leaves of `Σ output · grad`, for a
    /// `grad` of `output`'s shape: the backward pass from a value that is
    /// not a scalar.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `grad` does not have `output`'s shape;
    /// those of the backend.
    pub fn backward_with(&self, output: &Var<B>, grad: Tensor) -> Result<Gradients> {
        if grad.shape() != output.shape() {
            return Err(Error::ShapeMismatch {
                op: "backward",
                expected: output.shape(),
                got: grad.shape().to_vec(),
            });
        }
        let backend = &self.inner.backend;
        let mut pending: HashMap<usize, Vec<Grad>> = HashMap::new();
        pending.insert(output.index, vec![Grad::Dense(grad)]);
        let mut gradients = Gradients::default();
        for index in (0..=output.index).rev() {
            let Some(parts) = pending.remove(&index) else {
                continue;
            };
            let (shape, dtype, wanted, leaf, op) = {
                let mut nodes = self.inner.nodes.borrow_mut();
                let node = &mut nodes[index];
                let (leaf, op) = match &mut node.kind {
                    Kind::Leaf { sparse } => (Some(*sparse), None),
                    Kind::Constant => (None, None),
                    Kind::Op { inputs, derivative } => {
                        (None, derivative.take().map(|d| (inputs.clone(), d)))
                    }
                };
                (node.shape.clone(), node.dtype, node.grad, leaf, op)
            };
            if !wanted {
                continue;
            }
            if let Some(sparse) = leaf {
                gradients
                    .grads
                    .insert(index, combine(backend, parts, &shape, dtype, sparse)?);
                continue;
            }
            let Some((inputs, derivative)) = op else {
                continue;
            };
            let g = combine(backend, parts, &shape, dtype, false)?;
            let Grad::Dense(g) = g else {
                unreachable!("combined into a dense gradient")
            };
            let needed: Vec<bool> = {
                let nodes = self.inner.nodes.borrow();
                inputs.iter().map(|&i| nodes[i].grad).collect()
            };
            let grads = derivative.backward(backend, g, &needed)?;
            for ((input, g), need) in inputs.into_iter().zip(grads).zip(needed) {
                if let (Some(g), true) = (g, need) {
                    pending.entry(input).or_default().push(g);
                }
            }
        }
        // The derivatives the pass did not reach go too.
        for node in self.inner.nodes.borrow_mut().iter_mut() {
            if let Kind::Op { derivative, .. } = &mut node.kind {
                *derivative = None;
            }
        }
        Ok(gradients)
    }

    /// Joins tensors along `dim`.
    ///
    /// # Errors
    ///
    /// Those of [`Concat`].
    pub fn concat(&self, xs: &[&Var<B>], dim: usize) -> Result<Var<B>> {
        let values: Vec<Tensor> = xs.iter().map(|x| x.value()).collect();
        let refs: Vec<&Tensor> = values.iter().collect();
        let y = self.backend().forward(&Concat { dim }, &refs)?;
        let sizes: Vec<usize> = values.iter().map(|v| v.shape()[dim]).collect();
        Ok(self.record(xs, y, || {
            Box::new(move |_: &B, g: Tensor, needed: &[bool]| {
                let mut start = 0;
                sizes
                    .iter()
                    .zip(needed)
                    .map(|(&n, &need)| {
                        let part = need.then(|| g.narrow(dim, start, n)).transpose();
                        start += n;
                        Ok(part?.map(Grad::Dense))
                    })
                    .collect()
            })
        }))
    }

    /// `f(inputs)`, computed without keeping anything for its derivative,
    /// and computed again during the backward pass to have it: activation
    /// checkpointing. Only the inputs are kept (the parameters a region
    /// reads are leaves, kept anyway).
    ///
    /// `f` receives the inputs on a tape of its own, and must give the same
    /// result each time: no randomness that is not drawn again the same.
    /// Its gradients with respect to the inputs are those of the region.
    ///
    /// # Errors
    ///
    /// Those of `f`.
    pub fn checkpoint<F>(&self, inputs: &[&Var<B>], f: F) -> Result<Var<B>>
    where
        F: Fn(&Tape<B>, &[Var<B>]) -> Result<Var<B>> + 'static,
    {
        let values: Vec<Tensor> = inputs.iter().map(|x| x.value()).collect();
        let y = {
            let inner = Tape::new(self.backend().clone());
            let xs: Vec<Var<B>> = values.iter().map(|v| inner.constant(v.clone())).collect();
            f(&inner, &xs)?.value()
        };
        let sparse: Vec<bool> = {
            let nodes = self.inner.nodes.borrow();
            inputs
                .iter()
                .map(|x| matches!(nodes[x.index].kind, Kind::Leaf { sparse: true }))
                .collect()
        };
        Ok(self.record(inputs, y, move || {
            Box::new(move |backend: &B, g: Tensor, needed: &[bool]| {
                let inner = Tape::new(backend.clone());
                let xs: Vec<Var<B>> = values
                    .into_iter()
                    .zip(needed)
                    .zip(&sparse)
                    .map(|((v, &need), &sparse)| match (need, sparse) {
                        (true, true) => inner.sparse_leaf(v),
                        (true, false) => inner.leaf(v),
                        _ => inner.constant(v),
                    })
                    .collect();
                let y = f(&inner, &xs)?;
                let mut grads = inner.backward_with(&y, g)?;
                Ok(xs.iter().map(|x| grads.take(x)).collect())
            })
        }))
    }
}

/// `x` in `dtype`, unchanged if it is already.
fn cast<B: Compute>(backend: &B, x: &Tensor, dtype: DType) -> Result<Tensor> {
    if x.dtype() == dtype {
        Ok(x.clone())
    } else {
        backend.forward(&Cast { dtype }, &[x])
    }
}

fn add<B: Compute>(backend: &B, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let b = cast(backend, b, a.dtype())?;
    backend.forward(
        &Binary {
            kind: BinaryKind::Add,
        },
        &[a, &b],
    )
}

/// The parts of a value's gradient, as one: rows stay rows for a sparse
/// leaf, slices that tile the value are joined, and the rest is added.
fn combine<B: Compute>(
    backend: &B,
    parts: Vec<Grad>,
    shape: &[usize],
    dtype: DType,
    sparse: bool,
) -> Result<Grad> {
    let (mut dense, mut rows, mut slices) = (Vec::new(), Vec::new(), Vec::new());
    for part in parts {
        match part {
            Grad::Dense(g) => dense.push(g),
            Grad::Rows { indices, values } => rows.push((indices, values)),
            Grad::Slice { dim, start, values } => slices.push((dim, start, values)),
        }
    }
    if sparse && dense.is_empty() && slices.is_empty() && !rows.is_empty() {
        return join_rows(backend, rows, shape);
    }
    // Slices along one dimension that do not overlap are joined, the holes
    // filled with zeros: the gradient of the pieces of a split.
    if !slices.is_empty() {
        let dim = slices[0].0;
        slices.sort_by_key(|s| s.1);
        let disjoint = slices.iter().all(|s| s.0 == dim)
            && slices
                .windows(2)
                .all(|w| w[0].1 + w[0].2.shape()[dim] <= w[1].1);
        if disjoint {
            let mut pieces = Vec::new();
            let mut at = 0;
            for (_, start, values) in slices {
                if start > at {
                    let mut s = shape.to_vec();
                    s[dim] = start - at;
                    pieces.push(backend.zeros(&s, values.dtype())?);
                }
                at = start + values.shape()[dim];
                pieces.push(values);
            }
            if at < shape[dim] {
                let mut s = shape.to_vec();
                s[dim] = shape[dim] - at;
                pieces.push(backend.zeros(&s, pieces[0].dtype())?);
            }
            let dtype = pieces[0].dtype();
            let pieces: Vec<Tensor> = pieces
                .iter()
                .map(|p| cast(backend, p, dtype))
                .collect::<Result<_>>()?;
            let refs: Vec<&Tensor> = pieces.iter().collect();
            dense.push(if refs.len() == 1 {
                refs[0].clone()
            } else {
                backend.forward(&Concat { dim }, &refs)?
            });
        } else {
            for (dim, start, values) in slices {
                dense.push(pad(backend, &values, dim, start, shape)?);
            }
        }
    }
    if !rows.is_empty() {
        let joined = join_rows(backend, rows, shape)?;
        dense.push(joined.dense(backend, shape)?);
    }
    let mut total: Option<Tensor> = None;
    for g in dense {
        total = Some(match total {
            None => g,
            Some(t) => add(backend, &t, &g)?,
        });
    }
    let total = total.expect("at least one part");
    let total = if total.shape() == shape {
        total
    } else {
        total.reshape(shape)?
    };
    let total = if total.is_contiguous() {
        total
    } else {
        backend.forward(&Contiguous, &[&total])?
    };
    Ok(Grad::Dense(cast(backend, &total, dtype)?))
}

/// Several lists of rows as one.
fn join_rows<B: Compute>(
    backend: &B,
    mut rows: Vec<(Tensor, Tensor)>,
    shape: &[usize],
) -> Result<Grad> {
    if rows.len() == 1 {
        let (indices, values) = rows.pop().expect("one");
        return Ok(Grad::Rows { indices, values });
    }
    let width: Vec<usize> = shape[1..].to_vec();
    let flat = |(i, v): (Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
        let n = i.numel();
        let mut s = vec![n];
        s.extend(&width);
        Ok((i.reshape(&[n])?, v.reshape(&s)?))
    };
    let flat: Vec<(Tensor, Tensor)> = rows.into_iter().map(flat).collect::<Result<_>>()?;
    let dtype = flat[0].1.dtype();
    let index_dtype = flat[0].0.dtype();
    let indices: Vec<Tensor> = flat
        .iter()
        .map(|(i, _)| cast(backend, i, index_dtype))
        .collect::<Result<_>>()?;
    let values: Vec<Tensor> = flat
        .iter()
        .map(|(_, v)| cast(backend, v, dtype))
        .collect::<Result<_>>()?;
    Ok(Grad::Rows {
        indices: backend.forward(&Concat { dim: 0 }, &indices.iter().collect::<Vec<_>>())?,
        values: backend.forward(&Concat { dim: 0 }, &values.iter().collect::<Vec<_>>())?,
    })
}

/// `g`, summed over the dimensions along which a value of `shape` was
/// broadcast to `g`'s shape.
fn unbroadcast<B: Compute>(backend: &B, g: Tensor, shape: &[usize]) -> Result<Tensor> {
    if g.shape() == shape {
        return Ok(g);
    }
    let lead = g.rank() - shape.len();
    let dims: Vec<usize> = (0..g.rank())
        .filter(|&d| d < lead || (shape[d - lead] == 1 && g.shape()[d] != 1))
        .collect();
    let summed = backend.forward(
        &Sum {
            dims,
            keepdim: true,
        },
        &[&g],
    )?;
    summed.reshape(shape)
}

/// `g` broadcast back over the reduced `dims` of `shape`, materialized.
fn expand<B: Compute>(
    backend: &B,
    g: &Tensor,
    shape: &[usize],
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor> {
    let kept: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(d, &n)| if dims.contains(&d) { 1 } else { n })
        .collect();
    let g = if keepdim {
        g.clone()
    } else {
        g.reshape(&kept)?
    };
    let g = if g.is_contiguous() {
        g
    } else {
        backend.forward(&Contiguous, &[&g])?
    };
    let g = g.reshape(&kept)?.broadcast_to(shape)?;
    backend.forward(&Contiguous, &[&g])
}

/// The derivative computed by the backend from the saved inputs.
fn by_backend<B: Compute + Run<O>, O: Operation + 'static>(
    op: O,
    saved: Vec<Tensor>,
) -> Box<dyn Derivative<B>> {
    Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
        let refs: Vec<&Tensor> = saved.iter().collect();
        Ok(backend
            .backward(&op, &refs, &g)?
            .into_iter()
            .map(|g| g.map(Grad::Dense))
            .collect())
    })
}

impl<B: Compute> Var<B> {
    /// The value.
    ///
    /// # Panics
    ///
    /// Never: a handle keeps its value.
    pub fn value(&self) -> Tensor {
        self.tape.nodes.borrow()[self.index]
            .value
            .clone()
            .expect("a handle keeps its value")
    }

    /// The shape of the value.
    pub fn shape(&self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.index].shape.clone()
    }

    /// The number of elements.
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// The type of the value.
    pub fn dtype(&self) -> DType {
        self.tape.nodes.borrow()[self.index].dtype
    }

    /// Whether a leaf reaches it: whether it will receive a gradient.
    pub fn requires_grad(&self) -> bool {
        self.tape.nodes.borrow()[self.index].grad
    }

    /// The tape it is on.
    pub fn tape(&self) -> Tape<B> {
        Tape {
            inner: Rc::clone(&self.tape),
        }
    }

    fn backend(&self) -> &B {
        &self.tape.backend
    }

    fn record(
        &self,
        inputs: &[&Var<B>],
        value: Tensor,
        derivative: impl FnOnce() -> Box<dyn Derivative<B>>,
    ) -> Var<B> {
        self.tape().record(inputs, value, derivative)
    }

    /// An operation of the catalogue whose derivative the backend computes
    /// from all its inputs, which are kept.
    ///
    /// # Errors
    ///
    /// Those of the operation.
    pub fn apply<O>(op: O, inputs: &[&Var<B>]) -> Result<Var<B>>
    where
        O: Operation + 'static,
        B: Run<O>,
    {
        let first = inputs.first().ok_or(Error::Arity {
            op: O::NAME,
            expected: "at least 1",
            got: 0,
        })?;
        let values: Vec<Tensor> = inputs.iter().map(|x| x.value()).collect();
        let refs: Vec<&Tensor> = values.iter().collect();
        let y = first.backend().forward(&op, &refs)?;
        Ok(first.record(inputs, y, || by_backend(op, values)))
    }

    // Element-wise.

    /// `f(x)` for each element.
    ///
    /// # Errors
    ///
    /// Those of [`Unary`].
    pub fn unary(&self, kind: UnaryKind) -> Result<Self> {
        Self::apply(Unary { kind }, &[self])
    }

    /// The gated unit `f(x[…, :n]) · x[…, n:]` ([`Glu`]).
    ///
    /// # Errors
    ///
    /// Those of [`Glu`].
    pub fn glu(&self, activation: UnaryKind) -> Result<Self> {
        Self::apply(Glu { activation }, &[self])
    }

    fn binary(&self, kind: BinaryKind, other: &Self) -> Result<Self> {
        match kind {
            BinaryKind::Add | BinaryKind::Sub => {
                let (a, b) = (self.value(), other.value());
                let y = self.backend().forward(&Binary { kind }, &[&a, &b])?;
                let shapes = (a.shape().to_vec(), b.shape().to_vec());
                drop((a, b));
                Ok(self.record(&[self, other], y, move || {
                    Box::new(move |backend: &B, g: Tensor, needed: &[bool]| {
                        let ga = needed[0]
                            .then(|| unbroadcast(backend, g.clone(), &shapes.0))
                            .transpose()?;
                        let gb = needed[1]
                            .then(|| {
                                let g = unbroadcast(backend, g.clone(), &shapes.1)?;
                                if kind == BinaryKind::Sub {
                                    backend.forward(
                                        &Affine {
                                            scale: -1.0,
                                            bias: 0.0,
                                        },
                                        &[&g],
                                    )
                                } else {
                                    Ok(g)
                                }
                            })
                            .transpose()?;
                        Ok(vec![ga.map(Grad::Dense), gb.map(Grad::Dense)])
                    })
                }))
            }
            _ => Self::apply(Binary { kind }, &[self, other]),
        }
    }

    /// `self + other`, broadcast.
    ///
    /// # Errors
    ///
    /// Those of [`Binary`].
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary(BinaryKind::Add, other)
    }

    /// `self − other`, broadcast.
    ///
    /// # Errors
    ///
    /// Those of [`Binary`].
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.binary(BinaryKind::Sub, other)
    }

    /// `self · other`, broadcast.
    ///
    /// # Errors
    ///
    /// Those of [`Binary`].
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.binary(BinaryKind::Mul, other)
    }

    /// `self / other`, broadcast.
    ///
    /// # Errors
    ///
    /// Those of [`Binary`].
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary(BinaryKind::Div, other)
    }

    /// `scale · x + bias`.
    ///
    /// # Errors
    ///
    /// Those of [`Affine`].
    pub fn affine(&self, scale: f64, bias: f64) -> Result<Self> {
        let y = self
            .backend()
            .forward(&Affine { scale, bias }, &[&self.value()])?;
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                let g = backend.forward(&Affine { scale, bias: 0.0 }, &[&g])?;
                Ok(vec![Some(Grad::Dense(g))])
            })
        }))
    }

    /// `scale · x`.
    ///
    /// # Errors
    ///
    /// Those of [`Affine`].
    pub fn scale(&self, scale: f64) -> Result<Self> {
        self.affine(scale, 0.0)
    }

    /// The value converted to `dtype`; the gradient comes back in the
    /// value's type.
    ///
    /// # Errors
    ///
    /// Those of [`Cast`].
    pub fn cast(&self, dtype: DType) -> Result<Self> {
        let from = self.dtype();
        if from == dtype {
            return Ok(self.clone());
        }
        let y = self.backend().forward(&Cast { dtype }, &[&self.value()])?;
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                Ok(vec![
                    from.is_float()
                        .then(|| cast(backend, &g, from))
                        .transpose()?
                        .map(Grad::Dense),
                ])
            })
        }))
    }

    // Shapes: views, without copies.

    fn view(
        &self,
        y: Tensor,
        back: impl FnOnce(&B, Tensor) -> Result<Grad> + 'static,
    ) -> Result<Self> {
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| Ok(vec![Some(back(backend, g)?)]))
        }))
    }

    /// The value with the elements in row-major order.
    ///
    /// # Errors
    ///
    /// Those of [`Contiguous`].
    pub fn contiguous(&self) -> Result<Self> {
        let x = self.value();
        if x.is_contiguous() {
            return Ok(self.clone());
        }
        let y = self.backend().forward(&Contiguous, &[&x])?;
        self.view(y, |_, g| Ok(Grad::Dense(g)))
    }

    /// The value with another shape of as many elements: a view if its
    /// layout allows, a copy otherwise.
    ///
    /// # Errors
    ///
    /// If the numbers of elements differ.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let x = self.value();
        let from = x.shape().to_vec();
        let y = match x.reshape(shape) {
            Ok(y) => y,
            Err(_) => self.backend().forward(&Contiguous, &[&x])?.reshape(shape)?,
        };
        self.view(y, move |backend, g| {
            let g = match g.reshape(&from) {
                Ok(g) => g,
                Err(_) => backend.forward(&Contiguous, &[&g])?.reshape(&from)?,
            };
            Ok(Grad::Dense(g))
        })
    }

    /// Dimensions `a` and `b` exchanged, as a view.
    ///
    /// # Errors
    ///
    /// If a dimension is out of range.
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self> {
        let y = self.value().transpose(a, b)?;
        self.view(y, move |_, g| Ok(Grad::Dense(g.transpose(a, b)?)))
    }

    /// The dimensions in the order `order`, as a view.
    ///
    /// # Errors
    ///
    /// If `order` is not a permutation of the dimensions.
    pub fn permute(&self, order: &[usize]) -> Result<Self> {
        let y = self.value().permute(order)?;
        let mut inverse = vec![0; order.len()];
        for (i, &o) in order.iter().enumerate() {
            inverse[o] = i;
        }
        self.view(y, move |_, g| Ok(Grad::Dense(g.permute(&inverse)?)))
    }

    /// `len` indices along `dim` from `start`, as a view.
    ///
    /// # Errors
    ///
    /// If the range is out of the dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let y = self.value().narrow(dim, start, len)?;
        self.view(y, move |_, values| Ok(Grad::Slice { dim, start, values }))
    }

    /// The value broadcast to `shape`, as a view.
    ///
    /// # Errors
    ///
    /// If the shapes are not compatible.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let x = self.value();
        let from = x.shape().to_vec();
        let y = x.broadcast_to(shape)?;
        self.view(y, move |backend, g| {
            Ok(Grad::Dense(unbroadcast(backend, g, &from)?))
        })
    }

    // Products.

    /// The batched matrix product ([`Matmul`]).
    ///
    /// # Errors
    ///
    /// Those of [`Matmul`].
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        Self::apply(Matmul, &[self, other])
    }

    /// `x · wᵀ (+ bias)` for a weight `w [out, in]` as `transformers` stores
    /// it.
    ///
    /// # Errors
    ///
    /// Those of [`Matmul`] and [`Binary`].
    pub fn linear(&self, w: &Self, bias: Option<&Self>) -> Result<Self> {
        let y = self.matmul(&w.transpose(0, 1)?)?;
        match bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }

    /// The product by frozen quantized weights `[N, K]` ([`QuantizedMatmul`]):
    /// only `x` receives a gradient.
    ///
    /// # Errors
    ///
    /// Those of [`QuantizedMatmul`].
    pub fn quantized_matmul(&self, weights: &[&Tensor]) -> Result<Self> {
        let x = self.value();
        let mut inputs = vec![&x];
        inputs.extend_from_slice(weights);
        let y = self.backend().forward(&QuantizedMatmul, &inputs)?;
        let weights: Vec<Tensor> = weights.iter().map(|&w| w.clone()).collect();
        let dtype = x.dtype();
        let shape = x.shape().to_vec();
        drop(x);
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                // The backend's derivative reads only the shape of `x`.
                let zero = backend.zeros(&[1], dtype)?;
                let x = zero.reshape(&vec![1; shape.len()])?.broadcast_to(&shape)?;
                let mut inputs = vec![&x];
                inputs.extend(weights.iter());
                let mut grads = backend.backward(&QuantizedMatmul, &inputs, &g)?;
                Ok(vec![grads.swap_remove(0).map(Grad::Dense)])
            })
        }))
    }

    // Reductions.

    /// The sum over `dims`.
    ///
    /// # Errors
    ///
    /// Those of [`Sum`].
    pub fn sum(&self, dims: &[usize], keepdim: bool) -> Result<Self> {
        let x = self.value();
        let shape = x.shape().to_vec();
        let dims = dims.to_vec();
        let y = self.backend().forward(
            &Sum {
                dims: dims.clone(),
                keepdim,
            },
            &[&x],
        )?;
        self.view(y, move |backend, g| {
            Ok(Grad::Dense(expand(backend, &g, &shape, &dims, keepdim)?))
        })
    }

    /// The sum of all elements, a tensor `[1]`.
    ///
    /// # Errors
    ///
    /// Those of [`Sum`].
    pub fn sum_all(&self) -> Result<Self> {
        let n = self.numel();
        self.reshape(&[n])?.sum(&[0], true)
    }

    /// The mean over `dims`.
    ///
    /// # Errors
    ///
    /// Those of [`Mean`].
    pub fn mean(&self, dims: &[usize], keepdim: bool) -> Result<Self> {
        let x = self.value();
        let shape = x.shape().to_vec();
        let dims = dims.to_vec();
        let count: usize = dims.iter().map(|&d| shape[d]).product();
        let y = self.backend().forward(
            &Mean {
                dims: dims.clone(),
                keepdim,
            },
            &[&x],
        )?;
        self.view(y, move |backend, g| {
            let g = backend.forward(
                &Affine {
                    scale: 1.0 / count as f64,
                    bias: 0.0,
                },
                &[&g],
            )?;
            Ok(Grad::Dense(expand(backend, &g, &shape, &dims, keepdim)?))
        })
    }

    /// The maximum along `dim` ([`Max`]).
    ///
    /// # Errors
    ///
    /// Those of [`Max`].
    pub fn max(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Self::apply(Max { dim, keepdim }, &[self])
    }

    // Normalizations.

    /// The softmax along `dim`.
    ///
    /// # Errors
    ///
    /// Those of [`Softmax`].
    pub fn softmax(&self, dim: usize) -> Result<Self> {
        Self::apply(Softmax { dim }, &[self])
    }

    /// The log-softmax along `dim`.
    ///
    /// # Errors
    ///
    /// Those of [`LogSoftmax`].
    pub fn log_softmax(&self, dim: usize) -> Result<Self> {
        Self::apply(LogSoftmax { dim }, &[self])
    }

    /// The layer normalization over the last dimension ([`LayerNorm`]).
    ///
    /// # Errors
    ///
    /// Those of [`LayerNorm`].
    pub fn layer_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f64) -> Result<Self> {
        let mut inputs = vec![self];
        inputs.extend(weight);
        inputs.extend(bias);
        Self::apply(LayerNorm { eps }, &inputs)
    }

    /// The RMS normalization over the last dimension ([`RmsNorm`]).
    ///
    /// # Errors
    ///
    /// Those of [`RmsNorm`].
    pub fn rms_norm(&self, weight: Option<&Self>, eps: f64, weight_offset: f64) -> Result<Self> {
        let mut inputs = vec![self];
        inputs.extend(weight);
        Self::apply(RmsNorm { eps, weight_offset }, &inputs)
    }

    // Attention and positions.

    /// Scaled dot-product attention ([`Attention`]): `self` is the queries
    /// `[B, Hq, Tq, D]`; the mask, if any, is a constant.
    ///
    /// # Errors
    ///
    /// Those of [`Attention`].
    pub fn attention(
        &self,
        k: &Self,
        v: &Self,
        mask: Option<&Tensor>,
        op: Attention,
    ) -> Result<Self> {
        let mut values = vec![self.value(), k.value(), v.value()];
        values.extend(mask.cloned());
        let refs: Vec<&Tensor> = values.iter().collect();
        let y = self.backend().forward(&op, &refs)?;
        drop(refs);
        Ok(self.record(&[self, k, v], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                let refs: Vec<&Tensor> = values.iter().collect();
                let mut grads = backend.backward(&op, &refs, &g)?;
                grads.truncate(3);
                Ok(grads.into_iter().map(|g| g.map(Grad::Dense)).collect())
            })
        }))
    }

    /// Rotary position embeddings ([`Rope`]), at the integer `positions`.
    /// Only the positions are kept: the derivative of a rotation does not
    /// read what it rotated.
    ///
    /// # Errors
    ///
    /// Those of [`Rope`].
    pub fn rope(&self, positions: &Tensor, op: Rope) -> Result<Self> {
        let x = self.value();
        let y = self.backend().forward(&op, &[&x, positions])?;
        let (shape, dtype) = (x.shape().to_vec(), x.dtype());
        drop(x);
        let positions = positions.clone();
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                // The backends read only the shape of `x`: a broadcast zero.
                let zero = backend.zeros(&[1], DType::F32)?;
                let x = zero.reshape(&vec![1; shape.len()])?.broadcast_to(&shape)?;
                let g = cast(backend, &g, DType::F32)?;
                let mut grads = backend.backward(&op, &[&x, &positions], &g)?;
                let g = grads.swap_remove(0).expect("x receives a gradient");
                Ok(vec![Some(Grad::Dense(cast(backend, &g, dtype)?))])
            })
        }))
    }

    // Rows.

    /// The rows `indices` of the table `self` ([`GatherRows`]). Only the
    /// indices are kept; the table's gradient comes as [`Grad::Rows`].
    ///
    /// # Errors
    ///
    /// Those of [`GatherRows`].
    pub fn gather_rows(&self, indices: &Tensor) -> Result<Self> {
        let y = self
            .backend()
            .forward(&GatherRows, &[&self.value(), indices])?;
        let indices = indices.clone();
        let dtype = self.dtype();
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                Ok(vec![Some(Grad::Rows {
                    indices,
                    values: cast(backend, &g, dtype)?,
                })])
            })
        }))
    }

    /// The adjoint of [`Var::gather_rows`] ([`ScatterAddRows`]).
    ///
    /// # Errors
    ///
    /// Those of [`ScatterAddRows`].
    pub fn scatter_add_rows(&self, indices: &Tensor, rows: usize) -> Result<Self> {
        let y = self
            .backend()
            .forward(&ScatterAddRows { rows }, &[&self.value(), indices])?;
        let indices = indices.clone();
        Ok(self.record(&[self], y, move || {
            Box::new(move |backend: &B, g: Tensor, _: &[bool]| {
                Ok(vec![Some(Grad::Dense(
                    backend.forward(&GatherRows, &[&g, &indices])?,
                ))])
            })
        }))
    }
}
