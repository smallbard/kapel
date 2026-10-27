//! Tensor parallelism (Shoeybi et al., *Megatron-LM*, 2019): each block's
//! heads and the inner width of its MLP shared out between the ranks.
//!
//! For `t` ranks, rank `r` holds the `r`-th `1/t` of the query, key and
//! value projections' rows (its heads) and of the gate and up projections'
//! rows, and the same columns of the output and down projections. A block's
//! input is the same on every rank; each rank computes its heads' attention
//! and its part of the MLP, and the products of the output and down
//! projections are summed over the ranks ([`reduce`]: a sum forward, the
//! gradient as it is backward). Their input enters with [`enter`] (as it is
//! forward, the gradients summed backward), so that every rank receives the
//! whole gradient of the block's input. The output and down projections'
//! biases are added once, after the sum. The norms, the tables and the
//! output stay whole on every rank, and their gradients are the same on
//! every rank; those of the query and key norms (Qwen 3), which each rank
//! applies to its heads only, are summed ([`partial`]).
//!
//! The sums go through the host ([`crate::dist`]): this checks the numbers
//! of tensor parallelism on one machine; NCCL makes it fast on several
//! cards.

use std::sync::Arc;

use kapel_core::{DType, Error, Result, Tensor};
use kapel_models::{Block, Config};
use kapel_ops::Cast;

use crate::dist::Collective;
use crate::model::Parameters;
use crate::tape::{Compute, Grad, Var};

/// The ranks a block is shared out between.
#[derive(Clone)]
pub struct Group(pub Arc<dyn Collective>);

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rank {} of {}", self.0.rank(), self.0.size())
    }
}

fn failed(op: &'static str) -> impl Fn(crate::dist::DistError) -> Error {
    move |e| Error::Device {
        op,
        message: e.to_string(),
    }
}

/// The sum over the ranks of `t`, in `f32` on the host, back on `t`'s
/// device in its type.
fn summed<B: Compute>(backend: &B, group: &dyn Collective, t: &Tensor) -> Result<Tensor> {
    let host = backend.download(t)?;
    let mut values: Vec<f32> = host.to_f64_vec()?.into_iter().map(|x| x as f32).collect();
    group
        .all_reduce_sum(&mut values)
        .map_err(failed("all_reduce"))?;
    let sum = backend.upload(&Tensor::from_slice(&values, t.shape())?)?;
    if t.dtype() == DType::F32 {
        Ok(sum)
    } else {
        backend.forward(&Cast { dtype: t.dtype() }, &[&sum])
    }
}

/// `x` as it is, its gradient summed over the ranks: the input of the
/// projections shared out by rows.
///
/// # Errors
///
/// If another rank stops.
pub fn enter<B: Compute>(x: &Var<B>, group: &Group) -> Result<Var<B>> {
    let g = group.0.clone();
    Ok(x.tape().custom(&[x], x.value(), move || {
        move |backend: &B, grad: Tensor, _: &[bool]| -> Result<Vec<Option<Grad>>> {
            Ok(vec![Some(Grad::Dense(summed(backend, g.as_ref(), &grad)?))])
        }
    }))
}

/// The sum over the ranks of `x`, its gradient as it is: the output of the
/// projections shared out by columns.
///
/// # Errors
///
/// If another rank stops.
pub fn reduce<B: Compute>(x: &Var<B>, group: &Group) -> Result<Var<B>> {
    let sum = summed(x.tape().backend(), group.0.as_ref(), &x.value())?;
    Ok(x.tape().custom(&[x], sum, || {
        |_: &B, grad: Tensor, _: &[bool]| -> Result<Vec<Option<Grad>>> {
            Ok(vec![Some(Grad::Dense(grad))])
        }
    }))
}

/// Whether a parameter, whole on every rank, receives from each a part of
/// its gradient, to be summed: the query and key norms.
pub fn partial(name: &str) -> bool {
    name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight")
}

/// How a parameter is shared out: by rows (dimension 0), by columns
/// (dimension 1), or whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Each rank its rows (dimension 0): the projections of the heads and
    /// of the MLP's input.
    Rows,
    /// Each rank its columns (dimension 1): the output and down projections.
    Columns,
    /// Whole on every rank.
    Whole,
}

/// How the parameter `name` is shared out.
pub fn split(name: &str) -> Split {
    const ROWS: [&str; 5] = [
        "self_attn.q_proj.",
        "self_attn.k_proj.",
        "self_attn.v_proj.",
        "mlp.gate_proj.",
        "mlp.up_proj.",
    ];
    if ROWS.iter().any(|m| name.contains(m)) {
        return Split::Rows;
    }
    if (name.contains("self_attn.o_proj.") || name.contains("mlp.down_proj."))
        && name.ends_with(".weight")
    {
        return Split::Columns;
    }
    Split::Whole
}

/// The configuration a rank of `ranks` computes with: its heads and its part
/// of the MLP.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for blocks other than Llama's (and Mistral's,
/// Qwen's, Gemma 2's), or heads, key and value heads, or an inner width that
/// the ranks do not divide.
pub fn local_config(config: &Config, ranks: usize) -> Result<Config> {
    let refuse = |message: String| {
        Err(Error::InvalidArgument {
            op: "tensor parallelism",
            message,
        })
    };
    if !matches!(config.block, Block::Sequential | Block::Sandwich) {
        return refuse(format!("blocks of {:?} are not shared out", config.block));
    }
    for (what, n) in [
        ("heads", config.heads),
        ("key and value heads", config.kv_heads),
        ("the inner width", config.intermediate),
    ] {
        if n % ranks != 0 {
            return refuse(format!("{n} {what} for {ranks} ranks"));
        }
    }
    let mut local = config.clone();
    local.heads /= ranks;
    local.kv_heads /= ranks;
    local.intermediate /= ranks;
    Ok(local)
}

/// Rank `rank`'s part of `params` (whole ones as they are), on the host.
///
/// # Errors
///
/// If a shared parameter's dimension is not a multiple of the ranks.
pub fn shard(params: &Parameters, rank: usize, ranks: usize) -> Result<Parameters> {
    let mut out = Parameters::default();
    for (name, p) in params.iter() {
        let mut p = p.clone();
        let dim = match split(name) {
            Split::Rows => 0,
            Split::Columns => 1,
            Split::Whole => {
                out.insert(name, p);
                continue;
            }
        };
        let n = p.value.shape()[dim];
        if n % ranks != 0 {
            return Err(Error::InvalidArgument {
                op: "tensor parallelism",
                message: format!("{name}: {n} for {ranks} ranks"),
            });
        }
        let each = n / ranks;
        p.value = crate::model::owned(&p.value.narrow(dim, rank * each, each)?)?;
        out.insert(name, p);
    }
    Ok(out)
}

/// The whole parameter from every rank's part, in the order of the ranks:
/// the inverse of [`shard`] for one parameter.
///
/// # Errors
///
/// If the parts do not join.
pub fn join(name: &str, parts: &[Tensor]) -> Result<Tensor> {
    match split(name) {
        Split::Whole => parts.first().cloned().ok_or(Error::Arity {
            op: "join",
            expected: "a part",
            got: 0,
        }),
        split => {
            let dim = usize::from(split == Split::Columns);
            let refs: Vec<&Tensor> = parts.iter().collect();
            kapel_cpu::Cpu::new().forward(&kapel_ops::Concat { dim }, &refs)
        }
    }
}

use kapel_ops::Run as _;

/// Fully sharded weights (FSDP), each trained parameter whole on the rank
/// that owns it ([`crate::dist::owners`]), an empty tensor on the others.
/// The decoder [`fetch`]es a parameter where it uses it: inside a block
/// whose activations are computed again (`training.checkpoint_activations`),
/// a block's weights are whole on every rank only while its forward pass
/// runs, and while it runs again for the backward pass.
#[derive(Clone)]
pub struct Shards {
    /// The ranks.
    pub group: Group,
    /// The rank that owns each trained parameter.
    pub owners: Arc<std::collections::BTreeMap<String, usize>>,
    /// The shape of each trained parameter.
    pub shapes: Arc<std::collections::BTreeMap<String, Vec<usize>>>,
}

impl std::fmt::Debug for Shards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shards of {} parameters, {:?}",
            self.owners.len(),
            self.group
        )
    }
}

impl Shards {
    /// The parameter `name` whole, from `v`, this rank's leaf for it: `v`
    /// itself for a parameter not shared out.
    ///
    /// # Errors
    ///
    /// If another rank stops.
    pub fn fetch<B: Compute>(&self, name: &str, v: &Var<B>) -> Result<Var<B>> {
        match (self.owners.get(name), self.shapes.get(name)) {
            (Some(&owner), Some(shape)) => fetch(v, owner, shape, &self.group),
            _ => Ok(v.clone()),
        }
    }
}

/// The parameter `owner` holds, whole on every rank, from `v`: the owner's
/// leaf, an empty one on the others. Its gradient goes back to the owner,
/// every rank's summed in the order of the ranks; the others' leaves
/// receive an empty one.
///
/// # Errors
///
/// If another rank stops.
pub fn fetch<B: Compute>(
    v: &Var<B>,
    owner: usize,
    shape: &[usize],
    group: &Group,
) -> Result<Var<B>> {
    let g = group.0.clone();
    let backend = v.tape().backend().clone();
    let mine = if g.rank() == owner {
        let host = backend.download(&v.value())?;
        host.to_f64_vec()?
            .into_iter()
            .flat_map(|x| (x as f32).to_le_bytes())
            .collect()
    } else {
        Vec::new()
    };
    let all = g.exchange(&mine).map_err(failed("fetch"))?;
    let whole = backend.upload(&Tensor::from_bytes(DType::F32, shape, &all[owner])?)?;
    let empty = v.shape();
    Ok(v.tape().custom(&[v], whole, move || {
        move |backend: &B, grad: Tensor, _: &[bool]| -> Result<Vec<Option<Grad>>> {
            let host = backend.download(&grad)?;
            let mut values: Vec<f32> = host.to_f64_vec()?.into_iter().map(|x| x as f32).collect();
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            let all = g.exchange(&bytes).map_err(failed("fetch"))?;
            if g.rank() != owner {
                let none = Tensor::zeros(DType::F32, &empty)?;
                return Ok(vec![Some(Grad::Dense(backend.upload(&none)?))]);
            }
            for (k, bytes) in all.iter().enumerate() {
                for (x, b) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    let y = f32::from_le_bytes(*b);
                    *x = if k == 0 { y } else { *x + y };
                }
            }
            let sum = Tensor::from_slice(&values, grad.shape())?;
            Ok(vec![Some(Grad::Dense(backend.upload(&sum)?))])
        }
    }))
}
