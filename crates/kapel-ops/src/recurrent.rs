//! The states of linear attention, carried from one token to the next and
//! written in place: a causal convolution's last inputs, and the matrix of the
//! gated delta rule (Gated DeltaNet: Yang, Kautz and Hatamizadeh, 2024).
//!
//! Both read a state per sequence from a table of states, `[N, ...]`, the
//! row of each token given by `slots [R]`: the tokens of a pass, several
//! sequences together, each sequence's in the order of its positions. A row
//! reads the state its sequence's previous row left. Like the caches of
//! [`Write`](crate::Write), the tables are written only through their only
//! handle.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{float, indices, rank, same_device};

/// Linear attention's states, written in place.
pub trait Recurrent {
    /// A depthwise causal convolution of `K` taps over the rows of `x [R,
    /// C]`, the last `K − 1` inputs of each sequence kept in `state [N, K −
    /// 1, C]`. For each row `r` in order, with `s = slots[r]` and the window
    /// `w = (state[s, 0], …, state[s, K − 2], x[r])`:
    ///
    /// - `y[r, c] = Σⱼ weight[c, j] · w[j][c]`, `j` from 0 to `K − 1`;
    /// - `state[s] ← (w[1], …, w[K − 1])`, in the state's type.
    ///
    /// No bias, no activation. `weight [C, K]`; any float types, the output
    /// in that of `x`, the sums in `f32` at least. See [`check_conv`].
    ///
    /// # Errors
    ///
    /// Those of [`check_conv`]; [`Error::IndexOutOfRange`] for a slot out of
    /// range.
    fn causal_conv(
        &self,
        state: &mut Tensor,
        x: &Tensor,
        weight: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor>;

    /// The gated delta rule over the rows of a pass, each sequence's state
    /// `S [Dk, Dv]` per value head kept in `state [N, H, Dk, Dv]`, in `f32`.
    /// For each row `r` in order, with `s = slots[r]`, and each value head
    /// `h`, which reads key head `h / (H / Hk)`:
    ///
    /// - `S ← exp(g[r, h]) · S`;
    /// - `u = Sᵀ k`, `δ = (v[r, h] − u) · beta[r, h]`, `S ← S + k δᵀ`;
    /// - `o[r, h] = Sᵀ q`,
    ///
    /// with `q = q[r, h / (H / Hk)]` and `k = k[r, h / (H / Hk)]`, which the
    /// caller has normalized and scaled. `q, k [R, Hk, Dk]`, `v [R, H, Dv]`,
    /// `g, beta [R, H]`: any float types, computed in `f32` at least, the
    /// output `o [R, H, Dv]` in the type of `v`. See [`check_delta`].
    ///
    /// # Errors
    ///
    /// Those of [`check_delta`]; [`Error::IndexOutOfRange`] for a slot out of
    /// range.
    #[allow(clippy::too_many_arguments)]
    fn delta_rule(
        &self,
        state: &mut Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor>;
}

/// The name of [`Recurrent::causal_conv`] in errors.
pub const CAUSAL_CONV: &str = "causal_conv";

/// The name of [`Recurrent::delta_rule`] in errors.
pub const DELTA_RULE: &str = "delta_rule";

/// Checks that `state` is a table that can be written in place: row-major
/// from the start of its storage, and its only handle.
fn writable(op: &'static str, state: &mut Tensor) -> Result<()> {
    if !state.is_contiguous() || state.layout().offset() != 0 {
        return Err(Error::InvalidArgument {
            op,
            message: "the state must be row-major from the start of its storage".into(),
        });
    }
    if state.storage_mut().is_none() {
        return Err(Error::InvalidArgument {
            op,
            message: "the state is shared: writing in place needs its only handle".into(),
        });
    }
    Ok(())
}

fn shape(op: &'static str, what: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(Error::InvalidArgument {
            op,
            message: format!(
                "{what} must have shape {expected:?}, got {:?}",
                tensor.shape()
            ),
        })
    }
}

/// Checks a call of [`Recurrent::causal_conv`]: `state [N, K − 1, C]`,
/// `x [R, C]`, `weight [C, K]` of float types with `K ≥ 1`, `slots [R]` of
/// indices, all on one device, and `state` writable in place. The output's
/// shape and type.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a shape that does not fit or a state that
/// cannot be written, [`Error::UnsupportedDType`], [`Error::DeviceMismatch`].
pub fn check_conv(
    state: &mut Tensor,
    x: &Tensor,
    weight: &Tensor,
    slots: &Tensor,
) -> Result<(Vec<usize>, DType)> {
    let op = CAUSAL_CONV;
    same_device(op, &[state, x, weight, slots])?;
    for t in [&*state, x, weight] {
        float(op, t)?;
    }
    indices(op, slots)?;
    rank(op, x, 2, "x")?;
    rank(op, weight, 2, "the weight")?;
    rank(op, state, 3, "the state")?;
    rank(op, slots, 1, "the slots")?;
    let (r, c) = (x.shape()[0], x.shape()[1]);
    let taps = weight.shape()[1];
    if taps == 0 {
        return Err(Error::InvalidArgument {
            op,
            message: "a convolution of no tap".into(),
        });
    }
    shape(op, "the weight", weight, &[c, taps])?;
    shape(op, "the state", state, &[state.shape()[0], taps - 1, c])?;
    shape(op, "the slots", slots, &[r])?;
    writable(op, state)?;
    Ok((vec![r, c], x.dtype()))
}

/// Checks a call of [`Recurrent::delta_rule`]: `state [N, H, Dk, Dv]` in
/// `f32`, `q` and `k [R, Hk, Dk]`, `v [R, H, Dv]`, `g` and `beta [R, H]` of
/// float types with `H` a multiple of `Hk`, `slots [R]` of indices, all on
/// one device, and `state` writable in place. The output's shape and type.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a shape that does not fit or a state that
/// cannot be written, [`Error::UnsupportedDType`] (a state not in `f32`
/// among them), [`Error::DeviceMismatch`].
pub fn check_delta(
    state: &mut Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    slots: &Tensor,
) -> Result<(Vec<usize>, DType)> {
    let op = DELTA_RULE;
    same_device(op, &[state, q, k, v, g, beta, slots])?;
    for t in [q, k, v, g, beta] {
        float(op, t)?;
    }
    if state.dtype() != DType::F32 {
        return Err(Error::UnsupportedDType {
            op,
            dtype: state.dtype(),
        });
    }
    indices(op, slots)?;
    rank(op, q, 3, "q")?;
    rank(op, v, 3, "v")?;
    rank(op, state, 4, "the state")?;
    let (r, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (h, dv) = (v.shape()[1], v.shape()[2]);
    if hk == 0 || h % hk != 0 {
        return Err(Error::InvalidArgument {
            op,
            message: format!("{h} value heads are not a multiple of {hk} key heads"),
        });
    }
    shape(op, "k", k, &[r, hk, dk])?;
    shape(op, "v", v, &[r, h, dv])?;
    shape(op, "g", g, &[r, h])?;
    shape(op, "beta", beta, &[r, h])?;
    shape(op, "the slots", slots, &[r])?;
    shape(op, "the state", state, &[state.shape()[0], h, dk, dv])?;
    writable(op, state)?;
    Ok((vec![r, h, dv], v.dtype()))
}

/// Checks the slots a recurrent operation reads, as a backend reads them:
/// each below `n`, the rows of the table of states.
///
/// # Errors
///
/// [`Error::IndexOutOfRange`] for the first one out of range.
pub fn check_slots(op: &'static str, slots: &[usize], n: usize) -> Result<()> {
    match slots.iter().find(|&&s| s >= n) {
        Some(&s) => Err(Error::IndexOutOfRange {
            op,
            index: s as i64,
            size: n,
        }),
        None => Ok(()),
    }
}
