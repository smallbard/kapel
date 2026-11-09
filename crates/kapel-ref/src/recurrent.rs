//! Linear attention's states, in `f64`: each row of a pass in order, as the
//! definitions say.

use kapel_core::DType;
use kapel_core::{Result, Tensor};
use kapel_ops::{
    CAUSAL_CONV, Cast, DELTA_RULE, Recurrent, Run, check_conv, check_delta, check_slots,
};

use crate::Reference;
use crate::array::Array;
use crate::paged::{index_values, write_host_rows};

/// The value `v` once stored in the type of `state`.
fn stored(state: &Tensor, v: f64) -> Result<f64> {
    let dtype = state.dtype();
    let mut bytes = vec![0; dtype.block_bytes()];
    dtype.encode(v, &mut bytes)?;
    dtype.decode(&bytes)
}

/// `y`, computed in `f64`, in the type the operation gives.
fn typed(y: &Array, dtype: DType) -> Result<Tensor> {
    let y = y.tensor()?;
    if dtype == DType::F64 {
        return Ok(y);
    }
    Reference.forward(&Cast { dtype }, &[&y])
}

/// Writes `values`, the whole table, into `state`.
fn write_all(state: &mut Tensor, values: &[f64]) -> Result<()> {
    let rows: Vec<usize> = (0..state.shape()[0]).collect();
    write_host_rows(state, &rows, values)
}

impl Recurrent for Reference {
    fn causal_conv(
        &self,
        state: &mut Tensor,
        x: &Tensor,
        weight: &Tensor,
        slots: &Tensor,
    ) -> Result<Tensor> {
        let (shape, dtype) = check_conv(state, x, weight, slots)?;
        let slots = index_values(CAUSAL_CONV, slots)?;
        let n = state.shape()[0];
        check_slots(CAUSAL_CONV, &slots, n)?;
        let (c, taps) = (weight.shape()[0], weight.shape()[1]);
        let x = Array::of(x)?;
        let w = Array::of(weight)?;
        let mut table = state.to_f64_vec()?;
        let kept = taps - 1;
        let mut y = Array::zeros(&shape);
        for (r, &s) in slots.iter().enumerate() {
            let rows = &mut table[s * kept * c..][..kept * c];
            let input = &x.data[r * c..][..c];
            for ch in 0..c {
                let mut sum = 0.0;
                for j in 0..kept {
                    sum += w.data[ch * taps + j] * rows[j * c + ch];
                }
                sum += w.data[ch * taps + kept] * input[ch];
                y.data[r * c + ch] = sum;
            }
            if kept > 0 {
                rows.copy_within(c.., 0);
                for (ch, &v) in input.iter().enumerate() {
                    rows[(kept - 1) * c + ch] = stored(state, v)?;
                }
            }
        }
        write_all(state, &table)?;
        typed(&y, dtype)
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
    ) -> Result<Tensor> {
        let (shape, dtype) = check_delta(state, q, k, v, g, beta, slots)?;
        let slots = index_values(DELTA_RULE, slots)?;
        let n = state.shape()[0];
        check_slots(DELTA_RULE, &slots, n)?;
        let (hk, dk) = (q.shape()[1], q.shape()[2]);
        let (h, dv) = (v.shape()[1], v.shape()[2]);
        let group = h / hk;
        let (q, k, v, g, beta) = (
            Array::of(q)?,
            Array::of(k)?,
            Array::of(v)?,
            Array::of(g)?,
            Array::of(beta)?,
        );
        let mut table = state.to_f64_vec()?;
        let mut o = Array::zeros(&shape);
        let mut u = vec![0.0; dv];
        for (r, &s) in slots.iter().enumerate() {
            for head in 0..h {
                let kh = head / group;
                let qr = &q.data[(r * hk + kh) * dk..][..dk];
                let kr = &k.data[(r * hk + kh) * dk..][..dk];
                let vr = &v.data[(r * h + head) * dv..][..dv];
                let (decay, b) = (g.data[r * h + head].exp(), beta.data[r * h + head]);
                let m = &mut table[(s * h + head) * dk * dv..][..dk * dv];
                for x in m.iter_mut() {
                    *x *= decay;
                }
                u.fill(0.0);
                for (i, &ki) in kr.iter().enumerate() {
                    for (uj, &sij) in u.iter_mut().zip(&m[i * dv..][..dv]) {
                        *uj += sij * ki;
                    }
                }
                for (i, &ki) in kr.iter().enumerate() {
                    for (j, sij) in m[i * dv..][..dv].iter_mut().enumerate() {
                        *sij += ki * (vr[j] - u[j]) * b;
                    }
                }
                let out = &mut o.data[(r * h + head) * dv..][..dv];
                for (i, &qi) in qr.iter().enumerate() {
                    for (oj, &sij) in out.iter_mut().zip(&m[i * dv..][..dv]) {
                        *oj += sij * qi;
                    }
                }
            }
        }
        write_all(state, &table)?;
        typed(&o, dtype)
    }
}
