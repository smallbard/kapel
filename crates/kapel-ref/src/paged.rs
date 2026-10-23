//! Attention over a cache kept by pages, and rows written in place.

use kapel_core::{DType, Error, HostStorage, Result, Tensor};
use kapel_ops::{
    Operation, PagedAttention, Record, Recording, Run, WRITE_ROWS, Write, check_copy, check_write,
    check_written,
};

use crate::Reference;
use crate::array::Array;

/// Index values, read at their exact value; they are non-negative integers
/// (`u32`, or `i32` checked here).
pub(crate) fn index_values(op: &'static str, tensor: &Tensor) -> Result<Vec<usize>> {
    tensor
        .to_f64_vec()?
        .into_iter()
        .map(|v| {
            usize::try_from(v as i64).map_err(|_| Error::IndexOutOfRange {
                op,
                index: v as i64,
                size: 0,
            })
        })
        .collect()
}

impl Run<PagedAttention> for Reference {
    fn forward(&self, op: &PagedAttention, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let name = PagedAttention::NAME;
        let (q, keys, values) = (
            Array::of(inputs[0])?,
            Array::of(inputs[1])?,
            Array::of(inputs[2])?,
        );
        let pages = index_values(name, inputs[3])?;
        let rows = index_values(name, inputs[4])?;
        let (sequences, per_sequence) = (inputs[3].shape()[0], inputs[3].shape()[1]);
        let [lines, hkv, d] = [keys.shape[0], keys.shape[1], keys.shape[2]];
        op.check_indices(&pages, sequences, &rows, lines)?;
        let (hq, dv) = (q.shape[1], values.shape[2]);
        let group = hq / hkv;
        let scale = op.scale_for(d);
        let line =
            |s: usize, j: usize| pages[s * per_sequence + j / op.page] * op.page + j % op.page;
        let mut y = Array::zeros(&signature.shape);
        for (r, row) in rows.as_chunks::<2>().0.iter().enumerate() {
            let (s, count) = (row[0], row[1]);
            let first = op.first_visible(count);
            for h in 0..hq {
                let kv = h / group;
                let query = &q.data[(r * hq + h) * d..][..d];
                let scores: Vec<f64> = (first..count)
                    .map(|j| {
                        let key = &keys.data[(line(s, j) * hkv + kv) * d..][..d];
                        let dot: f64 = query.iter().zip(key).map(|(a, b)| a * b).sum();
                        let scaled = scale * dot;
                        match op.softcap {
                            Some(cap) => cap * (scaled / cap).tanh(),
                            None => scaled,
                        }
                    })
                    .collect();
                if scores.is_empty() {
                    continue;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                let out = &mut y.data[(r * hq + h) * dv..][..dv];
                for (w, j) in weights.iter().zip(first..count) {
                    let value = &values.data[(line(s, j) * hkv + kv) * dv..][..dv];
                    for (o, v) in out.iter_mut().zip(value) {
                        *o += w / total * v;
                    }
                }
            }
        }
        y.tensor()
    }

    fn backward(
        &self,
        op: &PagedAttention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None; inputs.len()])
    }
}

impl Write for Reference {
    fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        Tensor::zeros(dtype, shape)
    }

    fn write_rows(&self, table: &mut Tensor, index: &Tensor, rows: &Tensor) -> Result<()> {
        check_write(table, index, rows)?;
        let lines = table.shape()[0];
        let written = index_values(WRITE_ROWS, index)?;
        check_written(&written, lines)?;
        write_host_rows(table, &written, &rows.to_f64_vec()?)
    }

    fn copy_from_host(&self, target: &mut Tensor, values: &Tensor) -> Result<()> {
        copy_host(target, values)
    }
}

/// [`Write::copy_from_host`] for host tensors.
pub(crate) fn copy_host(target: &mut Tensor, values: &Tensor) -> Result<()> {
    let bytes = check_copy(target, values)?;
    let storage = target
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| Error::InvalidArgument {
            op: kapel_ops::COPY_FROM_HOST,
            message: "the target is not in host memory that can be written".into(),
        })?;
    storage.bytes_mut()[..bytes.len()].copy_from_slice(&bytes);
    Ok(())
}

impl Record for Reference {
    /// Computes as it goes: nothing to record.
    fn records(&self) -> bool {
        false
    }

    fn record(&self, _pass: &mut dyn FnMut() -> Result<Tensor>) -> Result<Option<Recording>> {
        Ok(None)
    }
}

/// Writes `values`, row by row, into the rows `written` of the host tensor
/// `table`, each value encoded in the table's type. The caller has checked
/// the call ([`check_write`], [`check_written`]).
pub(crate) fn write_host_rows(table: &mut Tensor, written: &[usize], values: &[f64]) -> Result<()> {
    let dtype = table.dtype();
    let width: usize = table.shape()[1..].iter().product();
    let size = dtype.block_bytes();
    let wrong = || Error::InvalidArgument {
        op: WRITE_ROWS,
        message: "the table is not in host memory that can be written".into(),
    };
    let storage = table
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(wrong)?;
    let bytes = storage.bytes_mut();
    for (&r, row) in written.iter().zip(values.chunks_exact(width.max(1))) {
        for (c, &value) in row.iter().enumerate() {
            let at = (r * width + c) * size;
            dtype.encode(value, &mut bytes[at..at + size])?;
        }
    }
    Ok(())
}
