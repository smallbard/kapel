//! Mixtures of experts on the card: the choice of the experts and the
//! products by them, without reading the choice back to the host, so that a
//! pass of a model with experts can be recorded.

use kapel_core::{DType, Error, Result, Tensor};
use kapel_ops::{ExpertMatmul, GatherColumns, Operation, Run, TopK};

use crate::device::{Args, Cuda, float_code, small};

impl Cuda {
    /// The device address of the `u32` values of an index tensor, row-major:
    /// in place when it holds them so (as `TopK` makes them), otherwise
    /// converted on the host; with the storage that keeps them alive. Index
    /// values uploaded from the host are checked by `check`; those the card
    /// made, which reading back would wait for, are not: the kernels give
    /// NaN for one out of range rather than read beyond their tables.
    fn device_indices(
        &self,
        op: &'static str,
        idx: &Tensor,
        check: impl Fn(&[usize]) -> Result<()>,
    ) -> Result<(u64, std::sync::Arc<dyn kapel_core::Storage>)> {
        let storage = self.storage(op, idx)?;
        if let Some(bytes) = storage.host() {
            let host = Tensor::from_storage(
                std::sync::Arc::new(kapel_core::HostStorage::from_bytes(bytes)),
                idx.layout().clone(),
                idx.dtype(),
            )?;
            let values: Vec<usize> = host
                .to_f64_vec()?
                .into_iter()
                .map(|v| v.max(0.0) as usize)
                .collect();
            check(&values)?;
        }
        if idx.is_contiguous() && idx.dtype() == DType::U32 {
            let pointer = storage.pointer() + idx.layout().offset() as u64 * 4;
            return Ok((pointer, std::sync::Arc::clone(idx.storage())));
        }
        let values = self.indices(op, idx, usize::MAX)?;
        let (pointer, keep) = self.row_list(op, idx, &values)?;
        Ok((pointer, keep))
    }
}

impl Run<TopK> for Cuda {
    /// One thread per row.
    fn forward(&self, op: &TopK, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = TopK::NAME;
        let signature = op.infer(inputs)?;
        let x = inputs[0];
        let e = *x.shape().last().expect("checked by infer");
        let rows = x.numel().checked_div(e).unwrap_or(0);
        let xa = self.float_arg(name, x, x.layout())?;
        self.output_as(&signature.shape, DType::U32, false, |out, n| {
            self.launch(
                "top_k",
                rows,
                Args::new()
                    .float(&xa)
                    .u32(small(name, rows)?)
                    .u32(small(name, e)?)
                    .u32(small(name, op.k)?)
                    .slice(out, n),
            )
        })
    }

    fn backward(
        &self,
        op: &TopK,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None])
    }
}

impl Run<GatherColumns> for Cuda {
    /// One thread per output.
    fn forward(&self, op: &GatherColumns, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = GatherColumns::NAME;
        let signature = op.infer(inputs)?;
        let (x, idx) = (inputs[0], inputs[1]);
        let e = *x.shape().last().expect("checked by infer");
        let k = *signature.shape.last().expect("a last dimension");
        let xa = self.float_arg(name, x, x.layout())?;
        let (list, _keep) = self.device_indices(name, idx, |v| GatherColumns::check(v, e))?;
        let count = idx.numel();
        let dtype = signature.dtype;
        self.output_as(&signature.shape, dtype, false, |out, n| {
            self.launch(
                "gather_columns",
                count,
                Args::new()
                    .float(&xa)
                    .slice(list, count)
                    .u32(small(name, e)?)
                    .u32(small(name, k)?)
                    .u32(float_code(name, dtype)?)
                    .slice(out, dtype.storage_bytes(n)?),
            )
        })
    }

    fn backward(
        &self,
        op: &GatherColumns,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None, None])
    }
}

impl Run<ExpertMatmul> for Cuda {
    /// One warp per route and row of its expert's weights.
    fn forward(&self, op: &ExpertMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = ExpertMatmul::NAME;
        let signature = op.infer(inputs)?;
        let (x, w, idx) = (inputs[0], inputs[1], inputs[2]);
        let [t, k, n] = [signature.shape[0], signature.shape[1], signature.shape[2]];
        let width = w.shape()[1];
        let xa = self.float_arg(name, x, x.layout())?;
        let (list, _keep) = self.device_indices(name, idx, |v| op.check(v))?;
        let (format, group, kw, pointer, bytes, view) = if w.dtype().is_quantized() {
            let Some(codes) = self.codes(name, w)? else {
                return Err(Error::UnsupportedDType {
                    op: name,
                    dtype: w.dtype(),
                });
            };
            let flat = self.float_arg(name, x, x.layout())?.view;
            (codes.code, codes.group, 0, codes.pointer, codes.bytes, flat)
        } else {
            let wa = self.float_arg(name, w, w.layout())?;
            (u32::MAX, 0, wa.code, wa.base, wa.bytes, wa.view)
        };
        let dtype = signature.dtype;
        let routes = t * k;
        self.output_as(&signature.shape, dtype, false, |out, count| {
            self.launch(
                "expert_gemv",
                routes
                    .checked_mul(n)
                    .and_then(|v| v.checked_mul(32))
                    .ok_or(Error::Overflow { op: name })?,
                Args::new()
                    .u32(format)
                    .u32(group)
                    .u32(kw)
                    .slice(pointer, bytes)
                    .view(view)
                    .float(&xa)
                    .slice(list, routes)
                    .u32(small(name, k)?)
                    .u32(u32::from(x.rank() == 3))
                    .u32(small(name, width)?)
                    .u32(small(name, n)?)
                    .u32(small(name, op.experts)?)
                    .u32(float_code(name, dtype)?)
                    .slice(out, dtype.storage_bytes(count)?),
            )
        })
    }

    fn backward(
        &self,
        op: &ExpertMatmul,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None, None, None])
    }
}
