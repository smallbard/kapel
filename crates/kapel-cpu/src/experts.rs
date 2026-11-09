//! Mixtures of experts: the choice of each token's experts, and the products
//! by them, each expert's tokens gathered and multiplied together by the
//! products of the backend.

use kapel_core::{DType, Result, Tensor};
use kapel_ops::{ExpertMatmul, GatherColumns, Matmul, Operation, QuantizedMatmul, Run, TopK};

use crate::backend::Cpu;
use crate::io;

impl Run<TopK> for Cpu {
    fn forward(&self, op: &TopK, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(TopK::NAME, inputs[0])?;
        let e = *inputs[0].shape().last().expect("checked by infer");
        let mut out = Vec::with_capacity(signature.shape.iter().product());
        if e > 0 {
            for row in x.chunks_exact(e) {
                out.extend(TopK::choose(op.k, row));
            }
        }
        Tensor::from_slice(&out, &signature.shape)
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

impl Run<GatherColumns> for Cpu {
    fn forward(&self, op: &GatherColumns, inputs: &[&Tensor]) -> Result<Tensor> {
        let signature = op.infer(inputs)?;
        let x = io::contiguous(GatherColumns::NAME, inputs[0])?;
        let e = *inputs[0].shape().last().expect("checked by infer");
        let idx = io::indices(GatherColumns::NAME, inputs[1], e)?;
        let k = *signature.shape.last().expect("a last dimension");
        let values: Vec<f32> = idx
            .iter()
            .enumerate()
            .map(|(i, &c)| x[i / k.max(1) * e + c])
            .collect();
        io::write(&self.buffers, &values, &signature.shape, signature.dtype)
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

impl Run<ExpertMatmul> for Cpu {
    /// Each expert's rows gathered (in the order of the tokens) and
    /// multiplied by its weights in one product.
    fn forward(&self, op: &ExpertMatmul, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = ExpertMatmul::NAME;
        let signature = op.infer(inputs)?;
        let (x, w) = (inputs[0], inputs[1]);
        let idx = io::indices(name, inputs[2], usize::MAX)?;
        op.check(&idx)?;
        let [t, k, n] = [signature.shape[0], signature.shape[1], signature.shape[2]];
        let width = w.shape()[1];
        let per_route = x.rank() == 3;
        let values = io::contiguous(name, x)?;
        // The products read the weights' type: float weights take their
        // inputs in it.
        let kind = if w.dtype().is_quantized() {
            DType::F32
        } else {
            w.dtype()
        };
        let mut out = vec![0.0f32; t * k * n];
        let mut routes: Vec<Vec<usize>> = vec![Vec::new(); op.experts];
        for (route, &expert) in idx.iter().enumerate() {
            routes[expert].push(route);
        }
        for (expert, routes) in routes.iter().enumerate() {
            if routes.is_empty() || n == 0 {
                continue;
            }
            let mut rows = Vec::with_capacity(routes.len() * width);
            for &route in routes {
                let row = if per_route { route } else { route / k };
                rows.extend_from_slice(&values[row * width..][..width]);
            }
            let rows = io::write(&self.buffers, &rows, &[routes.len(), width], kind)?;
            let weights = w.narrow(0, expert * n, n)?;
            let y = if w.dtype().is_quantized() {
                self.forward(&QuantizedMatmul, &[&rows, &weights])?
            } else {
                self.forward(&Matmul, &[&rows, &weights.transpose(0, 1)?])?
            };
            let y = io::contiguous(name, &y)?;
            for (i, &route) in routes.iter().enumerate() {
                out[route * n..][..n].copy_from_slice(&y[i * n..][..n]);
            }
        }
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
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
