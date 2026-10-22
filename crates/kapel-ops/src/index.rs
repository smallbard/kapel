//! Gathering and scattering rows: embeddings and their gradients.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, indices, same_device};

/// Rows of a table: `src [R, ...]` and integer `idx` of any shape give
/// `y [idx.shape..., ...]` with `y[i, ...] = src[idx[i], ...]`. An embedding
/// lookup is a gather of the embedding table by token ids.
///
/// The table may be quantized, its rows a whole number of blocks: the rows are
/// then given at their exact dequantized values, in `f32`. This is how a model
/// read from GGUF looks up its quantized embeddings.
///
/// The gradient of `src` adds each row of the output gradient to the row it came
/// from; a quantized table, frozen, and `idx` receive none. An index outside
/// `0..R` is an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherRows;

impl Operation for GatherRows {
    const NAME: &'static str = "gather_rows";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
        same_device(Self::NAME, inputs)?;
        let (src, idx) = (inputs[0], inputs[1]);
        if !src.dtype().is_quantized() {
            float(Self::NAME, src)?;
        }
        indices(Self::NAME, idx)?;
        if src.rank() == 0 {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: "the table needs at least one dimension".into(),
            });
        }
        let width: usize = src.shape()[1..].iter().product();
        if !width.is_multiple_of(src.dtype().block_len()) {
            return Err(Error::BlockMisaligned {
                dtype: src.dtype(),
                elements: width,
            });
        }
        let mut shape = idx.shape().to_vec();
        shape.extend_from_slice(&src.shape()[1..]);
        Ok(Signature {
            shape,
            dtype: if src.dtype().is_quantized() {
                DType::F32
            } else {
                src.dtype()
            },
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}

/// The adjoint of [`GatherRows`]: `src [idx.shape..., ...]` and integer `idx`
/// give `y [rows, ...]`, zeros except `y[idx[i], ...] += src[i, ...]`. Rows that
/// appear several times are summed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScatterAddRows {
    /// The number of rows of the output.
    pub rows: usize,
}

impl Operation for ScatterAddRows {
    const NAME: &'static str = "scatter_add_rows";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        arity(Self::NAME, inputs, &[2], "2")?;
        same_device(Self::NAME, inputs)?;
        let (src, idx) = (inputs[0], inputs[1]);
        float(Self::NAME, src)?;
        indices(Self::NAME, idx)?;
        if !src.shape().starts_with(idx.shape()) {
            return Err(Error::InvalidArgument {
                op: Self::NAME,
                message: format!(
                    "the source shape {:?} must start with the index shape {:?}",
                    src.shape(),
                    idx.shape()
                ),
            });
        }
        let mut shape = vec![self.rows];
        shape.extend_from_slice(&src.shape()[idx.rank()..]);
        Ok(Signature {
            shape,
            dtype: src.dtype(),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}
