//! Products by quantized weights, and rows of quantized tables, on the device
//! (`qgemv`, `dequantize`, `dequantize_rows` in `kernels/`).
//!
//! With few rows (decoding), the product reads the codes in place, one warp
//! per weight row. With many, the weights are dequantized by pieces of rows
//! into the activations' type, and multiplied by the tiled product. The
//! formats without kernels (Q2_K, Q3_K) go through the host.

use kapel_core::{DType, Error, Layout, QuantFormat, Result, Tensor};
use kapel_cuda_abi::quant;
use kapel_ops::QuantizedMatmul;

use crate::device::{Args, Cuda, float_code, small, view_of};

/// The most rows of activations of the product that reads the codes in place.
const GEMV_ROWS: usize = 8;

/// The most bytes of one piece of dequantized weights, for the products with
/// many rows.
const PIECE_BYTES: usize = 32 << 20;

/// The kernels' code and group of a quantized type, if they decode it.
pub(crate) fn code(dtype: DType) -> Option<(u32, u32)> {
    let DType::Quantized(format) = dtype else {
        return None;
    };
    let code = match format {
        QuantFormat::Int8 { group } if group % 8 == 0 => (quant::INT8, u32::from(group)),
        QuantFormat::Int4 { group } if group % 8 == 0 => (quant::INT4, u32::from(group)),
        QuantFormat::Q8_0 => (quant::Q8_0, 0),
        QuantFormat::Q4_0 => (quant::Q4_0, 0),
        QuantFormat::Q4_1 => (quant::Q4_1, 0),
        QuantFormat::Q5_0 => (quant::Q5_0, 0),
        QuantFormat::Q5_1 => (quant::Q5_1, 0),
        QuantFormat::Q4K => (quant::Q4K, 0),
        QuantFormat::Q5K => (quant::Q5K, 0),
        QuantFormat::Q6K => (quant::Q6K, 0),
        _ => return None,
    };
    Some(code)
}

/// A quantized tensor of this backend as the kernels take it: its code,
/// group, device address and bytes, from its first block.
struct Codes {
    code: u32,
    group: u32,
    pointer: u64,
    bytes: usize,
}

impl Cuda {
    fn codes(&self, op: &'static str, w: &Tensor) -> Result<Option<Codes>> {
        let Some((code, group)) = code(w.dtype()) else {
            return Ok(None);
        };
        let storage = self.storage(op, w)?;
        let start = w.dtype().storage_bytes(w.layout().offset())?;
        let bytes = w.dtype().storage_bytes(w.numel())?;
        Ok(Some(Codes {
            code,
            group,
            pointer: storage.pointer() + start as u64,
            bytes,
        }))
    }

    /// `x [..., K] · wᵢᵀ`, the weights stacked, on the device when every
    /// weight's format has kernels; `None` otherwise.
    pub(crate) fn quantized_product(
        &self,
        x: &Tensor,
        weights: &[&Tensor],
        shape: &[usize],
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let op = <QuantizedMatmul as kapel_ops::Operation>::NAME;
        let mut codes = Vec::with_capacity(weights.len());
        for w in weights {
            match self.codes(op, w)? {
                Some(c) => codes.push(c),
                None => return Ok(None),
            }
        }
        let k = weights[0].shape()[1];
        let width = *shape.last().expect("an output dimension");
        let rows: usize = x.shape()[..x.rank() - 1].iter().product();
        if k == 0 || rows == 0 || width == 0 {
            // A sum over nothing.
            return Ok(Some(self.output_as(shape, dtype, true, |_, _| Ok(()))?));
        }
        let x = self.contiguous_x(op, x)?.reshape(&[rows, k])?;
        let xa = self.float_arg(op, &x, x.layout())?;
        let ko = float_code(op, dtype)?;
        debug_assert_eq!(dtype, x.dtype(), "the product has the activations' type");
        let half = match dtype {
            DType::BF16 => Some(kapel_cuda_abi::half::BF16),
            DType::F16 => Some(kapel_cuda_abi::half::F16),
            _ => None,
        };
        // The formats with kernels of their own, by name.
        let fast = |c: &Codes| -> Option<&'static str> {
            let name = match (c.code, rows) {
                (quant::INT8, 1) => "qgemv_int8_1",
                (quant::INT8, _) => "qgemv_int8",
                (quant::INT4, 1) => "qgemv_int4_1",
                (quant::INT4, _) => "qgemv_int4",
                (quant::Q8_0, 1) => "qgemv_q8_0_1",
                (quant::Q8_0, _) => "qgemv_q8_0",
                (quant::Q4K, 1) => "qgemv_q4k_1",
                (quant::Q4K, _) => "qgemv_q4k",
                (quant::Q6K, 1) => "qgemv_q6k_1",
                (quant::Q6K, _) => "qgemv_q6k",
                _ => return None,
            };
            c.pointer.is_multiple_of(4).then_some(name)
        };
        if rows <= GEMV_ROWS
            && let Some(kind) = half
            && codes.iter().all(|c| fast(c).is_some())
            && k.is_multiple_of(8)
            && xa.base % 16 == 0
        {
            // Aligned loads, eight weights at a time, the activations in
            // 128-bit loads; a kernel per format, and per one row or more.
            let out = self.output_as(&[rows, width], dtype, false, |out, n| {
                let mut c0 = 0;
                for (w, c) in weights.iter().zip(&codes) {
                    let columns = w.shape()[0];
                    let kernel = fast(c).expect("checked above");
                    let args = Args::new()
                        .u32(c.group)
                        .u32(small(op, k)?)
                        .slice(c.pointer, c.bytes)
                        .u32(kind)
                        .slice(xa.base, xa.bytes / 4);
                    let args = if rows == 1 {
                        args
                    } else {
                        args.u32(small(op, rows)?)
                    };
                    self.launch(
                        kernel,
                        columns.checked_mul(32).ok_or(Error::Overflow { op })?,
                        args.u32(small(op, c0)?)
                            .u32(small(op, width)?)
                            .slice(out, n),
                    )?;
                    c0 += columns;
                }
                Ok(())
            })?;
            return Ok(Some(out.reshape(shape)?));
        }
        if rows <= GEMV_ROWS {
            let out = self.output_as(&[rows, width], dtype, false, |out, n| {
                let mut c0 = 0;
                for (w, c) in weights.iter().zip(&codes) {
                    let columns = w.shape()[0];
                    self.launch(
                        "qgemv",
                        columns.checked_mul(32).ok_or(Error::Overflow { op })?,
                        Args::new()
                            .u32(c.code)
                            .u32(c.group)
                            .u32(small(op, k)?)
                            .slice(c.pointer, c.bytes)
                            .u32(xa.code)
                            .slice(xa.base, xa.bytes)
                            .u32(small(op, rows)?)
                            .u32(small(op, c0)?)
                            .u32(small(op, width)?)
                            .u32(ko)
                            .slice(out, dtype.storage_bytes(n)?),
                    )?;
                    c0 += columns;
                }
                Ok(())
            })?;
            return Ok(Some(out.reshape(shape)?));
        }
        // Many rows: pieces of the weights dequantized to the activations'
        // type, each multiplied by the tiled product into its columns.
        let piece_type = x.dtype();
        let row_bytes = piece_type.storage_bytes(k)?;
        let piece_rows = (PIECE_BYTES / row_bytes.max(1)).max(64);
        let out = self.output_as(&[rows, width], dtype, false, |_, _| Ok(()))?;
        let out_layout = Layout::contiguous(&[rows, width])?;
        let mut c0 = 0;
        for (w, c) in weights.iter().zip(&codes) {
            let columns = w.shape()[0];
            let mut first = 0;
            while first < columns {
                let count = piece_rows.min(columns - first);
                let piece = self.output_as(&[count, k], piece_type, false, |p, n| {
                    self.launch(
                        "dequantize",
                        n / 8,
                        Args::new()
                            .u32(c.code)
                            .u32(c.group)
                            .u32(small(op, k)?)
                            .slice(c.pointer, c.bytes)
                            .u32(small(op, first)?)
                            .u32(float_code(op, piece_type)?)
                            .slice(p, piece_type.storage_bytes(n)?),
                    )
                })?;
                // The output has the activations' type: so has the product.
                let part = self.matmul(&x, &piece.transpose(0, 1)?, &[rows, count])?;
                self.place(op, &part, &out, &out_layout.narrow(1, c0 + first, count)?)?;
                first += count;
            }
            c0 += columns;
        }
        Ok(Some(out.reshape(shape)?))
    }

    /// `x` row-major from the start of its storage, in its own type.
    fn contiguous_x(&self, op: &'static str, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && x.layout().offset() == 0 {
            self.storage(op, x)?;
            return Ok(x.clone());
        }
        self.materialize(op, x, x.layout())
    }

    /// Copies `part` into the region `to` (a view of `out`'s storage) of
    /// `out`, a tensor this backend just made: bit for bit, `part` of `out`'s
    /// type.
    pub(crate) fn place(
        &self,
        op: &'static str,
        part: &Tensor,
        out: &Tensor,
        to: &Layout,
    ) -> Result<()> {
        let kernel = match part.dtype().element_size() {
            Some(2) => "copy_into_16",
            Some(4) => "copy_into_32",
            _ => "copy_into_64",
        };
        let from = self.any_arg(op, part, part.layout())?;
        let storage = self.storage(op, out)?;
        let elements = out.numel();
        let count = part.numel();
        self.launch(
            kernel,
            count,
            Args::new()
                .strided(&from)
                .slice(storage.pointer(), elements)
                .view(view_of(op, to)?)
                .u32(small(op, count)?),
        )
    }

    /// Rows `rows` of a quantized table, dequantized to `f32` on the device;
    /// `None` for a format without kernels.
    pub(crate) fn quantized_rows(
        &self,
        table: &Tensor,
        index: &Tensor,
        rows: &[usize],
        shape: &[usize],
    ) -> Result<Option<Tensor>> {
        let op = "gather_rows";
        let Some(c) = self.codes(op, table)? else {
            return Ok(None);
        };
        let k = table.shape()[1];
        let (list, _keep) = self.row_list(op, index, rows)?;
        let out = self.output_as(shape, DType::F32, false, |out, n| {
            self.launch(
                "dequantize_rows",
                n / 8,
                Args::new()
                    .u32(c.code)
                    .u32(c.group)
                    .u32(small(op, k)?)
                    .slice(c.pointer, c.bytes)
                    .slice(list, rows.len())
                    .u32(kapel_cuda_abi::float::F32)
                    .slice(out, n * 4),
            )
        })?;
        Ok(Some(out))
    }
}
