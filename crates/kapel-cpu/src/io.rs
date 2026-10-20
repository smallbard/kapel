//! Reading tensors as `f32` and writing results.
//!
//! An `f32` tensor in host memory is read in place, strides included. A tensor of
//! another float type is converted once into a contiguous `f32` copy. Results are
//! computed in `f32` and written in the output's type, into buffers taken from the
//! backend's pool.

use std::borrow::Cow;
use std::sync::Arc;

use kapel_core::{BufferPool, DType, Error, Layout, Result, Tensor};

/// The types the backend reads and writes for floating-point values.
pub(crate) fn supported(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::BF16 | DType::F16 => Ok(()),
        dtype => Err(Error::UnsupportedDType { op, dtype }),
    }
}

/// The host bytes of a tensor: an owned buffer or a mapped file.
pub(crate) fn host_bytes<'a>(op: &'static str, tensor: &'a Tensor) -> Result<&'a [u8]> {
    tensor.storage().host_bytes().ok_or(Error::WrongDevice {
        op,
        expected: kapel_core::Device::Cpu,
        got: tensor.device(),
    })
}

/// Host bytes seen as `f32`, if they are aligned for it. Pooled buffers always
/// are; a tensor of a mapped file may not be.
fn as_f32(bytes: &[u8]) -> Option<&[f32]> {
    // SAFETY: every bit pattern is a valid `f32`; `align_to` only returns a
    // middle part that is correctly aligned.
    let (head, values, _) = unsafe { bytes.align_to::<f32>() };
    head.is_empty().then_some(values)
}

/// Host bytes seen as `f32`, mutably.
pub(crate) fn as_f32_mut(bytes: &mut [u8]) -> &mut [f32] {
    // SAFETY: as in `as_f32`.
    let (head, values, _) = unsafe { bytes.align_to_mut::<f32>() };
    assert!(head.is_empty(), "host storage is 64-byte aligned");
    values
}

/// A tensor's values as `f32`, with the layout to read them through.
pub(crate) struct Strided<'a> {
    pub(crate) data: Cow<'a, [f32]>,
    pub(crate) layout: Layout,
}

/// Reads a float tensor: in place for `f32`, converted otherwise.
pub(crate) fn strided<'a>(op: &'static str, tensor: &'a Tensor) -> Result<Strided<'a>> {
    supported(op, tensor.dtype())?;
    let bytes = host_bytes(op, tensor)?;
    if let (DType::F32, Some(values)) = (tensor.dtype(), as_f32(bytes)) {
        return Ok(Strided {
            data: Cow::Borrowed(values),
            layout: tensor.layout().clone(),
        });
    }
    Ok(Strided {
        data: Cow::Owned(convert(tensor)?),
        layout: Layout::contiguous(tensor.shape())?,
    })
}

/// A float tensor's values as `f32`, in row-major order: borrowed when the
/// tensor is a contiguous `f32` one.
pub(crate) fn contiguous<'a>(op: &'static str, tensor: &'a Tensor) -> Result<Cow<'a, [f32]>> {
    supported(op, tensor.dtype())?;
    let bytes = host_bytes(op, tensor)?;
    let layout = tensor.layout();
    if let (DType::F32, Some(values)) = (tensor.dtype(), as_f32(bytes)) {
        if layout.is_contiguous() {
            let start = layout.offset();
            return Ok(Cow::Borrowed(&values[start..start + layout.numel()]));
        }
        return Ok(Cow::Owned(layout.offsets().map(|at| values[at]).collect()));
    }
    convert(tensor).map(Cow::Owned)
}

/// The values of an `f32` (unaligned), `bf16` or `f16` tensor, in row-major order.
fn convert(tensor: &Tensor) -> Result<Vec<f32>> {
    let bytes = host_bytes("convert", tensor)?;
    let layout = tensor.layout();
    let read = |at: usize| u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]]);
    Ok(match tensor.dtype() {
        DType::F32 => layout
            .offsets()
            .map(|at| {
                f32::from_le_bytes([
                    bytes[4 * at],
                    bytes[4 * at + 1],
                    bytes[4 * at + 2],
                    bytes[4 * at + 3],
                ])
            })
            .collect(),
        DType::BF16 => layout
            .offsets()
            .map(|at| f32::from_bits(u32::from(read(at)) << 16))
            .collect(),
        DType::F16 => layout
            .offsets()
            .map(|at| kapel_core::F16(read(at)).to_f64() as f32)
            .collect(),
        dtype => {
            return Err(Error::UnsupportedDType {
                op: "convert",
                dtype,
            });
        }
    })
}

/// Index values (`u32` or `i32`), each checked against `size`.
pub(crate) fn indices(op: &'static str, tensor: &Tensor, size: usize) -> Result<Vec<usize>> {
    let bytes = host_bytes(op, tensor)?;
    let signed = tensor.dtype() == DType::I32;
    tensor
        .layout()
        .offsets()
        .map(|at| {
            let raw = [
                bytes[4 * at],
                bytes[4 * at + 1],
                bytes[4 * at + 2],
                bytes[4 * at + 3],
            ];
            let index = if signed {
                i64::from(i32::from_le_bytes(raw))
            } else {
                i64::from(u32::from_le_bytes(raw))
            };
            usize::try_from(index)
                .ok()
                .filter(|&i| i < size)
                .ok_or(Error::IndexOutOfRange { op, index, size })
        })
        .collect()
}

/// `f32` to `bf16`, rounded to nearest even; NaN stays NaN.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Writes `f32` results as a tensor of `dtype` and `shape`, in a pooled buffer.
pub(crate) fn write(
    pool: &Arc<BufferPool>,
    values: &[f32],
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let layout = Layout::contiguous(shape)?;
    debug_assert_eq!(layout.numel(), values.len());
    let mut storage = pool.take(dtype.storage_bytes(values.len())?);
    match dtype {
        DType::F32 => as_f32_mut(storage.bytes_mut()).copy_from_slice(values),
        DType::BF16 => {
            for (out, &v) in storage
                .bytes_mut()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(values)
            {
                *out = to_bf16(v).to_le_bytes();
            }
        }
        DType::F16 => {
            for (out, &v) in storage
                .bytes_mut()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(values)
            {
                *out = kapel_core::F16::from_f64(f64::from(v)).0.to_le_bytes();
            }
        }
        dtype => return Err(Error::UnsupportedDType { op: "write", dtype }),
    }
    Tensor::from_storage(Arc::new(storage), layout, dtype)
}

/// For each element of `to`, in row-major order, the position of the element of
/// a tensor laid out as `from` that it reads: `from` is broadcast to `to`.
pub(crate) fn broadcast_offsets(from: &Layout, to: &[usize]) -> Result<Vec<usize>> {
    Ok(from.broadcast_to(to)?.offsets().collect())
}

/// A tensor of `dtype` and `shape` whose `f32` values `compute` writes, every
/// one of them. For `f32`, `compute` writes straight into a pooled buffer, whose
/// previous contents it must overwrite; for another type, into a temporary that
/// is then converted.
pub(crate) fn output(
    pool: &Arc<BufferPool>,
    shape: &[usize],
    dtype: DType,
    compute: impl FnOnce(&mut [f32]),
) -> Result<Tensor> {
    let layout = Layout::contiguous(shape)?;
    if dtype == DType::F32 {
        let mut storage = pool.take(dtype.storage_bytes(layout.numel())?);
        compute(as_f32_mut(storage.bytes_mut()));
        return Tensor::from_storage(Arc::new(storage), layout, dtype);
    }
    let mut values = vec![0.0; layout.numel()];
    compute(&mut values);
    write(pool, &values, shape, dtype)
}
