//! Tensors: a storage, a layout and a data type.

use std::fmt;
use std::sync::Arc;

use crate::dtype::dequantize;
use crate::{DType, Device, Element, Error, HostStorage, Layout, Result, Storage};

/// A tensor: shared storage, seen through a layout, with a data type.
///
/// Cloning a tensor and taking views (transpose, slice, broadcast, reshape) share
/// the storage: no data is copied. A view of a quantized tensor is not allowed,
/// since its elements are stored in blocks; dequantize it first.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<dyn Storage>,
    layout: Layout,
    dtype: DType,
}

impl Tensor {
    /// A tensor over an existing storage.
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if the layout reaches beyond the storage,
    /// [`Error::NotContiguous`] for a strided quantized layout,
    /// [`Error::BlockMisaligned`] if a quantized layout does not start or end on a
    /// block.
    pub fn from_storage(storage: Arc<dyn Storage>, layout: Layout, dtype: DType) -> Result<Self> {
        if dtype.is_quantized() {
            if !layout.is_contiguous() {
                return Err(Error::NotContiguous {
                    op: "Tensor::from_storage",
                });
            }
            if !layout.offset().is_multiple_of(dtype.block_len()) {
                return Err(Error::BlockMisaligned {
                    dtype,
                    elements: layout.offset(),
                });
            }
        }
        let needed = dtype.storage_bytes(layout.storage_extent())?;
        if needed > storage.len() {
            return Err(Error::BufferSize {
                op: "Tensor::from_storage",
                expected: needed,
                got: storage.len(),
            });
        }
        Ok(Self {
            storage,
            layout,
            dtype,
        })
    }

    /// A host tensor holding a copy of `data`, with `shape`.
    ///
    /// # Errors
    ///
    /// [`Error::ElementCountMismatch`] if `data` does not have one value per
    /// element of `shape`.
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize]) -> Result<Self> {
        let layout = Layout::contiguous(shape)?;
        if layout.numel() != data.len() {
            return Err(Error::ElementCountMismatch {
                op: "Tensor::from_slice",
                expected: data.len(),
                got: layout.numel(),
            });
        }
        let size = T::DTYPE.block_bytes();
        let mut storage = HostStorage::new(data.len() * size);
        for (value, out) in data.iter().zip(storage.bytes_mut().chunks_exact_mut(size)) {
            value.write_le(out);
        }
        Self::from_storage(Arc::new(storage), layout, T::DTYPE)
    }

    /// A host tensor of `dtype` and `shape` from its raw little-endian bytes; for
    /// a quantized type, whole blocks.
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if `bytes` is not the size `shape` and `dtype` need.
    pub fn from_bytes(dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<Self> {
        let layout = Layout::contiguous(shape)?;
        let expected = dtype.storage_bytes(layout.numel())?;
        if bytes.len() != expected {
            return Err(Error::BufferSize {
                op: "Tensor::from_bytes",
                expected,
                got: bytes.len(),
            });
        }
        Self::from_storage(Arc::new(HostStorage::from_bytes(bytes)), layout, dtype)
    }

    /// A host tensor of zeros.
    ///
    /// # Errors
    ///
    /// [`Error::BlockMisaligned`] for a quantized type whose blocks do not divide
    /// the number of elements.
    pub fn zeros(dtype: DType, shape: &[usize]) -> Result<Self> {
        let layout = Layout::contiguous(shape)?;
        let bytes = dtype.storage_bytes(layout.numel())?;
        Self::from_storage(Arc::new(HostStorage::new(bytes)), layout, dtype)
    }

    /// The size of each dimension.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// The strides, in elements.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// The layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device that holds the data.
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// The number of elements.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Whether the elements are stored in row-major order, without gaps.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// The storage, shared with every view of it.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// The storage, to write into in place, if this tensor is its only
    /// handle: no clone nor view of it is alive, so that nothing else sees
    /// the change. `None` otherwise.
    pub fn storage_mut(&mut self) -> Option<&mut dyn Storage> {
        Arc::get_mut(&mut self.storage)
    }

    fn view(&self, op: &'static str, layout: Result<Layout>) -> Result<Self> {
        if self.dtype.is_quantized() {
            return Err(Error::UnsupportedDType {
                op,
                dtype: self.dtype,
            });
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: layout?,
            dtype: self.dtype,
        })
    }

    /// Swaps two dimensions, without copying.
    ///
    /// # Errors
    ///
    /// See [`Layout::transpose`]; [`Error::UnsupportedDType`] for a quantized tensor.
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self> {
        self.view("transpose", self.layout.transpose(a, b))
    }

    /// Reorders the dimensions, without copying.
    ///
    /// # Errors
    ///
    /// See [`Layout::permute`]; [`Error::UnsupportedDType`] for a quantized tensor.
    pub fn permute(&self, order: &[usize]) -> Result<Self> {
        self.view("permute", self.layout.permute(order))
    }

    /// Keeps `len` elements of dimension `dim` from `start`, without copying.
    ///
    /// # Errors
    ///
    /// See [`Layout::narrow`]; [`Error::UnsupportedDType`] for a quantized tensor.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        self.view("narrow", self.layout.narrow(dim, start, len))
    }

    /// Repeats the tensor to `shape`, without copying.
    ///
    /// # Errors
    ///
    /// See [`Layout::broadcast_to`]; [`Error::UnsupportedDType`] for a quantized
    /// tensor.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        self.view("broadcast_to", self.layout.broadcast_to(shape))
    }

    /// The same elements with another shape, without copying. A strided view must
    /// be made contiguous first. A quantized tensor can be reshaped as long as its
    /// blocks are kept whole, which a contiguous reshape always does.
    ///
    /// # Errors
    ///
    /// See [`Layout::reshape`].
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.reshape(shape)?,
            dtype: self.dtype,
        })
    }

    fn host(&self, op: &'static str) -> Result<&[u8]> {
        self.storage.host_bytes().ok_or(Error::WrongDevice {
            op,
            expected: Device::Cpu,
            got: self.device(),
        })
    }

    /// The elements, in row-major order, as `T`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongDevice`] if the tensor is not in host memory,
    /// [`Error::DTypeMismatch`] if `T` is not the tensor's data type.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if T::DTYPE != self.dtype {
            return Err(Error::DTypeMismatch {
                op: "to_vec",
                expected: self.dtype,
                got: T::DTYPE,
            });
        }
        let bytes = self.host("to_vec")?;
        let size = self.dtype.block_bytes();
        Ok(self
            .layout
            .offsets()
            .map(|at| T::read_le(&bytes[at * size..(at + 1) * size]))
            .collect())
    }

    /// The exact value of every element, in row-major order, as `f64`: the
    /// reference view of any tensor, quantized ones included.
    ///
    /// # Errors
    ///
    /// [`Error::WrongDevice`] if the tensor is not in host memory;
    /// [`Error::UnsupportedDType`] for a quantized format without a reference
    /// decoder.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        let bytes = self.host("to_f64_vec")?;
        if self.dtype.is_quantized() {
            let start = self.dtype.storage_bytes(self.layout.offset())?;
            let len = self.dtype.storage_bytes(self.numel())?;
            return dequantize(self.dtype, &bytes[start..start + len]);
        }
        let size = self.dtype.block_bytes();
        // The float types in order, decoded without a call per element: the
        // logits of a large vocabulary are read at every generated token.
        if self.layout.is_contiguous() {
            let start = self.layout.offset() * size;
            let bytes = &bytes[start..start + self.numel() * size];
            match self.dtype {
                DType::F32 => {
                    let (values, _) = bytes.as_chunks::<4>();
                    return Ok(values
                        .iter()
                        .map(|b| f64::from(f32::from_le_bytes(*b)))
                        .collect());
                }
                DType::F64 => {
                    let (values, _) = bytes.as_chunks::<8>();
                    return Ok(values.iter().map(|b| f64::from_le_bytes(*b)).collect());
                }
                DType::BF16 => {
                    let (values, _) = bytes.as_chunks::<2>();
                    return Ok(values
                        .iter()
                        .map(|b| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)))
                        .collect());
                }
                _ => {}
            }
        }
        self.layout
            .offsets()
            .map(|at| self.dtype.decode(&bytes[at * size..(at + 1) * size]))
            .collect()
    }

    /// The tensor with its elements stored in row-major order: the same tensor if
    /// it already is, otherwise a copy in host memory.
    ///
    /// # Errors
    ///
    /// [`Error::WrongDevice`] for a strided view outside host memory: each backend
    /// makes its own tensors contiguous.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let source = self.host("contiguous")?;
        let size = self.dtype.block_bytes();
        let mut storage = HostStorage::new(self.numel() * size);
        for (at, out) in self
            .layout
            .offsets()
            .zip(storage.bytes_mut().chunks_exact_mut(size))
        {
            out.copy_from_slice(&source[at * size..(at + 1) * size]);
        }
        Self::from_storage(
            Arc::new(storage),
            Layout::contiguous(self.shape())?,
            self.dtype,
        )
    }

    /// `tensors` joined along `dim`, in host memory: they have the same type
    /// (not quantized) and the same shape but along `dim`. The bytes are
    /// copied as they are.
    ///
    /// # Errors
    ///
    /// [`Error::Arity`] without tensors, [`Error::InvalidDim`],
    /// [`Error::DTypeMismatch`], [`Error::ShapeMismatch`] or
    /// [`Error::UnsupportedDType`] for tensors that cannot be joined,
    /// [`Error::WrongDevice`] for a tensor outside host memory: each backend
    /// joins its own tensors.
    pub fn concat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let op = "concat";
        let first = tensors.first().ok_or(Error::Arity {
            op,
            expected: "at least 1",
            got: 0,
        })?;
        let (dtype, rank) = (first.dtype, first.rank());
        if dim >= rank {
            return Err(Error::InvalidDim { op, dim, rank });
        }
        let size = dtype
            .element_size()
            .ok_or(Error::UnsupportedDType { op, dtype })?;
        let mut shape = first.shape().to_vec();
        shape[dim] = 0;
        for x in tensors {
            if x.dtype != dtype {
                return Err(Error::DTypeMismatch {
                    op,
                    expected: dtype,
                    got: x.dtype,
                });
            }
            if x.rank() != rank || (0..rank).any(|d| d != dim && x.shape()[d] != first.shape()[d]) {
                return Err(Error::ShapeMismatch {
                    op,
                    expected: first.shape().to_vec(),
                    got: x.shape().to_vec(),
                });
            }
            shape[dim] = shape[dim]
                .checked_add(x.shape()[dim])
                .ok_or(Error::Overflow { op })?;
        }
        let layout = Layout::contiguous(&shape)?;
        // Each input contributes, for every index of the dimensions before
        // `dim`, one run of `its size along dim × inner` elements.
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let parts = tensors
            .iter()
            .map(|x| x.contiguous())
            .collect::<Result<Vec<_>>>()?;
        let mut storage = HostStorage::new(layout.numel() * size);
        let out = storage.bytes_mut();
        let mut at = 0;
        for o in 0..outer {
            for part in &parts {
                let run = part.shape()[dim] * inner * size;
                let bytes = part.host(op)?;
                let start = part.layout.offset() * size + o * run;
                out[at..at + run].copy_from_slice(&bytes[start..start + run]);
                at += run;
            }
        }
        Self::from_storage(Arc::new(storage), layout, dtype)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("dtype", &self.dtype)
            .field("device", &self.device())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BF16;

    #[test]
    fn views_share_the_storage() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert!(Arc::ptr_eq(t.storage(), tt.storage()));
        assert_eq!(
            tt.to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let column = t.narrow(1, 1, 1).unwrap();
        assert_eq!(column.to_vec::<f32>().unwrap(), vec![2.0, 5.0]);
        let wide = column.broadcast_to(&[2, 4]).unwrap();
        assert_eq!(
            wide.to_vec::<f32>().unwrap(),
            vec![2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0]
        );
    }

    #[test]
    fn contiguous_copies_a_strided_view() {
        let t = Tensor::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert!(matches!(tt.reshape(&[6]), Err(Error::NotContiguous { .. })));
        let copy = tt.contiguous().unwrap();
        assert!(copy.is_contiguous());
        assert_eq!(
            copy.reshape(&[6]).unwrap().to_vec::<i32>().unwrap(),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn small_floats_go_in_and_out() {
        let values = [BF16::from_f64(1.5), BF16::from_f64(-0.25)];
        let t = Tensor::from_slice(&values, &[2]).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.to_f64_vec().unwrap(), vec![1.5, -0.25]);
    }

    #[test]
    fn wrong_calls_explain_themselves() {
        let t = Tensor::from_slice(&[1.0f32; 6], &[2, 3]).unwrap();
        let error = t.to_vec::<f64>().unwrap_err();
        assert_eq!(
            error,
            Error::DTypeMismatch {
                op: "to_vec",
                expected: DType::F32,
                got: DType::F64
            }
        );
        assert!(
            error
                .to_string()
                .contains("expected data type f32, got f64")
        );
        assert!(Tensor::from_slice(&[1.0f32; 5], &[2, 3]).is_err());
        let message = t.transpose(0, 2).unwrap_err().to_string();
        assert!(message.contains("valid dimensions are 0..2"), "{message}");
    }

    #[test]
    fn quantized_tensors_reshape_but_do_not_view() {
        let dtype = DType::int4(4).unwrap();
        let mut bytes = 1.0f32.to_le_bytes().to_vec();
        bytes.extend([0x98, 0xa8]);
        bytes.extend(2.0f32.to_le_bytes());
        bytes.extend([0x88, 0x88]);
        let t = Tensor::from_bytes(dtype, &[2, 4], &bytes).unwrap();
        assert_eq!(
            t.to_f64_vec().unwrap(),
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(t.reshape(&[8]).unwrap().numel(), 8);
        assert!(matches!(
            t.transpose(0, 1),
            Err(Error::UnsupportedDType { .. })
        ));
        assert!(matches!(
            Tensor::from_bytes(dtype, &[6], &bytes),
            Err(Error::BlockMisaligned { .. })
        ));
    }
}
