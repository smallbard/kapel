//! Typed errors.
//!
//! Every error names the operation that failed and says what was expected, so that
//! the message tells the caller what to fix.

use std::fmt;

use crate::{DType, Device};

/// The result type of Kapel operations.
pub type Result<T> = std::result::Result<T, Error>;

/// What can go wrong when calling a Kapel operation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Two shapes that must agree do not.
    ShapeMismatch {
        /// The operation that failed.
        op: &'static str,
        /// The shape the operation needed.
        expected: Vec<usize>,
        /// The shape it received.
        got: Vec<usize>,
    },
    /// A shape cannot be broadcast to another.
    BroadcastIncompatible {
        /// The operation that failed.
        op: &'static str,
        /// The shape being broadcast.
        from: Vec<usize>,
        /// The target shape.
        to: Vec<usize>,
    },
    /// A reshape asks for a different number of elements.
    ElementCountMismatch {
        /// The operation that failed.
        op: &'static str,
        /// The number of elements of the tensor.
        expected: usize,
        /// The number of elements of the requested shape.
        got: usize,
    },
    /// A dimension index is out of bounds for the rank of the tensor.
    InvalidDim {
        /// The operation that failed.
        op: &'static str,
        /// The dimension that was asked for.
        dim: usize,
        /// The rank of the tensor.
        rank: usize,
    },
    /// The same dimension appears twice where each must appear once.
    DuplicateDim {
        /// The operation that failed.
        op: &'static str,
        /// The repeated dimension.
        dim: usize,
    },
    /// A range does not fit in a dimension.
    OutOfRange {
        /// The operation that failed.
        op: &'static str,
        /// The dimension.
        dim: usize,
        /// The start of the range.
        start: usize,
        /// The length of the range.
        len: usize,
        /// The size of the dimension.
        size: usize,
    },
    /// The operation needs a contiguous tensor, and this one is a strided view.
    NotContiguous {
        /// The operation that failed.
        op: &'static str,
    },
    /// Two data types that must agree do not.
    DTypeMismatch {
        /// The operation that failed.
        op: &'static str,
        /// The data type the operation needed.
        expected: DType,
        /// The data type it received.
        got: DType,
    },
    /// The operation does not support this data type.
    UnsupportedDType {
        /// The operation that failed.
        op: &'static str,
        /// The data type it received.
        dtype: DType,
    },
    /// The tensors of an operation live on different devices.
    DeviceMismatch {
        /// The operation that failed.
        op: &'static str,
        /// The device of the first tensor.
        expected: Device,
        /// The device of the other tensor.
        got: Device,
    },
    /// The operation needs the data on another device.
    WrongDevice {
        /// The operation that failed.
        op: &'static str,
        /// The device the operation needed.
        expected: Device,
        /// The device of the tensor.
        got: Device,
    },
    /// A quantized type stores whole blocks, and this element count is not a
    /// multiple of the block size.
    BlockMisaligned {
        /// The quantized type.
        dtype: DType,
        /// The number of elements.
        elements: usize,
    },
    /// A buffer does not have the size its shape and data type require.
    BufferSize {
        /// The operation that failed.
        op: &'static str,
        /// The size in bytes the shape and data type require.
        expected: usize,
        /// The size in bytes of the buffer.
        got: usize,
    },
    /// A size computation overflows `usize`.
    Overflow {
        /// The operation that failed.
        op: &'static str,
    },
    /// The operation received the wrong number of inputs.
    Arity {
        /// The operation that failed.
        op: &'static str,
        /// What the operation accepts, such as "2" or "3 or 4".
        expected: &'static str,
        /// The number of inputs it received.
        got: usize,
    },
    /// An index points outside the tensor it indexes.
    IndexOutOfRange {
        /// The operation that failed.
        op: &'static str,
        /// The index.
        index: i64,
        /// The number of rows that can be indexed.
        size: usize,
    },
    /// The device or its driver reported an error.
    Device {
        /// The operation that failed.
        op: &'static str,
        /// The driver's message.
        message: String,
    },
    /// A parameter of the operation is invalid; the message says why.
    InvalidArgument {
        /// The operation that failed.
        op: &'static str,
        /// What is wrong, and what is expected.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { op, expected, got } => {
                write!(f, "{op}: expected shape {expected:?}, got {got:?}")
            }
            Self::BroadcastIncompatible { op, from, to } => write!(
                f,
                "{op}: shape {from:?} cannot be broadcast to {to:?}; each dimension \
                 must be equal to the target or 1"
            ),
            Self::ElementCountMismatch { op, expected, got } => write!(
                f,
                "{op}: the tensor has {expected} elements, the requested shape has {got}"
            ),
            Self::InvalidDim { op, dim, rank } => write!(
                f,
                "{op}: dimension {dim} does not exist in a tensor of rank {rank}; \
                 valid dimensions are 0..{rank}"
            ),
            Self::DuplicateDim { op, dim } => {
                write!(f, "{op}: dimension {dim} appears more than once")
            }
            Self::OutOfRange {
                op,
                dim,
                start,
                len,
                size,
            } => write!(
                f,
                "{op}: range {start}..{} does not fit in dimension {dim} of size {size}",
                start.saturating_add(*len)
            ),
            Self::NotContiguous { op } => write!(
                f,
                "{op}: the tensor is a strided view; call `contiguous()` first"
            ),
            Self::DTypeMismatch { op, expected, got } => {
                write!(f, "{op}: expected data type {expected}, got {got}")
            }
            Self::UnsupportedDType { op, dtype } => {
                write!(f, "{op}: data type {dtype} is not supported")
            }
            Self::DeviceMismatch { op, expected, got } => write!(
                f,
                "{op}: tensors live on different devices ({expected} and {got}); move \
                 them to the same device first"
            ),
            Self::WrongDevice { op, expected, got } => write!(
                f,
                "{op}: needs a tensor on {expected}, this one is on {got}"
            ),
            Self::BlockMisaligned { dtype, elements } => write!(
                f,
                "{dtype} stores blocks of {} elements; {elements} elements is not a \
                 multiple of it",
                dtype.block_len()
            ),
            Self::BufferSize { op, expected, got } => write!(
                f,
                "{op}: the shape and data type need {expected} bytes, the buffer has {got}"
            ),
            Self::Overflow { op } => write!(f, "{op}: the size overflows `usize`"),
            Self::Arity { op, expected, got } => {
                write!(f, "{op}: expected {expected} inputs, got {got}")
            }
            Self::IndexOutOfRange { op, index, size } => {
                write!(f, "{op}: index {index} is out of range for {size} rows")
            }
            Self::InvalidArgument { op, message } => write!(f, "{op}: {message}"),
            Self::Device { op, message } => write!(f, "{op}: device error: {message}"),
        }
    }
}

impl std::error::Error for Error {}
