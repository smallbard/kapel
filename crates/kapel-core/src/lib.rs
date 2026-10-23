//! Tensors, data types, layouts and errors shared by every Kapel crate.
//!
//! - [`DType`]: the element types, from `f64` to `f8` and quantized block formats,
//!   each with an exact reference conversion to `f64`.
//! - [`Layout`]: shapes, strides and offsets; views without copies.
//! - [`Tensor`]: a shared [`Storage`] seen through a layout.
//! - [`BufferPool`]: host buffers reused by size.
//! - [`Error`]: what went wrong, and what to fix.
//!
//! ```
//! use kapel_core::{DType, Tensor};
//!
//! let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
//! let columns = t.transpose(0, 1)?; // a view: no copy
//! assert_eq!(columns.shape(), &[3, 2]);
//! assert_eq!(columns.to_vec::<f32>()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//! assert_eq!(t.dtype(), DType::F32);
//! # Ok::<(), kapel_core::Error>(())
//! ```

mod device;
mod dtype;
mod error;
mod float;
mod layout;
mod scalar;
mod storage;
mod tensor;

pub use device::Device;
pub use dtype::{
    BlockScale, DType, QuantFormat, block_code, block_scale, block_value, dequantize, encode_block,
    quantize,
};
pub use error::{Error, Result};
pub use layout::{Layout, Offsets};
pub use scalar::{BF16, Element, F8E4M3, F8E5M2, F16};
pub use storage::{BufferPool, HostStorage, Storage};
pub use tensor::Tensor;
