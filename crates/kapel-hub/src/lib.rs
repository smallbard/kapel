//! Model files: safetensors, model directories in the layout of the Hugging Face
//! Hub, and GGUF.
//!
//! Weights are mapped in memory: opening a model of several gigabytes is
//! immediate, and its tensors are read in place by the CPU backend.

mod bytes;
mod checkpoint;
#[cfg(feature = "download")]
pub mod download;
mod error;
pub mod gguf;
pub mod safetensors;

pub use checkpoint::{Checkpoint, read_json};
pub use error::{HubError, Result};
pub use gguf::Gguf;
pub use safetensors::{SafeTensors, TensorInfo};
