//! Model files: safetensors, model directories in the layout of the Hugging Face
//! Hub, and GGUF.
//!
//! Weights are mapped in memory: opening a model of several gigabytes is
//! immediate, and its tensors are read in place by the CPU backend.

mod atomic;
mod bytes;
mod checkpoint;
#[cfg(feature = "download")]
pub mod download;
mod error;
pub mod gguf;
pub mod safetensors;

pub use atomic::write_atomic;
pub use checkpoint::{Checkpoint, read_json};
pub use error::{HubError, Result};
pub use gguf::Gguf;
pub use safetensors::{SafeTensors, TensorInfo};

/// Reports a parity test that cannot run because its reference files (written
/// by `tools/parity`) are missing; with `KAPEL_REQUIRE_PARITY=1`, fails instead.
#[cfg(test)]
pub(crate) fn parity_skipped(what: &str) {
    use std::io::Write;
    assert!(
        std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
        "{what} is missing, and KAPEL_REQUIRE_PARITY=1 (see tools/parity/README.md)"
    );
    // Straight to stderr, past the capture of the test harness: a skip is seen.
    let _ = writeln!(
        std::io::stderr(),
        "skipped: {what} missing (see tools/parity/README.md)"
    );
}
