//! File contents shared by the tensors read from them.

use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use kapel_core::{Device, Storage};

use crate::error::{HubError, Result};

/// The bytes of a file: mapped in memory, or read into a buffer.
pub(crate) enum Bytes {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Bytes {
    /// Maps a file in memory. Its pages are read when first touched.
    pub(crate) fn map(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| HubError::io(path, e))?;
        // SAFETY: the mapping is read-only. As with every memory-mapped file, a
        // process that truncates the file while it is mapped can make reads
        // fault; model files are not modified while in use.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| HubError::io(path, e))?;
        Ok(Self::Mapped(map))
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Owned(bytes) => bytes,
        }
    }
}

/// The bytes of one tensor, inside a shared file.
pub(crate) struct FileStorage {
    pub(crate) file: Arc<Bytes>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl fmt::Debug for FileStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileStorage")
            .field("start", &self.start)
            .field("len", &self.len)
            .finish()
    }
}

impl Storage for FileStorage {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.file.as_slice()[self.start..self.start + self.len])
    }
}
