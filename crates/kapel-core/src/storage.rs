//! Where a tensor's bytes live, and the host buffer pool.
//!
//! Each backend has its own storage type. The host one, [`HostStorage`], is a
//! 64-byte aligned buffer (one AVX-512 register, one cache line) taken from a
//! [`BufferPool`]. A freed buffer goes back to the pool, sorted by size, and the
//! next request of the same size takes it: without a pool, every new tensor would
//! come fresh from the operating system, one zeroed page at a time.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::Device;

/// The bytes of a tensor, on some device.
pub trait Storage: Send + Sync + fmt::Debug + 'static {
    /// The device that holds the bytes.
    fn device(&self) -> Device;

    /// The size in bytes.
    fn len(&self) -> usize;

    /// Whether the storage holds no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The storage as [`Any`], for a backend to recover its own storage type.
    fn as_any(&self) -> &dyn Any;

    /// The bytes, if they are in host memory and can be read in place: an owned
    /// host buffer, or a file mapped in memory.
    fn host_bytes(&self) -> Option<&[u8]> {
        None
    }
}

/// The unit of host allocation: 64 bytes, aligned on 64.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; 64]);

impl fmt::Debug for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Line")
    }
}

const ZERO_LINE: Line = Line([0; 64]);

/// Host buffers kept for reuse, by size in 64-byte lines.
#[derive(Debug)]
pub struct BufferPool {
    free: Mutex<Pool>,
    /// At most this many bytes are kept; beyond it, freed buffers are released.
    capacity: usize,
}

#[derive(Debug, Default)]
struct Pool {
    buffers: HashMap<usize, Vec<Box<[Line]>>>,
    bytes: usize,
}

impl BufferPool {
    /// A pool that keeps at most `capacity` bytes of freed buffers.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(Pool::default()),
            capacity,
        })
    }

    /// A host storage of `len` bytes, taken from the pool if a buffer of the same
    /// size is free. Its contents are unspecified: a fresh buffer is zeroed, a
    /// reused one holds what its last owner wrote. Use [`BufferPool::zeroed`] when
    /// the contents matter.
    pub fn take(self: &Arc<Self>, len: usize) -> HostStorage {
        let lines = len.div_ceil(64);
        let reused = self.lock().take(lines);
        let buffer = reused.unwrap_or_else(|| vec![ZERO_LINE; lines].into_boxed_slice());
        HostStorage {
            buffer: Some(buffer),
            len,
            pool: Some(Arc::clone(self)),
        }
    }

    /// A host storage of `len` bytes, all zero.
    pub fn zeroed(self: &Arc<Self>, len: usize) -> HostStorage {
        let mut storage = self.take(len);
        storage.bytes_mut().fill(0);
        storage
    }

    /// The number of bytes held for reuse.
    pub fn retained_bytes(&self) -> usize {
        self.lock().bytes
    }

    /// Releases every buffer held for reuse.
    pub fn clear(&self) {
        let mut pool = self.lock();
        pool.buffers.clear();
        pool.bytes = 0;
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pool> {
        // A panic while holding the lock leaves the pool consistent: every
        // operation on it is a single push or pop.
        self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn give_back(&self, buffer: Box<[Line]>) {
        let bytes = buffer.len() * 64;
        let mut pool = self.lock();
        if pool.bytes + bytes <= self.capacity {
            pool.bytes += bytes;
            pool.buffers.entry(buffer.len()).or_default().push(buffer);
        }
    }
}

impl Pool {
    fn take(&mut self, lines: usize) -> Option<Box<[Line]>> {
        let buffer = self.buffers.get_mut(&lines)?.pop()?;
        self.bytes -= lines * 64;
        Some(buffer)
    }
}

/// Bytes in host memory, aligned on 64 bytes.
pub struct HostStorage {
    buffer: Option<Box<[Line]>>,
    len: usize,
    pool: Option<Arc<BufferPool>>,
}

impl HostStorage {
    /// A zeroed storage of `len` bytes, outside any pool.
    pub fn new(len: usize) -> Self {
        let lines = len.div_ceil(64);
        Self {
            buffer: Some(vec![ZERO_LINE; lines].into_boxed_slice()),
            len,
            pool: None,
        }
    }

    /// A storage holding a copy of `bytes`, outside any pool.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut storage = Self::new(bytes.len());
        storage.bytes_mut().copy_from_slice(bytes);
        storage
    }

    /// The bytes.
    pub fn bytes(&self) -> &[u8] {
        let lines = self
            .buffer
            .as_deref()
            .expect("a live storage has its buffer");
        // SAFETY: `Line` is `repr(C)` over `[u8; 64]`: the lines are
        // `lines.len() * 64` initialized bytes, and `self.len` never exceeds that.
        unsafe { std::slice::from_raw_parts(lines.as_ptr().cast::<u8>(), self.len) }
    }

    /// The bytes, mutably.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        let lines = self
            .buffer
            .as_deref_mut()
            .expect("a live storage has its buffer");
        // SAFETY: as in `bytes`, and the exclusive borrow of `self` makes the
        // slice the only access to the buffer.
        unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr().cast::<u8>(), self.len) }
    }
}

impl Drop for HostStorage {
    fn drop(&mut self) {
        if let (Some(pool), Some(buffer)) = (self.pool.take(), self.buffer.take()) {
            pool.give_back(buffer);
        }
    }
}

impl fmt::Debug for HostStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostStorage")
            .field("len", &self.len)
            .field("pooled", &self.pool.is_some())
            .finish()
    }
}

impl Storage for HostStorage {
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
        Some(self.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_storage_is_aligned_for_avx512() {
        for len in [1, 63, 64, 65, 4096] {
            let storage = HostStorage::new(len);
            assert_eq!(storage.bytes().as_ptr() as usize % 64, 0);
            assert_eq!(storage.bytes().len(), len);
        }
    }

    #[test]
    fn a_freed_buffer_serves_the_next_request_of_its_size() {
        let pool = BufferPool::new(1 << 20);
        let mut first = pool.take(1000);
        first.bytes_mut()[0] = 42;
        let address = first.bytes().as_ptr();
        drop(first);
        assert_eq!(pool.retained_bytes(), 1024);
        let second = pool.take(1000);
        assert_eq!(second.bytes().as_ptr(), address);
        assert_eq!(pool.retained_bytes(), 0);
        drop(second);
        let zeroed = pool.zeroed(1000);
        assert!(zeroed.bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn the_pool_keeps_no_more_than_its_capacity() {
        let pool = BufferPool::new(1024);
        let a = pool.take(1024);
        let b = pool.take(1024);
        drop(a);
        drop(b);
        assert_eq!(pool.retained_bytes(), 1024);
        pool.clear();
        assert_eq!(pool.retained_bytes(), 0);
    }
}
