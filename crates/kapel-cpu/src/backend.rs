//! The backend value: its threads, its instructions, its buffers.

use std::sync::Arc;

use kapel_core::{BufferPool, Device, Error, Result, Tensor};
use kapel_ops::Transfer;

use crate::pool::{Pool, SharedMut};
use crate::{Isa, default_threads};

/// Elements per task for operations that treat elements independently: large
/// enough that a task outweighs its scheduling, small enough to spread a
/// medium tensor over every core.
pub(crate) const CHUNK: usize = 16_384;

/// The CPU backend.
///
/// Cloning it shares its threads and its buffer pool.
#[derive(Clone)]
pub struct Cpu {
    pub(crate) pool: Arc<Pool>,
    pub(crate) isa: Isa,
    pub(crate) buffers: Arc<BufferPool>,
}

impl Cpu {
    /// A backend with one thread per physical core and the best instructions of
    /// the processor.
    pub fn new() -> Self {
        Self::with_threads(default_threads())
    }

    /// A backend with `threads` threads, the calling thread included.
    pub fn with_threads(threads: usize) -> Self {
        Self {
            pool: Arc::new(Pool::new(threads)),
            isa: Isa::detect(),
            buffers: BufferPool::new(1 << 30),
        }
    }

    /// The same backend, restricted to the instructions `isa`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the processor does not have them.
    pub fn with_isa(mut self, isa: Isa) -> Result<Self> {
        if !isa.available() {
            return Err(Error::InvalidArgument {
                op: "Cpu::with_isa",
                message: format!("this processor does not support {isa:?}"),
            });
        }
        self.isa = isa;
        Ok(self)
    }

    /// The number of threads, the calling thread included.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// The instructions the kernels use.
    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// Fills `out` in chunks of `chunk` elements, spread over the threads:
    /// `fill(start, part)` writes `part`, which begins at element `start`.
    pub(crate) fn fill(
        &self,
        out: &mut [f32],
        chunk: usize,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) {
        let len = out.len();
        let shared = SharedMut::new(out);
        self.pool.run(len.div_ceil(chunk.max(1)), |task| {
            let start = task * chunk;
            let end = (start + chunk).min(len);
            // SAFETY: tasks write disjoint ranges of `out`, which outlives the
            // call to `run`.
            let part = unsafe { shared.slice(start, end - start) };
            fill(start, part);
        });
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Cpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cpu")
            .field("threads", &self.threads())
            .field("isa", &self.isa)
            .finish()
    }
}

impl Transfer for Cpu {
    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor.device() {
            Device::Cpu => Ok(tensor.clone()),
            device => Err(Error::WrongDevice {
                op: "upload",
                expected: Device::Cpu,
                got: device,
            }),
        }
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        self.upload(tensor)
    }
}
