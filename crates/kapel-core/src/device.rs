//! Where a tensor lives.

use std::fmt;

/// The device that holds a tensor's data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// Host memory, used by the CPU and reference backends.
    Cpu,
    /// The memory of a CUDA device, by ordinal.
    Cuda(u32),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(ordinal) => write!(f, "cuda:{ordinal}"),
        }
    }
}
