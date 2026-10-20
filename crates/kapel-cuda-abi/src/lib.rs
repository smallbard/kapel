//! The arguments shared by Kapel's CUDA kernels and its CUDA backend.
//!
//! The kernels are compiled to PTX by cuda-oxide, apart from the backend, which
//! loads them at run time. Values passed by value must have the same layout on
//! both sides: they are defined once, here, `#[repr(C)]`, without dependencies,
//! so that both the kernel crate and the backend compile this crate.
//!
//! Operation codes are plain `u32` constants rather than enums: a kernel receives
//! a number, and an unknown one must not be undefined behavior.

#![no_std]

/// The most dimensions a strided view passed to a kernel may have.
pub const MAX_RANK: usize = 8;

/// Threads per block of the element-wise kernels.
pub const BLOCK: u32 = 256;

/// A strided view of a tensor: element `(i₀, …)` of `shape` is at
/// `offset + Σ iₖ·strides[k]`. Unused leading dimensions have size 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    /// The size of each dimension, the unused ones first, set to 1.
    pub shape: [u32; MAX_RANK],
    /// The step of each dimension, in elements; 0 for a broadcast one.
    pub strides: [u32; MAX_RANK],
    /// The position of the first element.
    pub offset: u32,
}

impl View {
    /// The storage position of the element at row-major position `flat` of the
    /// view's shape.
    #[inline(always)]
    pub fn at(&self, mut flat: u32) -> u32 {
        let mut position = self.offset;
        let mut dim = MAX_RANK;
        while dim > 0 {
            dim -= 1;
            let size = self.shape[dim];
            position += (flat % size) * self.strides[dim];
            flat /= size;
        }
        position
    }
}

/// Codes of the `unary` kernels.
pub mod unary {
    /// `−x`
    pub const NEG: u32 = 0;
    /// `eˣ`
    pub const EXP: u32 = 1;
    /// `ln x`
    pub const LOG: u32 = 2;
    /// `√x`
    pub const SQRT: u32 = 3;
    /// `1/√x`
    pub const RSQRT: u32 = 4;
    /// `x²`
    pub const SQUARE: u32 = 5;
    /// `|x|`
    pub const ABS: u32 = 6;
    /// `1/(1 + e⁻ˣ)`
    pub const SIGMOID: u32 = 7;
    /// `tanh x`
    pub const TANH: u32 = 8;
    /// `max(x, 0)`
    pub const RELU: u32 = 9;
    /// `x · sigmoid(x)`
    pub const SILU: u32 = 10;
    /// GELU, tanh approximation.
    pub const GELU_TANH: u32 = 11;
    /// GELU, exact.
    pub const GELU_ERF: u32 = 12;
}

/// Codes of the `binary` kernels.
pub mod binary {
    /// `a + b`
    pub const ADD: u32 = 0;
    /// `a − b`
    pub const SUB: u32 = 1;
    /// `a · b`
    pub const MUL: u32 = 2;
    /// `a / b`
    pub const DIV: u32 = 3;
    /// `max(a, b)`, ties to `a`.
    pub const MAX: u32 = 4;
    /// `min(a, b)`, ties to `a`.
    pub const MIN: u32 = 5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_reads_a_transpose() {
        let mut view = View {
            shape: [1; MAX_RANK],
            strides: [0; MAX_RANK],
            offset: 0,
        };
        view.shape[6] = 3;
        view.shape[7] = 2;
        view.strides[6] = 1;
        view.strides[7] = 3;
        let positions: [u32; 6] = core::array::from_fn(|i| view.at(i as u32));
        assert_eq!(positions, [0, 3, 1, 4, 2, 5]);
    }
}
