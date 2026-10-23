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
    ///
    /// The arithmetic is in `u32`, without overflow checks: the backend only
    /// passes views of storages of at most `u32::MAX` elements, in which every
    /// position, and every partial sum of one, fits.
    #[inline(always)]
    pub fn at(&self, mut flat: u32) -> u32 {
        let mut position = self.offset;
        let mut dim = MAX_RANK;
        // Once `flat` is 0, the remaining dimensions add nothing: a view
        // whose dimensions the backend merged ends early.
        while dim > 0 && flat > 0 {
            dim -= 1;
            let size = self.shape[dim];
            position += (flat % size) * self.strides[dim];
            flat /= size;
        }
        position
    }
}

/// The shape and parameters of an attention over a cache kept by pages
/// (`paged_attention`, `paged_combine`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paged {
    /// The step between two query rows of `q`, in elements.
    pub q_row: u32,
    /// The step between two heads of a query row, in elements; each head's
    /// `D` elements are contiguous.
    pub q_head: u32,
    /// Query heads.
    pub heads: u32,
    /// Key and value heads.
    pub kv_heads: u32,
    /// The width of a key head.
    pub d: u32,
    /// The width of a value head.
    pub dv: u32,
    /// Lines of a page.
    pub page: u32,
    /// Pages of each sequence in the table of pages.
    pub per_sequence: u32,
    /// The window, or 0 for none.
    pub window: u32,
    /// Positions per tile block.
    pub chunk: u32,
    /// Tile blocks per query row and group of heads.
    pub chunks: u32,
    /// The factor of the scores.
    pub scale: f32,
    /// The bound of the scores, or 0 for none.
    pub softcap: f32,
    /// The float codes (`float`) of `q`, the cache and the output, a byte each
    /// from the lowest.
    pub kinds: u32,
}

/// The most positions a block of `paged_attention` takes.
pub const PAGED_CHUNK: u32 = 256;

/// The most query heads a block of `paged_attention` takes: those of one key
/// head, by groups of this many.
pub const PAGED_HEADS: u32 = 8;

/// The widest key or value head `paged_attention` takes.
pub const PAGED_WIDTH: u32 = 256;

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

/// Codes of the float types of the kernels that read and write any of them:
/// their slices are bytes, read and written as elements of the code's type.
pub mod float {
    /// IEEE 754 binary32.
    pub const F32: u32 = 0;
    /// bfloat16.
    pub const BF16: u32 = 1;
    /// IEEE 754 binary16.
    pub const F16: u32 = 2;
    /// 8-bit float, E4M3 without infinities (OCP "FN").
    pub const F8E4M3: u32 = 3;
    /// 8-bit float, E5M2 (OCP).
    pub const F8E5M2: u32 = 4;
}

/// Codes of the quantized formats the kernels decode (`qgemv`,
/// `dequantize`): those of `kapel_core::QuantFormat` but Q2_K and Q3_K.
pub mod quant {
    /// Kapel int8: per group, an `f32` scale and signed bytes.
    pub const INT8: u32 = 0;
    /// Kapel int4: per group, an `f32` scale and nibbles, offset by 8.
    pub const INT4: u32 = 1;
    /// GGUF Q8_0.
    pub const Q8_0: u32 = 2;
    /// GGUF Q4_0.
    pub const Q4_0: u32 = 3;
    /// GGUF Q4_1.
    pub const Q4_1: u32 = 4;
    /// GGUF Q5_0.
    pub const Q5_0: u32 = 5;
    /// GGUF Q5_1.
    pub const Q5_1: u32 = 6;
    /// GGUF Q4_K.
    pub const Q4K: u32 = 7;
    /// GGUF Q5_K.
    pub const Q5K: u32 = 8;
    /// GGUF Q6_K.
    pub const Q6K: u32 = 9;
}

/// Codes of the 16-bit float formats of the conversion kernels.
pub mod half {
    /// bfloat16.
    pub const BF16: u32 = 0;
    /// IEEE 754 binary16.
    pub const F16: u32 = 1;
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
