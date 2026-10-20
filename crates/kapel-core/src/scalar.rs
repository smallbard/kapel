//! Rust types for the scalar data types, to move data in and out of tensors.

use crate::DType;
use crate::float;

/// A Rust type stored as one element of a [`DType`], in little-endian bytes.
pub trait Element: Copy + Send + Sync + 'static {
    /// The data type of this element type.
    const DTYPE: DType;

    /// Writes the element into `out`, exactly `DTYPE`'s element size long.
    fn write_le(self, out: &mut [u8]);

    /// Reads an element from `bytes`, exactly `DTYPE`'s element size long.
    fn read_le(bytes: &[u8]) -> Self;
}

macro_rules! native {
    ($($t:ty => $dtype:ident),* $(,)?) => {$(
        impl Element for $t {
            const DTYPE: DType = DType::$dtype;

            fn write_le(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_le_bytes());
            }

            fn read_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("one element"))
            }
        }
    )*};
}

native!(f64 => F64, f32 => F32, i8 => I8, u8 => U8, i32 => I32, u32 => U32);

macro_rules! small_float {
    ($(#[$doc:meta])* $name:ident($bits:ty), $dtype:ident, $format:path) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
        pub struct $name(pub $bits);

        impl $name {
            /// The value nearest to `x`, ties to even.
            pub fn from_f64(x: f64) -> Self {
                Self($format.encode(x) as $bits)
            }

            /// The exact value.
            pub fn to_f64(self) -> f64 {
                $format.decode(u32::from(self.0))
            }
        }

        impl Element for $name {
            const DTYPE: DType = DType::$dtype;

            fn write_le(self, out: &mut [u8]) {
                out.copy_from_slice(&self.0.to_le_bytes());
            }

            fn read_le(bytes: &[u8]) -> Self {
                Self(<$bits>::from_le_bytes(bytes.try_into().expect("one element")))
            }
        }
    };
}

small_float!(
    /// An IEEE 754 binary16 value, by its bits.
    F16(u16), F16, float::F16
);
small_float!(
    /// A bfloat16 value, by its bits.
    BF16(u16), BF16, float::BF16
);
small_float!(
    /// An OCP E4M3 ("FN") 8-bit float, by its bits.
    F8E4M3(u8), F8E4M3, float::F8_E4M3
);
small_float!(
    /// An OCP E5M2 8-bit float, by its bits.
    F8E5M2(u8), F8E5M2, float::F8_E5M2
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_floats_convert() {
        assert_eq!(F16::from_f64(1.0).0, 0x3c00);
        assert_eq!(BF16::from_f64(1.0).0, 0x3f80);
        assert_eq!(F8E4M3::from_f64(448.0).0, 0x7e);
        assert_eq!(F8E5M2::from_f64(-2.0).to_f64(), -2.0);
    }
}
