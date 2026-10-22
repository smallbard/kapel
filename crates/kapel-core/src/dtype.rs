//! Data types: scalars, small floats, and quantized block formats.

use std::fmt;

use crate::float::{self, Format};
use crate::{Error, Result};

/// The type of a tensor's elements.
///
/// Scalar types store one element per value. Quantized types store elements in
/// blocks: a block of [`DType::block_len`] elements takes [`DType::block_bytes`]
/// bytes, scales included, and a tensor holds whole blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE 754 binary64. The type of the reference backend.
    F64,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16: the exponent range of `f32`, 8 bits of precision.
    BF16,
    /// 8-bit float, 4 exponent and 3 mantissa bits, no infinities (OCP "E4M3FN").
    F8E4M3,
    /// 8-bit float, 5 exponent and 2 mantissa bits (OCP "E5M2").
    F8E5M2,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer, used for token ids and indices.
    U32,
    /// A quantized block format.
    Quantized(QuantFormat),
}

/// A quantized block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantFormat {
    /// Kapel int8: per group of `group` elements, an `f32` scale then `group`
    /// signed bytes. Element `i` is `scale × code[i]`.
    Int8 {
        /// Elements per group.
        group: u16,
    },
    /// Kapel int4: per group of `group` elements, an `f32` scale then `group / 2`
    /// bytes, two codes per byte, the low nibble first. Element `i` is
    /// `scale × (code[i] − 8)`.
    Int4 {
        /// Elements per group, even.
        group: u16,
    },
    /// GGUF Q8_0: 32 elements, an `f16` scale then 32 signed bytes.
    Q8_0,
    /// GGUF Q4_0: 32 elements, an `f16` scale then 16 bytes. Byte `j` holds
    /// element `j` in its low nibble and element `j + 16` in its high nibble;
    /// element values are `scale × (code − 8)`.
    Q4_0,
    /// GGUF Q4_1: as Q4_0, with an `f16` minimum after the scale; element values
    /// are `scale × code + minimum`.
    Q4_1,
    /// GGUF Q5_0: 32 elements, 22 bytes.
    Q5_0,
    /// GGUF Q5_1: 32 elements, 24 bytes.
    Q5_1,
    /// GGUF Q2_K: 256 elements, 84 bytes.
    Q2K,
    /// GGUF Q3_K: 256 elements, 110 bytes.
    Q3K,
    /// GGUF Q4_K: 256 elements, 144 bytes.
    Q4K,
    /// GGUF Q5_K: 256 elements, 176 bytes.
    Q5K,
    /// GGUF Q6_K: 256 elements, 210 bytes.
    Q6K,
}

impl DType {
    /// Kapel int8 with groups of `group` elements.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedDType`] if `group` is zero.
    pub fn int8(group: u16) -> Result<Self> {
        let dtype = Self::Quantized(QuantFormat::Int8 { group });
        if group == 0 {
            return Err(Error::UnsupportedDType {
                op: "DType::int8",
                dtype,
            });
        }
        Ok(dtype)
    }

    /// Kapel int4 with groups of `group` elements.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedDType`] if `group` is zero or odd.
    pub fn int4(group: u16) -> Result<Self> {
        let dtype = Self::Quantized(QuantFormat::Int4 { group });
        if group == 0 || !group.is_multiple_of(2) {
            return Err(Error::UnsupportedDType {
                op: "DType::int4",
                dtype,
            });
        }
        Ok(dtype)
    }

    /// Whether the type is a floating-point scalar.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F64 | Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2
        )
    }

    /// Whether the type is a quantized block format.
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// The number of elements in a block: 1 for scalar types.
    pub fn block_len(self) -> usize {
        match self {
            Self::Quantized(format) => match format {
                QuantFormat::Int8 { group } | QuantFormat::Int4 { group } => usize::from(group),
                QuantFormat::Q8_0
                | QuantFormat::Q4_0
                | QuantFormat::Q4_1
                | QuantFormat::Q5_0
                | QuantFormat::Q5_1 => 32,
                QuantFormat::Q2K
                | QuantFormat::Q3K
                | QuantFormat::Q4K
                | QuantFormat::Q5K
                | QuantFormat::Q6K => 256,
            },
            _ => 1,
        }
    }

    /// The number of bytes of a block: the element size for scalar types.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 => 1,
            Self::Quantized(format) => match format {
                QuantFormat::Int8 { group } => 4 + usize::from(group),
                QuantFormat::Int4 { group } => 4 + usize::from(group) / 2,
                QuantFormat::Q8_0 => 34,
                QuantFormat::Q4_0 => 18,
                QuantFormat::Q4_1 => 20,
                QuantFormat::Q5_0 => 22,
                QuantFormat::Q5_1 => 24,
                QuantFormat::Q2K => 84,
                QuantFormat::Q3K => 110,
                QuantFormat::Q4K => 144,
                QuantFormat::Q5K => 176,
                QuantFormat::Q6K => 210,
            },
        }
    }

    /// The size of one element in bytes, for scalar types.
    pub fn element_size(self) -> Option<usize> {
        (!self.is_quantized()).then(|| self.block_bytes())
    }

    /// The number of bytes that `elements` elements of this type take.
    ///
    /// # Errors
    ///
    /// [`Error::BlockMisaligned`] if `elements` is not a whole number of blocks,
    /// [`Error::Overflow`] if the size does not fit in `usize`.
    pub fn storage_bytes(self, elements: usize) -> Result<usize> {
        let block = self.block_len();
        if !elements.is_multiple_of(block) {
            return Err(Error::BlockMisaligned {
                dtype: self,
                elements,
            });
        }
        (elements / block)
            .checked_mul(self.block_bytes())
            .ok_or(Error::Overflow {
                op: "storage_bytes",
            })
    }

    /// Decodes one element of a scalar type from its little-endian bytes, exactly.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedDType`] for a quantized type (use [`dequantize`]),
    /// [`Error::BufferSize`] if `bytes` is not one element long.
    pub fn decode(self, bytes: &[u8]) -> Result<f64> {
        let size = self.element_size().ok_or(Error::UnsupportedDType {
            op: "DType::decode",
            dtype: self,
        })?;
        if bytes.len() != size {
            return Err(Error::BufferSize {
                op: "DType::decode",
                expected: size,
                got: bytes.len(),
            });
        }
        let b = |i: usize| bytes[i];
        Ok(match self {
            Self::F64 => f64::from_le_bytes(std::array::from_fn(b)),
            Self::F32 => f64::from(f32::from_le_bytes(std::array::from_fn(b))),
            Self::F16 => float::F16.decode(u32::from(u16::from_le_bytes([b(0), b(1)]))),
            Self::BF16 => float::BF16.decode(u32::from(u16::from_le_bytes([b(0), b(1)]))),
            Self::F8E4M3 => float::F8_E4M3.decode(u32::from(b(0))),
            Self::F8E5M2 => float::F8_E5M2.decode(u32::from(b(0))),
            Self::I8 => f64::from(b(0) as i8),
            Self::U8 => f64::from(b(0)),
            Self::I32 => f64::from(i32::from_le_bytes(std::array::from_fn(b))),
            Self::U32 => f64::from(u32::from_le_bytes(std::array::from_fn(b))),
            Self::Quantized(_) => unreachable!("quantized types have no element size"),
        })
    }

    /// Encodes `value` as one element of a scalar type, in little-endian bytes.
    ///
    /// Floats round to nearest, ties to even. Integers round to nearest, ties to
    /// even, and saturate at their bounds; NaN gives zero.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedDType`] for a quantized type,
    /// [`Error::BufferSize`] if `out` is not one element long.
    pub fn encode(self, value: f64, out: &mut [u8]) -> Result<()> {
        let size = self.element_size().ok_or(Error::UnsupportedDType {
            op: "DType::encode",
            dtype: self,
        })?;
        if out.len() != size {
            return Err(Error::BufferSize {
                op: "DType::encode",
                expected: size,
                got: out.len(),
            });
        }
        let small = |format: Format| format.encode(value);
        let integer = |min: f64, max: f64| {
            if value.is_nan() {
                0.0
            } else {
                value.round_ties_even().clamp(min, max)
            }
        };
        match self {
            Self::F64 => out.copy_from_slice(&value.to_le_bytes()),
            Self::F32 => out.copy_from_slice(&(value as f32).to_le_bytes()),
            Self::F16 => out.copy_from_slice(&(small(float::F16) as u16).to_le_bytes()),
            Self::BF16 => out.copy_from_slice(&(small(float::BF16) as u16).to_le_bytes()),
            Self::F8E4M3 => out[0] = small(float::F8_E4M3) as u8,
            Self::F8E5M2 => out[0] = small(float::F8_E5M2) as u8,
            Self::I8 => out[0] = (integer(-128.0, 127.0) as i8) as u8,
            Self::U8 => out[0] = integer(0.0, 255.0) as u8,
            Self::I32 => out.copy_from_slice(
                &(integer(f64::from(i32::MIN), f64::from(i32::MAX)) as i32).to_le_bytes(),
            ),
            Self::U32 => {
                out.copy_from_slice(&(integer(0.0, f64::from(u32::MAX)) as u32).to_le_bytes());
            }
            Self::Quantized(_) => unreachable!("quantized types have no element size"),
        }
        Ok(())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Quantized(QuantFormat::Int8 { group }) => return write!(f, "int8/{group}"),
            Self::Quantized(QuantFormat::Int4 { group }) => return write!(f, "int4/{group}"),
            Self::Quantized(QuantFormat::Q8_0) => "q8_0",
            Self::Quantized(QuantFormat::Q4_0) => "q4_0",
            Self::Quantized(QuantFormat::Q4_1) => "q4_1",
            Self::Quantized(QuantFormat::Q5_0) => "q5_0",
            Self::Quantized(QuantFormat::Q5_1) => "q5_1",
            Self::Quantized(QuantFormat::Q2K) => "q2_k",
            Self::Quantized(QuantFormat::Q3K) => "q3_k",
            Self::Quantized(QuantFormat::Q4K) => "q4_k",
            Self::Quantized(QuantFormat::Q5K) => "q5_k",
            Self::Quantized(QuantFormat::Q6K) => "q6_k",
        };
        f.write_str(name)
    }
}

/// The reference dequantization: the exact `f64` value of every element of
/// `bytes`, a whole number of blocks of `dtype`.
///
/// Every format is decoded here: Kapel's int8 and int4, and GGUF's, as
/// llama.cpp's reference (`gguf-py`) defines them.
///
/// # Errors
///
/// [`Error::UnsupportedDType`] for a scalar type or a format not decoded here,
/// [`Error::BufferSize`] if `bytes` is not a whole number of blocks.
pub fn dequantize(dtype: DType, bytes: &[u8]) -> Result<Vec<f64>> {
    let DType::Quantized(format) = dtype else {
        return Err(Error::UnsupportedDType {
            op: "dequantize",
            dtype,
        });
    };
    let block_bytes = dtype.block_bytes();
    if !bytes.len().is_multiple_of(block_bytes) {
        let blocks = bytes.len().div_ceil(block_bytes);
        return Err(Error::BufferSize {
            op: "dequantize",
            expected: blocks * block_bytes,
            got: bytes.len(),
        });
    }
    let half =
        |b: &[u8], at: usize| float::F16.decode(u32::from(u16::from_le_bytes([b[at], b[at + 1]])));
    let single = |b: &[u8]| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let mut out = Vec::with_capacity(bytes.len() / block_bytes * dtype.block_len());
    for block in bytes.chunks_exact(block_bytes) {
        match format {
            QuantFormat::Int8 { .. } => {
                let scale = single(block);
                out.extend(block[4..].iter().map(|&q| scale * f64::from(q as i8)));
            }
            QuantFormat::Int4 { .. } => {
                let scale = single(block);
                for &byte in &block[4..] {
                    out.push(scale * (f64::from(byte & 0x0f) - 8.0));
                    out.push(scale * (f64::from(byte >> 4) - 8.0));
                }
            }
            QuantFormat::Q8_0 => {
                let scale = half(block, 0);
                out.extend(block[2..].iter().map(|&q| scale * f64::from(q as i8)));
            }
            QuantFormat::Q4_0 | QuantFormat::Q4_1 => {
                let scale = half(block, 0);
                let (minimum, codes) = match format {
                    QuantFormat::Q4_1 => (half(block, 2), &block[4..]),
                    _ => (0.0, &block[2..]),
                };
                let value = |code: u8| match format {
                    QuantFormat::Q4_1 => scale * f64::from(code) + minimum,
                    _ => scale * (f64::from(code) - 8.0),
                };
                out.extend(codes.iter().map(|&byte| value(byte & 0x0f)));
                out.extend(codes.iter().map(|&byte| value(byte >> 4)));
            }
            QuantFormat::Q5_0 | QuantFormat::Q5_1 => {
                let scale = half(block, 0);
                let (minimum, rest) = match format {
                    QuantFormat::Q5_1 => (half(block, 2), &block[4..]),
                    _ => (0.0, &block[2..]),
                };
                let high = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
                let codes = &rest[4..];
                for j in 0..32 {
                    let low = if j < 16 {
                        codes[j] & 0x0f
                    } else {
                        codes[j - 16] >> 4
                    };
                    let q = f64::from(low | (((high >> j) & 1) as u8) << 4);
                    out.push(match format {
                        QuantFormat::Q5_1 => scale * q + minimum,
                        _ => scale * (q - 16.0),
                    });
                }
            }
            QuantFormat::Q2K => {
                let (scales, codes) = (&block[..16], &block[16..80]);
                let (d, dmin) = (half(block, 80), half(block, 82));
                for e in 0..256 {
                    let (h, s, l) = (e / 128, e % 128 / 32, e % 32);
                    let q = f64::from((codes[h * 32 + l] >> (2 * s)) & 3);
                    let k = scales[e / 16];
                    out.push(d * f64::from(k & 0x0f) * q - dmin * f64::from(k >> 4));
                }
            }
            QuantFormat::Q3K => {
                let (mask, codes, packed) = (&block[..32], &block[32..96], &block[96..108]);
                let d = half(block, 108);
                // Sixteen 6-bit scales: 4 low bits from the first 8 bytes, 2 high
                // bits from the last 4, then offset by 32.
                let scale = |i: usize| {
                    let low = (packed[i % 8] >> (4 * (i / 8))) & 0x0f;
                    let high = (packed[8 + i % 4] >> (2 * (i / 4))) & 0x03;
                    f64::from(((low | (high << 4)) as i8) - 32)
                };
                for e in 0..256 {
                    let (h, s, l) = (e / 128, e % 128 / 32, e % 32);
                    let low = i32::from((codes[h * 32 + l] >> (2 * s)) & 3);
                    // The high bit subtracts 4 when it is clear.
                    let high = i32::from(((mask[e % 32] >> (e / 32)) & 1) ^ 1);
                    out.push(d * scale(e / 16) * f64::from(low - 4 * high));
                }
            }
            QuantFormat::Q4K | QuantFormat::Q5K => {
                let (d, dmin) = (half(block, 0), half(block, 2));
                let (scale, minimum) = k_scales(&block[4..16]);
                let (high, codes) = match format {
                    QuantFormat::Q5K => (Some(&block[16..48]), &block[48..176]),
                    _ => (None, &block[16..144]),
                };
                for e in 0..256 {
                    let (c, t, l) = (e / 64, e % 64 / 32, e % 32);
                    let mut q = (codes[c * 32 + l] >> (4 * t)) & 0x0f;
                    if let Some(high) = high {
                        q |= ((high[l] >> (e / 32)) & 1) << 4;
                    }
                    let k = e / 32;
                    out.push(d * scale[k] * f64::from(q) - dmin * minimum[k]);
                }
            }
            QuantFormat::Q6K => {
                let (low, high, scales) = (&block[..128], &block[128..192], &block[192..208]);
                let d = half(block, 208);
                for f in 0..256 {
                    let (h, t, m) = (f / 128, f % 128 / 64, f % 64);
                    let ql = (low[h * 64 + m] >> (4 * t)) & 0x0f;
                    let (s, l) = (f % 128 / 32, f % 32);
                    let qh = (high[h * 32 + l] >> (2 * s)) & 0x03;
                    let q = i32::from(ql | (qh << 4)) - 32;
                    out.push(d * f64::from(scales[f / 16] as i8) * f64::from(q));
                }
            }
        }
    }
    Ok(out)
}

/// Quantizes `values`, a whole number of blocks, to `dtype` by rounding to
/// the nearest code, block by block: the reference quantization.
///
/// - Kapel's int8, as Smallem does: a scale `max |x| / 127` per group, codes
///   rounded to the nearest, ties away from zero.
/// - Kapel's int4: the scale of each group that, with codes rounded to the
///   nearest between −8 and 7, makes the smallest squared error, among the
///   scales that put the largest magnitude at either end and those between.
/// - GGUF's Q8_0 and Q4_0, as llama.cpp's reference (`quantize_row_q8_0_ref`,
///   `quantize_row_q4_0_ref`) does: Q8_0 like int8 with an `f16` scale; Q4_0
///   with the scale `m / −8`, `m` the value of largest magnitude, and codes
///   `x / scale + 8`, truncated after adding a half, at most 15.
///
/// A block of zeros has a scale of zero.
///
/// # Errors
///
/// [`Error::UnsupportedDType`] for another type, [`Error::BlockMisaligned`] if
/// `values` is not a whole number of blocks.
pub fn quantize(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    let unsupported = Err(Error::UnsupportedDType {
        op: "quantize",
        dtype,
    });
    let DType::Quantized(format) = dtype else {
        return unsupported;
    };
    let len = dtype.block_len();
    let mut out = Vec::with_capacity(dtype.storage_bytes(values.len())?);
    let half = |x: f64| float::F16.encode(x).to_le_bytes();
    let largest = |block: &[f32]| block.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    for block in values.chunks_exact(len) {
        match format {
            QuantFormat::Int8 { .. } => {
                let scale = (largest(block) / 127.0) as f32;
                let inverse = if scale > 0.0 {
                    1.0 / f64::from(scale)
                } else {
                    0.0
                };
                out.extend(scale.to_le_bytes());
                out.extend(
                    block.iter().map(|&x| {
                        (f64::from(x) * inverse).round().clamp(-127.0, 127.0) as i8 as u8
                    }),
                );
            }
            QuantFormat::Int4 { .. } => {
                let (scale, codes) = int4_codes(block);
                out.extend(scale.to_le_bytes());
                for pair in codes.as_chunks::<2>().0 {
                    out.push(pair[0] | (pair[1] << 4));
                }
            }
            QuantFormat::Q8_0 => {
                let d = largest(block) / 127.0;
                let inverse = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend(half(d)[..2].iter());
                out.extend(
                    block.iter().map(|&x| {
                        (f64::from(x) * inverse).round().clamp(-127.0, 127.0) as i8 as u8
                    }),
                );
            }
            QuantFormat::Q4_0 => {
                // The value of largest magnitude, with its sign.
                let m = block
                    .iter()
                    .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                let d = f64::from(m) / -8.0;
                let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
                let code = |x: f32| ((f64::from(x) * inverse + 8.5) as i32).min(15) as u8;
                out.extend(half(d)[..2].iter());
                for j in 0..16 {
                    out.push(code(block[j]) | (code(block[j + 16]) << 4));
                }
            }
            _ => return unsupported,
        }
    }
    if !values.len().is_multiple_of(len) {
        return Err(Error::BlockMisaligned {
            dtype,
            elements: values.len(),
        });
    }
    Ok(out)
}

/// The scale and the codes (0 to 15, for `scale × (code − 8)`) of a group of
/// Kapel int4 that bring it closest to `block`, in squared error. With 16
/// levels, `−8` to `7`, the largest magnitude `m` may take either end, and
/// the scale that rounds best is not always `|m| / 7`: scales around it are
/// tried, each followed by the least-squares scale of its codes.
fn int4_codes(block: &[f32]) -> (f32, Vec<u8>) {
    let m = block
        .iter()
        .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    let mut best = (0.0f32, vec![8u8; block.len()], f64::INFINITY);
    if m == 0.0 {
        return (best.0, best.1);
    }
    let codes_of = |scale: f64| -> Vec<u8> {
        block
            .iter()
            .map(|&x| ((f64::from(x) / scale).round().clamp(-8.0, 7.0) + 8.0) as u8)
            .collect()
    };
    let error = |scale: f32, codes: &[u8]| -> f64 {
        block
            .iter()
            .zip(codes)
            .map(|(&x, &c)| {
                let e = f64::from(x) - f64::from(scale) * (f64::from(c) - 8.0);
                e * e
            })
            .sum()
    };
    // `m` at −8 (the end with more room), then at 7, and 18 levels between.
    for step in 0..=20 {
        let levels = -8.0 + 15.0 * f64::from(step) / 20.0;
        if levels.abs() < 6.5 {
            continue;
        }
        let scale = f64::from(m) / levels;
        let codes = codes_of(scale);
        // The best scale for these codes, then the codes of that scale.
        let (num, den) = block
            .iter()
            .zip(&codes)
            .fold((0.0, 0.0), |(n, d), (&x, &c)| {
                let c = f64::from(c) - 8.0;
                (n + f64::from(x) * c, d + c * c)
            });
        for (scale, codes) in [(scale, codes.clone()), (num / den, codes_of(num / den))] {
            let scale = scale as f32;
            if !(scale.is_finite() && scale != 0.0) {
                continue;
            }
            let e = error(scale, &codes);
            if e < best.2 {
                best = (scale, codes, e);
            }
        }
    }
    (best.0, best.1)
}

/// The eight 6-bit scales and minimums of a Q4_K or Q5_K block, packed in 12
/// bytes: the first four of each in the low 6 bits of bytes 0–3 and 4–7, the
/// last four split between the low and high nibbles of bytes 8–11 and the top
/// 2 bits of bytes 0–7.
fn k_scales(packed: &[u8]) -> ([f64; 8], [f64; 8]) {
    let (d, m, md) = (&packed[..4], &packed[4..8], &packed[8..12]);
    let mut scale = [0.0; 8];
    let mut minimum = [0.0; 8];
    for i in 0..4 {
        scale[i] = f64::from(d[i] & 0x3f);
        minimum[i] = f64::from(m[i] & 0x3f);
        scale[4 + i] = f64::from((md[i] & 0x0f) | ((d[i] >> 2) & 0x30));
        minimum[4 + i] = f64::from((md[i] >> 4) | ((m[i] >> 2) & 0x30));
    }
    (scale, minimum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_types_round_trip_their_values() {
        let cases: [(DType, f64); 10] = [
            (DType::F64, 0.1),
            (DType::F32, 1.5),
            (DType::F16, -2.25),
            (DType::BF16, 3.0),
            (DType::F8E4M3, 0.875),
            (DType::F8E5M2, -96.0),
            (DType::I8, -100.0),
            (DType::U8, 200.0),
            (DType::I32, -70_000.0),
            (DType::U32, 4_000_000_000.0),
        ];
        for (dtype, value) in cases {
            let mut bytes = vec![0; dtype.block_bytes()];
            dtype.encode(value, &mut bytes).unwrap();
            assert_eq!(dtype.decode(&bytes).unwrap(), value, "{dtype}");
        }
    }

    #[test]
    fn integers_round_and_saturate() {
        let mut byte = [0u8];
        DType::I8.encode(300.0, &mut byte).unwrap();
        assert_eq!(DType::I8.decode(&byte).unwrap(), 127.0);
        DType::U8.encode(-4.0, &mut byte).unwrap();
        assert_eq!(DType::U8.decode(&byte).unwrap(), 0.0);
        DType::I8.encode(2.5, &mut byte).unwrap();
        assert_eq!(DType::I8.decode(&byte).unwrap(), 2.0);
        DType::I8.encode(f64::NAN, &mut byte).unwrap();
        assert_eq!(DType::I8.decode(&byte).unwrap(), 0.0);
    }

    #[test]
    fn quantized_sizes_count_whole_blocks() {
        let int4 = DType::int4(64).unwrap();
        assert_eq!(int4.storage_bytes(128).unwrap(), 2 * 36);
        assert_eq!(
            int4.storage_bytes(100),
            Err(Error::BlockMisaligned {
                dtype: int4,
                elements: 100
            })
        );
        assert_eq!(
            DType::Quantized(QuantFormat::Q4K)
                .storage_bytes(512)
                .unwrap(),
            288
        );
        assert!(DType::int4(7).is_err());
        assert!(DType::int8(0).is_err());
    }

    #[test]
    fn kapel_formats_dequantize() {
        let mut block = 0.5f32.to_le_bytes().to_vec();
        block.extend([0x0f, 0x80]); // low nibble 15, high 0; then 0, 8
        assert_eq!(
            dequantize(DType::int4(4).unwrap(), &block).unwrap(),
            vec![3.5, -4.0, -4.0, 0.0]
        );
        let mut block = 2.0f32.to_le_bytes().to_vec();
        block.extend([0xff, 0x02]);
        assert_eq!(
            dequantize(DType::int8(2).unwrap(), &block).unwrap(),
            vec![-2.0, 4.0]
        );
    }

    #[test]
    fn q4_0_puts_low_nibbles_first() {
        let mut block = 0x3c00u16.to_le_bytes().to_vec(); // f16 1.0
        block.extend((0..16u8).map(|j| (j % 16) | (15 - j % 16) << 4));
        let values = dequantize(DType::Quantized(QuantFormat::Q4_0), &block).unwrap();
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], -8.0);
        assert_eq!(values[15], 7.0);
        assert_eq!(values[16], 7.0);
        assert_eq!(values[31], -8.0);
    }

    #[test]
    fn quantization_rounds_to_the_nearest_code() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 10.0).collect();
        for dtype in [
            DType::int8(32).unwrap(),
            DType::int4(64).unwrap(),
            DType::Quantized(QuantFormat::Q8_0),
            DType::Quantized(QuantFormat::Q4_0),
        ] {
            let bytes = quantize(dtype, &values).unwrap();
            assert_eq!(bytes.len(), dtype.storage_bytes(64).unwrap());
            let back = dequantize(dtype, &bytes).unwrap();
            // Within half a step of the largest code of each block.
            let top = match dtype {
                DType::Quantized(QuantFormat::Q4_0) => 8.0,
                _ => 127.0,
            };
            let len = dtype.block_len();
            if let DType::Quantized(QuantFormat::Int4 { .. }) = dtype {
                // Squared error at most that of the symmetric scale `max / 7`.
                for (block, back) in values.chunks(len).zip(back.chunks(len)) {
                    let step = block.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())) / 7.0;
                    let plain: f64 = block
                        .iter()
                        .map(|&x| {
                            let x = f64::from(x);
                            let e = x - step * (x / step).round().clamp(-7.0, 7.0);
                            e * e
                        })
                        .sum();
                    let chosen: f64 = block
                        .iter()
                        .zip(back)
                        .map(|(&x, &y)| (f64::from(x) - y).powi(2))
                        .sum();
                    assert!(
                        chosen <= plain * (1.0 + 1e-9),
                        "{dtype}: {chosen} > {plain}"
                    );
                }
                continue;
            }
            for (block, back) in values.chunks(len).zip(back.chunks(len)) {
                let step = block.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())) / top;
                for (&x, &y) in block.iter().zip(back) {
                    // GGUF's codes are those of the scale before its
                    // rounding to `f16`, as in llama.cpp: up to 127 times
                    // that rounding more.
                    let slack = match dtype {
                        DType::Quantized(QuantFormat::Q8_0 | QuantFormat::Q4_0) => 0.07,
                        _ => 0.001,
                    };
                    let error = (f64::from(x) - y).abs();
                    assert!(error <= (0.5 + slack) * step, "{dtype}: {x} → {y}");
                }
            }
        }
        // Zeros stay zeros.
        let zeros = quantize(DType::int4(32).unwrap(), &[0.0; 32]).unwrap();
        assert!(
            dequantize(DType::int4(32).unwrap(), &zeros)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(quantize(DType::int8(32).unwrap(), &[1.0; 31]).is_err());
    }

    #[test]
    fn errors_say_what_to_fix() {
        let message = DType::int4(64)
            .unwrap()
            .storage_bytes(100)
            .unwrap_err()
            .to_string();
        assert!(message.contains("blocks of 64"), "{message}");
    }
}
