//! Small floating-point formats: `f16`, `bf16`, and the two `f8` formats.
//!
//! Each format is a sign bit, `E` exponent bits and `M` mantissa bits. Decoding is
//! exact: every value of these formats is a `f64`. Encoding rounds a `f64` to the
//! nearest representable value, ties to even, once: the result is the correctly
//! rounded value, as hardware conversions give it.
//!
//! | Format | E | M | Largest finite | Infinity |
//! |---|---|---|---|---|
//! | `f16` (IEEE 754 binary16) | 5 | 10 | 65 504 | yes |
//! | `bf16` (bfloat16) | 8 | 7 | 3.39 × 10³⁸ | yes |
//! | `f8` E4M3 (OCP, "FN") | 4 | 3 | 448 | no: overflow gives NaN |
//! | `f8` E5M2 (OCP) | 5 | 2 | 57 344 | yes |

/// The parameters of a small floating-point format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Format {
    exponent_bits: u32,
    mantissa_bits: u32,
    /// IEEE-like: the all-ones exponent holds infinities and NaNs. Otherwise
    /// (E4M3 "FN"), it holds finite values, and only the all-ones pattern is NaN.
    ieee: bool,
}

pub(crate) const F16: Format = Format {
    exponent_bits: 5,
    mantissa_bits: 10,
    ieee: true,
};
pub(crate) const BF16: Format = Format {
    exponent_bits: 8,
    mantissa_bits: 7,
    ieee: true,
};
pub(crate) const F8_E4M3: Format = Format {
    exponent_bits: 4,
    mantissa_bits: 3,
    ieee: false,
};
pub(crate) const F8_E5M2: Format = Format {
    exponent_bits: 5,
    mantissa_bits: 2,
    ieee: true,
};

impl Format {
    fn bias(self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    fn sign_bit(self) -> u32 {
        1 << (self.exponent_bits + self.mantissa_bits)
    }

    fn max_exponent_field(self) -> u32 {
        (1 << self.exponent_bits) - 1
    }

    fn mantissa_mask(self) -> u32 {
        (1 << self.mantissa_bits) - 1
    }

    /// The canonical quiet NaN, positive.
    fn nan(self) -> u32 {
        if self.ieee {
            (self.max_exponent_field() << self.mantissa_bits) | (1 << (self.mantissa_bits - 1))
        } else {
            (self.max_exponent_field() << self.mantissa_bits) | self.mantissa_mask()
        }
    }

    /// The bit pattern of the largest finite value, positive.
    fn max_finite(self) -> u32 {
        if self.ieee {
            ((self.max_exponent_field() - 1) << self.mantissa_bits) | self.mantissa_mask()
        } else {
            (self.max_exponent_field() << self.mantissa_bits) | (self.mantissa_mask() - 1)
        }
    }

    /// The exact value of a bit pattern.
    pub(crate) fn decode(self, bits: u32) -> f64 {
        let sign = if bits & self.sign_bit() != 0 {
            -1.0
        } else {
            1.0
        };
        let exponent = (bits >> self.mantissa_bits) & self.max_exponent_field();
        let mantissa = bits & self.mantissa_mask();
        let m = self.mantissa_bits as i32;
        if self.ieee && exponent == self.max_exponent_field() {
            return if mantissa == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }
        if !self.ieee && exponent == self.max_exponent_field() && mantissa == self.mantissa_mask() {
            return f64::NAN;
        }
        if exponent == 0 {
            // Subnormal: no implicit leading one.
            return sign * f64::from(mantissa) * pow2(1 - self.bias() - m);
        }
        let significand = f64::from((1 << self.mantissa_bits) | mantissa);
        sign * significand * pow2(exponent as i32 - self.bias() - m)
    }

    /// The bit pattern nearest to `x`, ties to even. Values beyond the largest
    /// finite value give infinity, or NaN for a format without infinities.
    pub(crate) fn encode(self, x: f64) -> u32 {
        if x.is_nan() {
            return self.nan();
        }
        let sign = if x.is_sign_negative() {
            self.sign_bit()
        } else {
            0
        };
        let a = x.abs();
        let overflow = if self.ieee {
            self.max_exponent_field() << self.mantissa_bits
        } else {
            self.nan()
        };
        if a.is_infinite() {
            return sign | overflow;
        }
        if a == 0.0 {
            return sign;
        }
        let m = self.mantissa_bits as i32;
        let min_normal_exponent = 1 - self.bias();
        let exponent = exponent_of(a);
        if exponent < min_normal_exponent {
            // Subnormal, counted in units of the smallest subnormal. Rounding up
            // to `2^M` gives the smallest normal, whose bit pattern is the same
            // integer.
            let units = (a * pow2(m - min_normal_exponent)).round_ties_even();
            return sign | units as u32;
        }
        let mut significand = (a * pow2(m - exponent)).round_ties_even() as u32;
        let mut exponent = exponent;
        if significand == 1 << (m + 1) {
            significand >>= 1;
            exponent += 1;
        }
        let field = (exponent + self.bias()) as u32;
        let bits = (field << self.mantissa_bits) | (significand & self.mantissa_mask());
        if field > self.max_exponent_field() || bits > self.max_finite() {
            return sign | overflow;
        }
        sign | bits
    }
}

/// `2^e`, exactly, for the exponents these formats use.
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The exponent `e` such that `2^e <= a < 2^(e+1)`, for a positive normal `f64`.
fn exponent_of(a: f64) -> i32 {
    let bits = a.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    if field == 0 {
        // A subnormal `f64` is far below the smallest subnormal of these formats.
        -1075
    } else {
        field - 1023
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [(Format, u32); 4] = [(F16, 16), (BF16, 16), (F8_E4M3, 8), (F8_E5M2, 8)];

    #[test]
    fn every_value_round_trips() {
        for (format, width) in ALL {
            for bits in 0..(1u32 << width) {
                let value = format.decode(bits);
                if value.is_nan() {
                    assert!(format.decode(format.encode(value)).is_nan());
                } else {
                    assert_eq!(format.encode(value), bits, "{format:?} {bits:#x}");
                }
            }
        }
    }

    #[test]
    fn the_largest_values_are_the_published_ones() {
        assert_eq!(F16.decode(F16.max_finite()), 65_504.0);
        assert_eq!(F8_E4M3.decode(F8_E4M3.max_finite()), 448.0);
        assert_eq!(F8_E5M2.decode(F8_E5M2.max_finite()), 57_344.0);
        assert_eq!(BF16.decode(BF16.max_finite()), (2.0 - pow2(-7)) * pow2(127));
        assert_eq!(F16.decode(1), pow2(-24));
        assert_eq!(F8_E4M3.decode(1), pow2(-9));
        assert_eq!(F8_E5M2.decode(1), pow2(-16));
    }

    #[test]
    fn ties_go_to_even() {
        // Between 1 and the next f16, 1 + 2^-10, the midpoint rounds down to 1
        // (even mantissa); the next midpoint rounds up to 1 + 2^-9.
        assert_eq!(F16.decode(F16.encode(1.0 + pow2(-11))), 1.0);
        assert_eq!(
            F16.decode(F16.encode(1.0 + 3.0 * pow2(-11))),
            1.0 + pow2(-9)
        );
        // Just above the midpoint rounds up.
        assert_eq!(
            F16.decode(F16.encode(1.0 + pow2(-11) + pow2(-30))),
            1.0 + pow2(-10)
        );
    }

    #[test]
    fn bf16_matches_rounding_an_f32() {
        // bf16 is the top half of an f32, rounded to nearest even.
        let values = [1.0f32, -2.5, 3.140_625, 1.0e-30, 6.5e30, 0.1, 1.0 / 3.0];
        for value in values {
            let bits = value.to_bits();
            let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
            assert_eq!(BF16.encode(f64::from(value)), rounded, "{value}");
        }
    }

    #[test]
    fn overflow_gives_infinity_or_nan() {
        assert!(F16.decode(F16.encode(70_000.0)).is_infinite());
        assert!(F8_E5M2.decode(F8_E5M2.encode(1.0e6)).is_infinite());
        assert!(F8_E4M3.decode(F8_E4M3.encode(500.0)).is_nan());
        assert!(F8_E4M3.decode(F8_E4M3.encode(f64::INFINITY)).is_nan());
        // Up to the midpoint 464 between 448 and the next step (a NaN pattern),
        // values still round to 448.
        assert_eq!(F8_E4M3.decode(F8_E4M3.encode(460.0)), 448.0);
    }

    #[test]
    fn signs_and_zeros_are_kept() {
        for (format, _) in ALL {
            assert_eq!(format.encode(-0.0), format.sign_bit());
            assert_eq!(format.decode(format.encode(-1.5)), -1.5);
        }
    }

    #[test]
    fn tiny_values_round_to_zero_or_the_smallest_subnormal() {
        let smallest = F16.decode(1);
        assert_eq!(F16.encode(smallest * 0.5), 0);
        assert_eq!(F16.encode(smallest * 0.51), 1);
        assert_eq!(F16.encode(1.0e-300), 0);
    }
}
