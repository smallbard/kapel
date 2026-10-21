//! Rotary position embeddings.

use std::f64::consts::PI;

use kapel_core::{Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, indices, same_device, shape};

/// How the pairs of rotated components are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeLayout {
    /// Component `j` pairs with `j + R/2` (GPT-NeoX, Llama and most models in
    /// `transformers`).
    Half,
    /// Component `2j` pairs with `2j + 1` (GPT-J, the original RoPE paper).
    Interleaved,
}

/// How the frequencies are adapted to contexts longer than training.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum RopeScaling {
    /// The frequencies of RoPE: `ωⱼ = θ^(−2j/R)`.
    None,
    /// Positions divided by `factor` ("linear", position interpolation).
    Linear {
        /// The divisor of the positions.
        factor: f64,
    },
    /// NTK-aware scaling: `θ` replaced by `θ · factor^(R/(R−2))`.
    Ntk {
        /// The context extension factor.
        factor: f64,
    },
    /// Llama 3.1: low frequencies divided by `factor`, high ones kept, and a
    /// smooth passage in between.
    Llama3 {
        /// The divisor of the low frequencies.
        factor: f64,
        /// Wavelengths above `original_max_position / low_freq_factor` are scaled.
        low_freq_factor: f64,
        /// Wavelengths below `original_max_position / high_freq_factor` are kept.
        high_freq_factor: f64,
        /// The context length of training.
        original_max_position: usize,
    },
    /// YaRN: a ramp between interpolated and original frequencies, by
    /// wavelength, and an attention factor that multiplies the rotation.
    Yarn {
        /// The context extension factor.
        factor: f64,
        /// The context length of training.
        original_max_position: usize,
        /// Rotations above which frequencies are kept (32 by default).
        beta_fast: f64,
        /// Rotations below which frequencies are interpolated (1 by default).
        beta_slow: f64,
        /// The factor of cos and sin; `None` for `0.1 · ln(factor) + 1`.
        attention_factor: Option<f64>,
    },
}

/// Rotary position embeddings.
///
/// Inputs: `x [..., D]`, and `positions`, integers that broadcast to
/// `x.shape[..rank−1]`. The first `R` components of each vector (`rotary_dim`,
/// all `D` by default, even) are rotated in `R/2` pairs `(a, b)` chosen by
/// `layout`; pair `j` at position `p` turns by the angle `p · ωⱼ`:
///
/// `a' = s · (a cos − b sin)`, `b' = s · (b cos + a sin)`,
///
/// with the frequencies `ωⱼ` and the factor `s` (1 except for YaRN) of
/// [`Rope::frequencies`] and [`Rope::attention_factor`]. The other components
/// pass unchanged. Positions receive no gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// The base `θ` of the frequencies (10 000 in the paper, 500 000 for Llama 3).
    pub theta: f64,
    /// The pairing of the components.
    pub layout: RopeLayout,
    /// The number of rotated components; `None` for all of them.
    pub rotary_dim: Option<usize>,
    /// The adaptation to longer contexts.
    pub scaling: RopeScaling,
}

impl Rope {
    /// The number of rotated components for vectors of width `d`.
    pub fn rotated(&self, d: usize) -> usize {
        self.rotary_dim.unwrap_or(d)
    }

    /// The `R/2` angular frequencies `ωⱼ`, in radians per position, and the
    /// division of positions for linear scaling folded in.
    pub fn frequencies(&self, r: usize) -> Vec<f64> {
        let half = r / 2;
        let base = |theta: f64| -> Vec<f64> {
            (0..half)
                .map(|j| theta.powf(-((2 * j) as f64) / r as f64))
                .collect()
        };
        match self.scaling {
            RopeScaling::None => base(self.theta),
            RopeScaling::Linear { factor } => base(self.theta).iter().map(|w| w / factor).collect(),
            RopeScaling::Ntk { factor } => {
                base(self.theta * factor.powf(r as f64 / (r as f64 - 2.0)))
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            } => {
                let original = original_max_position as f64;
                let low_wavelength = original / low_freq_factor;
                let high_wavelength = original / high_freq_factor;
                base(self.theta)
                    .into_iter()
                    .map(|w| {
                        let wavelength = 2.0 * PI / w;
                        if wavelength < high_wavelength {
                            w
                        } else if wavelength > low_wavelength {
                            w / factor
                        } else {
                            let smooth = (original / wavelength - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1.0 - smooth) * w / factor + smooth * w
                        }
                    })
                    .collect()
            }
            RopeScaling::Yarn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
                ..
            } => {
                // The dimension at which a frequency makes `rotations` turns over
                // the original context.
                let correction = |rotations: f64| {
                    r as f64 * (original_max_position as f64 / (rotations * 2.0 * PI)).ln()
                        / (2.0 * self.theta.ln())
                };
                let low = correction(beta_fast).floor().max(0.0);
                let mut high = correction(beta_slow).ceil().min(r as f64 - 1.0);
                if low == high {
                    high += 0.001;
                }
                base(self.theta)
                    .into_iter()
                    .enumerate()
                    .map(|(j, w)| {
                        let ramp = ((j as f64 - low) / (high - low)).clamp(0.0, 1.0);
                        let keep = 1.0 - ramp;
                        (w / factor) * (1.0 - keep) + w * keep
                    })
                    .collect()
            }
        }
    }

    /// The factor `s` of the rotation: `0.1 · ln(factor) + 1` for YaRN unless
    /// given, 1 otherwise.
    pub fn attention_factor(&self) -> f64 {
        match self.scaling {
            RopeScaling::Yarn {
                attention_factor: Some(s),
                ..
            } => s,
            RopeScaling::Yarn { factor, .. } if factor > 1.0 => 0.1 * factor.ln() + 1.0,
            _ => 1.0,
        }
    }
}

impl Rope {
    /// Checks that the frequencies of `r` rotated components are finite and
    /// positive.
    fn check_parameters(&self, r: usize) -> Result<()> {
        let invalid = |message: String| {
            Err(Error::InvalidArgument {
                op: Self::NAME,
                message,
            })
        };
        let positive = |what: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                invalid(format!("{what} must be finite and positive, got {value}"))
            }
        };
        positive("theta", self.theta)?;
        match self.scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => positive("factor", factor)?,
            RopeScaling::Ntk { factor } => {
                positive("factor", factor)?;
                if r <= 2 {
                    return invalid(format!(
                        "NTK scaling needs more than 2 rotated components, got {r}"
                    ));
                }
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            } => {
                positive("factor", factor)?;
                positive("low_freq_factor", low_freq_factor)?;
                positive("high_freq_factor", high_freq_factor)?;
                if high_freq_factor <= low_freq_factor {
                    return invalid(format!(
                        "high_freq_factor ({high_freq_factor}) must exceed low_freq_factor \
                         ({low_freq_factor})"
                    ));
                }
                if original_max_position == 0 {
                    return invalid("original_max_position must be positive".into());
                }
            }
            RopeScaling::Yarn {
                factor,
                original_max_position,
                beta_fast,
                beta_slow,
                attention_factor,
            } => {
                positive("factor", factor)?;
                positive("beta_fast", beta_fast)?;
                positive("beta_slow", beta_slow)?;
                if let Some(s) = attention_factor {
                    positive("attention_factor", s)?;
                }
                if self.theta <= 1.0 {
                    return invalid(format!("YaRN needs theta above 1, got {}", self.theta));
                }
                if original_max_position == 0 {
                    return invalid("original_max_position must be positive".into());
                }
            }
        }
        Ok(())
    }
}

impl Operation for Rope {
    const NAME: &'static str = "rope";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        arity(op, inputs, &[2], "2")?;
        same_device(op, inputs)?;
        let (x, positions) = (inputs[0], inputs[1]);
        float(op, x)?;
        indices(op, positions)?;
        let d = x.shape().last().copied().unwrap_or(0);
        let r = self.rotated(d);
        if r == 0 || !r.is_multiple_of(2) || r > d {
            return Err(Error::InvalidArgument {
                op,
                message: format!("{r} rotated components: need an even number, at most {d}"),
            });
        }
        self.check_parameters(r)?;
        let lead = &x.shape()[..x.rank() - 1];
        if shape::broadcast(op, positions.shape(), lead)? != lead {
            return Err(Error::BroadcastIncompatible {
                op,
                from: positions.shape().to_vec(),
                to: lead.to_vec(),
            });
        }
        Ok(Signature {
            shape: x.shape().to_vec(),
            dtype: x.dtype(),
        })
    }

    fn differentiable(&self, index: usize) -> bool {
        index == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rope(scaling: RopeScaling) -> Rope {
        Rope {
            theta: 10_000.0,
            layout: RopeLayout::Half,
            rotary_dim: None,
            scaling,
        }
    }

    #[test]
    fn plain_frequencies_follow_the_paper() {
        let w = rope(RopeScaling::None).frequencies(8);
        assert_eq!(w.len(), 4);
        assert_eq!(w[0], 1.0);
        assert!((w[1] - 10_000f64.powf(-0.25)).abs() < 1e-15);
    }

    #[test]
    fn llama3_keeps_high_frequencies_and_divides_low_ones() {
        let scaling = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        };
        let plain = Rope {
            theta: 500_000.0,
            ..rope(RopeScaling::None)
        }
        .frequencies(128);
        let scaled = Rope {
            theta: 500_000.0,
            ..rope(scaling)
        }
        .frequencies(128);
        assert_eq!(scaled[0], plain[0]);
        assert!((scaled[63] - plain[63] / 8.0).abs() < 1e-18);
    }

    #[test]
    fn yarn_interpolates_low_frequencies_and_scales_the_rotation() {
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        };
        let plain = rope(RopeScaling::None).frequencies(64);
        let yarn = rope(scaling);
        let scaled = yarn.frequencies(64);
        assert_eq!(scaled[0], plain[0]);
        assert!((scaled[31] - plain[31] / 4.0).abs() < 1e-15);
        assert!((yarn.attention_factor() - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-15);
    }
}
