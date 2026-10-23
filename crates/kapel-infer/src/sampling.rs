//! Choosing the next token from the logits.

use serde_json::Value;

/// How the next token is chosen, as `transformers` chooses it: the repetition
/// penalty first, then the temperature, `top_k`, `top_p` and `min_p`, then a
/// draw from what is left. The frequency and presence penalties of OpenAI's
/// API come before them all.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    /// The divisor of the logits. 0 for greedy decoding: always the most
    /// likely token, the first of them on a tie.
    pub temperature: f64,
    /// Only the `k` most likely tokens (and those tied with the last); `None`
    /// for all.
    pub top_k: Option<usize>,
    /// Only the most likely tokens whose probabilities add up to `top_p`; 1
    /// for all.
    pub top_p: f64,
    /// Only the tokens at least `min_p` times as likely as the most likely; 0
    /// for all.
    pub min_p: f64,
    /// The logit of every token already in the sequence, prompt included, is
    /// divided by this factor if positive, multiplied if negative; 1 for none.
    pub repetition_penalty: f64,
    /// Subtracted from the logit of each generated token, times the number
    /// of times it was generated (OpenAI's `frequency_penalty`); 0 for none.
    pub frequency_penalty: f64,
    /// Subtracted from the logit of each token generated at least once
    /// (OpenAI's `presence_penalty`); 0 for none.
    pub presence_penalty: f64,
    /// The seed of the draws: the same seed gives the same tokens.
    pub seed: u64,
}

impl Default for Sampling {
    /// Greedy decoding.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

impl Sampling {
    /// Greedy decoding.
    pub fn greedy() -> Self {
        Self::default()
    }

    /// The settings of a model's `generation_config.json`, with the defaults
    /// of `transformers` for those it leaves out: greedy unless `do_sample`;
    /// when sampling, temperature 1 and `top_k` 50.
    pub fn from_generation_config(config: &Value) -> Self {
        let number = |key: &str| config.get(key).and_then(Value::as_f64);
        let mut sampling = Self {
            repetition_penalty: number("repetition_penalty").unwrap_or(1.0),
            ..Self::default()
        };
        if config.get("do_sample").and_then(Value::as_bool) == Some(true) {
            sampling.temperature = number("temperature").unwrap_or(1.0);
            sampling.top_k = match config.get("top_k").and_then(Value::as_u64) {
                Some(0) => None,
                Some(k) => usize::try_from(k).ok(),
                None => Some(50),
            };
            sampling.top_p = number("top_p").unwrap_or(1.0);
            sampling.min_p = number("min_p").unwrap_or(0.0);
        }
        sampling
    }

    /// Whether every choice is the most likely token.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// A small generator of random numbers, the same on every machine:
/// xoshiro256**, seeded through SplitMix64.
#[derive(Clone, Debug)]
pub(crate) struct Rng([u64; 4]);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A number in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Chooses tokens with one [`Sampling`], drawing from its own generator.
#[derive(Clone, Debug)]
pub(crate) struct Sampler {
    settings: Sampling,
    rng: Rng,
}

impl Sampler {
    pub(crate) fn new(settings: Sampling) -> Self {
        let rng = Rng::new(settings.seed);
        Self { settings, rng }
    }

    /// Whether the choice is the most likely token, whatever the sequence:
    /// greedy, without a repetition penalty. It can then be made where the
    /// logits are (`kapel_ops::ArgMax`, the same rule as here).
    pub(crate) fn is_argmax(&self) -> bool {
        let s = &self.settings;
        s.is_greedy()
            && s.repetition_penalty == 1.0
            && s.frequency_penalty == 0.0
            && s.presence_penalty == 0.0
    }

    /// The next token, from the logits of the last position and the tokens
    /// of the sequence so far, the generated ones from `generated` on.
    pub(crate) fn choose(&mut self, logits: &mut [f64], sequence: &[u32], generated: usize) -> u32 {
        let choice = self.distribution(logits, sequence, generated);
        self.draw(&choice)
    }

    /// What the next token is drawn from: the logits of the last position
    /// with the penalties, the temperature and the filters applied, as
    /// [`Sampler::choose`] draws it. The logits are changed.
    pub(crate) fn distribution(
        &self,
        logits: &mut [f64],
        sequence: &[u32],
        generated: usize,
    ) -> Choice {
        let s = &self.settings;
        if s.frequency_penalty != 0.0 || s.presence_penalty != 0.0 {
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &id in &sequence[generated.min(sequence.len())..] {
                *counts.entry(id).or_default() += 1;
            }
            for (id, count) in counts {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l -= f64::from(count) * s.frequency_penalty + s.presence_penalty;
                }
            }
        }
        if s.repetition_penalty != 1.0 {
            let mut seen = vec![false; logits.len()];
            for &id in sequence {
                if let Some(done) = seen.get_mut(id as usize)
                    && !std::mem::replace(done, true)
                {
                    let l = &mut logits[id as usize];
                    *l = if *l < 0.0 {
                        *l * s.repetition_penalty
                    } else {
                        *l / s.repetition_penalty
                    };
                }
            }
        }
        if s.is_greedy() {
            return Choice::Token(argmax(logits));
        }
        for l in logits.iter_mut() {
            *l /= s.temperature;
        }
        if let Some(k) = s.top_k
            && k > 0
            && k < logits.len()
        {
            let mut sorted = logits.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let kth = sorted[k - 1];
            for l in logits.iter_mut() {
                if *l < kth {
                    *l = f64::NEG_INFINITY;
                }
            }
        }
        if s.top_p < 1.0 {
            let probs = softmax(logits);
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            // Kept while the tokens before it hold less than `top_p`: at least
            // the most likely one.
            let mut before = 0.0;
            for &i in &order {
                if before >= s.top_p {
                    logits[i] = f64::NEG_INFINITY;
                }
                before += probs[i];
            }
        }
        if s.min_p > 0.0 {
            let probs = softmax(logits);
            let floor = s.min_p * probs.iter().copied().fold(0.0, f64::max);
            for (l, p) in logits.iter_mut().zip(&probs) {
                if *p < floor {
                    *l = f64::NEG_INFINITY;
                }
            }
        }
        let probabilities = softmax(logits);
        if probabilities.iter().all(|&p| p == 0.0) {
            // Nothing left (every logit −∞ or NaN): the first largest.
            return Choice::Token(argmax(logits));
        }
        Choice::Probabilities(probabilities)
    }

    /// A token drawn from `choice`, with this sampler's generator.
    pub(crate) fn draw(&mut self, choice: &Choice) -> u32 {
        match choice {
            Choice::Token(token) => *token,
            Choice::Probabilities(probs) => {
                let u = self.rng.uniform();
                let mut total = 0.0;
                let mut last = 0;
                for (i, &p) in probs.iter().enumerate() {
                    if p > 0.0 {
                        total += p;
                        last = i as u32;
                        if u < total {
                            return i as u32;
                        }
                    }
                }
                // Rounding left `u` above the total: the last possible token.
                last
            }
        }
    }

    /// A number in `[0, 1)` from this sampler's generator.
    pub(crate) fn uniform(&mut self) -> f64 {
        self.rng.uniform()
    }
}

/// What a token is drawn from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Choice {
    /// One token, certainly: greedy decoding, or a proposal without
    /// probabilities.
    Token(u32),
    /// The probability of every token, adding up to 1.
    Probabilities(Vec<f64>),
}

impl Choice {
    /// The probability of `token`.
    pub(crate) fn probability(&self, token: u32) -> f64 {
        match self {
            Self::Token(t) => f64::from(u8::from(*t == token)),
            Self::Probabilities(p) => p.get(token as usize).copied().unwrap_or(0.0),
        }
    }
}

/// The index of the largest value, the first on a tie; NaNs are never chosen
/// unless every value is one.
fn argmax(values: &[f64]) -> u32 {
    let mut best: Option<usize> = None;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|b| v > values[b]) {
            best = Some(i);
        }
    }
    best.unwrap_or(0) as u32
}

/// The probabilities of `logits`, in `f64`; tokens at −∞ get 0.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return vec![0.0; logits.len()];
    }
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choose(settings: Sampling, logits: &[f64], sequence: &[u32]) -> u32 {
        Sampler::new(settings).choose(&mut logits.to_vec(), sequence, 0)
    }

    #[test]
    fn greedy_takes_the_first_largest() {
        assert_eq!(
            choose(Sampling::greedy(), &[1.0, 3.0, 3.0, f64::NAN], &[]),
            1
        );
    }

    #[test]
    fn the_penalty_divides_positive_logits_and_multiplies_negative_ones() {
        let s = Sampling {
            repetition_penalty: 2.0,
            ..Sampling::greedy()
        };
        // 4 → 2 falls below 3; −1 → −2 stays below 0.5.
        assert_eq!(choose(s.clone(), &[4.0, 3.0], &[0, 0]), 1);
        assert_eq!(choose(s, &[-1.0, -0.6], &[1]), 0);
    }

    #[test]
    fn openai_penalties_count_only_the_generated_tokens() {
        let s = Sampling {
            frequency_penalty: 1.0,
            presence_penalty: 0.5,
            ..Sampling::greedy()
        };
        let mut sampler = Sampler::new(s);
        // Token 0 twice in the answer: 3 − 2 − 0.5 < 1. Token 1 in the
        // prompt only keeps its logit.
        assert_eq!(sampler.choose(&mut [3.0, 1.0, 0.0], &[1, 1, 0, 0], 2), 1);
        assert!(!sampler.is_argmax());
    }

    fn counts(settings: &Sampling, logits: &[f64], draws: u64) -> Vec<u64> {
        let mut counts = vec![0; logits.len()];
        for seed in 0..draws {
            let s = Sampling {
                seed,
                ..settings.clone()
            };
            counts[choose(s, logits, &[]) as usize] += 1;
        }
        counts
    }

    #[test]
    fn draws_follow_the_probabilities() {
        let s = Sampling {
            temperature: 1.0,
            ..Sampling::greedy()
        };
        let logits = [0.0, 1.0f64.ln(), 2.0f64.ln()];
        // Probabilities 1/4, 1/4, 1/2.
        let c = counts(&s, &logits, 40_000);
        for (got, want) in c.iter().zip([10_000.0, 10_000.0, 20_000.0]) {
            assert!((*got as f64 - want).abs() < 600.0, "{c:?}");
        }
    }

    #[test]
    fn filters_keep_what_transformers_keeps() {
        let logits = [3.0, 2.0, 1.0, 0.0];
        let only = |settings: Sampling| {
            let c = counts(&settings, &logits, 2000);
            c.iter().map(|&n| n > 0).collect::<Vec<_>>()
        };
        let base = Sampling {
            temperature: 1.0,
            ..Sampling::greedy()
        };
        let top_k = Sampling {
            top_k: Some(2),
            ..base.clone()
        };
        assert_eq!(only(top_k), [true, true, false, false]);
        // Probabilities about 0.64, 0.24, 0.09, 0.03: the first two hold 0.88.
        let top_p = Sampling {
            top_p: 0.8,
            ..base.clone()
        };
        assert_eq!(only(top_p), [true, true, false, false]);
        let min_p = Sampling { min_p: 0.1, ..base };
        assert_eq!(only(min_p), [true, true, true, false]);
    }

    #[test]
    fn a_seed_gives_the_same_draws() {
        let s = Sampling {
            temperature: 1.0,
            seed: 7,
            ..Sampling::greedy()
        };
        let mut a = Sampler::new(s.clone());
        let mut b = Sampler::new(s);
        let logits = [0.1, 0.2, 0.3, 0.4];
        for _ in 0..50 {
            assert_eq!(
                a.choose(&mut logits.to_vec(), &[], 0),
                b.choose(&mut logits.to_vec(), &[], 0)
            );
        }
    }

    #[test]
    fn generation_configs_read_as_transformers_reads_them() {
        let greedy = Sampling::from_generation_config(&serde_json::json!({}));
        assert!(greedy.is_greedy());
        let sampled = Sampling::from_generation_config(&serde_json::json!({
            "do_sample": true, "temperature": 0.6, "top_p": 0.95
        }));
        assert_eq!(sampled.temperature, 0.6);
        assert_eq!(sampled.top_k, Some(50));
        assert_eq!(sampled.top_p, 0.95);
    }
}
