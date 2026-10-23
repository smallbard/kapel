//! Speculative decoding: a cheap proposer guesses the next tokens, the model
//! checks them all in one pass.
//!
//! In decoding, a pass reads every weight of the model for one token, and
//! costs about as much for a few tokens: the device waits for memory. A
//! proposer guesses the `k` tokens that follow; the model computes the
//! positions of all of them in one pass, as in training, where each position
//! predicts the next. When the guesses are good, a pass gives several tokens.
//!
//! The check does not keep the guesses the model would have drawn: that would
//! be another draw. It accepts each proposal `x`, drawn from the proposer's
//! distribution `q`, with probability `min(1, p(x) / q(x))`, `p` being the
//! model's distribution (after the request's temperature, filters and
//! penalties); at the first refusal, it draws a token from `max(0, p − q)`,
//! renormalized, and stops. When every proposal is accepted, the pass gives
//! one token more, drawn from `p`. The tokens then follow the model's
//! distribution exactly (Leviathan, Kalman and Matias, 2023; Chen et al.,
//! 2023): greedy decoding gives the same tokens as without a proposer.
//!
//! Three proposers ([`Draft`]): a smaller model of the same tokenizer; the
//! tokens that followed the last ones where they last appeared in the
//! sequence (prompt lookup, no second model); and the model's own
//! multi-token prediction layer, for models that have one.

use std::collections::VecDeque;

use kapel_core::Tensor;
use kapel_models::{Cache, RecordedStep};
use kapel_ops::{ArgMax, Backend, Transfer};

use crate::generate::{Finish, Piece, Pipeline, Request};
use crate::sampling::{Choice, Sampler, Sampling};
use crate::text::TextStream;
use crate::{InferError, Result};

/// What the check of `k` proposals gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Verdict {
    /// The proposals accepted, the first ones, from 0 to `k`.
    pub(crate) accepted: usize,
    /// The token that follows them: drawn from the corrected distribution
    /// after a refusal, from `p` when all were accepted.
    pub(crate) next: u32,
}

/// Checks `proposed`, drawn one by one from the proposer's distributions
/// `draft`, against the model's `target` at the same positions and one more
/// (`k + 1` in all), drawing with `sampler`'s generator.
pub(crate) fn verify(
    sampler: &mut Sampler,
    proposed: &[u32],
    draft: &[Choice],
    target: &[Choice],
) -> Verdict {
    assert!(
        draft.len() == proposed.len() && target.len() == proposed.len() + 1,
        "{} proposals, {} distributions of the proposer, {} of the model",
        proposed.len(),
        draft.len(),
        target.len()
    );
    for (i, &x) in proposed.iter().enumerate() {
        let (p, q) = (target[i].probability(x), draft[i].probability(x));
        // `q > 0`: the proposer drew `x`. A certain choice needs no draw.
        let keep = p >= q || (p > 0.0 && sampler.uniform() * q < p);
        if keep {
            continue;
        }
        let next = match &target[i] {
            Choice::Token(token) => *token,
            Choice::Probabilities(p) => {
                let residual: Vec<f64> = p
                    .iter()
                    .enumerate()
                    .map(|(t, &p)| (p - draft[i].probability(t as u32)).max(0.0))
                    .collect();
                let total: f64 = residual.iter().sum();
                if total > 0.0 {
                    let residual = residual.iter().map(|r| r / total).collect();
                    sampler.draw(&Choice::Probabilities(residual))
                } else {
                    // `p = q`: a refusal then has probability 0; to rounding,
                    // a draw from `p`.
                    sampler.draw(&target[i])
                }
            }
        };
        return Verdict { accepted: i, next };
    }
    Verdict {
        accepted: proposed.len(),
        next: sampler.draw(&target[proposed.len()]),
    }
}

/// How [`Pipeline::speculate`] proposes tokens.
pub enum Draft<'d, B> {
    /// A smaller model of the same tokenizer, on the same backend: it
    /// proposes `tokens` tokens per pass of the model, one at a time, drawn
    /// with the request's sampling.
    Model {
        /// The proposer.
        pipeline: &'d Pipeline<B>,
        /// The proposals per pass.
        tokens: usize,
    },
    /// The tokens that followed the last `n` tokens where they last appeared
    /// in the sequence (prompt lookup), `n` from `longest` down to
    /// `shortest`, up to `tokens` of them: no second model, and good guesses
    /// when the answer repeats its prompt (code edits, summaries, quotes).
    Ngram {
        /// The proposals per pass, at most.
        tokens: usize,
        /// The longest end of the sequence looked up.
        longest: usize,
        /// The shortest.
        shortest: usize,
    },
    /// The model's multi-token prediction layer (MiMo): one proposal per
    /// pass, from the hidden state the pass gives.
    MultiToken,
}

/// The counts of a speculative generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculationStats {
    /// Passes of the model.
    pub passes: usize,
    /// Tokens proposed.
    pub proposed: usize,
    /// Tokens accepted.
    pub accepted: usize,
}

impl SpeculationStats {
    /// The share of the proposals accepted.
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.proposed.max(1) as f64
    }
}

/// The proposer's side of a [`Speculative`] generation: one per generation,
/// so its size does not matter.
#[allow(clippy::large_enum_variant)]
enum Proposer<'d, B> {
    Model {
        pipeline: &'d Pipeline<B>,
        tokens: usize,
        cache: Cache,
        recorded: Option<RecordedStep>,
        sampler: Sampler,
    },
    Ngram {
        tokens: usize,
        longest: usize,
        shortest: usize,
    },
    MultiToken(crate::multi_token::MultiToken),
}

/// The tokens a [`Pipeline`] generates after a prompt with a proposer, as
/// they come; see the module's documentation. The same items as
/// [`crate::Generation`].
pub struct Speculative<'p, 'd, B> {
    pipeline: &'p Pipeline<B>,
    cache: Cache,
    proposer: Proposer<'d, B>,
    /// The prompt and the tokens generated so far; the last is not in the
    /// cache yet.
    sequence: Vec<u32>,
    prompt_len: usize,
    sampler: Sampler,
    text: TextStream,
    max_tokens: usize,
    ignore_eos: bool,
    finish: Option<Finish>,
    /// Pieces computed and not yet given.
    ready: VecDeque<Piece>,
    stats: SpeculationStats,
}

impl<B: Backend + Transfer> Pipeline<B> {
    /// The tokens that follow `prompt`, as [`Pipeline::generate`] gives them,
    /// several per pass of the model when `draft` guesses well. Greedy
    /// decoding gives the same tokens; sampling draws from the same
    /// distribution.
    ///
    /// # Errors
    ///
    /// If `prompt` is empty, the request has a constraint (not supported
    /// here), or the proposer does not fit the model: another vocabulary, or
    /// no multi-token layer.
    pub fn speculate<'d>(
        &self,
        prompt: &[u32],
        request: &Request,
        draft: Draft<'d, B>,
    ) -> Result<Speculative<'_, 'd, B>> {
        if prompt.is_empty() {
            return Err(InferError::EmptyPrompt);
        }
        if request.constraint.is_some() {
            return Err(InferError::Unsupported(
                "speculative decoding of an answer held to a constraint".into(),
            ));
        }
        let proposer = match draft {
            Draft::Model { pipeline, tokens } => {
                // The tables of logits may be padded differently; the tokens
                // must be the same.
                let (a, b) = (pipeline.tokenizer(), self.tokenizer());
                if a.vocab_size() != b.vocab_size()
                    || (0..a.vocab_size() as u32).any(|id| a.token_bytes(id) != b.token_bytes(id))
                {
                    return Err(InferError::Unsupported(
                        "a proposer with another vocabulary".into(),
                    ));
                }
                // Its own draws: the same seed would tie them to the model's.
                let sampling = Sampling {
                    seed: request.sampling.seed ^ 0x5eed_d4af_7000_0001,
                    ..request.sampling.clone()
                };
                Proposer::Model {
                    pipeline,
                    tokens,
                    cache: pipeline.new_cache(),
                    recorded: None,
                    sampler: Sampler::new(sampling),
                }
            }
            Draft::Ngram {
                tokens,
                longest,
                shortest,
            } => Proposer::Ngram {
                tokens,
                longest: longest.max(1),
                shortest: shortest.clamp(1, longest.max(1)),
            },
            Draft::MultiToken => Proposer::MultiToken(crate::multi_token::MultiToken::new(
                self,
                &request.sampling,
            )?),
        };
        Ok(Speculative {
            pipeline: self,
            cache: self.new_cache(),
            proposer,
            sequence: prompt.to_vec(),
            prompt_len: prompt.len(),
            sampler: Sampler::new(request.sampling.clone()),
            text: TextStream::new(request.stop.clone()),
            max_tokens: request.max_tokens,
            ignore_eos: request.ignore_eos,
            finish: (request.max_tokens == 0).then_some(Finish::Length),
            ready: VecDeque::new(),
            stats: SpeculationStats::default(),
        })
    }

    /// An empty cache of one sequence, of the type [`crate::LoadOptions`]
    /// chose.
    pub(crate) fn new_cache(&self) -> Cache {
        match self.cache_dtype() {
            Some(dtype) => Cache::with_dtype(dtype),
            None => Cache::new(),
        }
    }

    /// The most likely token of each row of `logits`, found on the backend.
    pub(crate) fn most_likely_rows(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let best = self.backend().forward(
            &ArgMax {
                dim: 1,
                keepdim: false,
            },
            &[logits],
        )?;
        Ok(self.backend().download(&best)?.to_vec::<u32>()?)
    }
}

/// The tokens that followed the last `n` tokens of `sequence` where they last
/// appeared before, `n` from `longest` down to `shortest`, at most `count`.
pub(crate) fn lookup(sequence: &[u32], count: usize, longest: usize, shortest: usize) -> Vec<u32> {
    let len = sequence.len();
    for n in (shortest..=longest.min(len.saturating_sub(1))).rev() {
        let end = &sequence[len - n..];
        // The latest earlier occurrence, so that what follows it is known.
        if let Some(at) = (0..len - n).rev().find(|&i| &sequence[i..i + n] == end) {
            let from = at + n;
            return sequence[from..(from + count).min(len)].to_vec();
        }
    }
    Vec::new()
}

impl<B: Backend + Transfer> Speculative<'_, '_, B> {
    /// The tokens generated so far.
    pub fn tokens(&self) -> &[u32] {
        &self.sequence[self.prompt_len..]
    }

    /// Why the generation ended, once it has.
    pub fn finish(&self) -> Option<&Finish> {
        self.finish.as_ref()
    }

    /// The counts so far.
    pub fn stats(&self) -> SpeculationStats {
        self.stats
    }

    /// Proposals for the tokens after the sequence, with what each was drawn
    /// from, `k` at most.
    fn propose(&mut self, k: usize) -> Result<(Vec<u32>, Vec<Choice>)> {
        let (mut proposed, mut drawn) = (Vec::new(), Vec::new());
        match &mut self.proposer {
            Proposer::Model {
                pipeline,
                tokens,
                cache,
                recorded,
                sampler,
            } => {
                let p = *pipeline;
                let k = k.min(*tokens);
                let mut pending = self.sequence[cache.len()..].to_vec();
                for _ in 0..k {
                    let logits = match pending[..] {
                        [token] if !cache.is_empty() => {
                            cache.recorded_step(p.model(), p.backend(), recorded, token)?
                        }
                        _ => None,
                    };
                    let logits = match logits {
                        Some(logits) => logits,
                        None => p.model().step(p.backend(), &pending, cache)?,
                    };
                    let choice = if sampler.is_argmax() {
                        Choice::Token(p.most_likely_rows(&logits)?[0])
                    } else {
                        let mut values = p.backend().download(&logits)?.to_f64_vec()?;
                        let mut context = self.sequence.clone();
                        context.extend_from_slice(&proposed);
                        sampler.distribution(&mut values, &context, self.prompt_len)
                    };
                    let token = sampler.draw(&choice);
                    proposed.push(token);
                    drawn.push(choice);
                    pending = vec![token];
                }
            }
            Proposer::Ngram {
                tokens,
                longest,
                shortest,
            } => {
                proposed = lookup(&self.sequence, k.min(*tokens), *longest, *shortest);
                drawn = proposed.iter().map(|&t| Choice::Token(t)).collect();
            }
            Proposer::MultiToken(m) => {
                if k > 0
                    && let Some((token, choice)) = m.proposal()
                {
                    proposed.push(token);
                    drawn.push(choice);
                }
            }
        }
        Ok((proposed, drawn))
    }

    /// One pass of the model over the tokens it has not seen and the
    /// proposals: the tokens it gives, one to `k + 1`.
    fn round(&mut self) -> Result<Vec<u32>> {
        let p = self.pipeline;
        let generated = self.sequence.len() - self.prompt_len;
        // The last token of a pass is never proposed: room for it.
        let room = self.max_tokens.saturating_sub(generated + 1);
        let (proposed, drawn) = self.propose(room)?;
        let known = self.sequence.len();
        let mut fresh = self.sequence[self.cache.len()..].to_vec();
        fresh.extend_from_slice(&proposed);
        let rows = proposed.len() + 1;
        let (logits, hidden) = match &self.proposer {
            Proposer::MultiToken(_) => {
                let (logits, hidden) =
                    p.model()
                        .extend_hidden(p.backend(), &fresh, &mut self.cache, rows)?;
                (logits, Some(hidden))
            }
            _ => (
                p.model()
                    .extend_last(p.backend(), &fresh, &mut self.cache, rows)?,
                None,
            ),
        };
        let target: Vec<Choice> = if self.sampler.is_argmax() {
            p.most_likely_rows(&logits)?
                .into_iter()
                .map(Choice::Token)
                .collect()
        } else {
            let values = p.backend().download(&logits)?.to_f64_vec()?;
            let vocab = values.len() / rows;
            let mut context = self.sequence.clone();
            values
                .chunks_exact(vocab)
                .enumerate()
                .map(|(i, row)| {
                    let mut row = row.to_vec();
                    let choice = self
                        .sampler
                        .distribution(&mut row, &context, self.prompt_len);
                    if let Some(&x) = proposed.get(i) {
                        context.push(x);
                    }
                    choice
                })
                .collect()
        };
        let verdict = verify(&mut self.sampler, &proposed, &drawn, &target);
        self.stats.passes += 1;
        self.stats.proposed += proposed.len();
        self.stats.accepted += verdict.accepted;
        // The cache forgets the proposals refused; the last token, drawn by
        // the check, is in no cache yet.
        let kept = known + verdict.accepted;
        self.cache.truncate(kept);
        let mut new: Vec<u32> = proposed[..verdict.accepted].to_vec();
        new.push(verdict.next);
        match &mut self.proposer {
            Proposer::Model { cache, .. } => {
                let keep = cache.len().min(kept);
                cache.truncate(keep);
            }
            Proposer::MultiToken(m) => {
                let hidden = hidden.expect("computed for the multi-token layer");
                let mut tokens = self.sequence.clone();
                tokens.extend_from_slice(&new);
                m.advance(p, &hidden, fresh.len(), kept, &tokens)?;
            }
            Proposer::Ngram { .. } => {}
        }
        Ok(new)
    }

    fn next_piece(&mut self) -> Result<Option<Piece>> {
        if let Some(piece) = self.ready.pop_front() {
            return Ok(Some(piece));
        }
        if self.finish.is_some() {
            return Ok(None);
        }
        let p = self.pipeline;
        for token in self.round()? {
            self.sequence.push(token);
            if !self.ignore_eos && p.eos().contains(&token) {
                self.finish = Some(Finish::Eos);
                let text = self.text.finish(p.tokenizer());
                self.ready.push_back(Piece { token, text });
                break;
            }
            let added = self.text.push(p.tokenizer(), token);
            let mut text = added.text;
            if let Some(stop) = added.stopped {
                self.finish = Some(Finish::Stop(stop));
            } else if self.tokens().len() >= self.max_tokens {
                self.finish = Some(Finish::Length);
                text.push_str(&self.text.finish(p.tokenizer()));
            }
            self.ready.push_back(Piece { token, text });
            if self.finish.is_some() {
                break;
            }
        }
        Ok(self.ready.pop_front())
    }
}

impl<B: Backend + Transfer> Iterator for Speculative<'_, '_, B> {
    type Item = Result<Piece>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_piece() {
            Ok(piece) => piece.map(Ok),
            Err(e) => {
                // The caches may be half filled: nothing follows an error.
                self.finish = Some(Finish::Length);
                self.ready.clear();
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(seed: u64) -> Sampler {
        Sampler::new(Sampling {
            temperature: 1.0,
            seed,
            ..Sampling::greedy()
        })
    }

    #[test]
    fn greedy_checks_keep_what_the_model_would_choose() {
        let mut s = Sampler::new(Sampling::greedy());
        let t = Choice::Token;
        // The model would choose 2, 1, then 3; the proposer said 2, 0.
        let verdict = verify(&mut s, &[2, 0], &[t(2), t(0)], &[t(2), t(1), t(3)]);
        assert_eq!(
            verdict,
            Verdict {
                accepted: 1,
                next: 1
            }
        );
        // All accepted: the token more comes from the last position.
        let verdict = verify(&mut s, &[2, 1], &[t(2), t(1)], &[t(2), t(1), t(3)]);
        assert_eq!(
            verdict,
            Verdict {
                accepted: 2,
                next: 3
            }
        );
    }

    /// The chi-square of counts against probabilities.
    fn chi_square(counts: &[u64], probabilities: &[f64]) -> f64 {
        let total: u64 = counts.iter().sum();
        counts
            .iter()
            .zip(probabilities)
            .map(|(&count, &p)| {
                let expected = p * total as f64;
                (count as f64 - expected).powi(2) / expected
            })
            .sum()
    }

    /// Two positions over five tokens, the model's second distribution
    /// depending on the first token: the pairs a check gives follow the
    /// model's distribution, for a proposer that draws and for one that
    /// proposes a fixed token (as prompt lookup does).
    #[test]
    fn checked_tokens_follow_the_models_distribution() {
        let draft = [vec![0.4, 0.3, 0.1, 0.1, 0.1], vec![0.1, 0.1, 0.2, 0.3, 0.3]];
        let first = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let second = |a: usize| -> Vec<f64> {
            let mut p = vec![0.1; 5];
            p[a] += 0.5;
            p
        };
        let expected: Vec<f64> = (0..25)
            .map(|i| first[i / 5] * second(i / 5)[i % 5])
            .collect();
        for fixed in [false, true] {
            let mut drafter = sampler(1);
            let mut checker = sampler(2);
            let mut counts = vec![0u64; 25];
            for _ in 0..200_000 {
                let drawn: Vec<Choice> = if fixed {
                    vec![Choice::Token(3), Choice::Token(4)]
                } else {
                    draft
                        .iter()
                        .map(|d| Choice::Probabilities(d.clone()))
                        .collect()
                };
                let proposed: Vec<u32> = drawn.iter().map(|c| drafter.draw(c)).collect();
                let target = [
                    Choice::Probabilities(first.clone()),
                    Choice::Probabilities(second(proposed[0] as usize)),
                    Choice::Probabilities(vec![0.2; 5]),
                ];
                let verdict = verify(&mut checker, &proposed, &drawn, &target);
                let pair = match verdict.accepted {
                    0 => {
                        let b = checker.draw(&Choice::Probabilities(second(verdict.next as usize)));
                        (verdict.next as usize, b as usize)
                    }
                    1 => (proposed[0] as usize, verdict.next as usize),
                    _ => (proposed[0] as usize, proposed[1] as usize),
                };
                counts[pair.0 * 5 + pair.1] += 1;
            }
            // 24 degrees of freedom: at the 0.1 % level, below 51.2.
            let chi = chi_square(&counts, &expected);
            assert!(chi < 51.2, "chi-square {chi} (fixed proposals: {fixed})");
            // And the same test rejects the proposer's distribution.
            let wrong: Vec<f64> = (0..25).map(|i| draft[0][i / 5] * draft[1][i % 5]).collect();
            assert!(chi_square(&counts, &wrong) > 1000.0);
        }
    }

    #[test]
    fn lookup_finds_what_followed_the_last_tokens() {
        let seq = [1, 2, 3, 4, 5, 9, 2, 3];
        assert_eq!(lookup(&seq, 3, 3, 1), [4, 5, 9]);
        assert_eq!(lookup(&seq, 10, 3, 1), [4, 5, 9, 2, 3]);
        // The longest end first: [8, 2, 3] was not seen before, [2, 3] was.
        assert_eq!(lookup(&[7, 2, 3, 8, 2, 3], 2, 3, 2), [8, 2]);
        assert!(lookup(&[1, 2, 3], 4, 3, 1).is_empty());
        assert!(lookup(&[5], 4, 3, 1).is_empty());
    }
}
