//! The log-likelihood of a continuation after a context: what evaluations
//! by choice (HellaSwag, ARC, MMLU) and perplexities are made of.

use kapel_core::Tensor;
use kapel_models::Cache;
use kapel_ops::{Backend, Transfer};

use crate::{Pipeline, Result};

/// What [`Scorer::score`] gives for a continuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    /// The sum of the natural logarithms of the probabilities of its tokens.
    pub log_likelihood: f64,
    /// Whether each of its tokens was the most likely one (greedy decoding
    /// would have given it).
    pub greedy: bool,
    /// The number of its tokens.
    pub tokens: usize,
}

/// Scores continuations after contexts with a pipeline's model, one cache
/// kept from one call to the next: the positions a sequence shares with
/// the previous one (the context common to the choices of a question, a
/// prompt common to the questions) are not computed again.
pub struct Scorer<'p, B> {
    pipeline: &'p Pipeline<B>,
    cache: Cache,
    /// The tokens the cache holds.
    held: Vec<u32>,
    /// The most tokens per pass of the model.
    piece: usize,
}

impl<'p, B: Backend + Transfer> Scorer<'p, B> {
    /// A scorer that computes at most `piece` tokens per pass of the model
    /// (the larger, the faster, and the more memory the logits take: `piece`
    /// rows of the vocabulary).
    pub fn new(pipeline: &'p Pipeline<B>, piece: usize) -> Self {
        Self {
            pipeline,
            cache: pipeline
                .cache_dtype()
                .map_or_else(Cache::new, Cache::with_dtype),
            held: Vec::new(),
            piece: piece.max(1),
        }
    }

    /// The score of `continuation` after `context`: the model reads
    /// `context` then every token of `continuation` but the last, and each
    /// token of `continuation` is scored by the logits of the position
    /// before it.
    ///
    /// # Errors
    ///
    /// If `context` or `continuation` is empty, or the model fails.
    pub fn score(&mut self, context: &[u32], continuation: &[u32]) -> Result<Score> {
        if context.is_empty() || continuation.is_empty() {
            return Err(crate::InferError::EmptyPrompt);
        }
        let input: Vec<u32> = context
            .iter()
            .chain(&continuation[..continuation.len() - 1])
            .copied()
            .collect();
        // The first scored row is that of the last token of the context: it
        // is computed again even when the cache holds it, its logits being
        // gone.
        let first = context.len() - 1;
        let shared = self
            .held
            .iter()
            .zip(&input)
            .take_while(|(a, b)| a == b)
            .count()
            .min(first);
        self.cache.truncate(shared);
        self.held.truncate(shared);
        let model = self.pipeline.model();
        let backend = self.pipeline.backend();
        let mut score = Score {
            log_likelihood: 0.0,
            greedy: true,
            tokens: continuation.len(),
        };
        let mut start = shared;
        while start < input.len() {
            let end = (start + self.piece).min(input.len());
            let ids = &input[start..end];
            // The rows of this piece that score a token: those from `first`.
            let wanted = end - start.max(first).min(end);
            let logits = model.extend_last(backend, ids, &mut self.cache, wanted.max(1))?;
            self.held.extend_from_slice(ids);
            if wanted > 0 {
                let logits = backend.download(&logits)?;
                let targets = &continuation[end - wanted - first..end - first];
                for (log_probability, greedy) in row_scores(&logits, targets)? {
                    score.log_likelihood += log_probability;
                    score.greedy &= greedy;
                }
            }
            start = end;
        }
        Ok(score)
    }
}

/// The log-probability of each target under its row of `logits` `[rows,
/// vocab]` (in host memory), and whether it was the most likely (the first
/// of equals). The rows are shared among the machine's threads, each
/// decoding its own to `f64`: a row is a whole vocabulary, and a text scores
/// thousands. Each row is computed alone, so the results do not depend on
/// the number of threads.
fn row_scores(logits: &Tensor, targets: &[u32]) -> Result<Vec<(f64, bool)>> {
    let one = |row: &[f64], target: u32| {
        let (max, best) =
            row.iter()
                .enumerate()
                .fold((f64::NEG_INFINITY, 0), |(max, best), (i, &v)| {
                    if v > max { (v, i) } else { (max, best) }
                });
        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
        (
            row[target as usize] - max - sum.ln(),
            best == target as usize,
        )
    };
    let rows = targets.len();
    let vocab = logits.shape().last().copied().unwrap_or(1);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(rows)
        .max(1);
    let per = rows.div_ceil(threads);
    let mut out = vec![(0.0, false); rows];
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = out
            .chunks_mut(per)
            .zip(targets.chunks(per))
            .enumerate()
            .map(|(i, (out, targets))| {
                scope.spawn(move || -> Result<()> {
                    let values = logits.narrow(0, i * per, targets.len())?.to_f64_vec()?;
                    for ((row, &target), slot) in values.chunks_exact(vocab).zip(targets).zip(out) {
                        *slot = one(row, target);
                    }
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("a scoring thread panicked")?;
        }
        Ok(())
    })?;
    Ok(out)
}
