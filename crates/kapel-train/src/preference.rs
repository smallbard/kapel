//! Preferences: two answers to one prompt, the better one first, and DPO
//! (Rafailov et al., 2023), which learns from them without a model of
//! rewards:
//!
//! ```text
//! loss = −ln σ(β · [(ln π(y⁺) − ln π_ref(y⁺)) − (ln π(y⁻) − ln π_ref(y⁻))])
//! ```
//!
//! `π_ref` is the model the run starts from, frozen: with LoRA, the model
//! without its adapters; otherwise a copy of the starting weights in the
//! computation's type. Both answers may grow less likely while their gap
//! grows: their probability then goes to answers neither of them is, often
//! degenerate (Razin et al., 2024, *likelihood displacement*), which
//! Smallem saw. The likelihood term of RPO (Pang et al., 2024), the
//! cross-entropy of the preferred answers weighted by `nll`, holds them up.
//!
//! A pair is read from a JSON object as TRL reads it: `prompt`, `chosen` and
//! `rejected`, the three texts or messages; or `chosen` and `rejected`
//! alone, whole conversations whose common beginning is the prompt. The
//! loss is computed on the host from the log-probabilities of the
//! sequences, and its gradient goes back through them
//! ([`crate::Tape::backward_with`]).

use std::path::Path;

use kapel_tokenizers::Message;
use serde_json::Value;

use crate::chat::{Encoder, Example, Turns};
use crate::data::{Cursor, DataError, Position};

type Result<T> = std::result::Result<T, DataError>;

fn fail(m: impl Into<String>) -> DataError {
    DataError(m.into())
}

/// A sequence of tokens, and which of them the loss counts (the answer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    /// The tokens: the prompt, then the answer.
    pub tokens: Vec<u32>,
    /// Whether each token is the answer's.
    pub mask: Vec<bool>,
}

impl Sequence {
    /// The tokens of the answer.
    pub fn answer_tokens(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }
}

/// A prompt with a preferred answer and a rejected one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The prompt and the preferred answer.
    pub chosen: Sequence,
    /// The prompt and the rejected answer.
    pub rejected: Sequence,
}

fn turns(value: &Value) -> Result<Turns> {
    match value {
        Value::String(s) => Ok(Turns::Text(s.clone())),
        Value::Array(_) => {
            let example =
                Example::from_json(&serde_json::json!({ "messages": value }), "messages")?;
            match example {
                Example::Messages(m) => Ok(Turns::Messages(m)),
                Example::Completion { .. } => unreachable!("messages read as messages"),
            }
        }
        other => Err(fail(format!("neither a text nor messages: {other}"))),
    }
}

/// The prompt and the two answers of a JSON object.
fn sides(value: &Value) -> Result<(Turns, Turns, Turns)> {
    let get = |k: &str| {
        value
            .get(k)
            .ok_or_else(|| fail(format!("a pair without {k:?}")))
    };
    let (chosen, rejected) = (turns(get("chosen")?)?, turns(get("rejected")?)?);
    if let Some(p) = value.get("prompt") {
        return Ok((turns(p)?, chosen, rejected));
    }
    // An implicit prompt: the messages both conversations begin with.
    let (Turns::Messages(c), Turns::Messages(r)) = (&chosen, &rejected) else {
        return Err(fail("a pair of texts without a prompt"));
    };
    let common = c.iter().zip(r).take_while(|(a, b)| a == b).count();
    if common == 0 || common == c.len() || common == r.len() {
        return Err(fail("a pair whose conversations share no prompt"));
    }
    let messages = |m: &[Message]| Turns::Messages(m.to_vec());
    Ok((
        messages(&c[..common]),
        messages(&c[common..]),
        messages(&r[common..]),
    ))
}

/// What reading pairs found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairCounts {
    /// The pairs kept.
    pub pairs: usize,
    /// The sequences cut to the length of the rows.
    pub truncated: usize,
}

/// Reads the pairs of the files `pattern` names, encoded by `encoder`; a
/// sequence longer than `length` tokens is cut to it (its end left out).
///
/// # Errors
///
/// If a file cannot be read, or a record is not a pair.
pub fn read_pairs(
    pattern: &str,
    encoder: &Encoder,
    length: usize,
    limit: Option<usize>,
) -> Result<(Vec<Pair>, PairCounts)> {
    let mut counts = PairCounts::default();
    let mut pairs = Vec::new();
    for (i, value) in crate::prepare::read_records(pattern, limit)?
        .into_iter()
        .enumerate()
    {
        let at = |e: DataError| fail(format!("{pattern}: pair {}: {}", i + 1, e.0));
        let (prompt, chosen, rejected) = sides(&value).map_err(at)?;
        let mut encode = |completion: Turns| -> Result<Sequence> {
            let example = Example::Completion {
                prompt: prompt.clone(),
                completion,
            };
            let (mut tokens, mut mask) = encoder.encode(&example).map_err(at)?;
            if tokens.len() > length {
                tokens.truncate(length);
                mask.truncate(length);
                counts.truncated += 1;
            }
            Ok(Sequence { tokens, mask })
        };
        let pair = Pair {
            chosen: encode(chosen)?,
            rejected: encode(rejected)?,
        };
        pairs.push(pair);
    }
    if pairs.is_empty() {
        return Err(fail(format!("{pattern}: no pair")));
    }
    counts.pairs = pairs.len();
    Ok((pairs, counts))
}

/// The ids, targets and answer tokens of sequences, one per row of
/// `length`, padded with `pad`: each target the next token where it is the
/// answer's, [`crate::IGNORE`] elsewhere.
pub fn rows(sequences: &[&Sequence], length: usize, pad: u32) -> (Vec<u32>, Vec<u32>) {
    let mut ids = Vec::with_capacity(sequences.len() * length);
    let mut targets = Vec::with_capacity(sequences.len() * length);
    for s in sequences {
        let n = s.tokens.len().min(length);
        ids.extend_from_slice(&s.tokens[..n]);
        ids.extend(std::iter::repeat_n(pad, length - n));
        for j in 0..length {
            let learnt = j + 1 < n && s.mask[j + 1];
            targets.push(if learnt {
                s.tokens[j + 1]
            } else {
                crate::IGNORE
            });
        }
    }
    (ids, targets)
}

/// Records held in memory (pairs, prompts), read in an order drawn from a
/// seed for each epoch, and where the reading is, for checkpoints.
#[derive(Clone, Debug)]
pub struct Shuffled<T> {
    items: Vec<T>,
    seed: u64,
    shuffle: bool,
    order: Vec<u32>,
    cursor: Cursor,
    read: u64,
}

/// The pairs of a run of DPO.
pub type PairLoader = Shuffled<Pair>;

impl<T> Shuffled<T> {
    /// A reader of `items`, shuffled from `seed` if `shuffle`.
    pub fn new(items: Vec<T>, seed: u64, shuffle: bool) -> Self {
        let mut loader = Self {
            items,
            seed,
            shuffle,
            order: Vec::new(),
            cursor: Cursor::default(),
            read: 0,
        };
        loader.order = loader.order_of(0);
        loader
    }

    fn order_of(&self, epoch: u64) -> Vec<u32> {
        let n = self.items.len();
        if self.shuffle {
            crate::data::permutation(n, crate::data::mix(&[self.seed, 4, epoch]))
        } else {
            (0..n as u32).collect()
        }
    }

    /// The records.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// The next `n` records.
    pub fn next(&mut self, n: usize) -> Vec<&T> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if self.cursor.row == self.items.len() {
                self.cursor.epoch += 1;
                self.cursor.row = 0;
                self.order = self.order_of(self.cursor.epoch);
            }
            out.push(self.order[self.cursor.row] as usize);
            self.cursor.row += 1;
            self.read += 1;
        }
        out.into_iter().map(|i| &self.items[i]).collect()
    }

    /// Where the reading is.
    pub fn position(&self) -> Position {
        Position {
            rows: self.read,
            sources: vec![self.cursor],
        }
    }

    /// Goes back to `position`.
    ///
    /// # Errors
    ///
    /// If it is not a position of these pairs.
    pub fn seek(&mut self, position: &Position) -> Result<()> {
        let [cursor] = position.sources[..] else {
            return Err(fail("a position of another source of data"));
        };
        if cursor.row > self.items.len() {
            return Err(fail("a position beyond the records"));
        }
        self.cursor = cursor;
        self.read = position.rows;
        self.order = self.order_of(cursor.epoch);
        Ok(())
    }
}

/// The settings of DPO.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct DpoConfig {
    /// The pairs: JSONL or Parquet files, a record each (see the module).
    pub train: String,
    /// `β`: how far from the reference the model may go; small, far.
    #[serde(default = "beta_default")]
    pub beta: f64,
    /// The weight of the likelihood of the preferred answers (RPO's `α`);
    /// 0 for DPO alone.
    #[serde(default)]
    pub nll: f64,
    /// Read the pairs in an order drawn anew for each epoch.
    #[serde(default = "yes")]
    pub shuffle: bool,
    /// The pairs read at most.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Pairs held out, as `train`: their loss, accuracy and margin measured
    /// every `eval.every` steps (`val_loss`, `val_accuracy`, `val_margin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<String>,
}

fn beta_default() -> f64 {
    0.1
}

fn yes() -> bool {
    true
}

impl DpoConfig {
    /// Checks the values.
    ///
    /// # Errors
    ///
    /// A message naming the first value refused.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if !(self.beta > 0.0 && self.beta.is_finite()) {
            return Err(format!("dpo.beta: positive, got {}", self.beta));
        }
        if !(self.nll >= 0.0 && self.nll.is_finite()) {
            return Err(format!("dpo.nll: not negative, got {}", self.nll));
        }
        Ok(())
    }
}

/// The loss of DPO for a batch of pairs and what it shows.
#[derive(Clone, Debug, PartialEq)]
pub struct DpoTerms {
    /// The loss: the mean of the pairs', plus the likelihood term.
    pub loss: f64,
    /// Its derivatives with respect to the log-probability of each
    /// preferred answer.
    pub chosen: Vec<f64>,
    /// … and of each rejected one.
    pub rejected: Vec<f64>,
    /// The share of pairs whose preferred answer has the larger implicit
    /// reward.
    pub accuracy: f64,
    /// The mean implicit reward of the preferred answers, `β (ln π − ln
    /// π_ref)`.
    pub chosen_reward: f64,
    /// … of the rejected ones.
    pub rejected_reward: f64,
    /// The cross-entropy of the preferred answers, by token.
    pub nll: f64,
}

/// `ln σ(x)`, without overflow.
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The loss of DPO, `mean −ln σ(β Δ)`, plus `nll` times the cross-entropy
/// of the preferred answers (all their tokens, `chosen_tokens` each), from
/// the log-probabilities of the answers by the model (`policy_*`) and by the
/// reference (`reference_*`).
pub fn dpo_terms(
    policy_chosen: &[f64],
    policy_rejected: &[f64],
    reference_chosen: &[f64],
    reference_rejected: &[f64],
    chosen_tokens: &[usize],
    beta: f64,
    nll: f64,
) -> DpoTerms {
    let p = policy_chosen.len();
    let n = p.max(1) as f64;
    let tokens = chosen_tokens.iter().sum::<usize>().max(1) as f64;
    let mut terms = DpoTerms {
        loss: 0.0,
        chosen: vec![0.0; p],
        rejected: vec![0.0; p],
        accuracy: 0.0,
        chosen_reward: 0.0,
        rejected_reward: 0.0,
        nll: -policy_chosen.iter().sum::<f64>() / tokens,
    };
    for i in 0..p {
        let rc = beta * (policy_chosen[i] - reference_chosen[i]);
        let rr = beta * (policy_rejected[i] - reference_rejected[i]);
        let margin = rc - rr;
        terms.loss -= log_sigmoid(margin) / n;
        // d(−ln σ(m))/dm = −σ(−m).
        let d = -sigmoid(-margin) * beta / n;
        terms.chosen[i] = d - nll / tokens;
        terms.rejected[i] = -d;
        terms.accuracy += f64::from(u8::from(margin > 0.0)) / n;
        terms.chosen_reward += rc / n;
        terms.rejected_reward += rr / n;
    }
    terms.loss += nll * terms.nll;
    terms
}

/// Whether `path` names files of pairs or prompts: a pattern of JSONL or
/// Parquet files.
pub fn is_records(path: &str) -> bool {
    let p = Path::new(path);
    matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("jsonl" | "ndjson" | "parquet")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loss_is_ln_2_when_the_model_is_its_reference() {
        let lp = [-3.0, -7.5];
        let t = dpo_terms(&lp, &[-4.0, -2.0], &lp, &[-4.0, -2.0], &[3, 4], 0.1, 0.0);
        assert!((t.loss - std::f64::consts::LN_2).abs() < 1e-12);
        assert_eq!(t.accuracy, 0.0);
        // Half of β over the pairs, each way.
        assert!((t.chosen[0] + 0.1 * 0.5 / 2.0).abs() < 1e-12);
        assert!((t.rejected[1] - 0.1 * 0.5 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_derivatives_are_those_of_the_loss() {
        let pc = [-3.0, -7.5, -1.25];
        let pr = [-4.0, -2.0, -6.0];
        let (rc, rr) = ([-3.5, -7.0, -2.0], [-3.0, -2.5, -5.0]);
        let lengths = [3, 4, 2];
        let (beta, nll) = (0.3, 0.7);
        let t = dpo_terms(&pc, &pr, &rc, &rr, &lengths, beta, nll);
        let h = 1e-6;
        for i in 0..3 {
            for side in 0..2 {
                let moved = |d: f64| {
                    let (mut c, mut r) = (pc, pr);
                    if side == 0 {
                        c[i] += d;
                    } else {
                        r[i] += d;
                    }
                    dpo_terms(&c, &r, &rc, &rr, &lengths, beta, nll).loss
                };
                let numeric = (moved(h) - moved(-h)) / (2.0 * h);
                let analytic = if side == 0 {
                    t.chosen[i]
                } else {
                    t.rejected[i]
                };
                assert!((numeric - analytic).abs() < 1e-8, "{numeric} {analytic}");
            }
        }
    }

    fn sequence(tokens: &[u32], answer_from: usize) -> Sequence {
        Sequence {
            tokens: tokens.to_vec(),
            mask: (0..tokens.len()).map(|i| i >= answer_from).collect(),
        }
    }

    #[test]
    fn rows_predict_the_answer_only() {
        let a = sequence(&[1, 2, 3, 4], 2);
        let (ids, targets) = rows(&[&a], 6, 0);
        let ignore = crate::IGNORE;
        assert_eq!(ids, vec![1, 2, 3, 4, 0, 0]);
        assert_eq!(targets, vec![ignore, 3, 4, ignore, ignore, ignore]);
    }

    #[test]
    fn a_loader_resumes_where_it_was() {
        let pairs: Vec<Pair> = (0..5)
            .map(|i| Pair {
                chosen: sequence(&[i], 0),
                rejected: sequence(&[i + 10], 0),
            })
            .collect();
        let mut a = PairLoader::new(pairs.clone(), 3, true);
        a.next(7);
        let position = a.position();
        let rest: Vec<Pair> = a.next(6).into_iter().cloned().collect();
        let mut b = PairLoader::new(pairs, 3, true);
        b.seek(&position).unwrap();
        let again: Vec<Pair> = b.next(6).into_iter().cloned().collect();
        assert_eq!(rest, again);
    }
}
