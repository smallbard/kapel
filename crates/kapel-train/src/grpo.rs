//! GRPO (Shao et al., *DeepSeekMath*, 2024): the model writes a group of
//! `G` answers to each prompt, a program scores them ([`crate::rewards`]),
//! and each answer's advantage is its reward against its group's:
//!
//! ```text
//! A = (r − mean of the group) / (standard deviation of the group + 10⁻⁴)
//! ```
//!
//! Each token of an answer then weighs in the loss
//!
//! ```text
//! −min(ρ · A, clip(ρ, 1 − ε, 1 + ε_high) · A) + β · (π_ref/π − ln(π_ref/π) − 1)
//! ```
//!
//! where `ρ = π / π_old` is the ratio of its probability now to that when
//! the answer was written, bounded so that one step moves little, and the
//! second term the divergence from the reference model (`k3`, Schulman's
//! estimator). With one step per generation, `π_old = π` and `ρ = 1`: the
//! gradient is REINFORCE's, the advantage times that of `ln π`.
//!
//! The losses of the tokens are averaged as TRL's `loss_type` says:
//!
//! - `"grpo"`, the paper's: over each answer, then over the answers; a long
//!   answer weighs each of its tokens less, which favours long wrong
//!   answers (Liu et al., *Dr. GRPO*, 2025);
//! - `"dr_grpo"`: their sum over the answers times the largest length of an
//!   answer, a constant; Dr. GRPO also leaves the advantages unscaled by
//!   the deviation (`scale_rewards = false`);
//! - `"token"` (DAPO's): over all the tokens of the step.
//!
//! The answers are written by Kapel's inference engine
//! ([`kapel_infer::Engine`]), all those of a step at once, from the
//! weights being trained ([`generate`]). The loss is computed on the host
//! from the log-probabilities of the tokens, and its gradient goes back
//! through them.

use std::collections::{BTreeMap, HashMap};

use kapel_infer::{EngineOptions, Pipeline, Request, Sampling};
use kapel_ops::{Backend, Transfer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{Encoder, Turns};
use crate::data::DataError;
use crate::rewards::RewardConfig;

type Result<T> = std::result::Result<T, DataError>;

fn fail(m: impl Into<String>) -> DataError {
    DataError(m.into())
}

/// The settings of GRPO.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GrpoConfig {
    /// The prompts: JSONL or Parquet files, a record each, its prompt in
    /// the field `prompt` (a text or messages), its expected answer in
    /// others, for the rewards.
    pub prompts: String,
    /// The prompts of a step.
    #[serde(default = "prompts_default")]
    pub prompts_per_step: usize,
    /// The answers to each prompt.
    #[serde(default = "group_default")]
    pub group: usize,
    /// The most tokens of an answer.
    #[serde(default = "max_tokens_default")]
    pub max_tokens: usize,
    /// The temperature of the answers.
    #[serde(default = "one")]
    pub temperature: f64,
    /// Only the most likely tokens whose probabilities add up to `top_p`.
    #[serde(default = "one")]
    pub top_p: f64,
    /// Only the `top_k` most likely tokens; absent, all.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// The weight of the divergence from the reference; 0 for none (and no
    /// reference computed).
    #[serde(default = "beta_default")]
    pub beta: f64,
    /// The bound of the ratio below 1.
    #[serde(default = "epsilon_default")]
    pub epsilon: f64,
    /// The bound above 1; absent, `epsilon` (DAPO raises it).
    #[serde(default)]
    pub epsilon_high: Option<f64>,
    /// `"grpo"`, `"dr_grpo"` or `"token"` (see the module).
    #[serde(default = "loss_default")]
    pub loss: String,
    /// Divide the advantages by their group's standard deviation; absent,
    /// yes but for `"dr_grpo"`.
    #[serde(default)]
    pub scale_rewards: Option<bool>,
    /// The optimizer's steps on each generation (`μ`); beyond 1, the
    /// ratio comes into play.
    #[serde(default = "iterations_default")]
    pub iterations: usize,
    /// Read the prompts in an order drawn anew for each epoch.
    #[serde(default = "yes")]
    pub shuffle: bool,
    /// The variables of the chat template (`enable_thinking = false` for
    /// Qwen 3).
    #[serde(default)]
    pub template: BTreeMap<String, Value>,
    /// The rewards, summed with their weights.
    pub rewards: Vec<RewardConfig>,
}

fn prompts_default() -> usize {
    8
}

fn group_default() -> usize {
    8
}

fn max_tokens_default() -> usize {
    256
}

fn one() -> f64 {
    1.0
}

fn beta_default() -> f64 {
    0.04
}

fn epsilon_default() -> f64 {
    0.2
}

fn loss_default() -> String {
    "grpo".into()
}

fn iterations_default() -> usize {
    1
}

fn yes() -> bool {
    true
}

impl GrpoConfig {
    /// Whether the advantages are scaled by their group's deviation.
    pub fn scales_rewards(&self) -> bool {
        self.scale_rewards.unwrap_or(self.loss != "dr_grpo")
    }

    /// The sampling of the answers, from `seed`.
    pub fn sampling(&self, seed: u64) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed,
            ..Sampling::default()
        }
    }

    /// Checks the values.
    ///
    /// # Errors
    ///
    /// A message naming the first value refused.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.prompts_per_step == 0 || self.group < 2 || self.max_tokens == 0 {
            return Err("grpo: prompts_per_step, max_tokens at least 1, group at least 2".into());
        }
        if !matches!(self.loss.as_str(), "grpo" | "dr_grpo" | "token") {
            return Err(format!(
                "grpo.loss: \"grpo\", \"dr_grpo\" or \"token\", got {:?}",
                self.loss
            ));
        }
        if self.iterations == 0 {
            return Err("grpo.iterations: at least 1".into());
        }
        let positive = self.temperature > 0.0 && self.beta >= 0.0 && self.epsilon >= 0.0;
        if !positive {
            return Err("grpo: temperature positive, beta and epsilon not negative".into());
        }
        if self.rewards.is_empty() {
            return Err("grpo.rewards: none".into());
        }
        for r in &self.rewards {
            r.build().map_err(|e| format!("grpo.rewards: {e}"))?;
        }
        Ok(())
    }
}

/// A task evaluated during training: prompts whose answers a program
/// checks, answered greedily, the accuracy the mean of the rewards.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskConfig {
    /// Its name, in the log.
    pub name: String,
    /// The prompts, as [`GrpoConfig::prompts`].
    pub prompts: String,
    /// The prompts answered at most, the first ones.
    #[serde(default)]
    pub limit: Option<usize>,
    /// The most tokens of an answer.
    #[serde(default = "task_tokens_default")]
    pub max_tokens: usize,
    /// The variables of the chat template.
    #[serde(default)]
    pub template: BTreeMap<String, Value>,
    /// The rewards, summed with their weights: 1 for a right answer.
    pub rewards: Vec<RewardConfig>,
}

fn task_tokens_default() -> usize {
    512
}

impl TaskConfig {
    /// Checks the values.
    ///
    /// # Errors
    ///
    /// A message naming the first value refused.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.rewards.is_empty() || self.max_tokens == 0 {
            return Err(format!(
                "eval.tasks {:?}: no reward, or no token",
                self.name
            ));
        }
        for r in &self.rewards {
            r.build()
                .map_err(|e| format!("eval.tasks {:?}: {e}", self.name))?;
        }
        Ok(())
    }
}

/// A prompt: its tokens, and the record it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    /// The tokens, the start of an answer included.
    pub tokens: Vec<u32>,
    /// The JSON object, for the rewards.
    pub record: Value,
}

/// Reads the prompts of the files `pattern` names, each in the field
/// `prompt` of its record, encoded by `encoder` with the template's
/// `variables`.
///
/// # Errors
///
/// If a file cannot be read, or a record has no prompt.
pub fn read_prompts(
    pattern: &str,
    encoder: &Encoder,
    variables: &BTreeMap<String, Value>,
    limit: Option<usize>,
) -> Result<Vec<Prompt>> {
    let records = crate::prepare::read_records(pattern, limit)?;
    let prompts: Vec<Prompt> = records
        .into_iter()
        .enumerate()
        .map(|(i, record)| {
            let at = |m: String| fail(format!("{pattern}: record {}: {m}", i + 1));
            let prompt = match record.get("prompt") {
                Some(Value::String(s)) => Turns::Text(s.clone()),
                Some(v @ Value::Array(_)) => {
                    let wrapped = serde_json::json!({ "messages": v });
                    match crate::chat::Example::from_json(&wrapped, "messages")
                        .map_err(|e| at(e.0))?
                    {
                        crate::chat::Example::Messages(m) => Turns::Messages(m),
                        crate::chat::Example::Completion { .. } => unreachable!("messages"),
                    }
                }
                _ => return Err(at("no field \"prompt\"".into())),
            };
            let tokens = encoder.prompt(&prompt, variables).map_err(|e| at(e.0))?;
            Ok(Prompt { tokens, record })
        })
        .collect::<Result<_>>()?;
    if prompts.is_empty() {
        return Err(fail(format!("{pattern}: no prompt")));
    }
    Ok(prompts)
}

/// An answer written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    /// Its tokens, the one that ended it included.
    pub tokens: Vec<u32>,
    /// Its text, without the end token.
    pub text: String,
    /// Whether it ended by itself, not cut at the most tokens.
    pub ended: bool,
}

/// The bytes of a page of the cache of `config`'s model, in `element`
/// bytes a value.
pub fn page_bytes(config: &kapel_models::Config, element: usize) -> usize {
    config.layers * 2 * config.kv_heads * config.head_dim * element * kapel_models::PAGE
}

/// The answers to `prompts`, written together by the engine of `pipeline`
/// in a cache of `cache_bytes` (as many as it holds at once, the prompts
/// they share computed once), each with `request` and a seed of its own
/// from `seed`.
///
/// # Errors
///
/// If the engine fails, or the cache cannot hold one answer.
pub fn generate<B: Backend + Transfer>(
    pipeline: &Pipeline<B>,
    prompts: &[&[u32]],
    request: &Request,
    seed: u64,
    cache_bytes: usize,
) -> std::result::Result<Vec<Completion>, kapel_infer::InferError> {
    let config = pipeline.model().config();
    let element = pipeline
        .cache_dtype()
        .or_else(|| {
            pipeline
                .model()
                .tensor("model.embed_tokens.weight")
                .or_else(|| pipeline.model().tensor("transformer.wte.weight"))
                .map(kapel_core::Tensor::dtype)
        })
        .and_then(|d| d.element_size())
        .unwrap_or(2);
    let pages = cache_bytes / page_bytes(config, element).max(1);
    let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0) + request.max_tokens;
    if pages * kapel_models::PAGE < longest + 2 * kapel_models::PAGE {
        return Err(kapel_infer::InferError::Unsupported(format!(
            "a cache of {pages} pages cannot hold an answer of {longest} positions"
        )));
    }
    let options = EngineOptions {
        pages: Some(pages),
        prefix_sharing: true,
        max_sequences: prompts.len().clamp(1, 256),
        ..EngineOptions::default()
    };
    let mut engine = pipeline.engine(options);
    let mut index = HashMap::new();
    for (i, p) in prompts.iter().enumerate() {
        let mut r = request.clone();
        r.sampling.seed = crate::data::mix(&[seed, i as u64]);
        index.insert(engine.add(p.to_vec(), &r)?, i);
    }
    let mut out = vec![
        Completion {
            tokens: Vec::new(),
            text: String::new(),
            ended: false,
        };
        prompts.len()
    ];
    while !engine.is_idle() {
        for event in engine.step().map_err(|e| e.error)? {
            let c = &mut out[index[&event.id]];
            if let Some(piece) = event.piece {
                c.tokens.push(piece.token);
                c.text.push_str(&piece.text);
            }
            match event.finish {
                Some(kapel_infer::Finish::Eos | kapel_infer::Finish::Stop(_)) => c.ended = true,
                Some(kapel_infer::Finish::Error(e)) => {
                    return Err(kapel_infer::InferError::Unsupported(e));
                }
                _ => {}
            }
        }
    }
    Ok(out)
}

/// The advantages of the answers of groups of `group` in a row: each
/// reward less its group's mean, divided by its group's standard deviation
/// (`n − 1` in the denominator, as `torch.std`) plus 10⁻⁴ if `scale`.
pub fn advantages(rewards: &[f64], group: usize, scale: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(rewards.len());
    for g in rewards.chunks(group) {
        let n = g.len() as f64;
        let mean = g.iter().sum::<f64>() / n;
        let var = g.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std = var.sqrt();
        out.extend(g.iter().map(|r| {
            let a = r - mean;
            if scale { a / (std + 1e-4) } else { a }
        }));
    }
    out
}

/// The loss of the tokens of some answers, and what it shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenTerms {
    /// Their part of the loss.
    pub loss: f64,
    /// Its derivative with respect to the log-probability of each token.
    pub grads: Vec<f64>,
    /// The sum of their divergences from the reference.
    pub kl: f64,
    /// Those whose ratio was bounded.
    pub clipped: usize,
}

/// The loss of GRPO over tokens: for token `t` of answer `answer[t]`, its
/// log-probabilities now (`logp`), when the answer was written (`old`) and
/// by the reference (`reference`, if `beta` > 0), each token's loss weighed
/// by `weight[t]` (the averaging of the loss).
#[allow(clippy::too_many_arguments)]
pub fn token_terms(
    logp: &[f64],
    old: &[f64],
    reference: Option<&[f64]>,
    advantage: &[f64],
    answer: &[usize],
    weight: &[f64],
    config: &GrpoConfig,
) -> TokenTerms {
    let (low, high) = (
        1.0 - config.epsilon,
        1.0 + config.epsilon_high.unwrap_or(config.epsilon),
    );
    let mut terms = TokenTerms {
        grads: vec![0.0; logp.len()],
        ..TokenTerms::default()
    };
    for t in 0..logp.len() {
        let a = advantage[answer[t]];
        let ratio = (logp[t] - old[t]).exp();
        let bounded = ratio.clamp(low, high);
        // −min(ρ A, clip(ρ) A): the bounded one is a constant.
        let (value, d) = if ratio * a <= bounded * a {
            (-ratio * a, -ratio * a)
        } else {
            terms.clipped += 1;
            (-bounded * a, 0.0)
        };
        let (mut value, mut d) = (value, d);
        if let Some(r) = reference.filter(|_| config.beta > 0.0) {
            let x = r[t] - logp[t];
            let kl = x.exp() - x - 1.0;
            terms.kl += kl;
            value += config.beta * kl;
            d += config.beta * (1.0 - x.exp());
        }
        terms.loss += weight[t] * value;
        terms.grads[t] = weight[t] * d;
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GrpoConfig {
        GrpoConfig {
            prompts: String::new(),
            prompts_per_step: 1,
            group: 4,
            max_tokens: 8,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            beta: 0.04,
            epsilon: 0.2,
            epsilon_high: None,
            loss: "grpo".into(),
            scale_rewards: None,
            iterations: 1,
            shuffle: true,
            template: BTreeMap::new(),
            rewards: Vec::new(),
        }
    }

    #[test]
    fn advantages_against_the_group() {
        let a = advantages(&[1.0, 0.0, 0.0, 1.0, 2.0, 2.0], 3, false);
        let third = 1.0 / 3.0;
        let expected = [2.0 * third, -third, -third, -2.0 * third, third, third]
            .map(|x| (x * 1e9_f64).round() / 1e9);
        let got: Vec<f64> = a.iter().map(|x| (x * 1e9).round() / 1e9).collect();
        assert_eq!(got, expected);
        // Scaled: a group all alike gives zeros, not a division by zero.
        let s = advantages(&[1.0, 1.0, 0.0, 2.0], 2, true);
        assert_eq!(&s[..2], &[0.0, 0.0]);
        assert!((s[2] + 1.0 / (2.0f64.sqrt() + 1e-4)).abs() < 1e-12);
    }

    #[test]
    fn with_its_own_answers_the_gradient_is_reinforce_s() {
        let logp = [-1.0, -2.0, -0.5];
        let c = GrpoConfig {
            beta: 0.0,
            ..config()
        };
        let t = token_terms(&logp, &logp, None, &[0.5, -2.0], &[0, 0, 1], &[0.5; 3], &c);
        assert_eq!(t.grads, vec![-0.25, -0.25, 1.0]);
        assert_eq!(t.clipped, 0);
    }

    #[test]
    fn a_ratio_out_of_bounds_stops_the_gradient() {
        let c = GrpoConfig {
            beta: 0.0,
            ..config()
        };
        // ρ = e^0.5 > 1.2 with A > 0, and e^-0.5 < 0.8 with A < 0: bounded;
        // the other way round they are not.
        let logp = [0.5, -0.5, 0.5, -0.5];
        let old = [0.0; 4];
        let t = token_terms(
            &logp,
            &old,
            None,
            &[1.0, -1.0],
            &[0, 1, 1, 0],
            &[1.0; 4],
            &c,
        );
        assert_eq!(t.clipped, 2);
        assert_eq!(t.grads[0], 0.0);
        assert_eq!(t.grads[1], 0.0);
        assert!(t.grads[2] > 0.0 && t.grads[3] < 0.0);
    }

    #[test]
    fn the_derivatives_are_those_of_the_loss() {
        let c = GrpoConfig {
            epsilon_high: Some(0.3),
            ..config()
        };
        let logp = [-1.0, -2.0, -0.5, -0.1];
        let old = [-1.1, -1.9, -0.45, -0.3];
        let reference = [-1.3, -1.5, -0.7, -0.2];
        let (adv, answer, weight) = ([0.7, -1.2], [0, 0, 1, 1], [0.25, 0.25, 0.5, 0.5]);
        let t = token_terms(&logp, &old, Some(&reference), &adv, &answer, &weight, &c);
        let h = 1e-6;
        for i in 0..4 {
            let at = |d: f64| {
                let mut l = logp;
                l[i] += d;
                token_terms(&l, &old, Some(&reference), &adv, &answer, &weight, &c).loss
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            assert!(
                (numeric - t.grads[i]).abs() < 1e-7,
                "{numeric} {}",
                t.grads[i]
            );
        }
        assert!(t.kl > 0.0);
    }
}
