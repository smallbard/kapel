//! The objectives of post-training: pairs of answers (DPO), answers the
//! model writes (GRPO), and the tasks evaluated along a run.
//!
//! Their losses are computed on the host from log-probabilities the model
//! gives on the device ([`crate::model::Decoder::token_losses`]), and their
//! gradients go back through them ([`Tape::backward_with`]). The answers are
//! written by the inference engine from a model of the current weights,
//! the adapters merged into it ([`Run::inference_model`]).

use std::collections::BTreeMap;
use std::time::Instant;

use kapel_core::Tensor;
use kapel_infer::{Pipeline, Request, Sampling};
use serde_json::{Value, json};

use super::{Objective, Result, Run, StepOutput, Summed, TrainError};
use crate::chat::Encoder;
use crate::grpo::{self, Completion, GrpoConfig, Prompt, TaskConfig};
use crate::model::{Batch, Parameters};
use crate::preference::{self, DpoConfig, PairLoader, Sequence, Shuffled};
use crate::rewards::Rewards;
use crate::tape::{Compute, Tape};
use crate::{flops, lora};

/// The pairs of a run of DPO.
pub(super) struct Pairs {
    pub(super) pairs: PairLoader,
    /// The pairs held out.
    val: Vec<preference::Pair>,
    config: DpoConfig,
    pad: u32,
}

impl Pairs {
    pub(super) fn new(
        config: &DpoConfig,
        encoder: &Encoder,
        length: usize,
        seed: u64,
        quiet: bool,
    ) -> Result<Self> {
        let (pairs, counts) = preference::read_pairs(&config.train, encoder, length, config.limit)?;
        if !quiet {
            eprintln!(
                "train: {} pairs{}",
                counts.pairs,
                if counts.truncated > 0 {
                    format!(", {} sequences cut to {length} tokens", counts.truncated)
                } else {
                    String::new()
                }
            );
        }
        let val = match &config.val {
            Some(path) => {
                let (val, counts) = preference::read_pairs(path, encoder, length, None)?;
                if !quiet {
                    eprintln!("train: {} pairs held out", counts.pairs);
                }
                val
            }
            None => Vec::new(),
        };
        Ok(Self {
            pairs: PairLoader::new(pairs, seed, config.shuffle),
            val,
            config: config.clone(),
            pad: encoder.eos().unwrap_or(0),
        })
    }
}

/// The answers of a step of GRPO, and what training on them needs.
struct Rollout {
    /// Each prompt with an answer, the answer the loss counts.
    sequences: Vec<Sequence>,
    /// The advantage of each.
    advantages: Vec<f64>,
    /// The weight of each counted token of each sequence in the loss.
    weights: Vec<f64>,
    /// The log-probabilities of the counted tokens of each sequence when it
    /// was written, if the run takes several steps on it.
    old: Option<Vec<Vec<f64>>>,
    /// … and by the reference, if the loss has a divergence.
    reference: Option<Vec<Vec<f64>>>,
    /// The steps left on it.
    left: usize,
    /// What the answers showed, for the log of each step on them.
    metrics: serde_json::Map<String, Value>,
    /// The operations of writing them, counted on the first step.
    flops: u64,
}

/// The prompts of a run of GRPO.
pub(super) struct Answers {
    pub(super) prompts: Shuffled<Prompt>,
    config: GrpoConfig,
    rewards: Rewards,
    pad: u32,
    rollout: Option<Rollout>,
}

impl Answers {
    pub(super) fn new(
        config: &GrpoConfig,
        encoder: &Encoder,
        length: usize,
        seed: u64,
        quiet: bool,
    ) -> Result<Self> {
        let all = grpo::read_prompts(&config.prompts, encoder, &config.template, None)?;
        let read = all.len();
        // The prompt and its longest answer fit in a row.
        let prompts: Vec<Prompt> = all
            .into_iter()
            .filter(|p| p.tokens.len() + config.max_tokens <= length)
            .collect();
        if prompts.is_empty() {
            return Err(TrainError(format!(
                "{}: no prompt leaves room in {length} tokens for an answer of {}",
                config.prompts, config.max_tokens
            )));
        }
        if !quiet {
            eprintln!(
                "train: {} prompts{}",
                prompts.len(),
                if read > prompts.len() {
                    format!(", {} too long left out", read - prompts.len())
                } else {
                    String::new()
                }
            );
        }
        Ok(Self {
            prompts: Shuffled::new(prompts, seed, config.shuffle),
            rewards: Rewards::new(&config.rewards)?,
            config: config.clone(),
            pad: encoder.eos().unwrap_or(0),
            rollout: None,
        })
    }
}

/// A task evaluated, its prompts read.
pub(super) struct Task {
    name: String,
    prompts: Vec<Prompt>,
    rewards: Rewards,
    max_tokens: usize,
}

impl Task {
    pub(super) fn new(config: &TaskConfig, encoder: &Encoder) -> Result<Self> {
        Ok(Self {
            name: config.name.clone(),
            prompts: grpo::read_prompts(&config.prompts, encoder, &config.template, config.limit)?,
            rewards: Rewards::new(&config.rewards)?,
            max_tokens: config.max_tokens,
        })
    }
}

/// The ids and targets of `sequences`, a row of `length` each.
fn rows_of(sequences: &[&Sequence], length: usize, pad: u32) -> (Vec<u32>, Vec<u32>) {
    preference::rows(sequences, length, pad)
}

impl<B: Compute> Run<'_, B> {
    /// `params` frozen: constants on a tape.
    fn frozen(params: &Parameters) -> Parameters {
        let mut out = params.clone();
        for (_, p) in out.iter_mut() {
            p.trainable = false;
        }
        out
    }

    /// The reference's parameters: the starting weights, or when only
    /// adapters train, the model without them.
    fn reference_params(&self) -> Parameters {
        match &self.reference {
            Some(r) => r.clone(),
            None => {
                let mut out = Parameters::default();
                for (n, p) in self.params.iter() {
                    if !lora::is_factor(n) {
                        let mut p = p.clone();
                        p.trainable = false;
                        out.insert(n, p);
                    }
                }
                out
            }
        }
    }

    /// The log-probability of each target of the batch by `params`, without
    /// gradients: `[rows · length]`, 0 where the target is ignored.
    fn token_logps(&self, params: &Parameters, ids: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        let rows = ids.len() / self.config.batch.length;
        let tape = Tape::new(self.backend.clone());
        let leaves = params.leaves(&tape);
        let batch = Batch {
            ids,
            targets,
            rows,
            segments: None,
        };
        let losses = self
            .decoder
            .token_losses(&leaves, &batch, &self.options())?;
        let values = self.backend.download(&losses.value())?.to_f64_vec()?;
        Ok(values.into_iter().map(|l| -l).collect())
    }

    /// The operations of the passes of training over `rows` rows, and of
    /// `references` forward passes over as many.
    fn pass_flops(&self, rows: usize, references: usize) -> u64 {
        let c = self.decoder.config();
        let length = self.config.batch.length;
        let tokens = rows * length;
        let train = if self.config.lora.is_some() {
            flops::frozen_step_flops(c, tokens, length)
        } else {
            flops::step_flops(c, tokens, length)
        };
        train + 2 * references as u64 * flops::forward_macs(c, rows, length)
    }

    /// A step of DPO: `accumulate` passes of `rows / 2` pairs.
    pub(super) fn pair_step(&mut self) -> Result<StepOutput> {
        let accumulate = self.config.batch.accumulate();
        let (rows, length) = (self.config.batch.rows, self.config.batch.length);
        let reference = self.reference_params();
        let mut sum = Summed::default();
        let (mut loss, mut tokens) = (0.0, 0u64);
        let mut means: BTreeMap<&str, f64> = BTreeMap::new();
        for _ in 0..accumulate {
            let Objective::Pairs(p) = &mut self.objective else {
                unreachable!("a step of pairs");
            };
            let (config, pad) = (p.config.clone(), p.pad);
            let pairs: Vec<preference::Pair> =
                p.pairs.next(rows / 2).into_iter().cloned().collect();
            let sequences: Vec<&Sequence> = pairs
                .iter()
                .flat_map(|p| [&p.chosen, &p.rejected])
                .collect();
            let (ids, targets) = rows_of(&sequences, length, pad);
            let sums = |logps: &[f64]| -> Vec<f64> {
                logps.chunks(length).map(|r| r.iter().sum()).collect()
            };
            let reference = sums(&self.token_logps(&reference, &ids, &targets)?);
            let tape = Tape::new(self.backend.clone());
            let leaves = self.params.leaves(&tape);
            let batch = Batch {
                ids: &ids,
                targets: &targets,
                rows,
                segments: None,
            };
            // `−ln π` of each token, summed on the host as the reference's.
            let losses = self
                .decoder
                .token_losses(&leaves, &batch, &self.options())?;
            let values = self.backend.download(&losses.value())?.to_f64_vec()?;
            let policy: Vec<f64> = values
                .chunks(length)
                .map(|r| -r.iter().sum::<f64>())
                .collect();
            let even = |v: &[f64]| v.iter().step_by(2).copied().collect::<Vec<_>>();
            let odd = |v: &[f64]| v.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
            let lengths: Vec<usize> = pairs.iter().map(|p| p.chosen.answer_tokens()).collect();
            let terms = preference::dpo_terms(
                &even(&policy),
                &odd(&policy),
                &even(&reference),
                &odd(&reference),
                &lengths,
                config.beta,
                config.nll,
            );
            // Each token of a row holds its part of `−ln π`: its gradient is
            // the opposite of the row's (the ignored ones give none).
            let g: Vec<f32> = terms
                .chosen
                .iter()
                .zip(&terms.rejected)
                .flat_map(|(c, r)| [-*c as f32, -*r as f32])
                .flat_map(|d| std::iter::repeat_n(d, length))
                .collect();
            let g = self
                .backend
                .upload(&Tensor::from_slice(&g, &[rows * length])?)?;
            let g = self.backend.forward(
                &kapel_ops::Cast {
                    dtype: losses.dtype(),
                },
                &[&g],
            )?;
            let mut grads = tape.backward_with(&losses, g)?;
            drop(losses);
            let grads = leaves.gradients(&mut grads);
            self.accumulate(&mut sum, grads)?;
            loss += terms.loss / accumulate as f64;
            tokens += (rows * length) as u64;
            for (k, v) in [
                ("accuracy", terms.accuracy),
                ("chosen_reward", terms.chosen_reward),
                ("rejected_reward", terms.rejected_reward),
                ("margin", terms.chosen_reward - terms.rejected_reward),
                ("nll", terms.nll),
            ] {
                *means.entry(k).or_default() += v / accumulate as f64;
            }
        }
        Ok(StepOutput {
            grads: sum.0.into_iter().collect(),
            scale: 1.0 / accumulate as f64,
            loss,
            tokens,
            flops: accumulate as u64 * self.pass_flops(rows, 1),
            metrics: means
                .into_iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect(),
        })
    }

    /// The loss of DPO on the pairs held out, with their accuracy and the
    /// mean rewards, by the current weights, without gradients; `None`
    /// without such pairs.
    pub(super) fn evaluate_pairs(&self) -> Result<Option<serde_json::Map<String, Value>>> {
        let Objective::Pairs(p) = &self.objective else {
            return Ok(None);
        };
        if p.val.is_empty() {
            return Ok(None);
        }
        let (rows, length) = (self.config.batch.rows, self.config.batch.length);
        let (policy_params, reference_params) =
            (Self::frozen(&self.params), self.reference_params());
        let mut sums: BTreeMap<&str, f64> = BTreeMap::new();
        for chunk in p.val.chunks((rows / 2).max(1)) {
            let sequences: Vec<&Sequence> = chunk
                .iter()
                .flat_map(|p| [&p.chosen, &p.rejected])
                .collect();
            let (ids, targets) = rows_of(&sequences, length, p.pad);
            let sums_of = |logps: Vec<f64>| -> Vec<f64> {
                logps.chunks(length).map(|r| r.iter().sum()).collect()
            };
            let policy = sums_of(self.token_logps(&policy_params, &ids, &targets)?);
            let reference = sums_of(self.token_logps(&reference_params, &ids, &targets)?);
            let even = |v: &[f64]| v.iter().step_by(2).copied().collect::<Vec<_>>();
            let odd = |v: &[f64]| v.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
            let lengths: Vec<usize> = chunk.iter().map(|p| p.chosen.answer_tokens()).collect();
            let terms = preference::dpo_terms(
                &even(&policy),
                &odd(&policy),
                &even(&reference),
                &odd(&reference),
                &lengths,
                p.config.beta,
                0.0,
            );
            let n = chunk.len() as f64;
            for (k, v) in [
                ("val_loss", terms.loss),
                ("val_accuracy", terms.accuracy),
                ("val_chosen_reward", terms.chosen_reward),
                ("val_rejected_reward", terms.rejected_reward),
                ("val_margin", terms.chosen_reward - terms.rejected_reward),
            ] {
                *sums.entry(k).or_default() += v * n;
            }
        }
        let n = p.val.len() as f64;
        Ok(Some(
            sums.into_iter()
                .map(|(k, v)| (k.to_string(), json!(v / n)))
                .collect(),
        ))
    }

    /// The model of the current weights, for inference: the adapters merged,
    /// the float weights in the computation's type, the projections stacked.
    pub(super) fn inference_model(&self) -> Result<kapel_models::Model> {
        let merged;
        let params = match &self.config.lora {
            Some(l) => {
                merged = lora::merge(
                    &self.backend,
                    &self.params,
                    self.decoder.config(),
                    l.scale(),
                )?;
                &merged
            }
            None => &self.params,
        };
        let dtype = self.config.dtype();
        let mut weights = BTreeMap::new();
        for (n, p) in params.iter() {
            let v = if p.value.dtype().is_float() && p.value.dtype() != dtype {
                self.backend
                    .forward(&kapel_ops::Cast { dtype }, &[&p.value])?
            } else {
                p.value.clone()
            };
            weights.insert(n.to_string(), v);
        }
        let raw = match &self.config.lora {
            Some(_) => lora::merged_config(&self.model_json, &self.params, self.decoder.config()),
            None => self.model_json.clone(),
        };
        Ok(kapel_models::Model::new(raw, weights)?.into_fused(&self.backend)?)
    }

    /// The pipeline of generation, its model the current weights.
    fn refreshed_generator(&mut self) -> Result<&Pipeline<B>> {
        let model = self.inference_model()?;
        match &mut self.generator {
            Some(g) => *g.model_mut() = model,
            None => {
                self.generator = Some(Pipeline::with_model(
                    &self.config.model.config,
                    self.backend.clone(),
                    model,
                )?);
            }
        }
        Ok(self.generator.as_ref().expect("a pipeline"))
    }

    /// Writes the answers of a step of GRPO and scores them.
    fn rollout(&mut self) -> Result<Rollout> {
        let started = Instant::now();
        let length = self.config.batch.length;
        let seed = crate::data::mix(&[self.config.data.seed, 5, self.step as u64]);
        let (prompts, config) = {
            let Objective::Answers(a) = &mut self.objective else {
                unreachable!("a step of answers");
            };
            let config = a.config.clone();
            let prompts: Vec<Prompt> = a
                .prompts
                .next(config.prompts_per_step)
                .into_iter()
                .cloned()
                .collect();
            (prompts, config)
        };
        let group = config.group;
        let request = Request {
            sampling: config.sampling(seed),
            max_tokens: config.max_tokens,
            ..Request::default()
        };
        let asked: Vec<&[u32]> = prompts
            .iter()
            .flat_map(|p| std::iter::repeat_n(p.tokens.as_slice(), group))
            .collect();
        let cache = (self.config.training.cache_gib * f64::from(1u32 << 30)) as usize;
        let completions: Vec<Completion> = {
            let pipeline = self.refreshed_generator()?;
            grpo::generate(pipeline, &asked, &request, seed, cache)?
        };
        let generated: usize = completions.iter().map(|c| c.tokens.len()).sum();
        let writing = started.elapsed().as_secs_f64();
        let Objective::Answers(a) = &self.objective else {
            unreachable!("a step of answers");
        };
        let pad = a.pad;
        let scored: Vec<(&str, &Value)> = completions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.text.as_str(), &prompts[i / group].record))
            .collect();
        let (rewards, by_reward) = a.rewards.score(&scored)?;
        let advantages = grpo::advantages(&rewards, group, config.scales_rewards());
        let sequences: Vec<Sequence> = completions
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let prompt = &prompts[i / group].tokens;
                let mut tokens = prompt.clone();
                tokens.extend_from_slice(&c.tokens);
                tokens.truncate(length);
                let mask = (0..tokens.len()).map(|j| j >= prompt.len()).collect();
                Sequence { tokens, mask }
            })
            .collect();
        // The averaging of the loss, as weights of the tokens.
        let count = sequences.len() as f64;
        let answer_tokens: Vec<usize> = sequences.iter().map(Sequence::answer_tokens).collect();
        let total: usize = answer_tokens.iter().sum();
        let weights: Vec<f64> = answer_tokens
            .iter()
            .map(|&n| match config.loss.as_str() {
                "dr_grpo" => 1.0 / (count * config.max_tokens as f64),
                "token" => 1.0 / total.max(1) as f64,
                _ => 1.0 / (count * n.max(1) as f64),
            })
            .collect();
        let mean = rewards.iter().sum::<f64>() / count;
        let deviation = (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / count).sqrt();
        let mut metrics = serde_json::Map::new();
        metrics.insert("reward".into(), json!(mean));
        metrics.insert("reward_std".into(), json!(deviation));
        for (name, m) in by_reward {
            metrics.insert(format!("reward_{name}"), json!(m));
        }
        metrics.insert("answer_tokens".into(), json!(total as f64 / count));
        metrics.insert(
            "ended".into(),
            json!(completions.iter().filter(|c| c.ended).count() as f64 / count),
        );
        metrics.insert("writing_s".into(), json!(writing));
        let mut rollout = Rollout {
            sequences,
            advantages,
            weights,
            old: None,
            reference: None,
            left: config.iterations,
            metrics,
            flops: 2 * generated as u64 * self.params.count_all() as u64,
        };
        // The log-probabilities that stay the same along the steps on these
        // answers.
        let counted = |s: &Sequence, logps: &[f64]| -> Vec<f64> {
            (0..length)
                .filter(|&j| j + 1 < s.tokens.len() && s.mask[j + 1])
                .map(|j| logps[j])
                .collect()
        };
        let by_rows = |params: &Parameters| -> Result<Vec<Vec<f64>>> {
            let mut out = Vec::with_capacity(rollout.sequences.len());
            for chunk in rollout.sequences.chunks(self.config.batch.rows) {
                let refs: Vec<&Sequence> = chunk.iter().collect();
                let (ids, targets) = rows_of(&refs, length, pad);
                let logps = self.token_logps(params, &ids, &targets)?;
                for (s, row) in chunk.iter().zip(logps.chunks(length)) {
                    out.push(counted(s, row));
                }
            }
            Ok(out)
        };
        if config.iterations > 1 {
            let now = Self::frozen(&self.params);
            rollout.old = Some(by_rows(&now)?);
        }
        if config.beta > 0.0 {
            let reference = self.reference_params();
            rollout.reference = Some(by_rows(&reference)?);
        }
        Ok(rollout)
    }

    /// A step of GRPO: answers written if the last ones are spent, then the
    /// passes over them.
    pub(super) fn answer_step(&mut self) -> Result<StepOutput> {
        let spent = match &self.objective {
            Objective::Answers(a) => a.rollout.as_ref().is_none_or(|r| r.left == 0),
            _ => unreachable!("a step of answers"),
        };
        let mut flops = 0;
        if spent {
            let r = self.rollout()?;
            flops += r.flops;
            let Objective::Answers(a) = &mut self.objective else {
                unreachable!("a step of answers");
            };
            a.rollout = Some(r);
        }
        let (rows, length) = (self.config.batch.rows, self.config.batch.length);
        let Objective::Answers(a) = &mut self.objective else {
            unreachable!("a step of answers");
        };
        let (config, pad) = (a.config.clone(), a.pad);
        let rollout = a.rollout.take().expect("answers written");
        let mut sum = Summed::default();
        let (mut loss, mut kl, mut clipped, mut counted) = (0.0, 0.0, 0usize, 0usize);
        for (c, chunk) in rollout.sequences.chunks(rows).enumerate() {
            let refs: Vec<&Sequence> = chunk.iter().collect();
            let (ids, targets) = rows_of(&refs, length, pad);
            let tape = Tape::new(self.backend.clone());
            let leaves = self.params.leaves(&tape);
            let batch = Batch {
                ids: &ids,
                targets: &targets,
                rows: chunk.len(),
                segments: None,
            };
            let losses = self
                .decoder
                .token_losses(&leaves, &batch, &self.options())?;
            let values = self.backend.download(&losses.value())?.to_f64_vec()?;
            // The counted positions, each with its sequence.
            let mut at = Vec::new();
            let (mut logp, mut answer, mut weight) = (Vec::new(), Vec::new(), Vec::new());
            let (mut old, mut reference) = (Vec::new(), Vec::new());
            for (k, s) in chunk.iter().enumerate() {
                let i = c * rows + k;
                let mut n = 0;
                for j in 0..length {
                    if j + 1 < s.tokens.len() && s.mask[j + 1] {
                        let p = k * length + j;
                        at.push(p);
                        logp.push(-values[p]);
                        answer.push(i);
                        weight.push(rollout.weights[i]);
                        old.push(rollout.old.as_ref().map_or(-values[p], |o| o[i][n]));
                        if let Some(r) = &rollout.reference {
                            reference.push(r[i][n]);
                        }
                        n += 1;
                    }
                }
            }
            let terms = grpo::token_terms(
                &logp,
                &old,
                rollout.reference.as_ref().map(|_| reference.as_slice()),
                &rollout.advantages,
                &answer,
                &weight,
                &config,
            );
            let mut g = vec![0f32; values.len()];
            for (&p, d) in at.iter().zip(&terms.grads) {
                // The losses are `−ln π`.
                g[p] = -*d as f32;
            }
            let g = self
                .backend
                .upload(&Tensor::from_slice(&g, &[values.len()])?)?;
            let g = self.backend.forward(
                &kapel_ops::Cast {
                    dtype: losses.dtype(),
                },
                &[&g],
            )?;
            let mut grads = tape.backward_with(&losses, g)?;
            drop(losses);
            let grads = leaves.gradients(&mut grads);
            self.accumulate(&mut sum, grads)?;
            loss += terms.loss;
            kl += terms.kl;
            clipped += terms.clipped;
            counted += at.len();
        }
        let mut metrics = rollout.metrics.clone();
        if config.beta > 0.0 {
            metrics.insert("kl".into(), json!(kl / counted.max(1) as f64));
        }
        metrics.insert(
            "clipped".into(),
            json!(clipped as f64 / counted.max(1) as f64),
        );
        let sequences = rollout.sequences.len();
        let Objective::Answers(a) = &mut self.objective else {
            unreachable!("a step of answers");
        };
        a.rollout = Some(Rollout {
            left: rollout.left - 1,
            flops: 0,
            ..rollout
        });
        let references =
            usize::from(config.beta > 0.0 && spent) + usize::from(config.iterations > 1 && spent);
        Ok(StepOutput {
            grads: sum.0.into_iter().collect(),
            scale: 1.0,
            loss,
            tokens: (sequences * length) as u64,
            flops: flops + self.pass_flops(sequences, references),
            metrics,
        })
    }

    /// The accuracy of each task, answered greedily, written to the log.
    pub(super) fn evaluate_tasks(&mut self) -> Result<()> {
        if self.tasks.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        self.refreshed_generator()?;
        let pipeline = self.generator.as_ref().expect("a pipeline");
        let cache = (self.config.training.cache_gib * f64::from(1u32 << 30)) as usize;
        let mut lines = Vec::new();
        for task in &self.tasks {
            let request = Request {
                sampling: Sampling::greedy(),
                max_tokens: task.max_tokens,
                ..Request::default()
            };
            let asked: Vec<&[u32]> = task.prompts.iter().map(|p| p.tokens.as_slice()).collect();
            let answers = grpo::generate(pipeline, &asked, &request, 0, cache)?;
            let scored: Vec<(&str, &Value)> = answers
                .iter()
                .zip(&task.prompts)
                .map(|(c, p)| (c.text.as_str(), &p.record))
                .collect();
            let (rewards, _) = task.rewards.score(&scored)?;
            let n = rewards.len().max(1) as f64;
            lines.push((
                task.name.clone(),
                rewards.iter().sum::<f64>() / n,
                answers.iter().map(|c| c.tokens.len()).sum::<usize>() as f64 / n,
                answers.iter().filter(|c| c.ended).count() as f64 / n,
            ));
        }
        let seconds = started.elapsed().as_secs_f64();
        for (name, accuracy, tokens, ended) in lines {
            self.record(json!({
                "step": self.step,
                "task": name,
                "accuracy": accuracy,
                "answer_tokens": tokens,
                "ended": ended,
                "tokens": self.tokens,
            }))?;
            if !self.options.quiet {
                eprintln!(
                    "step {:>6}/{} | {name} {:.2} % ({tokens:.0} tokens an answer, {:.0} % ended, {seconds:.1} s)",
                    self.step,
                    self.config.schedule.steps,
                    100.0 * accuracy,
                    100.0 * ended
                );
            }
        }
        Ok(())
    }
}
