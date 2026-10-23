//! `kapel eval`: the tasks of lm-evaluation-harness replayed by Kapel, from
//! the requests `tools/eval/export_tasks.py` writes.
//!
//! The requests are tokenized and scored as lm-evaluation-harness does with
//! a `transformers` model (`HFLM`): a context's trailing spaces moved to its
//! continuation, the pair encoded whole then split where the context alone
//! ends, the special tokens added unless the text begins with the prefix
//! token, an empty context replaced by the prefix token, sequences cut on
//! the left to the model's length, and texts scored by disjoint windows of
//! that length.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use kapel_infer::{Pipeline, Scorer};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::Tokenizer;
use serde_json::{Value, json};

use crate::args::{self, Device, Load};

pub(crate) const USAGE: &str = "\
usage: kapel eval MODEL --tasks FILE [options]

Replays the tasks of lm-evaluation-harness exported to FILE by
tools/eval/export_tasks.py with MODEL, and gives their scores: accuracies for
tasks by choice (acc, and acc_norm, normalized by the length of the answers),
perplexities by word and by byte and bits per byte for texts. The requests are
tokenized and scored as lm-evaluation-harness scores a transformers model.

options:
  --tasks FILE             the requests (JSON lines)
  --only TASK,...          only these tasks or groups
  --limit N                the first N documents of each task
  --max-length N           the longest sequence (default the model's
                           max_position_embeddings, as lm-evaluation-harness)
  --prefix-token ID        the token before an empty context and a text
                           (default the tokenizer's beginning of sequence, else
                           its end)
  --piece N                the most tokens per pass of the model (default 512)
  --output FILE            the scores as JSON, by task and by group
  --device, --dtype, --revision, --quantize, --quantize-output, --kv-cache,
  --byte-activations       as for kapel run
";

/// What `kapel eval` was asked.
#[derive(Clone, Debug)]
struct Eval {
    load: Load,
    tasks: PathBuf,
    only: Option<Vec<String>>,
    limit: Option<usize>,
    max_length: Option<usize>,
    prefix_token: Option<u32>,
    piece: usize,
    output: Option<PathBuf>,
}

fn parse(args: &[String]) -> Result<Eval, String> {
    let mut tasks = None;
    let mut eval = Eval {
        load: Load::default(),
        tasks: PathBuf::new(),
        only: None,
        limit: None,
        max_length: None,
        prefix_token: None,
        piece: 512,
        output: None,
    };
    let mut load = Load::default();
    args::walk("eval", args, &mut load, |option, value| {
        match option {
            "--tasks" => tasks = Some(PathBuf::from(value()?)),
            "--only" => {
                eval.only = Some(value()?.split(',').map(|t| t.trim().to_string()).collect());
            }
            "--limit" => eval.limit = Some(args::number(option, &value()?)?),
            "--max-length" => eval.max_length = Some(args::number(option, &value()?)?),
            "--prefix-token" => eval.prefix_token = Some(args::number(option, &value()?)?),
            "--piece" => eval.piece = args::number::<usize>(option, &value()?)?.max(1),
            "--output" => eval.output = Some(PathBuf::from(value()?)),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    eval.load = load;
    eval.tasks =
        tasks.ok_or("eval: which tasks? (--tasks FILE, from tools/eval/export_tasks.py)")?;
    if eval.max_length == Some(0) {
        return Err("--max-length: at least 1".into());
    }
    Ok(eval)
}

/// Runs `kapel eval`.
pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let eval = parse(args)?;
    let documents = read_documents(&eval)?;
    if documents.is_empty() {
        return Err(format!("{}: no document to evaluate", eval.tasks.display()));
    }
    let dir = crate::directory(&eval.load.model, &eval.load.revision)?;
    let options = crate::load_options(&eval.load);
    match eval.load.device {
        Device::Cpu => {
            let cpu = kapel_cpu::Cpu::new().with_byte_activations(eval.load.byte_activations);
            let pipeline = Pipeline::load_with(&dir, cpu, &options).map_err(|e| e.to_string())?;
            evaluate(&eval, &dir, &pipeline, &documents)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| e.to_string())?;
            let pipeline = Pipeline::load_with(&dir, cuda, &options).map_err(|e| e.to_string())?;
            evaluate(&eval, &dir, &pipeline, &documents)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err("this kapel was built without the feature `cuda`".into()),
    }
}

/// A document of a task: what to score, and how.
#[derive(Clone, Debug)]
struct Document {
    task: String,
    group: String,
    kind: Kind,
}

#[derive(Clone, Debug)]
enum Kind {
    /// Choices: the log-likelihood of each continuation after its context.
    Choice {
        requests: Vec<(String, String)>,
        gold: Vec<usize>,
        lengths: Vec<usize>,
    },
    /// A text scored whole (perplexity), with its words and bytes.
    Rolling {
        text: String,
        words: f64,
        bytes: f64,
    },
}

fn read_documents(eval: &Eval) -> Result<Vec<Document>, String> {
    let file =
        std::fs::File::open(&eval.tasks).map_err(|e| format!("{}: {e}", eval.tasks.display()))?;
    let mut documents = Vec::new();
    let mut per_task: BTreeMap<String, usize> = BTreeMap::new();
    for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", eval.tasks.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("{}:{}", eval.tasks.display(), n + 1);
        let value: Value = serde_json::from_str(&line).map_err(|e| format!("{}: {e}", at()))?;
        let text = |key: &str| {
            value[key]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{}: no {key:?}", at()))
        };
        let task = text("task")?;
        let group = value["group"].as_str().unwrap_or(&task).to_string();
        if let Some(only) = &eval.only
            && !only.iter().any(|o| *o == task || *o == group)
        {
            continue;
        }
        let count = per_task.entry(task.clone()).or_default();
        if eval.limit.is_some_and(|limit| *count >= limit) {
            continue;
        }
        *count += 1;
        let numbers = |key: &str| -> Result<Vec<usize>, String> {
            value[key]
                .as_array()
                .and_then(|a| a.iter().map(|v| v.as_u64().map(|v| v as usize)).collect())
                .ok_or_else(|| format!("{}: {key:?} is not a list of numbers", at()))
        };
        let kind = match value["type"].as_str() {
            Some("choice") => {
                let requests = value["requests"]
                    .as_array()
                    .and_then(|a| {
                        a.iter()
                            .map(|pair| {
                                Some((
                                    pair.get(0)?.as_str()?.to_string(),
                                    pair.get(1)?.as_str()?.to_string(),
                                ))
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| format!("{}: \"requests\" are not pairs of texts", at()))?;
                let (gold, lengths) = (numbers("gold")?, numbers("lengths")?);
                if requests.is_empty() || lengths.len() != requests.len() {
                    return Err(format!(
                        "{}: as many lengths as requests, at least one",
                        at()
                    ));
                }
                Kind::Choice {
                    requests,
                    gold,
                    lengths,
                }
            }
            Some("rolling") => Kind::Rolling {
                text: text("text")?,
                words: value["words"].as_f64().unwrap_or(0.0),
                bytes: value["bytes"].as_f64().unwrap_or(0.0),
            },
            other => return Err(format!("{}: unknown type {other:?}", at())),
        };
        documents.push(Document { task, group, kind });
    }
    Ok(documents)
}

/// The token lm-evaluation-harness puts before an empty context and a text:
/// the tokenizer's `bos_token`, else its `eos_token`.
fn prefix_token(dir: &Path, tokenizer: &Tokenizer) -> Result<u32, String> {
    let id = |name: &str| tokenizer.token_to_id(name);
    if dir.is_dir() {
        let config: Value = std::fs::read(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let token = |key: &str| {
            let v = &config[key];
            v.as_str().or_else(|| v["content"].as_str()).and_then(id)
        };
        token("bos_token").or_else(|| token("eos_token"))
    } else {
        let gguf = kapel_hub::gguf::Gguf::open(dir).map_err(|e| e.to_string())?;
        let token = |key: &str| match gguf.metadata.get(key) {
            Some(kapel_hub::gguf::Value::Uint(v)) => Some(*v as u32),
            Some(kapel_hub::gguf::Value::Int(v)) => Some(*v as u32),
            _ => None,
        };
        token("tokenizer.ggml.bos_token_id").or_else(|| token("tokenizer.ggml.eos_token_id"))
    }
    .ok_or_else(|| "the tokenizer has no beginning or end of sequence; give --prefix-token".into())
}

/// lm-evaluation-harness's tokenization.
struct Encoder<'t> {
    tokenizer: &'t Tokenizer,
    prefix: u32,
    /// The text of the prefix token.
    prefix_text: String,
}

impl Encoder<'_> {
    /// `tok_encode`: the special tokens added, unless `text` begins with
    /// the prefix token.
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let special = self.prefix_text.is_empty() || !text.starts_with(&self.prefix_text);
        self.tokenizer
            .encode(text, special)
            .map_err(|e| e.to_string())
    }

    /// The context and the continuation of a request, as tokens.
    fn pair(&self, context: &str, continuation: &str) -> Result<(Vec<u32>, Vec<u32>), String> {
        if context.is_empty() {
            let tokens = self
                .tokenizer
                .encode(continuation, false)
                .map_err(|e| e.to_string())?;
            return Ok(match tokens.first() {
                Some(&first) if first == self.prefix => {
                    (tokens[..1].to_vec(), tokens[1..].to_vec())
                }
                _ => (vec![self.prefix], tokens),
            });
        }
        // The trailing spaces of the context begin the continuation.
        let kept = context.trim_end().len();
        let (context, spaces) = context.split_at(kept);
        let whole = self.encode(&format!("{context}{spaces}{continuation}"))?;
        let context = self.encode(context)?;
        let continuation = whole.get(context.len()..).unwrap_or_default().to_vec();
        Ok((context, continuation))
    }
}

/// `context` and `continuation` cut on the left to `max_length + 1` tokens
/// in all: the context then the continuation, the last not read.
fn truncate(context: &[u32], continuation: &[u32], max_length: usize) -> Vec<u32> {
    let all: Vec<u32> = context.iter().chain(continuation).copied().collect();
    let from = all.len().saturating_sub(max_length + 1);
    all[from..all.len() - continuation.len()].to_vec()
}

/// The disjoint windows of a text of `tokens`, `(context, predicted)`:
/// `get_rolling_token_windows` with a context of one token, then
/// `make_disjoint_window`.
fn windows(tokens: &[u32], prefix: u32, max_length: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut windows = Vec::new();
    if tokens.is_empty() {
        return windows;
    }
    let first = max_length.min(tokens.len());
    let context: Vec<u32> = std::iter::once(prefix)
        .chain(tokens[..first - 1].iter().copied())
        .collect();
    windows.push((context, tokens[..first].to_vec()));
    let mut predicted = first;
    while predicted < tokens.len() {
        let length = (tokens.len() - predicted).min(max_length);
        let end = predicted + length;
        let context = &tokens[end - max_length - 1..end - 1];
        windows.push((context.to_vec(), tokens[end - length..end].to_vec()));
        predicted += length;
    }
    windows
        .into_iter()
        .map(|(context, predicted)| {
            let keep = context.len() - (predicted.len() - 1);
            (context[..keep].to_vec(), predicted)
        })
        .collect()
}

/// The sums of a task or a group.
#[derive(Clone, Debug, Default)]
struct Tally {
    documents: usize,
    /// Tasks by choice: right answers, by log-likelihood and normalized.
    right: f64,
    right_norm: f64,
    /// Texts: log-likelihood, words and bytes.
    log_likelihood: f64,
    words: f64,
    bytes: f64,
    rolling: bool,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.documents += other.documents;
        self.right += other.right;
        self.right_norm += other.right_norm;
        self.log_likelihood += other.log_likelihood;
        self.words += other.words;
        self.bytes += other.bytes;
        self.rolling |= other.rolling;
    }

    fn metrics(&self) -> Value {
        let n = self.documents as f64;
        if self.rolling {
            json!({
                "documents": self.documents,
                "word_perplexity": (-self.log_likelihood / self.words).exp(),
                "byte_perplexity": (-self.log_likelihood / self.bytes).exp(),
                "bits_per_byte": -self.log_likelihood / self.bytes / std::f64::consts::LN_2,
            })
        } else {
            json!({
                "documents": self.documents,
                "acc": self.right / n,
                "acc_norm": self.right_norm / n,
            })
        }
    }

    fn line(&self, name: &str) -> String {
        if self.rolling {
            format!(
                "{name:<40} {:>6}  word ppl {:>10.4}  byte ppl {:.4}  bits/byte {:.4}",
                self.documents,
                (-self.log_likelihood / self.words).exp(),
                (-self.log_likelihood / self.bytes).exp(),
                -self.log_likelihood / self.bytes / std::f64::consts::LN_2,
            )
        } else {
            let n = self.documents as f64;
            format!(
                "{name:<40} {:>6}  acc {:.4}  acc_norm {:.4}",
                self.documents,
                self.right / n,
                self.right_norm / n
            )
        }
    }
}

/// The index of the largest of `values`, the first of equals (`np.argmax`).
fn argmax(values: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (i, v) in values.enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

fn evaluate<B: Backend + Transfer>(
    eval: &Eval,
    dir: &Path,
    pipeline: &Pipeline<B>,
    documents: &[Document],
) -> Result<(), String> {
    let tokenizer = pipeline.tokenizer();
    let prefix = match eval.prefix_token {
        Some(id) => id,
        None => prefix_token(dir, tokenizer)?,
    };
    let encoder = Encoder {
        tokenizer,
        prefix,
        prefix_text: tokenizer.decode(&[prefix], false),
    };
    let max_length = eval
        .max_length
        .or(pipeline.model().config().max_positions)
        .unwrap_or(2048);
    eprintln!(
        "kapel: {} documents, sequences of up to {max_length} tokens, prefix token {prefix} \
         ({:?})",
        documents.len(),
        encoder.prefix_text
    );
    let mut scorer = Scorer::new(pipeline, eval.piece);
    let mut tasks: BTreeMap<String, (String, Tally)> = BTreeMap::new();
    let start = Instant::now();
    let mut last_report = Instant::now();
    for (n, document) in documents.iter().enumerate() {
        let mut tally = Tally {
            documents: 1,
            ..Tally::default()
        };
        match &document.kind {
            Kind::Choice {
                requests,
                gold,
                lengths,
            } => {
                let mut scores = Vec::with_capacity(requests.len());
                for (context, continuation) in requests {
                    let (context, continuation) = encoder.pair(context, continuation)?;
                    if continuation.is_empty() || continuation.len() > max_length {
                        return Err(format!(
                            "{}: a continuation of {} tokens (between 1 and {max_length})",
                            document.task,
                            continuation.len()
                        ));
                    }
                    let context = truncate(&context, &continuation, max_length);
                    let score = scorer
                        .score(&context, &continuation)
                        .map_err(|e| e.to_string())?;
                    scores.push(score.log_likelihood);
                }
                let pick = argmax(scores.iter().copied());
                let pick_norm = argmax(scores.iter().zip(lengths).map(|(s, &l)| s / l as f64));
                tally.right = f64::from(u8::from(gold.contains(&pick)));
                tally.right_norm = f64::from(u8::from(gold.contains(&pick_norm)));
            }
            Kind::Rolling { text, words, bytes } => {
                let tokens = encoder.encode(text)?;
                for (context, predicted) in windows(&tokens, prefix, max_length) {
                    let score = scorer
                        .score(&context, &predicted)
                        .map_err(|e| e.to_string())?;
                    tally.log_likelihood += score.log_likelihood;
                }
                tally.words = *words;
                tally.bytes = *bytes;
                tally.rolling = true;
            }
        }
        tasks
            .entry(document.task.clone())
            .or_insert_with(|| (document.group.clone(), Tally::default()))
            .1
            .add(&tally);
        if last_report.elapsed().as_secs() >= 10 || n + 1 == documents.len() {
            last_report = Instant::now();
            eprint!(
                "\rkapel: {}/{} documents, {:.0} s",
                n + 1,
                documents.len(),
                start.elapsed().as_secs_f64()
            );
            std::io::stderr().flush().ok();
        }
    }
    eprintln!();
    let mut groups: BTreeMap<String, (usize, Tally)> = BTreeMap::new();
    for (task, (group, tally)) in &tasks {
        if group != task {
            let entry = groups.entry(group.clone()).or_default();
            entry.0 += 1;
            entry.1.add(tally);
        }
    }
    for (task, (_, tally)) in &tasks {
        println!("{}", tally.line(task));
    }
    for (group, (count, tally)) in &groups {
        println!("{}", tally.line(&format!("{group} ({count} tasks)")));
    }
    if let Some(output) = &eval.output {
        let results = json!({
            "model": eval.load.model,
            "max_length": max_length,
            "prefix_token": prefix,
            "seconds": start.elapsed().as_secs_f64(),
            "tasks": tasks.iter().map(|(t, (_, tally))| (t.clone(), tally.metrics())).collect::<serde_json::Map<_, _>>(),
            "groups": groups.iter().map(|(g, (_, tally))| (g.clone(), tally.metrics())).collect::<serde_json::Map<_, _>>(),
        });
        let text = serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?;
        std::fs::write(output, text + "\n").map_err(|e| format!("{}: {e}", output.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_those_of_lm_evaluation_harness() {
        // get_rolling_token_windows(range(1, 11), prefix=0, max_seq_len=4,
        // context_len=1), then make_disjoint_window.
        let tokens: Vec<u32> = (1..=10).collect();
        let want: Vec<(Vec<u32>, Vec<u32>)> = vec![
            (vec![0], vec![1, 2, 3, 4]),
            (vec![4], vec![5, 6, 7, 8]),
            (vec![6, 7, 8], vec![9, 10]),
        ];
        assert_eq!(windows(&tokens, 0, 4), want);
        assert_eq!(windows(&[7], 0, 4), vec![(vec![0], vec![7])]);
        assert!(windows(&[], 0, 4).is_empty());
        // Every token predicted once, after at most max_length inputs.
        for max in 1..6 {
            let w = windows(&tokens, 0, max);
            let predicted: Vec<u32> = w.iter().flat_map(|(_, p)| p.clone()).collect();
            assert_eq!(predicted, tokens);
            assert!(
                w.iter()
                    .all(|(c, p)| !c.is_empty() && c.len() + p.len() - 1 <= max)
            );
        }
    }

    #[test]
    fn long_sequences_are_cut_on_the_left() {
        assert_eq!(truncate(&[1, 2, 3], &[4, 5], 10), vec![1, 2, 3]);
        // Five tokens, four read at most: the first goes.
        assert_eq!(truncate(&[1, 2, 3], &[4, 5], 3), vec![2, 3]);
    }

    #[test]
    fn argmax_takes_the_first_of_equals() {
        assert_eq!(argmax([1.0, 3.0, 3.0, 2.0].into_iter()), 1);
        assert_eq!(argmax([f64::NEG_INFINITY].into_iter()), 0);
    }
}
