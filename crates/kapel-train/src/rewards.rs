//! Rewards: how good an answer is, for GRPO and for the evaluations of
//! tasks whose answers a program checks.
//!
//! A reward reads the answer's text and the record its prompt came from
//! (the JSON object of the file of prompts, its expected answer in a
//! field). Kapel has four kinds:
//!
//! - `math`: 1 if the number the answer ends on (the content of its last
//!   `\boxed{…}`, else what follows its last `####`, else its last number)
//!   is the number of the record's field, 0 otherwise; GSM8K's answers end
//!   with `#### 72`, and its field is read the same way;
//! - `exact`: 1 if the answer, trimmed, is the field's text;
//! - `regex`: 1 if the answer matches the pattern, a format to follow;
//! - `command`: a program of one's own, started once per batch of answers:
//!   it reads one JSON object per line (`{"completion": …, "record": …}`)
//!   and writes one number per line.
//!
//! The reward of an answer is the sum of those of the run, each weighted.
//! A program reaches them through [`Reward`].

use std::io::{BufRead as _, Write as _};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A reward that cannot be computed.
#[derive(Debug)]
pub struct RewardError(pub String);

impl std::fmt::Display for RewardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RewardError {}

type Result<T> = std::result::Result<T, RewardError>;

/// A reward of answers.
pub trait Reward: Send + Sync {
    /// Its name, for the log.
    fn name(&self) -> &str;

    /// The rewards of `completions`, each with the record of its prompt.
    ///
    /// # Errors
    ///
    /// If the reward cannot be computed.
    fn score(&self, completions: &[(&str, &Value)]) -> Result<Vec<f64>>;
}

/// A reward of the configuration.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewardConfig {
    /// `"math"`, `"exact"`, `"regex"` or `"command"`.
    pub kind: String,
    /// Its weight in the sum.
    #[serde(default = "one")]
    pub weight: f64,
    /// The record's field of the expected answer, for `math` and `exact`.
    #[serde(default = "answer")]
    pub field: String,
    /// The pattern, for `regex`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// The program and its arguments, for `command`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
}

fn one() -> f64 {
    1.0
}

fn answer() -> String {
    "answer".into()
}

impl RewardConfig {
    /// The reward it describes.
    ///
    /// # Errors
    ///
    /// For an unknown kind, a pattern that does not compile, or a command
    /// without a program.
    pub fn build(&self) -> Result<Box<dyn Reward>> {
        let fail = |m: String| Err(RewardError(m));
        match self.kind.as_str() {
            "math" => Ok(Box::new(Math {
                field: self.field.clone(),
            })),
            "exact" => Ok(Box::new(Exact {
                field: self.field.clone(),
            })),
            "regex" => {
                let Some(p) = &self.pattern else {
                    return fail("a regex reward without a pattern".into());
                };
                let regex = fancy_regex::Regex::new(p)
                    .map_err(|e| RewardError(format!("pattern {p:?}: {e}")))?;
                Ok(Box::new(Pattern {
                    name: format!("regex {p}"),
                    regex,
                }))
            }
            "command" => {
                if self.command.is_empty() {
                    return fail("a command reward without a program".into());
                }
                Ok(Box::new(Command {
                    argv: self.command.clone(),
                }))
            }
            other => fail(format!(
                "reward kind: \"math\", \"exact\", \"regex\" or \"command\", got {other:?}"
            )),
        }
    }
}

/// The text of a record's field: a string, or a number written.
fn field_text(record: &Value, field: &str) -> Option<String> {
    match record.get(field)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// The numbers of `text`, in order: digits with an optional sign, decimal
/// point and thousands separators.
fn numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let starts = chars[i].is_ascii_digit()
            || (chars[i] == '-' && chars.get(i + 1).is_some_and(char::is_ascii_digit));
        if !starts {
            i += 1;
            continue;
        }
        let mut s = String::new();
        if chars[i] == '-' {
            s.push('-');
            i += 1;
        }
        while i < chars.len() {
            let c = chars[i];
            let next_digit = chars.get(i + 1).is_some_and(char::is_ascii_digit);
            if c.is_ascii_digit() {
                s.push(c);
            } else if c == ',' && next_digit {
                // A separator of thousands.
            } else if c == '.' && next_digit && !s.contains('.') {
                s.push(c);
            } else {
                break;
            }
            i += 1;
        }
        if let Ok(v) = s.parse() {
            out.push(v);
        }
    }
    out
}

/// The number an answer ends on: in its last `\boxed{…}`, else after its
/// last `####`, else its last number.
pub fn final_number(text: &str) -> Option<f64> {
    if let Some(at) = text.rfind("\\boxed{") {
        let inside = &text[at + 7..];
        let end = inside.find('}').unwrap_or(inside.len());
        if let Some(&v) = numbers(&inside[..end]).last() {
            return Some(v);
        }
    }
    if let Some(at) = text.rfind("####")
        && let Some(&v) = numbers(&text[at + 4..]).first()
    {
        return Some(v);
    }
    numbers(text).last().copied()
}

struct Math {
    field: String,
}

impl Reward for Math {
    fn name(&self) -> &str {
        "math"
    }

    fn score(&self, completions: &[(&str, &Value)]) -> Result<Vec<f64>> {
        completions
            .iter()
            .map(|(text, record)| {
                let expected = field_text(record, &self.field)
                    .and_then(|t| final_number(&t))
                    .ok_or_else(|| {
                        RewardError(format!("a record without a number in {:?}", self.field))
                    })?;
                let got = final_number(text);
                Ok(f64::from(u8::from(got.is_some_and(|g| {
                    (g - expected).abs() <= 1e-6 * expected.abs().max(1.0)
                }))))
            })
            .collect()
    }
}

struct Exact {
    field: String,
}

impl Reward for Exact {
    fn name(&self) -> &str {
        "exact"
    }

    fn score(&self, completions: &[(&str, &Value)]) -> Result<Vec<f64>> {
        completions
            .iter()
            .map(|(text, record)| {
                let expected = field_text(record, &self.field).ok_or_else(|| {
                    RewardError(format!("a record without the field {:?}", self.field))
                })?;
                Ok(f64::from(u8::from(text.trim() == expected.trim())))
            })
            .collect()
    }
}

struct Pattern {
    name: String,
    regex: fancy_regex::Regex,
}

impl Reward for Pattern {
    fn name(&self) -> &str {
        &self.name
    }

    fn score(&self, completions: &[(&str, &Value)]) -> Result<Vec<f64>> {
        completions
            .iter()
            .map(|(text, _)| {
                let found = self
                    .regex
                    .is_match(text)
                    .map_err(|e| RewardError(e.to_string()))?;
                Ok(f64::from(u8::from(found)))
            })
            .collect()
    }
}

struct Command {
    argv: Vec<String>,
}

impl Reward for Command {
    fn name(&self) -> &str {
        &self.argv[0]
    }

    fn score(&self, completions: &[(&str, &Value)]) -> Result<Vec<f64>> {
        let fail = |m: String| RewardError(format!("{}: {m}", self.argv[0]));
        let mut child = std::process::Command::new(&self.argv[0])
            .args(&self.argv[1..])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| fail(e.to_string()))?;
        let mut input = child.stdin.take().expect("a piped input");
        let lines: Vec<String> = completions
            .iter()
            .map(|(text, record)| {
                serde_json::json!({"completion": text, "record": record}).to_string()
            })
            .collect();
        // Written by a thread: a program that answers line by line would
        // otherwise block on a full pipe.
        let writer = std::thread::spawn(move || -> std::io::Result<()> {
            for line in lines {
                writeln!(input, "{line}")?;
            }
            Ok(())
        });
        let output = child.stdout.take().expect("a piped output");
        let mut scores = Vec::with_capacity(completions.len());
        for line in std::io::BufReader::new(output).lines() {
            let line = line.map_err(|e| fail(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            scores.push(
                line.trim()
                    .parse::<f64>()
                    .map_err(|_| fail(format!("not a number: {line:?}")))?,
            );
        }
        writer
            .join()
            .map_err(|_| fail("the writing thread panicked".into()))?
            .map_err(|e| fail(e.to_string()))?;
        let status = child.wait().map_err(|e| fail(e.to_string()))?;
        if !status.success() {
            return Err(fail(format!("ended with {status}")));
        }
        if scores.len() != completions.len() {
            return Err(fail(format!(
                "{} numbers for {} answers",
                scores.len(),
                completions.len()
            )));
        }
        Ok(scores)
    }
}

/// The weighted sum of the rewards of each answer, and each reward's mean
/// by name.
pub type Scores = (Vec<f64>, Vec<(String, f64)>);

/// The rewards of a run, each with its weight.
pub struct Rewards {
    rewards: Vec<(Box<dyn Reward>, f64)>,
}

impl Rewards {
    /// The rewards `configs` describe.
    ///
    /// # Errors
    ///
    /// As [`RewardConfig::build`].
    pub fn new(configs: &[RewardConfig]) -> Result<Self> {
        Ok(Self {
            rewards: configs
                .iter()
                .map(|c| Ok((c.build()?, c.weight)))
                .collect::<Result<_>>()?,
        })
    }

    /// Rewards of one's own, each with its weight.
    pub fn of(rewards: Vec<(Box<dyn Reward>, f64)>) -> Self {
        Self { rewards }
    }

    /// The weighted sum of the rewards of each completion, and each
    /// reward's mean, by name.
    ///
    /// # Errors
    ///
    /// The first reward's that fails.
    pub fn score(&self, completions: &[(&str, &Value)]) -> Result<Scores> {
        let mut total = vec![0.0; completions.len()];
        let mut means = Vec::with_capacity(self.rewards.len());
        for (reward, weight) in &self.rewards {
            let scores = reward.score(completions)?;
            for (t, s) in total.iter_mut().zip(&scores) {
                *t += weight * s;
            }
            let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
            means.push((reward.name().to_string(), mean));
        }
        Ok((total, means))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_final_number_of_an_answer() {
        assert_eq!(final_number("so 3 + 4 = 7.\n#### 7"), Some(7.0));
        assert_eq!(final_number("It costs $1,250.50 in all."), Some(1250.5));
        assert_eq!(
            final_number("first 3, then \\boxed{-12}. Check: 5"),
            Some(-12.0)
        );
        assert_eq!(final_number("#### 18 dollars, 4 left"), Some(18.0));
        assert_eq!(final_number("no number"), None);
        assert_eq!(final_number("1. 2. 3."), Some(3.0));
    }

    #[test]
    fn rewards_by_kind() {
        let record = serde_json::json!({"answer": "Natalia sold 48/2 = 24 clips.\n#### 72"});
        let config = |kind: &str| RewardConfig {
            kind: kind.into(),
            weight: 1.0,
            field: "answer".into(),
            pattern: Some(r"^<think>".into()),
            command: Vec::new(),
        };
        let math = config("math").build().unwrap();
        let done = math
            .score(&[("She sold 72 in all.", &record), ("71", &record)])
            .unwrap();
        assert_eq!(done, vec![1.0, 0.0]);
        let regex = config("regex").build().unwrap();
        assert_eq!(
            regex
                .score(&[("<think>a", &record), ("a", &record)])
                .unwrap(),
            vec![1.0, 0.0]
        );
        let exact = RewardConfig {
            field: "x".into(),
            ..config("exact")
        };
        let r = serde_json::json!({"x": " yes "});
        assert_eq!(
            exact
                .build()
                .unwrap()
                .score(&[("yes", &r), ("no", &r)])
                .unwrap(),
            vec![1.0, 0.0]
        );
        assert!(config("nope").build().is_err());
    }

    #[test]
    fn a_command_scores_each_line() {
        // `wc -c` would count; a shell loop gives one number per line.
        let c = RewardConfig {
            kind: "command".into(),
            weight: 2.0,
            field: "answer".into(),
            pattern: None,
            command: vec![
                "sh".into(),
                "-c".into(),
                "while read l; do echo ${#l}; done".into(),
            ],
        };
        let rewards = Rewards::new(&[c]).unwrap();
        let record = serde_json::json!({});
        let (total, means) = rewards.score(&[("a", &record), ("bb", &record)]).unwrap();
        // The lengths of the JSON lines, doubled.
        let line = |t: &str| {
            serde_json::json!({"completion": t, "record": record})
                .to_string()
                .len() as f64
        };
        assert_eq!(total, vec![2.0 * line("a"), 2.0 * line("bb")]);
        assert_eq!(means.len(), 1);
    }
}
