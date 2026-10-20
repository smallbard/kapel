//! Chat templates: the Jinja templates that turn a conversation into the text a
//! model was trained on.
//!
//! The template comes from `chat_template.jinja` or from the `chat_template`
//! field of `tokenizer_config.json`. It is rendered as `transformers` renders it:
//! blocks trimmed (`trim_blocks`, `lstrip_blocks`), Python string methods
//! available, `raise_exception` and `strftime_now` defined, and `tojson` writing
//! JSON as Python's `json.dumps` does.

use std::collections::BTreeMap;
use std::path::Path;

use minijinja::{Environment, Error as JinjaError, ErrorKind, Value as JinjaValue};
use serde_json::Value;

use crate::{TokenizerError, bad};

/// The most instructions one rendering may run.
const FUEL: u64 = 10_000_000;

/// A message of a conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// `"system"`, `"user"`, `"assistant"`…
    pub role: String,
    /// The text of the message.
    pub content: String,
}

impl Message {
    /// A message from `role`.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A chat template and the special tokens it may refer to.
#[derive(Clone, Debug)]
pub struct ChatTemplate {
    source: String,
    /// `bos_token`, `eos_token`… as `tokenizer_config.json` gives them.
    tokens: BTreeMap<String, String>,
}

/// A special token of `tokenizer_config.json`: a string, or an object with a
/// `content`.
fn token_text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string).or_else(|| {
        value
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

impl ChatTemplate {
    /// A template from its source and the special tokens it may use.
    pub fn new(source: impl Into<String>, tokens: BTreeMap<String, String>) -> Self {
        Self {
            source: source.into(),
            tokens,
        }
    }

    /// The template of a model directory, if it has one.
    ///
    /// # Errors
    ///
    /// If `tokenizer_config.json` cannot be read or parsed.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Option<Self>, TokenizerError> {
        let dir = dir.as_ref();
        let config: Value = match std::fs::read(dir.join("tokenizer_config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| bad(format!("tokenizer_config.json: {e}")))?,
            Err(_) => Value::Null,
        };
        let tokens = ["bos_token", "eos_token", "unk_token", "pad_token"]
            .into_iter()
            .filter_map(|k| {
                config
                    .get(k)
                    .and_then(token_text)
                    .map(|t| (k.to_string(), t))
            })
            .collect();
        if let Ok(source) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
            return Ok(Some(Self::new(source, tokens)));
        }
        let source = match config.get("chat_template") {
            Some(Value::String(s)) => Some(s.clone()),
            // Several named templates: the default one.
            Some(Value::Array(all)) => all
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some("default"))
                .and_then(|t| t.get("template").and_then(Value::as_str))
                .map(str::to_string),
            _ => None,
        };
        Ok(source.map(|s| Self::new(s, tokens)))
    }

    /// The text of `messages`, with the start of the assistant's answer if
    /// `add_generation_prompt`.
    ///
    /// # Errors
    ///
    /// If the template is not valid, or rejects the conversation (many raise an
    /// exception for a role they do not support).
    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // A bound on the work of one rendering: a template from an unknown model
        // cannot loop forever. Real templates use a few thousand units.
        env.set_fuel(Some(FUEL));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<JinjaValue, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_function("strftime_now", |format: String| strftime_now(&format));
        env.add_filter("tojson", tojson);
        let source = without_generation_tags(&self.source);
        env.add_template("chat", &source)
            .map_err(|e| TokenizerError::Template(e.to_string()))?;
        let messages: Vec<JinjaValue> = messages
            .iter()
            .map(|m| {
                JinjaValue::from_serialize(BTreeMap::from([
                    ("role", &m.role),
                    ("content", &m.content),
                ]))
            })
            .collect();
        let mut context: BTreeMap<String, JinjaValue> = self
            .tokens
            .iter()
            .map(|(k, v)| (k.clone(), JinjaValue::from(v.as_str())))
            .collect();
        context.insert("messages".into(), JinjaValue::from(messages));
        context.insert(
            "add_generation_prompt".into(),
            JinjaValue::from(add_generation_prompt),
        );
        env.get_template("chat")
            .and_then(|t| t.render(context))
            .map_err(|e| TokenizerError::Template(e.to_string()))
    }
}

/// The `{% generation %}…{% endgeneration %}` tags of `transformers` mark the
/// assistant's text for training masks and print nothing themselves. They become
/// `{% if true %}…{% endif %}`, which keeps their whitespace control (`-`).
fn without_generation_tags(source: &str) -> String {
    let tag =
        fancy_regex::Regex::new(r"\{%(-?)\s*(end)?generation\s*(-?)%\}").expect("a valid regex");
    tag.replace_all(source, |c: &fancy_regex::Captures<'_>| {
        let keyword = if c.get(2).is_some() {
            "endif"
        } else {
            "if true"
        };
        format!("{{%{} {keyword} {}%}}", &c[1], &c[3])
    })
    .into_owned()
}

/// `tojson` as `transformers` defines it: `json.dumps(value, ensure_ascii=False)`,
/// with `", "` and `": "` as separators, and an optional `indent`.
fn tojson(value: JinjaValue, kwargs: minijinja::value::Kwargs) -> Result<String, JinjaError> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let json = serde_json::to_value(&value)
        .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    write_json(&json, indent, 0, &mut out);
    Ok(out)
}

fn write_json(value: &Value, indent: Option<usize>, depth: usize, out: &mut String) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    };
    let separator = if indent.is_some() { "," } else { ", " };
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(separator);
                }
                newline(out, depth + 1);
                write_json(item, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(separator);
                }
                newline(out, depth + 1);
                out.push_str(&serde_json::to_string(key).expect("a string serializes"));
                out.push_str(": ");
                write_json(item, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
        other => out.push_str(&serde_json::to_string(other).expect("a JSON value serializes")),
    }
}

/// The current local date and time, formatted with the common `strftime`
/// directives (`%d %m %Y %y %b %B %a %A %H %M %S %j %%`). Templates use it to
/// write today's date in the system prompt.
fn strftime_now(format: &str) -> Result<String, JinjaError> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?
        .as_secs() as i64;
    Ok(format_time(seconds, format))
}

/// Formats a Unix time (UTC) with `strftime` directives.
fn format_time(seconds: i64, format: &str) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    let days = seconds.div_euclid(86_400);
    let time = seconds.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let cumulative = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let day_of_year = cumulative[(month - 1) as usize] + day + i64::from(leap && month > 2);
    let weekday = (days + 3).rem_euclid(7) as usize; // 1970-01-01 was a Thursday
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('e') => out.push_str(&format!("{day:2}")),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('Y') => out.push_str(&year.to_string()),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('b') => out.push_str(&MONTHS[(month - 1) as usize][..3]),
            Some('B') => out.push_str(MONTHS[(month - 1) as usize]),
            Some('a') => out.push_str(&DAYS[weekday][..3]),
            Some('A') => out.push_str(DAYS[weekday]),
            Some('H') => out.push_str(&format!("{:02}", time / 3600)),
            Some('M') => out.push_str(&format!("{:02}", time % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", time % 60)),
            Some('j') => out.push_str(&format!("{day_of_year:03}")),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn dates_are_formatted_as_strftime_does() {
        // 2024-02-29 13:05:09 UTC, a Thursday.
        assert_eq!(
            format_time(1_709_211_909, "%d %b %Y %H:%M:%S %a %j %y %B"),
            "29 Feb 2024 13:05:09 Thu 060 24 February"
        );
        assert_eq!(format_time(0, "%A %d %m %Y"), "Thursday 01 01 1970");
    }

    #[test]
    fn a_template_that_never_ends_is_stopped() {
        let template = ChatTemplate::new(
            "{% for i in range(10000) %}{% for j in range(10000) %}{% for k in range(10000) %}x{% endfor %}{% endfor %}{% endfor %}",
            BTreeMap::new(),
        );
        let error = template.render(&[], false).unwrap_err().to_string();
        assert!(error.contains("fuel"), "{error}");
    }

    #[test]
    fn tojson_writes_like_python() {
        let value: Value = serde_json::json!({"a": [1, "é"], "b": {}});
        let mut out = String::new();
        write_json(&value, None, 0, &mut out);
        assert_eq!(out, r#"{"a": [1, "é"], "b": {}}"#);
    }

    /// Every `chat_cases.json` written by `tools/parity/chat_reference.py`;
    /// without them (in continuous integration), the test checks nothing.
    #[test]
    fn renderings_match_transformers() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
        let Ok(dirs) = std::fs::read_dir(&root) else {
            return;
        };
        let mut failures = Vec::new();
        let mut checked = 0;
        for dir in dirs.flatten() {
            let Ok(text) = std::fs::read_to_string(dir.path().join("chat_cases.json")) else {
                continue;
            };
            let name = dir.file_name().to_string_lossy().into_owned();
            let template = ChatTemplate::from_dir(dir.path())
                .unwrap()
                .expect("a template");
            let cases: Value = serde_json::from_str(&text).unwrap();
            for case in cases.as_array().unwrap() {
                checked += 1;
                let messages: Vec<Message> = case["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| {
                        Message::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap())
                    })
                    .collect();
                let generation = case["add_generation_prompt"].as_bool().unwrap();
                let got = template.render(&messages, generation);
                match (case["text"].as_str(), got) {
                    (Some(want), Ok(got)) if got == want => {}
                    (None, Err(_)) => {}
                    (want, got) => failures.push(format!(
                        "{name} (generation {generation}): {got:?}\nexpected {want:?}"
                    )),
                }
            }
        }
        eprintln!("{checked} conversations checked");
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}
