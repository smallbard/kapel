//! Tokenizers read from `tokenizer.json`, the format of the `tokenizers` library.
//!
//! Encoding follows the same pipeline: added tokens (special or not) are cut out
//! of the text first; each remaining segment is normalized, split into pieces by
//! the pre-tokenizer, and each piece encoded by BPE; the post-processor then adds
//! the special tokens of the template, such as a beginning-of-sequence token.
//!
//! Supported: BPE models (GPT-2 style byte-level, Llama 3, Qwen, and
//! SentencePiece-style with byte fallback: Llama 2, Mistral, Gemma), the
//! normalizers, pre-tokenizers, post-processors and decoders they use.
//!
//! ```no_run
//! let tokenizer = kapel_tokenizers::Tokenizer::from_file("tokenizer.json")?;
//! let ids = tokenizer.encode("Hello world", true)?;
//! let text = tokenizer.decode(&ids, true);
//! # Ok::<(), kapel_tokenizers::TokenizerError>(())
//! ```

mod bpe;
mod chat;
mod decode;
mod text;

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use serde_json::Value;

pub use crate::chat::{ChatTemplate, Message};

use crate::bpe::Bpe;
use crate::decode::Decoder;
use crate::text::{Normalizer, Piece, PreTokenizer};

/// What can go wrong with a tokenizer.
#[derive(Debug)]
#[non_exhaustive]
pub enum TokenizerError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not a tokenizer Kapel can run; the message says why.
    Format(String),
    /// A chat template is not valid, or refused to render the conversation.
    Template(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tokenizer.json: {e}"),
            Self::Format(message) => write!(f, "tokenizer.json: {message}"),
            Self::Template(message) => write!(f, "chat template: {message}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub(crate) fn bad(message: impl Into<String>) -> TokenizerError {
    TokenizerError::Format(message.into())
}

pub(crate) fn str_field<'a>(json: &'a Value, key: &str) -> Result<&'a str, TokenizerError> {
    json.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("missing string field {key:?}")))
}

/// A token added to the vocabulary, cut out of the text before anything else.
#[derive(Debug)]
struct AddedToken {
    content: String,
    id: u32,
    special: bool,
    lstrip: bool,
    rstrip: bool,
}

/// An element of a template.
#[derive(Debug)]
enum Item {
    Special(Vec<u32>),
    Sequence,
}

/// A tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Bpe,
    /// The template of a single sequence, if the post-processor has one.
    template: Option<Vec<Item>>,
    decoder: Option<Decoder>,
    /// Added tokens, longest first, for leftmost-longest matching.
    added: Vec<AddedToken>,
    id_to_token: HashMap<u32, String>,
    special_ids: Vec<u32>,
}

/// The template of the single-sequence case in a post-processor, if any.
fn template(json: &Value) -> Result<Option<Vec<Item>>, TokenizerError> {
    match json.get("type").and_then(Value::as_str) {
        Some("TemplateProcessing") => {
            let specials = json.get("special_tokens").and_then(Value::as_object);
            let items = json
                .get("single")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("TemplateProcessing without single"))?
                .iter()
                .map(|item| {
                    if item.get("Sequence").is_some() {
                        return Ok(Item::Sequence);
                    }
                    let name = item
                        .get("SpecialToken")
                        .and_then(|s| s.get("id"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            bad("a template item is neither Sequence nor SpecialToken")
                        })?;
                    let ids = specials
                        .and_then(|s| s.get(name))
                        .and_then(|s| s.get("ids"))
                        .and_then(Value::as_array)
                        .ok_or_else(|| bad(format!("the template's token {name:?} has no ids")))?
                        .iter()
                        .filter_map(|v| v.as_u64().map(|v| v as u32))
                        .collect();
                    Ok(Item::Special(ids))
                })
                .collect::<Result<_, _>>()?;
            Ok(Some(items))
        }
        Some("Sequence") => {
            let mut found = None;
            for processor in json
                .get("processors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(t) = template(processor)? {
                    found = Some(t);
                }
            }
            Ok(found)
        }
        Some("ByteLevel") | None => Ok(None),
        Some(other) => Err(bad(format!("unsupported post-processor {other:?}"))),
    }
}

impl Tokenizer {
    /// Reads a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, or describes a tokenizer Kapel cannot run.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(path).map_err(TokenizerError::Io)?;
        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| bad(format!("not valid JSON: {e}")))?;
        Self::from_json(&json)
    }

    /// Builds a tokenizer from the parsed contents of `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// If it describes a tokenizer Kapel cannot run.
    pub fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        let part = |key: &str| json.get(key).filter(|v| !v.is_null());
        let model = Bpe::from_json(part("model").ok_or_else(|| bad("no model"))?)?;
        let mut added: Vec<AddedToken> = part("added_tokens")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|t| {
                let flag = |k: &str| t.get(k).and_then(Value::as_bool).unwrap_or(false);
                let id = t
                    .get("id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| bad("an added token has no id"))?;
                Ok(AddedToken {
                    content: str_field(t, "content")?.to_string(),
                    id: id as u32,
                    special: flag("special"),
                    lstrip: flag("lstrip"),
                    rstrip: flag("rstrip"),
                })
            })
            .collect::<Result<_, TokenizerError>>()?;
        added.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        let mut id_to_token: HashMap<u32, String> =
            model.vocab.iter().map(|(t, &id)| (id, t.clone())).collect();
        for token in &added {
            id_to_token.insert(token.id, token.content.clone());
        }
        let special_ids = added.iter().filter(|t| t.special).map(|t| t.id).collect();
        Ok(Self {
            normalizer: part("normalizer").map(Normalizer::from_json).transpose()?,
            pre_tokenizer: part("pre_tokenizer")
                .map(PreTokenizer::from_json)
                .transpose()?,
            template: part("post_processor").map(template).transpose()?.flatten(),
            decoder: part("decoder").map(Decoder::from_json).transpose()?,
            model,
            added,
            id_to_token,
            special_ids,
        })
    }

    /// The number of ids, added tokens included.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.keys().max().map_or(0, |&m| m as usize + 1)
    }

    /// The id of a token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|t| t.content == token)
            .map(|t| t.id)
            .or_else(|| self.model.vocab.get(token).copied())
    }

    /// The token of an id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// The text cut at the added tokens: segments of plain text with their byte
    /// offset in `text`, and the ids of the added tokens between them.
    fn split_added<'t>(&self, text: &'t str) -> Vec<Result<(usize, &'t str), u32>> {
        let mut out = Vec::new();
        let (mut start, mut at) = (0, 0);
        while at < text.len() {
            let found = self
                .added
                .iter()
                .find(|t| !t.content.is_empty() && text[at..].starts_with(&t.content));
            match found {
                Some(token) => {
                    let mut end = at;
                    if token.lstrip {
                        end = start + text[start..at].trim_end().len();
                    }
                    if end > start {
                        out.push(Ok((start, &text[start..end])));
                    }
                    out.push(Err(token.id));
                    at += token.content.len();
                    if token.rstrip {
                        at += text[at..].len() - text[at..].trim_start().len();
                    }
                    start = at;
                }
                None => at += text[at..].chars().next().map_or(1, char::len_utf8),
            }
        }
        if start < text.len() {
            out.push(Ok((start, &text[start..])));
        }
        out
    }

    /// The ids of `text`; with `add_special_tokens`, those of the
    /// post-processor's template too.
    ///
    /// # Errors
    ///
    /// If a regex of the tokenizer gives up on the text (see the backtracking
    /// limit of the regex engine).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        for part in self.split_added(text) {
            match part {
                Err(id) => ids.push(id),
                Ok((offset, segment)) => {
                    let normalized = match &self.normalizer {
                        Some(n) => n.apply(segment)?,
                        None => segment.to_string(),
                    };
                    let pieces = vec![Piece {
                        text: normalized,
                        first: offset == 0,
                    }];
                    let pieces = match &self.pre_tokenizer {
                        Some(p) => p.apply(pieces)?,
                        None => pieces,
                    };
                    for piece in pieces {
                        ids.extend(self.model.encode(&piece.text));
                    }
                }
            }
        }
        Ok(match (&self.template, add_special_tokens) {
            (Some(template), true) => template
                .iter()
                .flat_map(|item| match item {
                    Item::Special(special) => special.clone(),
                    Item::Sequence => ids.clone(),
                })
                .collect(),
            _ => ids,
        })
    }

    /// The text of `ids`; with `skip_special_tokens`, special tokens are left out.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter(|id| !(skip_special_tokens && self.special_ids.contains(id)))
            .filter_map(|id| self.id_to_token.get(id).cloned())
            .collect();
        match &self.decoder {
            Some(decoder) => decoder.apply(tokens).concat(),
            None => tokens.join(" "),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// A long piece without spaces: BPE must stay near-linear on it.
    #[test]
    fn a_long_word_encodes_quickly() {
        let mut vocab = serde_json::Map::new();
        let mut merges = Vec::new();
        let mut token = "a".to_string();
        vocab.insert(token.clone(), Value::from(0));
        for id in 1..12 {
            merges.push(Value::from(format!("{token} {token}")));
            token = token.repeat(2);
            vocab.insert(token.clone(), Value::from(id));
        }
        vocab.insert("b".into(), Value::from(12));
        let json = serde_json::json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": merges },
        });
        let tokenizer = Tokenizer::from_json(&json).unwrap();
        let text = "ab".repeat(50_000) + &"a".repeat(100_000);
        let start = std::time::Instant::now();
        let ids = tokenizer.encode(&text, false).unwrap();
        assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
        // 100 000 `a` make 48 tokens of 2048 and one of 1696 = 1024 + 512 + 128 + 32.
        assert_eq!(ids.len(), 100_000 + 48 + 4);
    }

    /// Every `tokenizer_cases.json` written by
    /// `tools/parity/tokenizers_reference.py`. Without them (in continuous
    /// integration), the test checks nothing.
    #[test]
    fn encodings_match_the_tokenizers_library() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
        let Ok(dirs) = std::fs::read_dir(&root) else {
            return;
        };
        let mut failures = Vec::new();
        let mut checked = 0;
        for dir in dirs.flatten() {
            let cases = dir.path().join("tokenizer_cases.json");
            let Ok(text) = std::fs::read_to_string(&cases) else {
                continue;
            };
            let name = dir.file_name().to_string_lossy().into_owned();
            let tokenizer = match Tokenizer::from_file(dir.path().join("tokenizer.json")) {
                Ok(t) => t,
                Err(e) => {
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let cases: Value = serde_json::from_str(&text).unwrap();
            for case in cases.as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let ids = |key: &str| -> Vec<u32> {
                    case[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect()
                };
                checked += 1;
                let got = tokenizer.encode(text, true).unwrap();
                if got != ids("ids") {
                    failures.push(format!(
                        "{name} {text:?}: {got:?}, expected {:?}",
                        ids("ids")
                    ));
                }
                let decoded = tokenizer.decode(&ids("ids_no_special"), false);
                if decoded != case["decoded"].as_str().unwrap() {
                    failures.push(format!(
                        "{name} decode {text:?}: {decoded:?}, expected {:?}",
                        case["decoded"]
                    ));
                }
            }
        }
        eprintln!("{checked} texts checked");
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
