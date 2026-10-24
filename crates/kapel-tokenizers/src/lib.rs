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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use aho_corasick::{AhoCorasick, Input, MatchKind};
use serde_json::Value;

pub use crate::chat::{ChatTemplate, FUEL, MAX_OUTPUT, Message};

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

/// A token id of `tokenizer.json`, which must fit in a `u32`.
pub(crate) fn id_of(value: &Value, what: &str) -> Result<u32, TokenizerError> {
    let id = value
        .as_u64()
        .ok_or_else(|| bad(format!("{what} is not an integer")))?;
    u32::try_from(id).map_err(|_| bad(format!("{what} {id} does not fit in 32 bits")))
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
    /// Added tokens, longest first; among equal contents, the first listed wins.
    added: Vec<AddedToken>,
    /// Finds the added tokens in a text, leftmost first and longest among those
    /// that start at the same place: one pattern per distinct non-empty content.
    matcher: Option<AhoCorasick>,
    /// For each pattern of `matcher`, its token in `added`.
    pattern_token: Vec<usize>,
    /// The id of each added token's content.
    added_ids: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_ids: HashSet<u32>,
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
                        .map(|v| id_of(v, &format!("an id of the template's token {name:?}")))
                        .collect::<Result<_, _>>()?;
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
                let id = id_of(
                    t.get("id").ok_or_else(|| bad("an added token has no id"))?,
                    "the id of an added token",
                )?;
                Ok(AddedToken {
                    content: str_field(t, "content")?.to_string(),
                    id,
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
        let mut added_ids = HashMap::new();
        let mut pattern_token = Vec::new();
        for (i, token) in added.iter().enumerate() {
            if !added_ids.contains_key(&token.content) {
                added_ids.insert(token.content.clone(), token.id);
                if !token.content.is_empty() {
                    pattern_token.push(i);
                }
            }
        }
        let matcher = if pattern_token.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(pattern_token.iter().map(|&i| &added[i].content))
                    .map_err(|e| bad(format!("the added tokens: {e}")))?,
            )
        };
        Ok(Self {
            normalizer: part("normalizer").map(Normalizer::from_json).transpose()?,
            pre_tokenizer: part("pre_tokenizer")
                .map(PreTokenizer::from_json)
                .transpose()?,
            template: part("post_processor").map(template).transpose()?.flatten(),
            decoder: part("decoder").map(Decoder::from_json).transpose()?,
            model,
            added,
            matcher,
            pattern_token,
            added_ids,
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
        self.added_ids
            .get(token)
            .or_else(|| self.model.vocab.get(token))
            .copied()
    }

    /// The token of an id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// The bytes of an id in the middle of a text: what decoding it adds
    /// there. A token of a byte-level vocabulary may be part of a character,
    /// so this is bytes, not text. A special token gives its content.
    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let token = self.id_to_token.get(&id)?;
        if self.special_ids.contains(&id) || self.decoder.is_none() {
            return Some(token.clone().into_bytes());
        }
        self.decoder.as_ref().map(|d| d.token_bytes(token))
    }

    /// Whether an id is a special token (an added token marked special).
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// The text cut at the added tokens: segments of plain text with their byte
    /// offset in `text`, and the ids of the added tokens between them.
    fn split_added<'t>(&self, text: &'t str) -> Vec<Result<(usize, &'t str), u32>> {
        let mut out = Vec::new();
        let (mut start, mut at) = (0, 0);
        if let Some(matcher) = &self.matcher {
            while at < text.len() {
                let Some(found) = matcher.find(Input::new(text).span(at..text.len())) else {
                    break;
                };
                let token = &self.added[self.pattern_token[found.pattern().as_usize()]];
                let mut end = found.start();
                if token.lstrip {
                    end = start + text[start..end].trim_end().len();
                }
                if end > start {
                    out.push(Ok((start, &text[start..end])));
                }
                out.push(Err(token.id));
                at = found.end();
                if token.rstrip {
                    at += text[at..].len() - text[at..].trim_start().len();
                }
                start = at;
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
                Ok((offset, segment)) => self.encode_plain(offset, segment, &mut ids)?,
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

    /// The ids of `text` taken as plain text: the added tokens it spells,
    /// such as `<|endoftext|>`, are encoded as their characters, never as
    /// the tokens themselves, and no template is applied. A corpus that
    /// quotes a special token thus cannot inject it (`encode_ordinary` of
    /// tiktoken).
    ///
    /// # Errors
    ///
    /// As [`Tokenizer::encode`].
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        if !text.is_empty() {
            self.encode_plain(0, text, &mut ids)?;
        }
        Ok(ids)
    }

    /// Appends the ids of `segment`, plain text at byte `offset` of its text.
    fn encode_plain(
        &self,
        offset: usize,
        segment: &str,
        ids: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
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
        Ok(())
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

/// Reports a parity test that cannot run because its reference files (written
/// by `tools/parity`) are missing; with `KAPEL_REQUIRE_PARITY=1`, fails instead.
#[cfg(test)]
pub(crate) fn parity_skipped(what: &str) {
    use std::io::Write;
    assert!(
        std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
        "{what} is missing, and KAPEL_REQUIRE_PARITY=1 (see tools/parity/README.md)"
    );
    // Straight to stderr, past the capture of the test harness: a skip is seen.
    let _ = writeln!(
        std::io::stderr(),
        "skipped: {what} missing (see tools/parity/README.md)"
    );
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// A tokenizer whose vocabulary is the characters `a` to `e` and space
    /// (ids 0 to 5), with `added` tokens.
    fn with_added(added: serde_json::Value) -> Result<Tokenizer, TokenizerError> {
        Tokenizer::from_json(&serde_json::json!({
            "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, " ": 5 }, "merges": [] },
            "added_tokens": added,
        }))
    }

    /// The added tokens are found leftmost first, longest among those that
    /// start at the same place, as `tokenizers` does.
    #[test]
    fn added_tokens_match_leftmost_longest() {
        let tokenizer = with_added(serde_json::json!([
            { "id": 10, "content": "ab" },
            { "id": 11, "content": "abc" },
            { "id": 12, "content": "bcd" },
            { "id": 13, "content": "abc" },
            { "id": 14, "content": "" },
            { "id": 15, "content": "é" },
            { "id": 16, "content": "d", "lstrip": true, "rstrip": true },
        ]))
        .unwrap();
        let encode = |text: &str| tokenizer.encode(text, false).unwrap();
        // "abc" beats "ab" at 0; "bcd" starts later, after it.
        assert_eq!(encode("abcd"), [11, 16]);
        assert_eq!(encode("abd"), [10, 16]);
        assert_eq!(encode("cbcda"), [2, 12, 0]);
        assert_eq!(encode("aébé"), [0, 15, 1, 15]);
        // `lstrip` and `rstrip` take the spaces around the token.
        assert_eq!(encode("a  d  e"), [0, 16, 4]);
        // The first of two equal contents wins, as in `token_to_id`.
        assert_eq!(tokenizer.token_to_id("abc"), Some(11));
        assert_eq!(tokenizer.token_to_id("e"), Some(4));
        assert_eq!(encode(""), Vec::<u32>::new());
    }

    /// `encode_ordinary` reads added tokens as text; a model without a type
    /// but with merges is a BPE, as in GPT-2's file on the Hub.
    #[test]
    fn ordinary_text_spells_the_added_tokens() {
        let tokenizer = Tokenizer::from_json(&serde_json::json!({
            "model": { "vocab": { "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, " ": 5 }, "merges": [] },
            "added_tokens": [{ "id": 10, "content": "ab" }],
        }))
        .unwrap();
        assert_eq!(tokenizer.encode("cab", false).unwrap(), [2, 10]);
        assert_eq!(tokenizer.encode_ordinary("cab").unwrap(), [2, 0, 1]);
        assert_eq!(tokenizer.encode_ordinary("").unwrap(), Vec::<u32>::new());
        let unknown = serde_json::json!({ "model": { "type": "WordPiece", "vocab": {} } });
        assert!(Tokenizer::from_json(&unknown).is_err());
    }

    /// The search by automaton cuts texts exactly as a scan of every added
    /// token at every character (Kapel's first implementation) does.
    #[test]
    fn added_tokens_are_cut_as_by_a_scan() {
        let tokenizer = with_added(serde_json::json!([
            { "id": 10, "content": "ab" },
            { "id": 11, "content": "abc" },
            { "id": 12, "content": "bcd", "rstrip": true },
            { "id": 13, "content": "b", "lstrip": true },
            { "id": 14, "content": "é a" },
            { "id": 15, "content": " " , "lstrip": true },
            { "id": 16, "content": "ddd" },
            { "id": 17, "content": "dd" },
        ]))
        .unwrap();
        fn scan<'t>(tokenizer: &Tokenizer, text: &'t str) -> Vec<Result<(usize, &'t str), u32>> {
            let mut out = Vec::new();
            let (mut start, mut at) = (0, 0);
            while at < text.len() {
                let found = tokenizer
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
        let alphabet = ['a', 'b', 'c', 'd', ' ', 'é'];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20_000 {
            let mut text = String::new();
            for _ in 0..(state % 24) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                text.push(alphabet[(state % alphabet.len() as u64) as usize]);
            }
            state = state.wrapping_add(1);
            assert_eq!(
                tokenizer.split_added(&text),
                scan(&tokenizer, &text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn ids_beyond_32_bits_are_errors() {
        let error = with_added(serde_json::json!([{ "id": 1u64 << 32, "content": "x" }]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("32 bits"), "{error}");
        let error = Tokenizer::from_json(&serde_json::json!({
            "model": { "type": "BPE", "vocab": { "a": 4_294_967_296u64 }, "merges": [] },
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("32 bits"), "{error}");
        let error = Tokenizer::from_json(&serde_json::json!({
            "model": { "type": "BPE", "vocab": { "a": 0 }, "merges": [] },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{ "SpecialToken": { "id": "<s>" } }, { "Sequence": { "id": "A" } }],
                "special_tokens": { "<s>": { "ids": [u64::MAX] } },
            },
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("32 bits"), "{error}");
    }

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
    /// integration), the test is skipped, or fails with `KAPEL_REQUIRE_PARITY=1`.
    #[test]
    fn encodings_match_the_tokenizers_library() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
        let Ok(dirs) = std::fs::read_dir(&root) else {
            parity_skipped("data/parity");
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
                // The bytes of the tokens, one after the other, are the
                // decoded text, but for the space taken off its start.
                let bytes: Vec<u8> = ids("ids_no_special")
                    .iter()
                    .flat_map(|&id| tokenizer.token_bytes(id).unwrap())
                    .collect();
                let whole = tokenizer.decode(&ids("ids_no_special"), false);
                if bytes != whole.as_bytes() && bytes.strip_prefix(b" ") != Some(whole.as_bytes()) {
                    failures.push(format!(
                        "{name} bytes {text:?}: {:?}, expected {whole:?}",
                        String::from_utf8_lossy(&bytes)
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
        if checked == 0 {
            parity_skipped("data/parity/*/tokenizer_cases.json");
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
