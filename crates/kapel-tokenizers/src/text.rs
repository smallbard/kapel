//! Normalizers and pre-tokenizers: from a text to the pieces BPE encodes.

use std::sync::OnceLock;

use fancy_regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::{TokenizerError, bad, str_field};

/// The pattern of a `Split` or `Replace`: a literal string or a regex.
#[derive(Debug)]
pub(crate) enum Pattern {
    Literal(String),
    Regex(Regex),
}

impl Pattern {
    pub(crate) fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        if let Some(s) = json.get("String").and_then(Value::as_str) {
            return Ok(Self::Literal(s.to_string()));
        }
        if let Some(r) = json.get("Regex").and_then(Value::as_str) {
            return Regex::new(r)
                .map(Self::Regex)
                .map_err(|e| bad(format!("regex {r:?}: {e}")));
        }
        Err(bad("a pattern is neither String nor Regex"))
    }

    /// The byte ranges of the matches in `text`.
    ///
    /// # Errors
    ///
    /// If the regex gives up: a pattern with backtracking can exceed the limit
    /// of the regex engine on a hostile text. The text is then rejected rather
    /// than cut wrongly.
    pub(crate) fn matches(&self, text: &str) -> Result<Vec<(usize, usize)>, TokenizerError> {
        Ok(match self {
            Self::Literal(s) if s.is_empty() => Vec::new(),
            Self::Literal(s) => text
                .match_indices(s.as_str())
                .map(|(i, m)| (i, i + m.len()))
                .collect(),
            Self::Regex(r) => {
                let mut ranges = Vec::new();
                for m in r.find_iter(text) {
                    let m = m.map_err(|e| bad(format!("the regex gave up on this text: {e}")))?;
                    if m.start() < m.end() {
                        ranges.push((m.start(), m.end()));
                    }
                }
                ranges
            }
        })
    }
}

/// A normalizer of `tokenizer.json`.
#[derive(Debug)]
pub(crate) enum Normalizer {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Lowercase,
    Prepend(String),
    Replace(Pattern, String),
    Strip { left: bool, right: bool },
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    pub(crate) fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        Ok(match str_field(json, "type")? {
            "NFC" => Self::Nfc,
            "NFD" => Self::Nfd,
            "NFKC" => Self::Nfkc,
            "NFKD" => Self::Nfkd,
            "Lowercase" => Self::Lowercase,
            "Prepend" => Self::Prepend(str_field(json, "prepend")?.to_string()),
            "Replace" => Self::Replace(
                Pattern::from_json(
                    json.get("pattern")
                        .ok_or_else(|| bad("Replace without a pattern"))?,
                )?,
                str_field(json, "content")?.to_string(),
            ),
            "Strip" => Self::Strip {
                left: json
                    .get("strip_left")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                right: json
                    .get("strip_right")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            },
            "Sequence" => Self::Sequence(
                json.get("normalizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| bad("Sequence without normalizers"))?
                    .iter()
                    .map(Self::from_json)
                    .collect::<Result<_, _>>()?,
            ),
            other => return Err(bad(format!("unsupported normalizer {other:?}"))),
        })
    }

    pub(crate) fn apply(&self, text: &str) -> Result<String, TokenizerError> {
        Ok(match self {
            Self::Nfc => text.nfc().collect(),
            Self::Nfd => text.nfd().collect(),
            Self::Nfkc => text.nfkc().collect(),
            Self::Nfkd => text.nfkd().collect(),
            Self::Lowercase => text.to_lowercase(),
            Self::Prepend(_) if text.is_empty() => String::new(),
            Self::Prepend(p) => format!("{p}{text}"),
            Self::Replace(pattern, content) => replace(pattern, text, content)?,
            Self::Strip { left, right } => {
                let t = if *left { text.trim_start() } else { text };
                (if *right { t.trim_end() } else { t }).to_string()
            }
            Self::Sequence(all) => {
                let mut text = text.to_string();
                for normalizer in all {
                    text = normalizer.apply(&text)?;
                }
                text
            }
        })
    }
}

pub(crate) fn replace(
    pattern: &Pattern,
    text: &str,
    content: &str,
) -> Result<String, TokenizerError> {
    let mut out = String::new();
    let mut last = 0;
    for (a, b) in pattern.matches(text)? {
        out.push_str(&text[last..a]);
        out.push_str(content);
        last = b;
    }
    out.push_str(&text[last..]);
    Ok(out)
}

/// How a split treats its delimiters.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Behavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// A pre-tokenizer of `tokenizer.json`.
#[derive(Debug)]
pub(crate) enum PreTokenizer {
    ByteLevel {
        add_prefix_space: bool,
        /// GPT-2's regex, compiled once, unless `use_regex` is false.
        regex: Option<Pattern>,
    },
    Split {
        pattern: Pattern,
        behavior: Behavior,
        invert: bool,
    },
    Metaspace {
        replacement: char,
        prepend: Prepend,
        split: bool,
    },
    /// Digits, one by one or in runs: the regex is compiled once.
    Digits(Pattern),
    Sequence(Vec<PreTokenizer>),
}

/// When Metaspace adds its replacement in front of a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prepend {
    Always,
    First,
    Never,
}

/// The regex of GPT-2's byte-level pre-tokenizer.
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// A piece of text, with whether it starts the original text (for Metaspace's
/// `first`).
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub(crate) text: String,
    pub(crate) first: bool,
}

impl PreTokenizer {
    pub(crate) fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        let flag =
            |key: &str, default: bool| json.get(key).and_then(Value::as_bool).unwrap_or(default);
        Ok(match str_field(json, "type")? {
            "ByteLevel" => Self::ByteLevel {
                add_prefix_space: flag("add_prefix_space", false),
                regex: flag("use_regex", true)
                    .then(|| Pattern::Regex(Regex::new(GPT2).expect("the GPT-2 regex is valid"))),
            },
            "Split" => Self::Split {
                pattern: Pattern::from_json(
                    json.get("pattern")
                        .ok_or_else(|| bad("Split without a pattern"))?,
                )?,
                behavior: match str_field(json, "behavior")? {
                    "Removed" => Behavior::Removed,
                    "Isolated" => Behavior::Isolated,
                    "MergedWithPrevious" => Behavior::MergedWithPrevious,
                    "MergedWithNext" => Behavior::MergedWithNext,
                    "Contiguous" => Behavior::Contiguous,
                    other => return Err(bad(format!("unknown split behavior {other:?}"))),
                },
                invert: flag("invert", false),
            },
            "Metaspace" => Self::Metaspace {
                replacement: str_field(json, "replacement")?
                    .chars()
                    .next()
                    .unwrap_or('▁'),
                prepend: match json.get("prepend_scheme").and_then(Value::as_str) {
                    Some("first") => Prepend::First,
                    Some("never") => Prepend::Never,
                    Some("always") => Prepend::Always,
                    _ if flag("add_prefix_space", true) => Prepend::Always,
                    _ => Prepend::Never,
                },
                split: flag("split", true),
            },
            "Digits" => Self::Digits(Pattern::Regex(
                Regex::new(if flag("individual_digits", false) {
                    r"\p{N}"
                } else {
                    r"\p{N}+"
                })
                .expect("the digit regexes are valid"),
            )),
            "Sequence" => Self::Sequence(
                json.get("pretokenizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| bad("Sequence without pretokenizers"))?
                    .iter()
                    .map(Self::from_json)
                    .collect::<Result<_, _>>()?,
            ),
            other => return Err(bad(format!("unsupported pre-tokenizer {other:?}"))),
        })
    }

    /// Splits each piece further.
    pub(crate) fn apply(&self, pieces: Vec<Piece>) -> Result<Vec<Piece>, TokenizerError> {
        match self {
            Self::Sequence(all) => {
                let mut pieces = pieces;
                for pre in all {
                    pieces = pre.apply(pieces)?;
                }
                Ok(pieces)
            }
            _ => {
                let mut out = Vec::new();
                for piece in pieces {
                    out.extend(self.apply_one(piece)?);
                }
                Ok(out)
            }
        }
    }

    fn apply_one(&self, piece: Piece) -> Result<Vec<Piece>, TokenizerError> {
        let first = piece.first;
        let make = |texts: Vec<String>| -> Vec<Piece> {
            texts
                .into_iter()
                .enumerate()
                .map(|(i, text)| Piece {
                    text,
                    first: first && i == 0,
                })
                .collect()
        };
        Ok(match self {
            Self::ByteLevel {
                add_prefix_space,
                regex,
            } => {
                let mut text = piece.text;
                if *add_prefix_space && !text.starts_with(' ') {
                    text.insert(0, ' ');
                }
                let parts = match regex {
                    Some(pattern) => split(&text, pattern, Behavior::Isolated, false)?,
                    None => vec![text],
                };
                make(parts.into_iter().map(|p| byte_level(&p)).collect())
            }
            Self::Split {
                pattern,
                behavior,
                invert,
            } => make(split(&piece.text, pattern, *behavior, *invert)?),
            Self::Metaspace {
                replacement,
                prepend,
                split: split_it,
            } => {
                let mut text: String = piece
                    .text
                    .chars()
                    .map(|c| if c == ' ' { *replacement } else { c })
                    .collect();
                let add = match prepend {
                    Prepend::Always => true,
                    Prepend::First => first,
                    Prepend::Never => false,
                };
                if add && !text.starts_with(*replacement) {
                    text.insert(0, *replacement);
                }
                if *split_it {
                    make(split(
                        &text,
                        &Pattern::Literal(replacement.to_string()),
                        Behavior::MergedWithNext,
                        false,
                    )?)
                } else {
                    make(vec![text])
                }
            }
            Self::Digits(pattern) => make(split(&piece.text, pattern, Behavior::Isolated, false)?),
            Self::Sequence(_) => unreachable!("sequences are applied piece list by piece list"),
        })
    }
}

/// Splits `text` around the matches of `pattern`, as `tokenizers` does.
fn split(
    text: &str,
    pattern: &Pattern,
    behavior: Behavior,
    invert: bool,
) -> Result<Vec<String>, TokenizerError> {
    // The text as alternating ranges: (start, end, is_delimiter).
    let mut ranges = Vec::new();
    let mut last = 0;
    for (a, b) in pattern.matches(text)? {
        if a > last {
            ranges.push((last, a, invert));
        }
        ranges.push((a, b, !invert));
        last = b;
    }
    if last < text.len() {
        ranges.push((last, text.len(), invert));
    }
    let mut out: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    let mut previous_delimiter = false;
    for (a, b, delimiter) in ranges {
        let part = &text[a..b];
        match (behavior, delimiter) {
            (Behavior::Removed, true) => {}
            (Behavior::MergedWithPrevious, true) => match out.last_mut() {
                Some(last) => last.push_str(part),
                None => out.push(part.to_string()),
            },
            (Behavior::MergedWithNext, true) => {
                pending.get_or_insert_with(String::new).push_str(part);
                previous_delimiter = true;
                continue;
            }
            (Behavior::Contiguous, true) if previous_delimiter => {
                out.last_mut().expect("a previous delimiter").push_str(part);
            }
            _ => {
                let mut piece = pending.take().unwrap_or_default();
                piece.push_str(part);
                out.push(piece);
            }
        }
        previous_delimiter = delimiter;
    }
    if let Some(rest) = pending {
        out.push(rest);
    }
    out.retain(|p| !p.is_empty());
    Ok(out)
}

/// GPT-2's map from bytes to printable characters: printable ASCII and most of
/// Latin-1 map to themselves, the other bytes to characters from U+0100 on.
pub(crate) fn bytes_to_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let printable = matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        map[usize::from(b)] = if printable {
            char::from(b)
        } else {
            let c = char::from_u32(next).expect("a valid code point");
            next += 1;
            c
        };
    }
    map
}

fn byte_level(text: &str) -> String {
    static MAP: OnceLock<[char; 256]> = OnceLock::new();
    let map = MAP.get_or_init(bytes_to_chars);
    text.bytes().map(|b| map[usize::from(b)]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_behaviors() {
        let space = Pattern::Literal(" ".into());
        assert_eq!(
            split("a b  c", &space, Behavior::Removed, false).unwrap(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            split("a b", &space, Behavior::Isolated, false).unwrap(),
            vec!["a", " ", "b"]
        );
        assert_eq!(
            split("a b", &space, Behavior::MergedWithPrevious, false).unwrap(),
            vec!["a ", "b"]
        );
        assert_eq!(
            split("a b", &space, Behavior::MergedWithNext, false).unwrap(),
            vec!["a", " b"]
        );
        assert_eq!(
            split("a  b", &space, Behavior::Contiguous, false).unwrap(),
            vec!["a", "  ", "b"]
        );
    }

    #[test]
    fn the_byte_map_covers_every_byte_once() {
        let map = bytes_to_chars();
        let mut seen: Vec<char> = map.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
        assert_eq!(map[usize::from(b' ')], 'Ġ');
        assert_eq!(map[usize::from(b'\n')], 'Ċ');
    }
}
