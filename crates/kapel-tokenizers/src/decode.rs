//! Decoders: from tokens back to text.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::text::{Pattern, bytes_to_chars};
use crate::{TokenizerError, bad, str_field};

/// The inverse of GPT-2's map from bytes to characters, built once.
fn chars_to_bytes() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        bytes_to_chars()
            .iter()
            .zip(0..=255u8)
            .map(|(&c, b)| (c, b))
            .collect()
    })
}

/// A decoder of `tokenizer.json`: it turns the list of token strings into another
/// list, joined at the end.
#[derive(Debug)]
pub(crate) enum Decoder {
    ByteLevel,
    Metaspace {
        replacement: char,
        prepend: bool,
    },
    Replace(Pattern, String),
    ByteFallback,
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Sequence(Vec<Decoder>),
}

impl Decoder {
    pub(crate) fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        let count = |key: &str| {
            json.get(key)
                .and_then(Value::as_u64)
                .map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX))
        };
        Ok(match str_field(json, "type")? {
            "ByteLevel" => Self::ByteLevel,
            "Metaspace" => Self::Metaspace {
                replacement: str_field(json, "replacement")?
                    .chars()
                    .next()
                    .unwrap_or('▁'),
                prepend: !matches!(
                    json.get("prepend_scheme").and_then(Value::as_str),
                    Some("never")
                ) && json
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            },
            "Replace" => Self::Replace(
                Pattern::from_json(
                    json.get("pattern")
                        .ok_or_else(|| bad("Replace without a pattern"))?,
                )?,
                str_field(json, "content")?.to_string(),
            ),
            "ByteFallback" => Self::ByteFallback,
            "Fuse" => Self::Fuse,
            "Strip" => Self::Strip {
                content: str_field(json, "content")?.chars().next().unwrap_or(' '),
                start: count("start"),
                stop: count("stop"),
            },
            "Sequence" => Self::Sequence(
                json.get("decoders")
                    .and_then(Value::as_array)
                    .ok_or_else(|| bad("Sequence without decoders"))?
                    .iter()
                    .map(Self::from_json)
                    .collect::<Result<_, _>>()?,
            ),
            other => return Err(bad(format!("unsupported decoder {other:?}"))),
        })
    }

    pub(crate) fn apply(&self, tokens: Vec<String>) -> Vec<String> {
        match self {
            Self::ByteLevel => {
                let to_byte = chars_to_bytes();
                let mut bytes: Vec<u8> = Vec::new();
                for c in tokens.iter().flat_map(|t| t.chars()) {
                    match to_byte.get(&c) {
                        Some(&b) => bytes.push(b),
                        None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                    }
                }
                vec![String::from_utf8_lossy(&bytes).into_owned()]
            }
            Self::Metaspace {
                replacement,
                prepend,
            } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let t: String = t
                        .chars()
                        .map(|c| if c == *replacement { ' ' } else { c })
                        .collect();
                    if i == 0 && *prepend {
                        t.strip_prefix(' ').map(str::to_string).unwrap_or(t)
                    } else {
                        t
                    }
                })
                .collect(),
            // A replacement that fails (a regex giving up) leaves the token as is:
            // decoding never fails.
            Self::Replace(pattern, content) => tokens
                .into_iter()
                .map(|t| crate::text::replace(pattern, &t, content).unwrap_or(t))
                .collect(),
            Self::ByteFallback => {
                let mut out = Vec::new();
                let mut bytes: Vec<u8> = Vec::new();
                let flush = |bytes: &mut Vec<u8>, out: &mut Vec<String>| {
                    if bytes.is_empty() {
                        return;
                    }
                    match String::from_utf8(std::mem::take(bytes)) {
                        Ok(s) => out.push(s),
                        // Invalid UTF-8: one replacement character per byte.
                        Err(e) => out.extend(e.as_bytes().iter().map(|_| "\u{FFFD}".to_string())),
                    }
                };
                for t in tokens {
                    let byte = t
                        .strip_prefix("<0x")
                        .and_then(|r| r.strip_suffix('>'))
                        .filter(|h| h.len() == 2)
                        .and_then(|h| u8::from_str_radix(h, 16).ok());
                    match byte {
                        Some(b) => bytes.push(b),
                        None => {
                            flush(&mut bytes, &mut out);
                            out.push(t);
                        }
                    }
                }
                flush(&mut bytes, &mut out);
                out
            }
            Self::Fuse => vec![tokens.concat()],
            Self::Strip {
                content,
                start,
                stop,
            } => tokens
                .into_iter()
                .map(|t| {
                    let chars: Vec<char> = t.chars().collect();
                    let lead = chars
                        .iter()
                        .take(*start)
                        .take_while(|&&c| c == *content)
                        .count();
                    let trail = chars
                        .iter()
                        .rev()
                        .take(*stop)
                        .take_while(|&&c| c == *content)
                        .count();
                    chars[lead..chars.len().saturating_sub(trail).max(lead)]
                        .iter()
                        .collect()
                })
                .collect(),
            Self::Sequence(all) => all.iter().fold(tokens, |t, d| d.apply(t)),
        }
    }
}
