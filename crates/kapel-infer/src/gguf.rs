//! The tokenizer, chat template and end tokens of a GGUF file.
//!
//! A GGUF file holds its tokenizer as metadata: the tokens, their kinds, the
//! merges and the name of the pre-tokenizer. They are written back as the
//! `tokenizer.json` of `transformers` they come from, which
//! `kapel-tokenizers` reads: the same tokens as `transformers` on the same
//! text.

use std::collections::BTreeMap;

use kapel_hub::Gguf;
use kapel_hub::gguf::Value as Meta;
use kapel_tokenizers::{ChatTemplate, Tokenizer};
use serde_json::{Value, json};

use crate::{InferError, Result};

/// The splitting pattern of Llama 3 (`tokenizer.ggml.pre` = `llama-bpe`).
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The splitting pattern of Qwen 2 and 3 (`qwen2`): digits one by one.
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The kinds of `tokenizer.ggml.token_type` that are added tokens.
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;

/// The texts of the tokens that end a turn, as llama.cpp recognizes them.
const END_TEXTS: [&str; 7] = [
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
];

fn unsupported(message: impl Into<String>) -> InferError {
    InferError::Unsupported(message.into())
}

fn string<'a>(gguf: &'a Gguf, key: &str) -> Option<&'a str> {
    match gguf.metadata.get(key)? {
        Meta::String(s) => Some(s),
        _ => None,
    }
}

fn integer(gguf: &Gguf, key: &str) -> Option<i64> {
    match gguf.metadata.get(key)? {
        Meta::Uint(v) => i64::try_from(*v).ok(),
        Meta::Int(v) => Some(*v),
        _ => None,
    }
}

fn array<'a>(gguf: &'a Gguf, key: &str) -> Result<&'a [Meta]> {
    match gguf.metadata.get(key) {
        Some(Meta::Array(values)) => Ok(values),
        _ => Err(unsupported(format!("the GGUF file has no {key}"))),
    }
}

/// What a GGUF file gives the pipeline besides its weights.
pub(crate) struct Vocabulary {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) template: Option<ChatTemplate>,
    pub(crate) eos: Vec<u32>,
}

/// The tokenizer, chat template and end tokens of `gguf`.
pub(crate) fn vocabulary(gguf: &Gguf) -> Result<Vocabulary> {
    let model = string(gguf, "tokenizer.ggml.model").unwrap_or("");
    if model != "gpt2" {
        return Err(unsupported(format!(
            "GGUF tokenizer {model:?}: only byte-level BPE (\"gpt2\") is read"
        )));
    }
    let pre = string(gguf, "tokenizer.ggml.pre").unwrap_or("default");
    // With llama.cpp's defaults for adding a beginning of sequence, which
    // `tokenizer.ggml.add_bos_token` overrides.
    let (pattern, normalizer, ignore_merges, bos_default) = match pre {
        "llama-bpe" | "llama3" => (LLAMA3, Value::Null, true, true),
        "qwen2" => (QWEN2, json!({"type": "NFC"}), false, false),
        other => {
            return Err(unsupported(format!(
                "GGUF pre-tokenizer {other:?} is not supported"
            )));
        }
    };
    let tokens: Vec<&str> = array(gguf, "tokenizer.ggml.tokens")?
        .iter()
        .map(|t| match t {
            Meta::String(s) => Ok(s.as_str()),
            _ => Err(unsupported("tokenizer.ggml.tokens holds a non-string")),
        })
        .collect::<Result<_>>()?;
    let kinds: Vec<i64> = match gguf.metadata.get("tokenizer.ggml.token_type") {
        Some(Meta::Array(kinds)) => kinds
            .iter()
            .map(|k| match k {
                Meta::Int(v) => *v,
                Meta::Uint(v) => i64::try_from(*v).unwrap_or(0),
                _ => 0,
            })
            .collect(),
        _ => vec![1; tokens.len()],
    };
    let merges: Vec<Value> = array(gguf, "tokenizer.ggml.merges")?
        .iter()
        .map(|m| match m {
            Meta::String(s) => s
                .split_once(' ')
                .map(|(a, b)| json!([a, b]))
                .ok_or_else(|| unsupported(format!("merge {s:?} has no space"))),
            _ => Err(unsupported("tokenizer.ggml.merges holds a non-string")),
        })
        .collect::<Result<_>>()?;
    let vocab: serde_json::Map<String, Value> = tokens
        .iter()
        .enumerate()
        .map(|(id, &t)| (t.to_string(), json!(id)))
        .collect();
    let added: Vec<Value> = tokens
        .iter()
        .zip(&kinds)
        .enumerate()
        .filter(|(_, (_, kind))| matches!(**kind, CONTROL | USER_DEFINED))
        .map(|(id, (&t, &kind))| {
            json!({
                "id": id, "content": t, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": kind == CONTROL,
            })
        })
        .collect();
    let token = |key: &str| {
        integer(gguf, key)
            .and_then(|id| usize::try_from(id).ok())
            .and_then(|id| tokens.get(id).copied())
    };
    let bos_id = integer(gguf, "tokenizer.ggml.bos_token_id");
    let bos = token("tokenizer.ggml.bos_token_id");
    let add_bos = match gguf.metadata.get("tokenizer.ggml.add_bos_token") {
        Some(Meta::Bool(add)) => *add,
        _ => bos_default,
    };
    let byte_level = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
        "use_regex": false,
    });
    let post_processor = match bos {
        Some(bos) if add_bos => json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [],
            "special_tokens": {bos: {"id": bos, "ids": [bos_id], "tokens": [bos]}},
        }),
        _ => byte_level.clone(),
    };
    let json = json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
                byte_level,
            ],
        },
        "post_processor": post_processor,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": ignore_merges,
            "vocab": vocab, "merges": merges,
        },
    });
    let tokenizer = Tokenizer::from_json(&json)?;

    let mut eos: Vec<u32> = [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
    ]
    .iter()
    .filter_map(|key| integer(gguf, key).and_then(|id| u32::try_from(id).ok()))
    .collect();
    for (id, (&t, &kind)) in tokens.iter().zip(&kinds).enumerate() {
        if kind == CONTROL && END_TEXTS.contains(&t) {
            eos.push(u32::try_from(id).map_err(|_| unsupported("more than 2³² tokens"))?);
        }
    }
    eos.sort_unstable();
    eos.dedup();

    let mut special = BTreeMap::new();
    for (name, key) in [
        ("bos_token", "tokenizer.ggml.bos_token_id"),
        ("eos_token", "tokenizer.ggml.eos_token_id"),
        ("unk_token", "tokenizer.ggml.unknown_token_id"),
        ("pad_token", "tokenizer.ggml.padding_token_id"),
    ] {
        if let Some(t) = token(key) {
            special.insert(name.to_string(), t.to_string());
        }
    }
    let template =
        string(gguf, "tokenizer.chat_template").map(|source| ChatTemplate::new(source, special));
    Ok(Vocabulary {
        tokenizer,
        template,
        eos,
    })
}
