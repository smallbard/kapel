//! The tokenizer, chat template and end tokens of a GGUF file.
//!
//! A GGUF file holds its tokenizer as metadata: the tokens, their kinds, the
//! merges and the name of the pre-tokenizer. They are written back as the
//! `tokenizer.json` of `transformers` they come from, which
//! `kapel-tokenizers` reads: the same tokens as `transformers` on the same
//! text.
//!
//! [`save_gguf`] goes the other way: a model and the tokenizer of its
//! directory (or of the GGUF file it was read from) written as llama.cpp's
//! converter writes them, so that llama.cpp reads the file too.

use std::collections::BTreeMap;
use std::path::Path;

use kapel_hub::Gguf;
use kapel_hub::gguf::{Entry, Value as Meta};
use kapel_models::Model;
use kapel_ops::Backend;
use kapel_tokenizers::{ChatTemplate, Tokenizer};
use serde_json::{Value, json};

use crate::{InferError, Result};

/// The splitting pattern of Llama 3 (`tokenizer.ggml.pre` = `llama-bpe`).
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The splitting pattern of Qwen 2 and 3 (`qwen2`): digits one by one.
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The kinds of `tokenizer.ggml.token_type`.
const NORMAL: i64 = 1;
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;
const UNUSED: i64 = 5;

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

/// Whether an added token looks like a control token to llama.cpp's
/// converter (`does_token_look_special`), though not marked special.
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<\u{ff5c}") && token.ends_with("\u{ff5c}>"))
        || (token.starts_with("<start_of_") && token.ends_with('>'))
        || (token.starts_with("<end_of_") && token.ends_with('>'))
}

/// The text of a special token in `tokenizer_config.json`: a string or an
/// object with its `content`.
fn token_text(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}

/// The splitting pattern of a `tokenizer.json`'s pre-tokenizer, if one.
fn split_pattern(pre: &Value) -> Option<&str> {
    match pre.get("type").and_then(Value::as_str)? {
        "Split" => pre.get("pattern")?.get("Regex")?.as_str(),
        "Sequence" => pre
            .get("pretokenizers")?
            .as_array()?
            .iter()
            .find_map(split_pattern),
        _ => None,
    }
}

/// Whether a post-processor puts the special token `id` at the start
/// (`first`) or at the end of a single sequence.
fn adds(post: &Value, token: Option<&str>, first: bool) -> Option<bool> {
    let token = token?;
    match post.get("type").and_then(Value::as_str)? {
        "TemplateProcessing" => {
            let single = post.get("single")?.as_array()?;
            let piece = if first { single.first() } else { single.last() }?;
            let id = piece.get("SpecialToken")?.get("id")?.as_str()?;
            Some(id == token)
        }
        "Sequence" => Some(
            post.get("processors")?
                .as_array()?
                .iter()
                .any(|p| adds(p, Some(token), first) == Some(true)),
        ),
        _ => Some(false),
    }
}

/// Whether a post-processor is, or holds, a template.
fn templated(post: &Value) -> bool {
    match post.get("type").and_then(Value::as_str) {
        Some("TemplateProcessing") => true,
        Some("Sequence") => post["processors"]
            .as_array()
            .is_some_and(|p| p.iter().any(templated)),
        _ => false,
    }
}

/// The settings of `generation_config.json` llama.cpp keeps
/// (`general.sampling.*`).
fn sampling_metadata(dir: &Path) -> Result<Vec<(String, Entry)>> {
    let path = dir.join("generation_config.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let config = kapel_hub::read_json(&path)?;
    let mut metadata = Vec::new();
    if let Some(k) = config.get("top_k").and_then(Value::as_i64) {
        metadata.push(("general.sampling.top_k".into(), Entry::I32(k as i32)));
    }
    for (key, name) in [
        ("top_p", "top_p"),
        ("min_p", "min_p"),
        ("temperature", "temp"),
        ("repetition_penalty", "penalty_repeat"),
    ] {
        if let Some(v) = config.get(key).and_then(Value::as_f64) {
            metadata.push((format!("general.sampling.{name}"), Entry::F32(v as f32)));
        }
    }
    Ok(metadata)
}

/// The default sampling of a GGUF file: its `general.sampling.*`, as
/// `generation_config.json` would give them; greedy without them.
pub(crate) fn sampling(gguf: &Gguf) -> crate::Sampling {
    let number = |key: &str| match gguf.metadata.get(&format!("general.sampling.{key}")) {
        Some(Meta::Float(v)) => Some(*v),
        Some(Meta::Int(v)) => Some(*v as f64),
        Some(Meta::Uint(v)) => Some(*v as f64),
        _ => None,
    };
    let mut config = serde_json::Map::new();
    for (key, name) in [
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("min_p", "min_p"),
        ("temp", "temperature"),
        ("penalty_repeat", "repetition_penalty"),
    ] {
        if let Some(v) = number(key) {
            config.insert(name.into(), json!(v));
        }
    }
    if config.is_empty() {
        return crate::Sampling::default();
    }
    if let Some(k) = config.get("top_k").and_then(Value::as_f64) {
        config.insert("top_k".into(), json!(k as i64));
    }
    config.insert("do_sample".into(), json!(true));
    crate::Sampling::from_generation_config(&Value::Object(config))
}

/// The tokenizer's metadata of a GGUF file, as llama.cpp's converter writes
/// it from a model directory: byte-level BPE tokenizers with the splitting
/// of Llama 3 or of Qwen 2 and 3. `vocab` is the size of the embedding
/// table: ids the tokenizer lacks are padding tokens.
fn directory_metadata(dir: &Path, vocab: usize) -> Result<Vec<(String, Entry)>> {
    let read = |name: &str| -> Result<Option<Value>> {
        let path = dir.join(name);
        if path.exists() {
            Ok(Some(kapel_hub::read_json(&path)?))
        } else {
            Ok(None)
        }
    };
    let json =
        read("tokenizer.json")?.ok_or_else(|| unsupported("the model has no tokenizer.json"))?;
    let config = read("tokenizer_config.json")?.unwrap_or(Value::Null);
    let model_config = read("config.json")?.unwrap_or(Value::Null);
    let model = &json["model"];
    if model["type"] != "BPE" {
        return Err(unsupported(
            "only byte-level BPE tokenizers are written to GGUF",
        ));
    }
    let pattern = split_pattern(&json["pre_tokenizer"]).unwrap_or("");
    let pre = match pattern {
        LLAMA3 => "llama-bpe",
        QWEN2 => "qwen2",
        _ => {
            return Err(unsupported(
                "this pre-tokenizer is not written to GGUF: Llama 3's and Qwen's are",
            ));
        }
    };
    let mut tokens: Vec<Option<(String, i64)>> = vec![None; vocab];
    if let Some(entries) = model["vocab"].as_object() {
        for (token, id) in entries {
            if let Some(slot) = id
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .and_then(|id| tokens.get_mut(id))
            {
                *slot = Some((token.clone(), NORMAL));
            }
        }
    }
    let mut ids: BTreeMap<String, u32> = BTreeMap::new();
    for added in json["added_tokens"].as_array().into_iter().flatten() {
        let (Some(id), Some(content)) = (added["id"].as_u64(), added["content"].as_str()) else {
            continue;
        };
        let special = added["special"].as_bool().unwrap_or(false);
        let kind = if special || looks_special(content) {
            CONTROL
        } else {
            USER_DEFINED
        };
        ids.insert(content.to_string(), id as u32);
        if let Some(slot) = usize::try_from(id).ok().and_then(|id| tokens.get_mut(id)) {
            *slot = Some((content.to_string(), kind));
        }
    }
    if let Some(entries) = model["vocab"].as_object() {
        for (token, id) in entries {
            if let Some(id) = id.as_u64() {
                ids.entry(token.clone()).or_insert(id as u32);
            }
        }
    }
    let (texts, kinds): (Vec<String>, Vec<i64>) = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.unwrap_or_else(|| (format!("[PAD{i}]"), UNUSED)))
        .unzip();
    let merges: Vec<String> = model["merges"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| match m {
            Value::String(s) => Some(s.clone()),
            Value::Array(pair) => match (pair.first()?.as_str(), pair.get(1)?.as_str()) {
                (Some(a), Some(b)) => Some(format!("{a} {b}")),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mut metadata = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Entry::String("gpt2".into()),
        ),
        ("tokenizer.ggml.pre".to_string(), Entry::String(pre.into())),
        ("tokenizer.ggml.tokens".to_string(), Entry::strings(texts)),
        (
            "tokenizer.ggml.token_type".to_string(),
            Entry::Array(kinds.into_iter().map(|k| Entry::I32(k as i32)).collect()),
        ),
        ("tokenizer.ggml.merges".to_string(), Entry::strings(merges)),
    ];
    // The special tokens: by their text in `tokenizer_config.json`, else by
    // their id in `config.json`, as llama.cpp's `SpecialVocab` finds them.
    let mut special = BTreeMap::new();
    for kind in ["bos", "eos", "unk", "pad"] {
        let id = config
            .get(format!("{kind}_token"))
            .and_then(token_text)
            .and_then(|t| ids.get(t).copied())
            .or_else(|| {
                model_config
                    .get(format!("{kind}_token_id"))
                    .and_then(Value::as_u64)
                    .and_then(|id| u32::try_from(id).ok())
            });
        if let Some(id) = id {
            special.insert(kind, id);
        }
    }
    for (kind, key) in [
        ("bos", "bos_token_id"),
        ("eos", "eos_token_id"),
        ("unk", "unknown_token_id"),
        ("pad", "padding_token_id"),
    ] {
        if let Some(&id) = special.get(kind) {
            metadata.push((format!("tokenizer.ggml.{key}"), Entry::U32(id)));
        }
    }
    let text_of = |kind: &str| {
        let id = *special.get(kind)?;
        ids.iter().find(|(_, i)| **i == id).map(|(t, _)| t.as_str())
    };
    for (kind, first) in [("bos", true), ("eos", false)] {
        let add = config
            .get(format!("add_{kind}_token"))
            .and_then(Value::as_bool)
            .or_else(|| adds(&json["post_processor"], text_of(kind), first))
            .unwrap_or(false);
        metadata.push((format!("tokenizer.ggml.add_{kind}_token"), Entry::Bool(add)));
    }
    // llama.cpp's converter reads a template of the post-processor for the
    // separator too.
    if templated(&json["post_processor"]) {
        metadata.push(("tokenizer.ggml.add_sep_token".into(), Entry::Bool(false)));
    }
    let template = match config.get("chat_template") {
        Some(Value::String(t)) => Some(t.clone()),
        _ => std::fs::read_to_string(dir.join("chat_template.jinja")).ok(),
    };
    if let Some(template) = template {
        metadata.push(("tokenizer.chat_template".into(), Entry::String(template)));
    }
    if json["normalizer"]["type"] == "NFC" {
        metadata.push(("tokenizer.ggml.normalizer.nfc".into(), Entry::Bool(true)));
    }
    if let Some(clean) = config
        .get("clean_up_tokenization_spaces")
        .and_then(Value::as_bool)
    {
        metadata.push(("tokenizer.ggml.clean_spaces".into(), Entry::Bool(clean)));
    }
    Ok(metadata)
}

/// The metadata of a GGUF file under `prefix`, copied with the types
/// llama.cpp reads them as.
fn gguf_metadata(gguf: &Gguf, prefix: &str) -> Result<Vec<(String, Entry)>> {
    fn entry(value: &Meta) -> Entry {
        match value {
            Meta::Uint(v) => Entry::U32(*v as u32),
            Meta::Int(v) => Entry::I32(*v as i32),
            Meta::Float(v) => Entry::F32(*v as f32),
            Meta::Bool(v) => Entry::Bool(*v),
            Meta::String(s) => Entry::String(s.clone()),
            Meta::Array(values) => Entry::Array(values.iter().map(entry).collect()),
        }
    }
    Ok(gguf
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with(prefix))
        .map(|(key, value)| (key.clone(), entry(value)))
        .collect())
}

/// Writes `model` to a GGUF file at `path`, named `name`, with the
/// tokenizer of `source`: the model's directory (and the defaults of its
/// `generation_config.json`), or the GGUF file it was read from. The
/// weights keep their types, quantized or not ([`Model::to_gguf`]), and
/// llama.cpp reads the file as it reads its own conversions.
///
/// # Errors
///
/// For a model or a tokenizer llama.cpp's converter does not write (see
/// [`Model::to_gguf`]), weights outside host memory, or a file that cannot
/// be read or written.
pub fn save_gguf<B: Backend>(
    model: &Model,
    backend: &B,
    source: impl AsRef<Path>,
    name: Option<&str>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let source = source.as_ref();
    let mut parts = model.to_gguf(backend)?;
    let (general, tokenizer) = if source.is_dir() {
        (
            sampling_metadata(source)?,
            directory_metadata(source, model.config().vocab)?,
        )
    } else {
        let gguf = Gguf::open(source)?;
        let general = gguf_metadata(&gguf, "general.sampling.")?;
        (general, gguf_metadata(&gguf, "tokenizer.")?)
    };
    if let Some(name) = name {
        parts.metadata.retain(|(k, _)| k != "general.name");
        parts
            .metadata
            .insert(2, ("general.name".into(), Entry::String(name.into())));
    }
    // After the architecture and the type, as llama.cpp's converter.
    let at = 2.min(parts.metadata.len());
    let metadata: Vec<(&str, Entry)> = parts.metadata[..at]
        .iter()
        .chain(&general)
        .chain(&parts.metadata[at..])
        .chain(&tokenizer)
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    let tensors: Vec<(&str, &kapel_core::Tensor)> =
        parts.tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    kapel_hub::gguf::write(path, &metadata, &tensors)?;
    Ok(())
}
