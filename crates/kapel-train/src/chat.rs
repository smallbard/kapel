//! Conversations made into tokens for training: the text the model's chat
//! template gives them, and which tokens the assistant wrote, the only ones
//! the loss counts.
//!
//! An example is a JSON object of one of two forms, those of TRL:
//!
//! - `{"messages": [{"role": "user", "content": …}, …]}`: every message of
//!   the assistant is learnt, with the text that ends its turn (the
//!   template's `<|im_end|>`, `<|eot_id|>`…);
//! - `{"prompt": …, "completion": …}`: the completion is learnt, everything
//!   after the prompt. Both are lists of messages, rendered by the template
//!   (the prompt with the start of the assistant's answer), or both texts:
//!   the prompt is encoded with the tokenizer's template (Llama's `<s>`),
//!   the completion as it is, followed by the end of text.
//!
//! The answer of a conversation starts where the rendering of the messages
//! before it, with the start of an answer, ends; the template must render
//! them the same with and without the answer, or the example is refused. It
//! ends where its text does, found by a mark appended to it while
//! rendering: a template may render an answer differently once others
//! follow (Qwen 3 writes the empty `<think>` block of the last answer only).
//!
//! The text is encoded piece by piece, the assistant's pieces apart, with
//! the special tokens the template writes read as such: a message that
//! spells one gets the token, as with `transformers`.

use std::collections::BTreeMap;
use std::path::Path;

use kapel_tokenizers::{ChatTemplate, Message, Tokenizer};
use serde_json::Value;

use crate::data::DataError;

type Result<T> = std::result::Result<T, DataError>;

fn fail(m: impl Into<String>) -> DataError {
    DataError(m.into())
}

/// The mark appended to each answer while rendering: a character of the
/// private use area, which no text of the examples may hold.
const MARK: char = '\u{F8FF}';

/// One side of an example: a text, or messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Turns {
    /// A plain text.
    Text(String),
    /// Messages, for the chat template.
    Messages(Vec<Message>),
}

/// An example of supervised fine-tuning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Example {
    /// A conversation: every answer of the assistant is learnt.
    Messages(Vec<Message>),
    /// A prompt and its completion, which is learnt.
    Completion {
        /// What the model reads.
        prompt: Turns,
        /// What it learns to write.
        completion: Turns,
    },
}

/// The text of a message's `content`: a string, or a list of parts of which
/// the texts are joined (`{"type": "text", "text": …}`).
fn content(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut text = String::new();
            for p in parts {
                match (p.get("type").and_then(Value::as_str), p.get("text")) {
                    (Some("text") | None, Some(Value::String(t))) => text.push_str(t),
                    _ => return Err(fail(format!("a part of content other than text: {p}"))),
                }
            }
            Ok(text)
        }
        other => Err(fail(format!("a content that is not text: {other}"))),
    }
}

fn messages(value: &Value) -> Result<Vec<Message>> {
    let Value::Array(all) = value else {
        return Err(fail("messages: not a list"));
    };
    all.iter()
        .map(|m| {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| fail(format!("a message without a role: {m}")))?;
            Ok(Message::new(
                role,
                content(m.get("content").unwrap_or(&Value::Null))?,
            ))
        })
        .collect()
}

fn turns(value: &Value) -> Result<Turns> {
    match value {
        Value::String(s) => Ok(Turns::Text(s.clone())),
        Value::Array(_) => Ok(Turns::Messages(messages(value)?)),
        other => Err(fail(format!("neither a text nor messages: {other}"))),
    }
}

impl Example {
    /// The example a JSON object holds, its conversation in field `field`
    /// (`messages` by default) or as `prompt` and `completion`.
    ///
    /// # Errors
    ///
    /// If it holds neither.
    pub fn from_json(value: &Value, field: &str) -> Result<Self> {
        if let Some(m) = value.get(field) {
            return Ok(Self::Messages(messages(m)?));
        }
        match (value.get("prompt"), value.get("completion")) {
            (Some(p), Some(c)) => Ok(Self::Completion {
                prompt: turns(p)?,
                completion: turns(c)?,
            }),
            _ => Err(fail(format!(
                "an example with neither {field:?} nor \"prompt\" and \"completion\""
            ))),
        }
    }

    /// The bytes of its texts, for the progress of a preparation.
    pub fn bytes(&self) -> usize {
        let of = |t: &Turns| match t {
            Turns::Text(s) => s.len(),
            Turns::Messages(m) => m.iter().map(|m| m.content.len()).sum(),
        };
        match self {
            Self::Messages(m) => m.iter().map(|m| m.content.len()).sum(),
            Self::Completion { prompt, completion } => of(prompt) + of(completion),
        }
    }
}

/// What makes examples into tokens: a tokenizer, its chat template, and the
/// texts that end a turn.
#[derive(Debug)]
pub struct Encoder {
    tokenizer: Tokenizer,
    template: Option<ChatTemplate>,
    /// The texts of the tokens that end an answer, longest first.
    ends: Vec<String>,
    /// The token that ends a completion of text.
    eos: Option<u32>,
}

/// The `eos_token` of a `tokenizer_config.json`, a string or `{content}`.
fn named_token(config: &Value, key: &str) -> Option<String> {
    match config.get(key)? {
        Value::String(s) => Some(s.clone()),
        v => v.get("content")?.as_str().map(str::to_string),
    }
}

impl Encoder {
    /// The encoder of a model directory, or of the directory of a
    /// `tokenizer.json`: its tokenizer, its chat template, and its end
    /// tokens (`eos_token` of `tokenizer_config.json`, and those of
    /// `generation_config.json`).
    ///
    /// # Errors
    ///
    /// If the tokenizer or the template cannot be read.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let at = |e: &dyn std::fmt::Display| fail(format!("{}: {e}", dir.display()));
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| at(&e))?;
        let template = ChatTemplate::from_dir(dir).map_err(|e| at(&e))?;
        let read = |name: &str| -> Option<Value> {
            serde_json::from_slice(&std::fs::read(dir.join(name)).ok()?).ok()
        };
        let mut ends = Vec::new();
        if let Some(t) = read("tokenizer_config.json").and_then(|c| named_token(&c, "eos_token")) {
            ends.push(t);
        }
        for file in ["generation_config.json", "config.json"] {
            let ids = read(file).and_then(|c| c.get("eos_token_id").cloned());
            let ids: Vec<u64> = match ids {
                Some(Value::Number(n)) => n.as_u64().into_iter().collect(),
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).collect(),
                _ => Vec::new(),
            };
            for id in ids {
                if let Some(t) = u32::try_from(id)
                    .ok()
                    .and_then(|i| tokenizer.id_to_token(i))
                {
                    ends.push(t.to_string());
                }
            }
        }
        ends.sort_by_key(|t| std::cmp::Reverse(t.len()));
        ends.dedup();
        let eos = ends.iter().find_map(|t| tokenizer.token_to_id(t));
        Ok(Self {
            tokenizer,
            template,
            ends,
            eos,
        })
    }

    /// An encoder of these parts.
    pub fn new(
        tokenizer: Tokenizer,
        template: Option<ChatTemplate>,
        ends: Vec<String>,
        eos: Option<u32>,
    ) -> Self {
        Self {
            tokenizer,
            template,
            ends,
            eos,
        }
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The chat template, if any.
    pub fn template(&self) -> Option<&ChatTemplate> {
        self.template.as_ref()
    }

    /// The token that ends a completion of text, if known.
    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    fn template_or_fail(&self) -> Result<&ChatTemplate> {
        self.template
            .as_ref()
            .ok_or_else(|| fail("messages, but the tokenizer has no chat template"))
    }

    fn render(&self, messages: &[Message], generation: bool) -> Result<String> {
        self.template_or_fail()?
            .render(messages, generation)
            .map_err(|e| fail(e.to_string()))
    }

    /// The tokens of `text`, its special tokens read as such.
    fn tokens(&self, text: &str, out: &mut Vec<u32>) -> Result<usize> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| fail(e.to_string()))?;
        out.extend_from_slice(&ids);
        Ok(ids.len())
    }

    /// The text of `messages` and the byte ranges of the assistant's
    /// answers in it, each with the text that ends its turn.
    pub fn spans(&self, messages: &[Message]) -> Result<(String, Vec<(usize, usize)>)> {
        if messages.iter().any(|m| m.content.contains(MARK)) {
            return Err(fail(format!(
                "a message holds the character {MARK:?}, which marks the answers"
            )));
        }
        let full = self.render(messages, false)?;
        let marked: Vec<Message> = messages
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if m.role == "assistant" {
                    m.content.push(MARK);
                }
                m
            })
            .collect();
        let with_marks = self.render(&marked, false)?;
        // Where each mark falls in the text without them.
        let mut ends = Vec::new();
        let mut plain = String::with_capacity(full.len());
        for c in with_marks.chars() {
            if c == MARK {
                ends.push(plain.len());
            } else {
                plain.push(c);
            }
        }
        if plain != full {
            return Err(fail(
                "the chat template renders an answer differently once it ends otherwise",
            ));
        }
        let answers: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "assistant")
            .map(|(i, _)| i)
            .collect();
        if ends.len() != answers.len() {
            return Err(fail(format!(
                "{} answers, but the template writes {} of them",
                answers.len(),
                ends.len()
            )));
        }
        let mut spans = Vec::with_capacity(answers.len());
        let mut last = 0;
        for (&i, &end) in answers.iter().zip(&ends) {
            let before = self.render(&messages[..i], true)?;
            if !full.starts_with(&before) || before.len() > end || before.len() < last {
                return Err(fail(format!(
                    "the chat template renders the conversation before answer {i} otherwise \
                     once the answer follows"
                )));
            }
            let close = self
                .ends
                .iter()
                .find(|t| full[end..].starts_with(t.as_str()))
                .map_or(0, String::len);
            spans.push((before.len(), end + close));
            last = end + close;
        }
        Ok((full, spans))
    }

    /// The tokens of `example`, and whether each is the assistant's: the
    /// tokens the loss counts as targets.
    ///
    /// # Errors
    ///
    /// For messages without a chat template, or a template that renders
    /// them in a way the answers cannot be found in ([`Encoder::spans`]).
    pub fn encode(&self, example: &Example) -> Result<(Vec<u32>, Vec<bool>)> {
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut push = |text: &str, learnt: bool, tokens: &mut Vec<u32>| -> Result<()> {
            let n = self.tokens(text, tokens)?;
            mask.extend(std::iter::repeat_n(learnt, n));
            Ok(())
        };
        match example {
            Example::Messages(messages) => {
                let (text, spans) = self.spans(messages)?;
                let mut at = 0;
                for (start, end) in spans {
                    push(&text[at..start], false, &mut tokens)?;
                    push(&text[start..end], true, &mut tokens)?;
                    at = end;
                }
                push(&text[at..], false, &mut tokens)?;
            }
            Example::Completion {
                prompt: Turns::Messages(p),
                completion: Turns::Messages(c),
            } => {
                let before = self.render(p, true)?;
                let all: Vec<Message> = p.iter().chain(c).cloned().collect();
                let full = self.render(&all, false)?;
                let Some(rest) = full.strip_prefix(before.as_str()) else {
                    return Err(fail(
                        "the chat template renders the prompt otherwise once the completion \
                         follows",
                    ));
                };
                push(&before, false, &mut tokens)?;
                push(rest, true, &mut tokens)?;
            }
            Example::Completion {
                prompt: Turns::Text(p),
                completion: Turns::Text(c),
            } => {
                let ids = self
                    .tokenizer
                    .encode(p, true)
                    .map_err(|e| fail(e.to_string()))?;
                mask.extend(std::iter::repeat_n(false, ids.len()));
                tokens.extend(ids);
                let ids = self
                    .tokenizer
                    .encode_ordinary(c)
                    .map_err(|e| fail(e.to_string()))?;
                mask.extend(std::iter::repeat_n(true, ids.len()));
                tokens.extend(ids);
                if let Some(eos) = self.eos {
                    tokens.push(eos);
                    mask.push(true);
                }
            }
            Example::Completion { .. } => {
                return Err(fail(
                    "a prompt and a completion of different forms: both texts, or both messages",
                ));
            }
        }
        Ok((tokens, mask))
    }

    /// The tokens of a prompt, for generation: the messages rendered with
    /// the start of an answer, or a text encoded with the tokenizer's
    /// template; `variables` are given to the chat template
    /// (`enable_thinking`…).
    ///
    /// # Errors
    ///
    /// As [`Encoder::encode`].
    pub fn prompt(&self, prompt: &Turns, variables: &BTreeMap<String, Value>) -> Result<Vec<u32>> {
        match prompt {
            Turns::Text(t) => self
                .tokenizer
                .encode(t, true)
                .map_err(|e| fail(e.to_string())),
            Turns::Messages(m) => {
                let text = self
                    .template_or_fail()?
                    .render_with(m, true, variables)
                    .map_err(|e| fail(e.to_string()))?;
                let mut out = Vec::new();
                self.tokens(&text, &mut out)?;
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer of bytes: each character of a small alphabet its own
    /// token, and the special tokens of ChatML.
    fn encoder(template: &str) -> Encoder {
        let mut vocab = serde_json::Map::new();
        let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyz \n:<>/|_".chars().collect();
        for (i, c) in alphabet.iter().enumerate() {
            vocab.insert(c.to_string(), Value::from(i));
        }
        let n = alphabet.len();
        let json = serde_json::json!({
            "added_tokens": [
                {"id": n, "content": "<|im_start|>", "special": true},
                {"id": n + 1, "content": "<|im_end|>", "special": true},
            ],
            "pre_tokenizer": null,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        });
        let tokenizer = Tokenizer::from_json(&json).unwrap();
        let template = ChatTemplate::new(template, BTreeMap::new());
        Encoder::new(
            tokenizer,
            Some(template),
            vec!["<|im_end|>".into()],
            Some(n as u32 + 1),
        )
    }

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn learnt(e: &Encoder, tokens: &[u32], mask: &[bool]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .zip(mask)
            .filter(|(_, m)| **m)
            .map(|(t, _)| *t)
            .collect();
        text(e, &ids)
    }

    /// The tokens' texts joined: the test tokenizer has no decoder.
    fn text(e: &Encoder, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| e.tokenizer().id_to_token(i).unwrap())
            .collect()
    }

    #[test]
    fn every_answer_and_its_end_are_learnt() {
        let e = encoder(CHATML);
        let conversation = Example::Messages(vec![
            Message::new("user", "hi"),
            Message::new("assistant", "ok"),
            Message::new("user", "and"),
            Message::new("assistant", "yes"),
        ]);
        let (tokens, mask) = e.encode(&conversation).unwrap();
        assert_eq!(tokens.len(), mask.len());
        assert_eq!(learnt(&e, &tokens, &mask), "ok<|im_end|>yes<|im_end|>");
        // All the text is there, in order.
        assert_eq!(
            text(&e, &tokens),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n\
             <|im_start|>user\nand<|im_end|>\n<|im_start|>assistant\nyes<|im_end|>\n"
        );
    }

    #[test]
    fn a_completion_is_learnt_to_the_end() {
        let e = encoder(CHATML);
        let example = Example::Completion {
            prompt: Turns::Messages(vec![Message::new("user", "hi")]),
            completion: Turns::Messages(vec![Message::new("assistant", "ok")]),
        };
        let (tokens, mask) = e.encode(&example).unwrap();
        assert_eq!(learnt(&e, &tokens, &mask), "ok<|im_end|>\n");
        let text = Example::Completion {
            prompt: Turns::Text("ab".into()),
            completion: Turns::Text("cd".into()),
        };
        let (tokens, mask) = e.encode(&text).unwrap();
        assert_eq!(learnt(&e, &tokens, &mask), "cd<|im_end|>");
        assert_eq!(mask, vec![false, false, true, true, true]);
    }

    #[test]
    fn the_last_answer_rendered_otherwise_is_found() {
        // Qwen 3's habit: an empty thinking block before the last answer only.
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.role == 'assistant' and loop.last %}<think></think>{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let e = encoder(template);
        let conversation = Example::Messages(vec![
            Message::new("user", "hi"),
            Message::new("assistant", "ok"),
            Message::new("user", "and"),
            Message::new("assistant", "yes"),
        ]);
        let (tokens, mask) = e.encode(&conversation).unwrap();
        assert_eq!(
            learnt(&e, &tokens, &mask),
            "ok<|im_end|><think></think>yes<|im_end|>"
        );
    }

    #[test]
    fn examples_read_from_json() {
        let v = serde_json::json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "c"}
        ]});
        assert_eq!(
            Example::from_json(&v, "messages").unwrap(),
            Example::Messages(vec![
                Message::new("user", "ab"),
                Message::new("assistant", "c")
            ])
        );
        let v = serde_json::json!({"prompt": "a", "completion": "b"});
        assert!(matches!(
            Example::from_json(&v, "messages").unwrap(),
            Example::Completion { .. }
        ));
        assert!(Example::from_json(&serde_json::json!({"text": "a"}), "messages").is_err());
        let marked = Example::Messages(vec![Message::new("user", "\u{F8FF}")]);
        assert!(encoder(CHATML).encode(&marked).is_err());
    }
}
