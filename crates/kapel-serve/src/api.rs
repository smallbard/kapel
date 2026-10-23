//! The requests and answers of OpenAI's API, as far as Kapel serves them.
//!
//! Requests are read leniently: fields Kapel does not know are ignored, as
//! clients send many. Fields it knows but cannot honour (tools, log
//! probabilities, images) are refused with an error that says so, rather
//! than ignored. vLLM's extensions are read too: `top_k`, `min_p`,
//! `repetition_penalty`, `ignore_eos`, and the constraints `guided_json`,
//! `guided_regex`, `guided_choice`, `guided_grammar`.

use kapel_infer::{Constraint, Request, Sampling};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The most answers to one request (`n`).
pub(crate) const MAX_CHOICES: usize = 16;

/// The most stop sequences of one request, as OpenAI allows.
const MAX_STOPS: usize = 4;

/// A request refused: the HTTP status and OpenAI's error body.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: u16,
    pub(crate) kind: &'static str,
    pub(crate) message: String,
    pub(crate) param: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn invalid(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
            param,
        }
    }

    pub(crate) fn body(&self) -> Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": Value::Null,
            }
        })
    }
}

/// A text, or several.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// The content of a chat message: a text, or parts of which only texts are
/// read.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<Value>),
}

/// A message of a conversation.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<Content>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Value>,
}

impl ChatMessage {
    /// The text of the message.
    pub(crate) fn text(&self) -> Result<String, ApiError> {
        if self.tool_calls.as_ref().is_some_and(|t| !t.is_null()) {
            return Err(ApiError::invalid(
                "tool calls are not supported",
                Some("messages"),
            ));
        }
        match &self.content {
            None => Ok(String::new()),
            Some(Content::Text(text)) => Ok(text.clone()),
            Some(Content::Parts(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                        }
                        other => {
                            return Err(ApiError::invalid(
                                format!("content parts of type {other:?} are not supported"),
                                Some("messages"),
                            ));
                        }
                    }
                }
                Ok(text)
            }
        }
    }
}

/// `stream_options`.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

/// What the requests of both endpoints share: how to generate.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct Generation {
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) max_completion_tokens: Option<u64>,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) n: Option<u64>,
    pub(crate) stop: Option<OneOrMany>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) seed: Option<i64>,
    pub(crate) frequency_penalty: Option<f64>,
    pub(crate) presence_penalty: Option<f64>,
    pub(crate) logprobs: Option<Value>,
    pub(crate) top_logprobs: Option<Value>,
    pub(crate) response_format: Option<Value>,
    pub(crate) tools: Option<Value>,
    pub(crate) best_of: Option<u64>,
    #[serde(default)]
    pub(crate) echo: bool,
    pub(crate) suffix: Option<String>,
    // vLLM's extensions.
    pub(crate) top_k: Option<i64>,
    pub(crate) min_p: Option<f64>,
    pub(crate) repetition_penalty: Option<f64>,
    #[serde(default)]
    pub(crate) ignore_eos: bool,
    pub(crate) guided_json: Option<Value>,
    pub(crate) guided_regex: Option<String>,
    pub(crate) guided_choice: Option<Vec<String>>,
    pub(crate) guided_grammar: Option<String>,
}

/// `/v1/chat/completions`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatRequest {
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub(crate) generation: Generation,
}

/// The prompt of `/v1/completions`: texts, or tokens.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Prompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenLists(Vec<Vec<u32>>),
}

/// `/v1/completions`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    pub(crate) prompt: Prompt,
    #[serde(flatten)]
    pub(crate) generation: Generation,
}

/// What a request asks of the engine, once read.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// The request of each answer, without its length, which depends on the
    /// prompt.
    pub(crate) request: Request,
    /// The tokens asked for, if the request says.
    pub(crate) max_tokens: Option<usize>,
    pub(crate) choices: usize,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
}

fn number_in(
    value: Option<f64>,
    range: std::ops::RangeInclusive<f64>,
    param: &'static str,
) -> Result<Option<f64>, ApiError> {
    match value {
        Some(v) if !range.contains(&v) || v.is_nan() => Err(ApiError::invalid(
            format!(
                "{param} must be between {} and {}",
                range.start(),
                range.end()
            ),
            Some(param),
        )),
        other => Ok(other),
    }
}

impl Generation {
    /// The request to the engine, the defaults of the model for what is not
    /// said.
    pub(crate) fn plan(&self, defaults: &Sampling, chat: bool) -> Result<Plan, ApiError> {
        if self
            .tools
            .as_ref()
            .is_some_and(|t| !t.is_null() && t != &Value::Array(vec![]))
        {
            return Err(ApiError::invalid("tools are not supported", Some("tools")));
        }
        let logprobs = match &self.logprobs {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(Value::Number(n)) => !chat && n.as_u64() != Some(0),
            Some(_) => true,
        };
        if logprobs || self.top_logprobs.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(ApiError::invalid(
                "log probabilities are not supported",
                Some("logprobs"),
            ));
        }
        if self.echo {
            return Err(ApiError::invalid("echo is not supported", Some("echo")));
        }
        if self.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
            return Err(ApiError::invalid("suffix is not supported", Some("suffix")));
        }
        let choices = self.n.unwrap_or(1);
        if choices == 0 || choices > MAX_CHOICES as u64 {
            return Err(ApiError::invalid(
                format!("n must be between 1 and {MAX_CHOICES}"),
                Some("n"),
            ));
        }
        if self.best_of.is_some_and(|b| b != choices) {
            return Err(ApiError::invalid(
                "best_of is not supported",
                Some("best_of"),
            ));
        }
        let mut sampling = defaults.clone();
        if let Some(t) = number_in(self.temperature, 0.0..=2.0, "temperature")? {
            sampling.temperature = t;
            if t == 0.0 {
                // Greedy: the filters would change nothing.
                sampling.top_k = None;
            }
        }
        if let Some(p) = number_in(self.top_p, 0.0..=1.0, "top_p")? {
            if p == 0.0 {
                return Err(ApiError::invalid("top_p must be above 0", Some("top_p")));
            }
            sampling.top_p = p;
        }
        if let Some(k) = self.top_k {
            sampling.top_k = match k {
                ..=0 => None,
                k => Some(usize::try_from(k).unwrap_or(usize::MAX)),
            };
        }
        if let Some(m) = number_in(self.min_p, 0.0..=1.0, "min_p")? {
            sampling.min_p = m;
        }
        if let Some(r) = number_in(self.repetition_penalty, 0.01..=10.0, "repetition_penalty")? {
            sampling.repetition_penalty = r;
        }
        if let Some(f) = number_in(self.frequency_penalty, -2.0..=2.0, "frequency_penalty")? {
            sampling.frequency_penalty = f;
        }
        if let Some(p) = number_in(self.presence_penalty, -2.0..=2.0, "presence_penalty")? {
            sampling.presence_penalty = p;
        }
        sampling.seed = match self.seed {
            Some(seed) => seed as u64,
            None => fresh_seed(),
        };
        let stop = self
            .stop
            .clone()
            .map(OneOrMany::into_vec)
            .unwrap_or_default();
        if stop.len() > MAX_STOPS {
            return Err(ApiError::invalid(
                format!("at most {MAX_STOPS} stop sequences"),
                Some("stop"),
            ));
        }
        let max_tokens = self.max_completion_tokens.or(self.max_tokens);
        if max_tokens == Some(0) {
            return Err(ApiError::invalid(
                "max_tokens must be at least 1",
                Some("max_tokens"),
            ));
        }
        Ok(Plan {
            request: Request {
                sampling,
                max_tokens: 0,
                stop,
                ignore_eos: self.ignore_eos,
                constraint: self.constraint()?,
            },
            max_tokens: max_tokens.map(|m| usize::try_from(m).unwrap_or(usize::MAX)),
            choices: choices as usize,
            stream: self.stream,
            include_usage: self
                .stream_options
                .as_ref()
                .is_some_and(|o| o.include_usage),
        })
    }

    /// The form the answer must take: `response_format`, or one of vLLM's
    /// `guided_*` fields.
    fn constraint(&self) -> Result<Option<Constraint>, ApiError> {
        let mut found = Vec::new();
        if let Some(format) = &self.response_format {
            match format.get("type").and_then(Value::as_str) {
                None | Some("text") => {}
                Some("json_object") => found.push(Constraint::JsonObject),
                Some("json_schema") => {
                    let schema = format
                        .get("json_schema")
                        .and_then(|s| s.get("schema"))
                        .cloned()
                        .ok_or_else(|| {
                            ApiError::invalid(
                                "response_format.json_schema.schema is missing",
                                Some("response_format"),
                            )
                        })?;
                    found.push(Constraint::JsonSchema(schema));
                }
                Some(other) => {
                    return Err(ApiError::invalid(
                        format!("response_format of type {other:?} is not supported"),
                        Some("response_format"),
                    ));
                }
            }
        }
        if let Some(schema) = &self.guided_json {
            found.push(match schema {
                Value::String(text) => Constraint::JsonSchema(
                    serde_json::from_str(text)
                        .map_err(|e| ApiError::invalid(e.to_string(), Some("guided_json")))?,
                ),
                schema => Constraint::JsonSchema(schema.clone()),
            });
        }
        if let Some(regex) = &self.guided_regex {
            found.push(Constraint::Regex(regex.clone()));
        }
        if let Some(choices) = &self.guided_choice {
            found.push(Constraint::Choice(choices.clone()));
        }
        if let Some(grammar) = &self.guided_grammar {
            found.push(Constraint::Lark(grammar.clone()));
        }
        if found.len() > 1 {
            return Err(ApiError::invalid(
                "at most one constraint on the answer",
                Some("response_format"),
            ));
        }
        Ok(found.pop())
    }
}

/// A seed for a request that gives none: different at each call.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    time ^ NEXT
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// The tokens of an answer and its prompt.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
    pub(crate) prompt_tokens_details: CachedTokens,
}

/// `prompt_tokens_details`.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct CachedTokens {
    pub(crate) cached_tokens: usize,
}

impl Usage {
    pub(crate) fn add(&mut self, prompt: usize, completion: usize, cached: usize) {
        self.prompt_tokens += prompt;
        self.completion_tokens += completion;
        self.total_tokens += prompt + completion;
        self.prompt_tokens_details.cached_tokens += cached;
    }
}

/// OpenAI's name of how an answer ended.
pub(crate) fn finish_reason(finish: &kapel_infer::Finish) -> &'static str {
    match finish {
        kapel_infer::Finish::Eos | kapel_infer::Finish::Stop(_) => "stop",
        kapel_infer::Finish::Length => "length",
        kapel_infer::Finish::Error(_) => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(body: Value) -> Result<Plan, ApiError> {
        let request: ChatRequest = serde_json::from_value(body).unwrap();
        request.generation.plan(&Sampling::greedy(), true)
    }

    #[test]
    fn requests_read_as_openai_writes_them() {
        let p = plan(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "temperature": 0.7, "top_p": 0.9, "n": 2, "stop": "\n",
            "max_completion_tokens": 5, "seed": 3, "stream": true,
            "stream_options": {"include_usage": true},
            "unknown_field": 1,
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
        }))
        .unwrap();
        assert_eq!(p.request.sampling.temperature, 0.7);
        assert_eq!(p.request.sampling.top_p, 0.9);
        assert_eq!(p.request.sampling.seed, 3);
        assert_eq!(p.request.stop, ["\n"]);
        assert_eq!(
            (p.max_tokens, p.choices, p.stream, p.include_usage),
            (Some(5), 2, true, true)
        );
        assert!(matches!(
            p.request.constraint,
            Some(Constraint::JsonSchema(_))
        ));
    }

    #[test]
    fn what_cannot_be_honoured_is_refused() {
        let base = |extra: Value| {
            let mut body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            plan(body)
        };
        for extra in [
            serde_json::json!({"logprobs": true}),
            serde_json::json!({"tools": [{"type": "function"}]}),
            serde_json::json!({"n": 0}),
            serde_json::json!({"temperature": 3}),
            serde_json::json!({"max_tokens": 0}),
            serde_json::json!({"stop": ["a", "b", "c", "d", "e"]}),
            serde_json::json!({"guided_regex": "a", "guided_choice": ["b"]}),
        ] {
            let error = base(extra.clone()).unwrap_err();
            assert_eq!(error.status, 400, "{extra}");
        }
        assert!(base(serde_json::json!({"tools": []})).is_ok());
    }
}
