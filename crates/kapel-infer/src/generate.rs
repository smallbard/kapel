//! Generation: a prompt, then one token at a time.

use std::path::Path;

use kapel_core::{DType, Tensor};
use kapel_hub::{Checkpoint, read_json};
use kapel_models::{Cache, Model};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::{ChatTemplate, Message, Tokenizer};
use serde_json::Value;

use crate::sampling::{Sampler, Sampling};
use crate::text::TextStream;
use crate::{InferError, Result};

/// A model ready to generate on a backend: its weights, its tokenizer, its
/// chat template and the tokens that end its answers.
pub struct Pipeline<B> {
    backend: B,
    model: Model,
    tokenizer: Tokenizer,
    template: Option<ChatTemplate>,
    eos: Vec<u32>,
    sampling: Sampling,
}

/// How [`Pipeline::load_with`] prepares the weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// The type of the float weights, if they are to be converted.
    pub dtype: Option<DType>,
    /// The format to quantize the linear projections to, before anything
    /// else ([`Model::quantize`]).
    pub quantize: Option<DType>,
    /// Whether the output table is quantized too.
    pub quantize_output: bool,
}

/// `model` quantized, moved to `backend` and converted as `options` say.
fn prepare<B: Backend + Transfer>(
    model: Model,
    backend: &B,
    options: &LoadOptions,
) -> Result<Model> {
    let model = match options.quantize {
        Some(format) => model.quantize(format, options.quantize_output)?,
        None => model,
    };
    let mut model = model.upload(backend)?;
    if let Some(dtype) = options.dtype {
        model = model.cast(backend, dtype)?;
    }
    Ok(model)
}

/// What to generate.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// How each token is chosen.
    pub sampling: Sampling,
    /// The most tokens to generate.
    pub max_tokens: usize,
    /// Texts that end the answer when they appear in it; the answer stops
    /// before them.
    pub stop: Vec<String>,
}

impl Default for Request {
    /// Greedy, up to 256 tokens.
    fn default() -> Self {
        Self {
            sampling: Sampling::greedy(),
            max_tokens: 256,
            stop: Vec::new(),
        }
    }
}

/// A generated token, and the text it makes final (see [`Generation`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The token.
    pub token: u32,
    /// The text that no later token can change any more; possibly empty.
    pub text: String,
}

/// Why a generation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model gave a token that ends its answers.
    Eos,
    /// A stop sequence appeared.
    Stop(String),
    /// [`Request::max_tokens`] were generated.
    Length,
}

/// The tokens of `eos_token_id`: a number or a list of them.
fn token_ids(value: Option<&Value>) -> Vec<u32> {
    match value {
        Some(Value::Number(n)) => n
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .into_iter()
            .collect(),
        Some(Value::Array(all)) => all
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|n| u32::try_from(n).ok())
            .collect(),
        _ => Vec::new(),
    }
}

impl<B: Backend + Transfer> Pipeline<B> {
    /// Loads the model directory `dir` onto `backend`, its float weights in
    /// `dtype` if given (as stored otherwise). A file is read as GGUF
    /// ([`Pipeline::load_gguf`]).
    ///
    /// The tokens that end an answer come from `generation_config.json`, else
    /// `config.json`, else the tokenizer's `eos_token`; the default sampling
    /// from `generation_config.json` ([`Sampling::from_generation_config`]).
    ///
    /// # Errors
    ///
    /// If the model, its tokenizer or its configurations cannot be read, or
    /// the weights cannot be moved to the backend.
    pub fn load(dir: impl AsRef<Path>, backend: B, dtype: Option<DType>) -> Result<Self> {
        let options = LoadOptions {
            dtype,
            ..LoadOptions::default()
        };
        Self::load_with(dir, backend, &options)
    }

    /// [`Pipeline::load`], with the weights prepared as `options` say:
    /// quantized first, then moved to the backend, then converted.
    ///
    /// # Errors
    ///
    /// As [`Pipeline::load`], and if the weights cannot be quantized.
    pub fn load_with(dir: impl AsRef<Path>, backend: B, options: &LoadOptions) -> Result<Self> {
        let dir = dir.as_ref();
        if dir.is_file() {
            return Self::load_gguf_with(dir, backend, options);
        }
        let model = prepare(Model::load(&Checkpoint::open(dir)?)?, &backend, options)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let template = ChatTemplate::from_dir(dir)?;
        let generation = match dir.join("generation_config.json") {
            path if path.exists() => read_json(&path)?,
            _ => Value::Null,
        };
        let mut eos = token_ids(generation.get("eos_token_id"));
        if eos.is_empty() {
            eos = token_ids(read_json(&dir.join("config.json"))?.get("eos_token_id"));
        }
        if eos.is_empty()
            && let Ok(config) = read_json(&dir.join("tokenizer_config.json"))
            && let Some(id) = config
                .get("eos_token")
                .and_then(|t| t.as_str().or_else(|| t.get("content")?.as_str()))
                .and_then(|t| tokenizer.token_to_id(t))
        {
            eos.push(id);
        }
        Ok(Self {
            backend,
            model,
            tokenizer,
            template,
            eos,
            sampling: Sampling::from_generation_config(&generation),
        })
    }

    /// Loads a GGUF file onto `backend`: the model, its tokenizer, its chat
    /// template and its end tokens, all from the file. The weights stay in
    /// their types (quantized ones included), except those that `dtype`
    /// converts, as in [`Pipeline::load`]. A GGUF file has no
    /// `generation_config.json`: the default sampling is greedy.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, or holds an architecture or a tokenizer
    /// that Kapel does not read.
    pub fn load_gguf(path: impl AsRef<Path>, backend: B, dtype: Option<DType>) -> Result<Self> {
        let options = LoadOptions {
            dtype,
            ..LoadOptions::default()
        };
        Self::load_gguf_with(path, backend, &options)
    }

    fn load_gguf_with(path: impl AsRef<Path>, backend: B, options: &LoadOptions) -> Result<Self> {
        let gguf = kapel_hub::Gguf::open(path)?;
        let model = prepare(Model::from_gguf(&gguf)?, &backend, options)?;
        let vocabulary = crate::gguf::vocabulary(&gguf)?;
        Ok(Self {
            backend,
            model,
            tokenizer: vocabulary.tokenizer,
            template: vocabulary.template,
            eos: vocabulary.eos,
            sampling: Sampling::default(),
        })
    }

    /// The model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tokens that end an answer.
    pub fn eos(&self) -> &[u32] {
        &self.eos
    }

    /// The sampling the model's authors chose (`generation_config.json`).
    pub fn default_sampling(&self) -> &Sampling {
        &self.sampling
    }

    /// Whether the model has a chat template.
    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    /// The tokens of a plain prompt, with the special tokens the tokenizer adds
    /// (such as a beginning of sequence).
    ///
    /// # Errors
    ///
    /// If the tokenizer fails.
    pub fn prompt(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?)
    }

    /// The tokens of a conversation in the model's chat template, ending with
    /// the start of the assistant's answer, as `transformers`'
    /// `apply_chat_template` gives them.
    ///
    /// # Errors
    ///
    /// If the model has no chat template, or the template rejects the
    /// conversation.
    pub fn chat_prompt(&self, messages: &[Message]) -> Result<Vec<u32>> {
        let template = self.template.as_ref().ok_or(InferError::NoChatTemplate)?;
        let text = template.render(messages, true)?;
        // The template writes the special tokens itself.
        Ok(self.tokenizer.encode(&text, false)?)
    }

    /// The tokens that follow `prompt`, one at a time, as an iterator.
    ///
    /// # Errors
    ///
    /// If `prompt` is empty.
    pub fn generate(&self, prompt: &[u32], request: &Request) -> Result<Generation<'_, B>> {
        if prompt.is_empty() {
            return Err(InferError::EmptyPrompt);
        }
        Ok(Generation {
            pipeline: self,
            cache: Cache::new(),
            sequence: prompt.to_vec(),
            prompt_len: prompt.len(),
            sampler: Sampler::new(request.sampling.clone()),
            text: TextStream::new(request.stop.clone()),
            max_tokens: request.max_tokens,
            finish: None,
        })
    }

    /// The logits of the last of `ids`, after `cache`, in host memory.
    fn logits(&self, ids: &[u32], cache: &mut Cache) -> Result<Vec<f64>> {
        let logits: Tensor = self.model.step(&self.backend, ids, cache)?;
        Ok(self.backend.download(&logits)?.to_f64_vec()?)
    }
}

/// The tokens a [`Pipeline`] generates after a prompt, as they come.
///
/// Each item is a generated token and the text it makes final: the text of the
/// whole answer is the concatenation of the pieces' texts. A token that ends
/// the answer (end of sequence, stop sequence, length) is the last item, with
/// the rest of the text.
pub struct Generation<'p, B> {
    pipeline: &'p Pipeline<B>,
    cache: Cache,
    /// The prompt and the tokens generated so far.
    sequence: Vec<u32>,
    prompt_len: usize,
    sampler: Sampler,
    text: TextStream,
    max_tokens: usize,
    finish: Option<Finish>,
}

impl<B: Backend + Transfer> Generation<'_, B> {
    /// The tokens generated so far.
    pub fn tokens(&self) -> &[u32] {
        &self.sequence[self.prompt_len..]
    }

    /// Why the generation ended, once it has.
    pub fn finish(&self) -> Option<&Finish> {
        self.finish.as_ref()
    }

    fn next_piece(&mut self) -> Result<Piece> {
        let p = self.pipeline;
        let fresh = if self.cache.is_empty() {
            &self.sequence[..]
        } else {
            &self.sequence[self.sequence.len() - 1..]
        };
        let mut logits = p.logits(fresh, &mut self.cache)?;
        let token = self.sampler.choose(&mut logits, &self.sequence);
        self.sequence.push(token);
        if p.eos.contains(&token) {
            self.finish = Some(Finish::Eos);
            let text = self.text.finish(&p.tokenizer);
            return Ok(Piece { token, text });
        }
        let added = self.text.push(&p.tokenizer, token);
        let mut text = added.text;
        if let Some(stop) = added.stopped {
            self.finish = Some(Finish::Stop(stop));
        } else if self.tokens().len() >= self.max_tokens {
            self.finish = Some(Finish::Length);
            text.push_str(&self.text.finish(&p.tokenizer));
        }
        Ok(Piece { token, text })
    }
}

impl<B: Backend + Transfer> Iterator for Generation<'_, B> {
    type Item = Result<Piece>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finish.is_some() || self.max_tokens == 0 {
            return None;
        }
        let piece = self.next_piece();
        if piece.is_err() {
            // The cache may be half filled: nothing follows an error.
            self.finish = Some(Finish::Length);
        }
        Some(piece)
    }
}
