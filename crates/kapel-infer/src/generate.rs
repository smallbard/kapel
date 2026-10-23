//! Generation: a prompt, then one token at a time.

use std::path::Path;
use std::sync::OnceLock;

use kapel_core::{DType, Tensor};
use kapel_hub::{Checkpoint, read_json};
use kapel_models::{Cache, Model, RecordedStep};
use kapel_ops::{ArgMax, Backend, Transfer};
use kapel_tokenizers::{ChatTemplate, Message, Tokenizer};
use serde_json::Value;

use crate::constraint::{Constraint, Grammars, Guide};
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
    /// The type of the caches of generation; `None` for the keys'.
    cache_dtype: Option<DType>,
    /// What compiles constraints, made at the first one.
    grammars: OnceLock<std::result::Result<Grammars, String>>,
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
    /// Whether the projections that read the same input are stacked into one
    /// weight, last ([`Model::fused`]): what a card computes faster.
    pub fuse: bool,
    /// The type of the caches of keys and values (`f16`, or `f8e4m3` for half
    /// the memory); `None` for that of the keys.
    pub cache_dtype: Option<DType>,
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
    if options.fuse {
        model = model.into_fused(backend)?;
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
    /// Whether the tokens that end an answer are taken as any other, so that
    /// the answer goes on to `max_tokens`: answers of known lengths, to
    /// measure speed.
    pub ignore_eos: bool,
    /// The form the answer must take, if any.
    pub constraint: Option<Constraint>,
}

impl Default for Request {
    /// Greedy, up to 256 tokens.
    fn default() -> Self {
        Self {
            sampling: Sampling::greedy(),
            max_tokens: 256,
            stop: Vec::new(),
            ignore_eos: false,
            constraint: None,
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
    /// The request failed on its own (its constraint's parser, say); the
    /// others go on.
    Error(String),
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
            cache_dtype: options.cache_dtype,
            grammars: OnceLock::new(),
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
            cache_dtype: options.cache_dtype,
            grammars: OnceLock::new(),
        })
    }

    /// The model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The backend the model runs on.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// An empty cache of generation of the type [`LoadOptions`] chose.
    pub fn cache(&self) -> kapel_models::KvCache {
        match self.cache_dtype {
            Some(dtype) => kapel_models::KvCache::new().with_dtype(dtype),
            None => kapel_models::KvCache::new(),
        }
    }

    /// The type of the caches of generation; `None` for that of the keys.
    pub fn cache_dtype(&self) -> Option<DType> {
        self.cache_dtype
    }

    /// The guide of one answer held to `constraint`; the model's tokens are
    /// read into llguidance at the first call.
    pub(crate) fn guide(&self, constraint: &Constraint) -> Result<Guide> {
        let grammars = self.grammars.get_or_init(|| {
            Grammars::new(&self.tokenizer, self.model.config().vocab, &self.eos)
                .map_err(|e| e.to_string())
        });
        match grammars {
            Ok(grammars) => grammars.guide(constraint),
            Err(e) => Err(InferError::Constraint(e.clone())),
        }
    }

    /// Checks that `constraint` compiles, so that a request can be refused
    /// before it is queued.
    ///
    /// # Errors
    ///
    /// If it does not: an invalid schema, regular expression or grammar.
    pub fn check_constraint(&self, constraint: &Constraint) -> Result<()> {
        self.guide(constraint).map(drop)
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
    /// If `prompt` is empty, or the request's constraint does not compile.
    pub fn generate(&self, prompt: &[u32], request: &Request) -> Result<Generation<'_, B>> {
        if prompt.is_empty() {
            return Err(InferError::EmptyPrompt);
        }
        let guide = request
            .constraint
            .as_ref()
            .map(|c| self.guide(c))
            .transpose()?;
        Ok(Generation {
            guide,
            pipeline: self,
            cache: match self.cache_dtype {
                Some(dtype) => Cache::with_dtype(dtype),
                None => Cache::new(),
            },
            sequence: prompt.to_vec(),
            prompt_len: prompt.len(),
            sampler: Sampler::new(request.sampling.clone()),
            text: TextStream::new(request.stop.clone()),
            max_tokens: request.max_tokens,
            ignore_eos: request.ignore_eos,
            finish: None,
            recorded: None,
        })
    }

    /// The most likely token of `logits` `[1, vocab]`, found on the backend.
    fn most_likely(&self, logits: &Tensor) -> Result<u32> {
        let best = self.backend.forward(
            &ArgMax {
                dim: 1,
                keepdim: false,
            },
            &[logits],
        )?;
        Ok(self.backend.download(&best)?.to_vec::<u32>()?[0])
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
    ignore_eos: bool,
    finish: Option<Finish>,
    /// The pass of the next tokens, recorded once.
    recorded: Option<RecordedStep>,
    /// The parser of the answer's form, if it has one.
    guide: Option<Guide>,
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
        // A generated token goes through a recorded pass when the backend
        // records them (a CUDA graph on a card); the prompt as it comes.
        let recorded = match fresh {
            [token] if !self.cache.is_empty() => {
                self.cache
                    .recorded_step(&p.model, &p.backend, &mut self.recorded, *token)?
            }
            _ => None,
        };
        let logits = match recorded {
            Some(logits) => logits,
            None => p.model.step(&p.backend, fresh, &mut self.cache)?,
        };
        let token = if self.sampler.is_argmax() && self.guide.is_none() {
            p.most_likely(&logits)?
        } else {
            let mut logits = p.backend.download(&logits)?.to_f64_vec()?;
            if let Some(guide) = &mut self.guide {
                guide.mask(&mut logits)?;
            }
            self.sampler
                .choose(&mut logits, &self.sequence, self.prompt_len)
        };
        if let Some(guide) = &mut self.guide {
            guide.advance(token)?;
        }
        self.sequence.push(token);
        if !self.ignore_eos && p.eos.contains(&token) {
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
