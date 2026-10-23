//! A model loaded on its device, whichever it is: what Kapel's bindings to
//! other languages share (`kapel-capi` for C, `kapel-python` for Python).
//!
//! A [`kapel_infer::Pipeline`] is generic over its backend. The bindings
//! cannot be: a program written in C or Python chooses the device when it
//! runs. [`Model`] holds a pipeline on the processor or on a card, and
//! answers the same calls for both. Its options are those of `kapel run`, in
//! the same words: `--device cuda:1` is `"cuda:1"`, `--quantize int4/64` is
//! `"int4/64"` ([`LoadOptions::parse`]).
//!
//! ```no_run
//! use kapel_runtime::{GenerateOptions, LoadOptions, Model};
//!
//! let model = Model::load("HuggingFaceTB/SmolLM2-135M-Instruct", &LoadOptions::default())?;
//! let prompt = model.chat_prompt(&[("user", "What is the capital of France?")])?;
//! let options = GenerateOptions::from(model.default_request());
//! let outcome = model.generate(&prompt, &options, |piece| {
//!     print!("{piece}");
//!     true
//! })?;
//! println!("\n{} tokens", outcome.completion_tokens);
//! # Ok::<(), kapel_runtime::Error>(())
//! ```
//!
//! The CUDA backend comes with the feature `cuda`.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use kapel_core::{DType, QuantFormat};
use kapel_infer::{EngineOptions, Finish as InferFinish, Pipeline, Request};
use kapel_ops::{Backend, Transfer};
use kapel_tokenizers::{Message, Tokenizer};

/// What failed, in words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<kapel_infer::InferError> for Error {
    fn from(e: kapel_infer::InferError) -> Self {
        Self(e.to_string())
    }
}

/// The result of the calls of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Where to compute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// The processor.
    #[default]
    Cpu,
    /// The NVIDIA card of this ordinal.
    Cuda(u32),
}

/// How to load a model: the options of `kapel run` that do so.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    /// Where to compute.
    pub device: Device,
    /// The type of the float weights; `None` for `f32` on the processor,
    /// `bf16` on a card.
    pub dtype: Option<DType>,
    /// The revision of a repository of the Hub.
    pub revision: String,
    /// The format to quantize the linear projections to, if any.
    pub quantize: Option<DType>,
    /// Whether the output table is quantized too.
    pub quantize_output: bool,
    /// The type of the cache of keys and values; `None` for that of the keys.
    pub kv_cache: Option<DType>,
    /// On the processor, the activations of the products by quantized
    /// weights quantized to bytes, as llama.cpp does.
    pub byte_activations: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: None,
            revision: "main".into(),
            quantize: None,
            quantize_output: false,
            kv_cache: None,
            byte_activations: false,
        }
    }
}

impl LoadOptions {
    /// Options from the words of `kapel run`, each `None` for its default:
    /// `device` is `cpu`, `cuda` or `cuda:N`; `dtype` is `f32`, `bf16` or
    /// `f16`; `quantize` is `int8`, `int4` (groups of 128 and 64, or
    /// `int8/G`, `int4/G`), `q8_0` or `q4_0`; `kv_cache` is `f32`, `bf16`,
    /// `f16`, `f8e4m3` or `f8e5m2`.
    ///
    /// # Errors
    ///
    /// If a word is none of those.
    pub fn parse(
        device: Option<&str>,
        dtype: Option<&str>,
        quantize: Option<&str>,
        kv_cache: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            device: device.map(parse_device).transpose()?.unwrap_or_default(),
            dtype: dtype.map(parse_dtype).transpose()?,
            quantize: quantize.map(parse_format).transpose()?,
            kv_cache: kv_cache.map(parse_kv_cache).transpose()?,
            ..Self::default()
        })
    }
}

/// `cpu`, `cuda` or `cuda:N`.
fn parse_device(value: &str) -> Result<Device> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        other => other
            .strip_prefix("cuda:")
            .and_then(|n| n.parse().ok())
            .map(Device::Cuda)
            .ok_or_else(|| Error::new(format!("device {value:?} is not cpu or cuda[:N]"))),
    }
}

/// `f32`, `bf16` or `f16`.
fn parse_dtype(value: &str) -> Result<DType> {
    match value {
        "f32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        _ => Err(Error::new(format!(
            "dtype {value:?} is not f32, bf16 or f16"
        ))),
    }
}

/// `int8[/G]`, `int4[/G]`, `q8_0` or `q4_0`.
fn parse_format(value: &str) -> Result<DType> {
    let bad = || {
        Error::new(format!(
            "quantize {value:?} is not int8[/G], int4[/G], q8_0 or q4_0"
        ))
    };
    let (name, group) = match value.split_once('/') {
        Some((name, group)) => (name, Some(group.parse::<u16>().map_err(|_| bad())?)),
        None => (value, None),
    };
    match name {
        "int8" => DType::int8(group.unwrap_or(128)).map_err(|e| Error::new(e.to_string())),
        "int4" => DType::int4(group.unwrap_or(64)).map_err(|e| Error::new(e.to_string())),
        "q8_0" if group.is_none() => Ok(DType::Quantized(QuantFormat::Q8_0)),
        "q4_0" if group.is_none() => Ok(DType::Quantized(QuantFormat::Q4_0)),
        _ => Err(bad()),
    }
}

/// `f32`, `bf16`, `f16`, `f8e4m3` or `f8e5m2`.
fn parse_kv_cache(value: &str) -> Result<DType> {
    match value {
        "f32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f8e4m3" => Ok(DType::F8E4M3),
        "f8e5m2" => Ok(DType::F8E5M2),
        _ => Err(Error::new(format!(
            "kv cache {value:?} is not f32, bf16, f16, f8e4m3 or f8e5m2"
        ))),
    }
}

/// The model directory or GGUF file: `model` itself, or its snapshot from
/// the Hub (downloaded once into `~/.cache/kapel/hub`, or `$KAPEL_CACHE`).
fn directory(model: &str, revision: &str) -> Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.is_dir() || path.is_file() {
        return Ok(path);
    }
    kapel_hub::download::snapshot(model, revision, &kapel_hub::download::default_cache())
        .map_err(|e| Error::new(e.to_string()))
}

/// How the pipeline prepares the weights on its device.
fn pipeline_options(options: &LoadOptions) -> kapel_infer::LoadOptions {
    let cuda = matches!(options.device, Device::Cuda(_));
    kapel_infer::LoadOptions {
        dtype: options
            .dtype
            .or(Some(if cuda { DType::BF16 } else { DType::F32 })),
        quantize: options.quantize,
        quantize_output: options.quantize_output,
        fuse: cuda,
        cache_dtype: options.kv_cache,
    }
}

/// The pipeline, on its backend.
enum Inner {
    Cpu(Pipeline<kapel_cpu::Cpu>),
    #[cfg(feature = "cuda")]
    Cuda(Pipeline<kapel_cuda::Cuda>, kapel_cuda::Cuda),
}

/// `$body` with `$p` the pipeline, whatever its backend.
macro_rules! dispatch {
    ($inner:expr, $p:ident => $body:expr) => {
        match $inner {
            Inner::Cpu($p) => $body,
            #[cfg(feature = "cuda")]
            Inner::Cuda($p, _) => $body,
        }
    };
}

/// A model loaded on its device, with its tokenizer and chat template.
///
/// It can be shared between threads: each generation has its own cache.
pub struct Model {
    inner: Inner,
    name: String,
    options: LoadOptions,
}

/// Why a generation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model gave a token that ends its answers.
    Eos,
    /// A stop sequence appeared; the answer ends before it.
    Stop(String),
    /// The most tokens asked for were generated.
    Length,
    /// The callback asked to stop.
    Cancelled,
}

/// A generation that ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The whole answer.
    pub text: String,
    /// Why it ended.
    pub finish: Finish,
    /// The tokens of the prompt.
    pub prompt_tokens: usize,
    /// The tokens generated.
    pub completion_tokens: usize,
}

/// What to generate: the model's sampling settings, and a few more.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateOptions {
    /// The most tokens to generate.
    pub max_tokens: usize,
    /// The divisor of the logits; 0 for greedy decoding.
    pub temperature: f64,
    /// Only the `k` most likely tokens; `None` for all.
    pub top_k: Option<usize>,
    /// Only the most likely tokens whose probabilities add up to `top_p`.
    pub top_p: f64,
    /// Only the tokens at least `min_p` times as likely as the most likely.
    pub min_p: f64,
    /// The penalty of the tokens already in the sequence; 1 for none.
    pub repetition_penalty: f64,
    /// The seed of the draws.
    pub seed: u64,
    /// Texts that end the answer; it stops before them.
    pub stop: Vec<String>,
}

impl From<Request> for GenerateOptions {
    fn from(request: Request) -> Self {
        let s = request.sampling;
        Self {
            max_tokens: request.max_tokens,
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            repetition_penalty: s.repetition_penalty,
            seed: s.seed,
            stop: request.stop,
        }
    }
}

/// How to serve a model: the options of `kapel serve`.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeOptions {
    /// The address to listen on.
    pub host: String,
    /// The port; 0 for any free one.
    pub port: u16,
    /// The model's name in the API; `None` for one from the name given.
    pub name: Option<String>,
    /// The most positions of a sequence; `None` for the model's, at most what
    /// the cache holds.
    pub max_model_len: Option<usize>,
    /// The memory of the cache, in GiB; `None` for 4 on the processor, and
    /// what the weights leave of `gpu_memory_utilization` on a card.
    pub cache_memory: Option<f64>,
    /// On a card, the share of its memory to use.
    pub gpu_memory_utilization: f64,
    /// The most answers computed at once.
    pub max_sequences: usize,
    /// The most tokens of one pass.
    pub max_pass_tokens: usize,
    /// The most answers in flight; beyond, requests get 429.
    pub max_requests: usize,
    /// Whether common prompt beginnings share their cache.
    pub prefix_sharing: bool,
    /// The longest a request may take.
    pub timeout: Duration,
}

impl Default for ServeOptions {
    /// The defaults of `kapel serve`.
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8000,
            name: None,
            max_model_len: None,
            cache_memory: None,
            gpu_memory_utilization: 0.85,
            max_sequences: 256,
            max_pass_tokens: 2048,
            max_requests: 1024,
            prefix_sharing: true,
            timeout: Duration::from_secs(600),
        }
    }
}

impl Model {
    /// Loads `model`: a model directory, a GGUF file, or a repository of the
    /// Hub (`owner/name`, downloaded once; a gated model needs `HF_TOKEN`).
    ///
    /// # Errors
    ///
    /// If it cannot be found or read, or its device is not available (a card
    /// without the feature `cuda`).
    pub fn load(model: &str, options: &LoadOptions) -> Result<Self> {
        let dir = directory(model, &options.revision)?;
        let prepare = pipeline_options(options);
        let inner = match options.device {
            Device::Cpu => {
                let cpu = kapel_cpu::Cpu::new().with_byte_activations(options.byte_activations);
                Inner::Cpu(Pipeline::load_with(&dir, cpu, &prepare)?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => {
                let cuda = kapel_cuda::Cuda::new(ordinal).map_err(|e| Error::new(e.to_string()))?;
                Inner::Cuda(Pipeline::load_with(&dir, cuda.clone(), &prepare)?, cuda)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                return Err(Error::new(
                    "this build of Kapel has no CUDA backend (the feature `cuda`)",
                ));
            }
        };
        Ok(Self {
            inner,
            name: model.to_string(),
            options: options.clone(),
        })
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        dispatch!(&self.inner, p => p.tokenizer())
    }

    /// The tokens of `text`, with the special tokens the tokenizer adds (a
    /// beginning of sequence, say) if `add_special_tokens`.
    ///
    /// # Errors
    ///
    /// If the tokenizer fails.
    pub fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer()
            .encode(text, add_special_tokens)
            .map_err(|e| Error::new(e.to_string()))
    }

    /// The text of `ids`, without the special tokens if `skip_special_tokens`.
    pub fn detokenize(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        self.tokenizer().decode(ids, skip_special_tokens)
    }

    /// Whether the model has a chat template.
    pub fn has_chat_template(&self) -> bool {
        dispatch!(&self.inner, p => p.has_chat_template())
    }

    /// The tokens of a conversation, `(role, content)` pairs, in the model's
    /// chat template, ending with the start of the assistant's answer.
    ///
    /// # Errors
    ///
    /// If the model has no chat template, or the template rejects the
    /// conversation.
    pub fn chat_prompt<R: AsRef<str>, C: AsRef<str>>(
        &self,
        messages: &[(R, C)],
    ) -> Result<Vec<u32>> {
        let messages: Vec<Message> = messages
            .iter()
            .map(|(role, content)| Message::new(role.as_ref(), content.as_ref()))
            .collect();
        Ok(dispatch!(&self.inner, p => p.chat_prompt(&messages))?)
    }

    /// What the model's authors chose to generate with
    /// (`generation_config.json`), up to 512 tokens, as `kapel run` does.
    pub fn default_request(&self) -> Request {
        Request {
            sampling: dispatch!(&self.inner, p => p.default_sampling().clone()),
            max_tokens: 512,
            ..Request::default()
        }
    }

    /// Generates what follows `prompt`, calling `on_piece` with each piece of
    /// text as it becomes final; generation stops when it returns `false`.
    ///
    /// # Errors
    ///
    /// If `prompt` is empty, or a pass of the model fails.
    pub fn generate(
        &self,
        prompt: &[u32],
        options: &GenerateOptions,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<Outcome> {
        let mut request = self.default_request();
        request.max_tokens = options.max_tokens;
        request.stop.clone_from(&options.stop);
        let s = &mut request.sampling;
        s.temperature = options.temperature;
        s.top_k = options.top_k.filter(|&k| k > 0);
        s.top_p = options.top_p;
        s.min_p = options.min_p;
        s.repetition_penalty = options.repetition_penalty;
        s.seed = options.seed;
        dispatch!(&self.inner, p => run(p, prompt, &request, on_piece))
    }

    /// Serves the model behind the HTTP API compatible with OpenAI's, as
    /// `kapel serve` does, until Ctrl-C.
    ///
    /// # Errors
    ///
    /// If the cache does not fit, the address cannot be bound, or serving
    /// fails.
    pub fn serve(self, options: &ServeOptions) -> Result<()> {
        let name = options
            .name
            .clone()
            .unwrap_or_else(|| model_name(&self.name));
        let load = self.options;
        match self.inner {
            Inner::Cpu(pipeline) => {
                let memory = options.cache_memory.unwrap_or(4.0);
                let pages =
                    (memory * f64::from(1u32 << 30)) as usize / page_bytes(&pipeline, &load);
                start(pipeline, pages, name, options)
            }
            #[cfg(feature = "cuda")]
            Inner::Cuda(pipeline, cuda) => {
                let page = page_bytes(&pipeline, &load);
                let pages = match options.cache_memory {
                    Some(memory) => (memory * f64::from(1u32 << 30)) as usize / page,
                    None => card_pages(&pipeline, &cuda, page, options)?,
                };
                start(pipeline, pages, name, options)
            }
        }
    }
}

/// Generates with `pipeline`.
fn run<B: Backend + Transfer>(
    pipeline: &Pipeline<B>,
    prompt: &[u32],
    request: &Request,
    mut on_piece: impl FnMut(&str) -> bool,
) -> Result<Outcome> {
    let mut generation = pipeline.generate(prompt, request)?;
    let mut text = String::new();
    let mut cancelled = false;
    for piece in &mut generation {
        let piece = piece?;
        if piece.text.is_empty() {
            continue;
        }
        text.push_str(&piece.text);
        if !on_piece(&piece.text) {
            cancelled = true;
            break;
        }
    }
    let finish = match generation.finish() {
        _ if cancelled => Finish::Cancelled,
        Some(InferFinish::Eos) => Finish::Eos,
        Some(InferFinish::Stop(s)) => Finish::Stop(s.clone()),
        Some(InferFinish::Length) | None => Finish::Length,
        Some(InferFinish::Error(e)) => return Err(Error::new(e.clone())),
    };
    Ok(Outcome {
        text,
        finish,
        prompt_tokens: prompt.len(),
        completion_tokens: generation.tokens().len(),
    })
}

/// The name of a model in the API: its Hub repository, from the name given
/// or from the path of a snapshot of the Hub's cache, else the last part of
/// its path.
fn model_name(model: &str) -> String {
    let path = std::path::Path::new(model);
    for part in path.components() {
        if let Some(repo) = part
            .as_os_str()
            .to_str()
            .and_then(|p| p.strip_prefix("models--"))
        {
            return repo.replacen("--", "/", 1);
        }
    }
    if !path.exists() {
        return model.to_string();
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(model)
        .to_string()
}

/// The bytes of one page of the cache of `pipeline`'s model.
fn page_bytes<B: Backend + Transfer>(pipeline: &Pipeline<B>, load: &LoadOptions) -> usize {
    let config = pipeline.model().config();
    let dtype = pipeline
        .cache_dtype()
        .or(pipeline_options(load).dtype)
        .unwrap_or(DType::F32);
    let element = dtype.element_size().unwrap_or(4);
    config.layers * 2 * config.kv_heads * config.head_dim * element * kapel_models::PAGE
}

/// The pages of cache that fit on the card besides the weights and what the
/// passes need.
#[cfg(feature = "cuda")]
fn card_pages(
    pipeline: &Pipeline<kapel_cuda::Cuda>,
    cuda: &kapel_cuda::Cuda,
    page: usize,
    options: &ServeOptions,
) -> Result<usize> {
    cuda.clear_pool();
    let (free, total) = cuda.memory().map_err(|e| Error::new(e.to_string()))?;
    let config = pipeline.model().config();
    // Activations of the widest pass, logits, and the recorded passes.
    let activations = options.max_pass_tokens * (config.hidden * 8 + config.intermediate * 4) * 4
        + options.max_sequences * config.vocab * 8
        + (5usize << 29);
    let budget = (total as f64 * options.gpu_memory_utilization) as usize;
    let left = budget.saturating_sub(total - free + activations);
    if left < 64 * page {
        return Err(Error::new(format!(
            "the card has room for {} pages of cache; raise the share of its memory \
             or give the memory of the cache",
            left / page
        )));
    }
    Ok(left / page)
}

/// Serves `pipeline` with a cache of `pages` pages, until Ctrl-C.
fn start<B>(pipeline: Pipeline<B>, pages: usize, name: String, options: &ServeOptions) -> Result<()>
where
    B: Backend + Transfer + Send + Sync + 'static,
{
    let held = pages * kapel_models::PAGE;
    let max_model_len = options.max_model_len.unwrap_or_else(|| {
        pipeline
            .model()
            .config()
            .max_positions
            .unwrap_or(held)
            .min(held)
    });
    let address = format!("{}:{}", options.host, options.port)
        .parse()
        .map_err(|e| Error::new(format!("host, port: {e}")))?;
    let mut serve = kapel_serve::ServeOptions::new(name, max_model_len);
    serve.address = address;
    serve.max_requests = options.max_requests;
    serve.request_timeout = options.timeout;
    let engine = EngineOptions {
        pages: Some(pages),
        prefix_sharing: options.prefix_sharing,
        max_pass_tokens: options.max_pass_tokens,
        max_sequences: options.max_sequences,
        reserve: 0.0,
    };
    kapel_serve::serve(pipeline, engine, serve).map_err(|e| Error::new(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_are_read_as_kapel_run_reads_them() {
        let o = LoadOptions::parse(
            Some("cuda:1"),
            Some("bf16"),
            Some("int4/32"),
            Some("f8e4m3"),
        )
        .unwrap();
        assert_eq!(o.device, Device::Cuda(1));
        assert_eq!(o.dtype, Some(DType::BF16));
        assert_eq!(o.quantize, Some(DType::int4(32).unwrap()));
        assert_eq!(o.kv_cache, Some(DType::F8E4M3));
        assert_eq!(
            LoadOptions::parse(None, None, None, None).unwrap(),
            LoadOptions::default()
        );
        assert_eq!(parse_format("int8").unwrap(), DType::int8(128).unwrap());
        assert!(parse_device("gpu").is_err());
        assert!(parse_format("q4_0/32").is_err());
        assert!(parse_dtype("f64").is_err());
        assert!(parse_kv_cache("int8").is_err());
    }

    #[test]
    fn names_in_the_api_come_from_the_hub() {
        assert_eq!(
            model_name("HuggingFaceTB/SmolLM2-135M"),
            "HuggingFaceTB/SmolLM2-135M"
        );
        assert_eq!(
            model_name("/c/hub/models--Qwen--Qwen3-0.6B/snapshots/abc"),
            "Qwen/Qwen3-0.6B"
        );
    }
}
