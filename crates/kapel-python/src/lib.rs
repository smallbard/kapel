//! The Python module of Kapel, `kapel`: load a model, generate, serve.
//!
//! ```python
//! import kapel
//!
//! model = kapel.Model("HuggingFaceTB/SmolLM2-135M-Instruct")
//! print(model.chat([{"role": "user", "content": "What is the capital of France?"}]))
//! for piece in model.generate("Once upon a time", max_tokens=64, stream=True):
//!     print(piece, end="", flush=True)
//! ```
//!
//! The work is done by [`kapel_runtime::Model`], with the interpreter's lock
//! released: other Python threads run while a model loads or generates. The
//! options are those of `kapel run` and `kapel serve`, in the same words.
//!
//! Built with maturin (`pyproject.toml`), for every Python from 3.9 (the
//! stable ABI). The CUDA backend comes with the feature `cuda`.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    kapel,
    KapelError,
    PyException,
    "A model that cannot be read, a pass of a model that failed."
);

fn failed(e: kapel_runtime::Error) -> PyErr {
    KapelError::new_err(e.to_string())
}

/// The options of loading, from the words of `kapel run`.
fn load_options(
    device: &str,
    dtype: Option<&str>,
    quantize: Option<&str>,
    kv_cache: Option<&str>,
    revision: &str,
) -> PyResult<kapel_runtime::LoadOptions> {
    let mut options = kapel_runtime::LoadOptions::parse(Some(device), dtype, quantize, kv_cache)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    options.revision = revision.to_string();
    Ok(options)
}

/// A model loaded on its device, with its tokenizer and chat template.
///
/// `name` is a model directory, a GGUF file, or a repository of the Hugging
/// Face Hub (`owner/name`, downloaded once into `~/.cache/kapel/hub`, or
/// `$KAPEL_CACHE`; a gated model needs `HF_TOKEN`).
#[pyclass(frozen, module = "kapel")]
struct Model {
    inner: Arc<kapel_runtime::Model>,
}

/// Pieces of text as they are generated, from a thread of their own.
/// Dropping the stream stops the generation.
#[pyclass(module = "kapel")]
struct Stream {
    pieces: Mutex<Receiver<Result<String, kapel_runtime::Error>>>,
}

#[pymethods]
impl Stream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let pieces = &self.pieces;
        let next = py.detach(|| pieces.lock().unwrap_or_else(|e| e.into_inner()).recv().ok());
        match next {
            Some(Ok(piece)) => Ok(Some(piece)),
            Some(Err(e)) => Err(failed(e)),
            None => Ok(None),
        }
    }
}

impl Model {
    /// The options of generation: the model's, overridden by those given.
    #[allow(clippy::too_many_arguments)]
    fn options(
        &self,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
    ) -> kapel_runtime::GenerateOptions {
        let mut o = kapel_runtime::GenerateOptions::from(self.inner.default_request());
        o.max_tokens = max_tokens.unwrap_or(o.max_tokens);
        o.temperature = temperature.unwrap_or(o.temperature);
        if let Some(k) = top_k {
            o.top_k = (k > 0).then_some(k);
        }
        o.top_p = top_p.unwrap_or(o.top_p);
        o.min_p = min_p.unwrap_or(o.min_p);
        o.repetition_penalty = repetition_penalty.unwrap_or(o.repetition_penalty);
        o.seed = seed.unwrap_or(o.seed);
        o.stop = stop.unwrap_or_default();
        o
    }

    /// The answer to `prompt`, whole or as a stream.
    fn run(
        &self,
        py: Python<'_>,
        prompt: Vec<u32>,
        options: kapel_runtime::GenerateOptions,
        stream: bool,
    ) -> PyResult<Py<PyAny>> {
        if !stream {
            let model = &self.inner;
            let outcome = py
                .detach(|| model.generate(&prompt, &options, |_| true))
                .map_err(failed)?;
            return Ok(outcome.text.into_pyobject(py)?.into_any().unbind());
        }
        // A few pieces ahead of the reader, no more.
        let (sender, receiver): (SyncSender<_>, _) = sync_channel(16);
        let model = Arc::clone(&self.inner);
        std::thread::Builder::new()
            .name("kapel-generate".into())
            .spawn(move || {
                let outcome = model.generate(&prompt, &options, |piece| {
                    // The reader is gone: stop.
                    sender.send(Ok(piece.to_string())).is_ok()
                });
                if let Err(e) = outcome {
                    let _ = sender.send(Err(e));
                }
            })
            .map_err(|e| KapelError::new_err(e.to_string()))?;
        let stream = Stream {
            pieces: Mutex::new(receiver),
        };
        Ok(Py::new(py, stream)?.into_any())
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (name, device="cpu", dtype=None, quantize=None, kv_cache=None, revision="main"))]
    fn new(
        py: Python<'_>,
        name: &str,
        device: &str,
        dtype: Option<&str>,
        quantize: Option<&str>,
        kv_cache: Option<&str>,
        revision: &str,
    ) -> PyResult<Self> {
        let options = load_options(device, dtype, quantize, kv_cache, revision)?;
        let model = py
            .detach(|| kapel_runtime::Model::load(name, &options))
            .map_err(failed)?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// The tokens of `text`, with the special tokens the tokenizer adds if
    /// `add_special_tokens`.
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn tokenize(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        self.inner
            .tokenize(text, add_special_tokens)
            .map_err(failed)
    }

    /// The text of `ids`, without the special tokens if
    /// `skip_special_tokens`.
    #[pyo3(signature = (ids, skip_special_tokens=false))]
    fn detokenize(&self, ids: Vec<u32>, skip_special_tokens: bool) -> String {
        self.inner.detokenize(&ids, skip_special_tokens)
    }

    /// Whether the model has a chat template.
    #[getter]
    fn has_chat_template(&self) -> bool {
        self.inner.has_chat_template()
    }

    /// The tokens of a conversation in the model's chat template, ending
    /// with the start of the assistant's answer.
    fn chat_prompt(&self, messages: Vec<HashMap<String, String>>) -> PyResult<Vec<u32>> {
        let pairs = messages
            .iter()
            .map(|m| match (m.get("role"), m.get("content")) {
                (Some(role), Some(content)) => Ok((role.as_str(), content.as_str())),
                _ => Err(PyValueError::new_err(
                    "each message needs a \"role\" and a \"content\"",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.chat_prompt(&pairs).map_err(failed)
    }

    /// What follows `prompt`, a text (with the special tokens the tokenizer
    /// adds) or a list of tokens: the whole text, or with `stream` an
    /// iterator of its pieces as they come. The options left to `None` are
    /// the model's (its `generation_config.json`), up to 512 tokens.
    #[pyo3(signature = (
        prompt, max_tokens=None, temperature=None, top_k=None, top_p=None, min_p=None,
        repetition_penalty=None, seed=None, stop=None, stream=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
        stream: bool,
    ) -> PyResult<Py<PyAny>> {
        let tokens = match prompt.extract::<String>() {
            Ok(text) => self.inner.tokenize(&text, true).map_err(failed)?,
            Err(_) => prompt
                .extract::<Vec<u32>>()
                .map_err(|_| PyValueError::new_err("the prompt is a text or a list of tokens"))?,
        };
        let options = self.options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
            stop,
        );
        self.run(py, tokens, options, stream)
    }

    /// The answer to a conversation, a list of `{"role": ..., "content":
    /// ...}`, in the model's chat template; the options as for `generate`.
    #[pyo3(signature = (
        messages, max_tokens=None, temperature=None, top_k=None, top_p=None, min_p=None,
        repetition_penalty=None, seed=None, stop=None, stream=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<HashMap<String, String>>,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
        stream: bool,
    ) -> PyResult<Py<PyAny>> {
        let tokens = self.chat_prompt(messages)?;
        let options = self.options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
            stop,
        );
        self.run(py, tokens, options, stream)
    }
}

/// Loads `name` and serves it behind the HTTP API compatible with OpenAI's
/// (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`,
/// `/metrics`), until Ctrl-C. The options are those of `kapel serve`.
#[pyfunction]
#[pyo3(signature = (
    name, host="127.0.0.1", port=8000, device="cpu", dtype=None, quantize=None, kv_cache=None,
    revision="main", served_model_name=None, max_model_len=None, cache_memory=None,
    gpu_memory_utilization=0.85, max_sequences=256, max_pass_tokens=2048, max_requests=1024,
    prefix_sharing=true, timeout=600.0,
))]
#[allow(clippy::too_many_arguments)]
fn serve(
    py: Python<'_>,
    name: &str,
    host: &str,
    port: u16,
    device: &str,
    dtype: Option<&str>,
    quantize: Option<&str>,
    kv_cache: Option<&str>,
    revision: &str,
    served_model_name: Option<String>,
    max_model_len: Option<usize>,
    cache_memory: Option<f64>,
    gpu_memory_utilization: f64,
    max_sequences: usize,
    max_pass_tokens: usize,
    max_requests: usize,
    prefix_sharing: bool,
    timeout: f64,
) -> PyResult<()> {
    let load = load_options(device, dtype, quantize, kv_cache, revision)?;
    let timeout = Duration::try_from_secs_f64(timeout)
        .map_err(|_| PyValueError::new_err("timeout is not a number of seconds"))?;
    let options = kapel_runtime::ServeOptions {
        host: host.to_string(),
        port,
        name: served_model_name,
        max_model_len,
        cache_memory,
        gpu_memory_utilization,
        max_sequences,
        max_pass_tokens,
        max_requests,
        prefix_sharing,
        timeout,
    };
    py.detach(|| kapel_runtime::Model::load(name, &load)?.serve(&options))
        .map_err(failed)
}

/// Kapel, an inference engine for small language models.
#[pymodule]
fn kapel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("KapelError", m.py().get_type::<KapelError>())?;
    m.add_class::<Model>()?;
    m.add_class::<Stream>()?;
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    Ok(())
}
