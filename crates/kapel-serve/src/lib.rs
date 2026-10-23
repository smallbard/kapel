//! An inference server compatible with OpenAI's API.
//!
//! [`serve`] puts a [`kapel_infer::Pipeline`] behind HTTP:
//!
//! - `POST /v1/chat/completions` and `POST /v1/completions`, whole answers
//!   or streams of server-sent events (`"stream": true`), `n` answers per
//!   request, stop sequences, the sampling settings of OpenAI and vLLM, and
//!   answers held to a form (`response_format`, vLLM's `guided_*`);
//! - `GET /v1/models`, the model served;
//! - `GET /health`, 200 while the engine runs;
//! - `GET /metrics`, Prometheus' text format: requests by outcome, tokens,
//!   latency of the first token and between tokens, use of the cache.
//!
//! The requests are served together by a [`kapel_infer::Engine`] on a thread
//! of its own (continuous batching): each pass of the model computes a token
//! of every answer in progress, and pieces of new prompts. A client that
//! closes its connection cancels its request. Requests beyond the limits
//! (body size, positions of the model, answers in flight) are refused with
//! OpenAI's errors; a request that takes too long ends with a timeout.
//!
//! Every request leaves one line of JSON on the standard error: its
//! outcome, its tokens, its latencies.
//!
//! ```no_run
//! use kapel_infer::{EngineOptions, Pipeline};
//! use kapel_serve::{ServeOptions, serve};
//!
//! let pipeline = Pipeline::load("SmolLM2-135M-Instruct", kapel_cpu::Cpu::new(), None)?;
//! let engine = EngineOptions {
//!     pages: Some(4096),
//!     prefix_sharing: true,
//!     ..EngineOptions::default()
//! };
//! serve(pipeline, engine, ServeOptions::new("smollm2", 8192))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod api;
mod log;
mod metrics;
mod server;
mod worker;

pub use server::{ServeOptions, Server, serve};
