//! The training engine of Kapel.
//!
//! - [`tape`]: reverse-mode differentiation on any backend, with values let
//!   go as soon as no gradient needs them, activation checkpointing, and
//!   gradients of table rows kept sparse;
//! - [`loss`]: cross-entropy, and its fused form over a large vocabulary;
//! - [`optim`]: AdamW (dense and by rows), Muon and Adafactor;
//! - [`chat`]: conversations made into tokens, with the assistant's marked;
//! - [`model`]: the decoders of `kapel-models` on a tape, to train them,
//!   with [`lora`] adapters if asked;
//! - [`preference`]: pairs of answers, and the loss of DPO;
//! - [`grpo`]: answers written by the model, scored by [`rewards`], and the
//!   loss of GRPO;
//! - [`profile`]: the time of each operation;
//! - [`trainer`]: the training loop, driven by a [`config`] file: batches of
//!   [`data`] in order, accumulation, clipping, a [`schedule`] of learning
//!   rates, evaluation, [`checkpoint`]s that resume exactly, and the
//!   model's FLOP utilization ([`flops`]).

pub mod chat;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod dataset;
pub mod flops;
pub mod grpo;
pub mod lora;
pub mod loss;
pub mod model;
pub mod optim;
pub mod preference;
pub mod prepare;
pub mod profile;
pub mod rewards;
pub mod schedule;
pub mod tape;
pub mod trainer;

pub use loss::{IGNORE, cross_entropy, linear_cross_entropy};
pub use tape::{Compute, Derivative, Grad, Gradients, Tape, Var};
