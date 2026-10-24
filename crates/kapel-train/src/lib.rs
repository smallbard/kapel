//! The training engine of Kapel.
//!
//! - [`tape`]: reverse-mode differentiation on any backend, with values let
//!   go as soon as no gradient needs them, activation checkpointing, and
//!   gradients of table rows kept sparse;
//! - [`loss`]: cross-entropy, and its fused form over a large vocabulary;
//! - [`optim`]: the optimizers, dense and by rows;
//! - [`model`]: the decoders of `kapel-models` on a tape, to train them;
//! - [`profile`]: the time of each operation.

pub mod loss;
pub mod model;
pub mod optim;
pub mod profile;
pub mod tape;

pub use loss::{IGNORE, cross_entropy, linear_cross_entropy};
pub use tape::{Compute, Derivative, Grad, Gradients, Tape, Var};
