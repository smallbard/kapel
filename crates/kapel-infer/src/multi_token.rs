//! The proposer of speculative decoding that a model carries itself: its
//! multi-token prediction layer (MiMo).
//!
//! The layer reads, at position `i`, the model's final hidden state there and
//! the token at `i + 1`, and predicts the token at `i + 2`. Each pass of the
//! model reads the last accepted token and the proposal after it: it checks
//! the proposal, and gives the hidden states the layer needs for the next
//! one. The proposer costs one block and the output head, and sees all the
//! model computed (Gloeckle et al., 2024; DeepSeek-V3).

use kapel_core::Tensor;
use kapel_models::Cache;
use kapel_ops::{Backend, Transfer};

use crate::generate::Pipeline;
use crate::sampling::{Choice, Sampler, Sampling};
use crate::{InferError, Result};

/// The multi-token layer's side of a speculative generation.
pub(crate) struct MultiToken {
    /// The layer's own keys and values, one per position it has read.
    cache: Cache,
    sampler: Sampler,
    /// The proposal for the token after the sequence, and what it was drawn
    /// from.
    proposal: Option<(u32, Choice)>,
}

impl MultiToken {
    pub(crate) fn new<B: Backend + Transfer>(
        pipeline: &Pipeline<B>,
        sampling: &Sampling,
    ) -> Result<Self> {
        if pipeline.model().config().multi_token_layers == 0 {
            return Err(InferError::Unsupported(
                "the model has no multi-token prediction layer".into(),
            ));
        }
        Ok(Self {
            cache: pipeline.new_cache(),
            sampler: Sampler::new(Sampling {
                seed: sampling.seed ^ 0x6d74_7000_0000_0001,
                ..sampling.clone()
            }),
            proposal: None,
        })
    }

    /// The proposal for the token after the sequence, if there is one.
    pub(crate) fn proposal(&mut self) -> Option<(u32, Choice)> {
        self.proposal.take()
    }

    /// After a pass of the model: `hidden` holds the hidden states of the
    /// `fresh` positions it read, from the first the layer has not read;
    /// `kept` positions are now final, and `tokens` are the sequence's,
    /// `kept + 1` of them. The layer reads the positions it can (those whose
    /// next token is known), and proposes the token after `tokens`.
    pub(crate) fn advance<B: Backend + Transfer>(
        &mut self,
        pipeline: &Pipeline<B>,
        hidden: &Tensor,
        fresh: usize,
        kept: usize,
        tokens: &[u32],
    ) -> Result<()> {
        let first = self.cache.len();
        debug_assert_eq!(tokens.len(), kept + 1);
        debug_assert!(kept - first <= fresh);
        let count = kept - first;
        if count == 0 {
            return Ok(());
        }
        let p = pipeline;
        let logits = p.model().multi_token(
            p.backend(),
            &mut self.cache,
            &hidden.narrow(0, 0, count)?,
            &tokens[first + 1..=kept],
        )?;
        let choice = if self.sampler.is_argmax() {
            Choice::Token(p.most_likely_rows(&logits)?[0])
        } else {
            let mut values = p.backend().download(&logits)?.to_f64_vec()?;
            self.sampler.distribution(&mut values, tokens, 0)
        };
        let token = self.sampler.draw(&choice);
        self.proposal = Some((token, choice));
        Ok(())
    }
}
