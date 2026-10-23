//! Answers held to a form: a JSON schema, a regular expression, a grammar,
//! one text of a list.
//!
//! The form is compiled by llguidance into a parser that reads the answer as
//! it is generated. Before each token, the parser gives the tokens that can
//! continue the answer in that form, and the others get a logit of −∞: the
//! sampling then chooses among the allowed tokens only, as it would have
//! among all. An end token is allowed once the answer is complete, and is
//! the only one allowed once nothing can follow.

use std::sync::Arc;

use kapel_tokenizers::Tokenizer;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;

use crate::{InferError, Result};

/// The form an answer must take.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// A JSON value valid for this JSON schema.
    JsonSchema(Value),
    /// Any JSON object.
    JsonObject,
    /// A text that this regular expression matches whole.
    Regex(String),
    /// A text of this grammar, in llguidance's Lark syntax.
    Lark(String),
    /// One of these texts.
    Choice(Vec<String>),
}

impl Constraint {
    fn grammar(&self) -> Result<TopLevelGrammar> {
        Ok(match self {
            Self::JsonSchema(schema) => TopLevelGrammar::from_json_schema(schema.clone()),
            Self::JsonObject => {
                TopLevelGrammar::from_json_schema(serde_json::json!({ "type": "object" }))
            }
            Self::Regex(regex) => TopLevelGrammar::from_regex(regex),
            Self::Lark(grammar) => TopLevelGrammar::from_lark(grammar.clone()),
            Self::Choice(choices) => {
                if choices.is_empty() {
                    return Err(InferError::Constraint("an empty list of choices".into()));
                }
                // Lark reads its strings as JSON reads them.
                let alternatives: Vec<String> = choices
                    .iter()
                    .map(|c| serde_json::to_string(c).expect("a string serializes"))
                    .collect();
                TopLevelGrammar::from_lark(format!("start: {}", alternatives.join(" | ")))
            }
        })
    }
}

/// What compiles constraints for one model: its tokens as bytes, in
/// llguidance's tree of tokens. Made once per model, it is shared by every
/// answer.
pub(crate) struct Grammars {
    factory: ParserFactory,
    /// The tokens that end an answer.
    eos: Arc<[u32]>,
    /// The end token llguidance allows when an answer may end.
    trie_eos: u32,
}

impl Grammars {
    /// The grammars of a model whose logits have `vocab` entries.
    pub(crate) fn new(tokenizer: &Tokenizer, vocab: usize, eos: &[u32]) -> Result<Self> {
        let vocab_size =
            u32::try_from(vocab).map_err(|_| InferError::Constraint("too many tokens".into()))?;
        let trie_eos = *eos
            .first()
            .ok_or_else(|| InferError::Constraint("the model has no end token".into()))?;
        let words: Vec<Vec<u8>> = (0..vocab_size)
            .map(|id| match tokenizer.token_bytes(id) {
                // Special tokens are marked, so that no text spells them.
                Some(bytes) if tokenizer.is_special(id) => {
                    let mut marked = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                    marked.extend(bytes);
                    marked
                }
                Some(bytes) => bytes,
                None => Vec::new(),
            })
            .collect();
        let info = TokRxInfo::new(vocab_size, trie_eos);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
        let mut factory =
            ParserFactory::new_simple(&env).map_err(|e| InferError::Constraint(e.to_string()))?;
        factory.quiet();
        Ok(Self {
            factory,
            eos: eos.into(),
            trie_eos,
        })
    }

    /// A guide that holds one answer to `constraint`.
    ///
    /// # Errors
    ///
    /// If the constraint does not compile: an invalid schema, regular
    /// expression or grammar.
    pub(crate) fn guide(&self, constraint: &Constraint) -> Result<Guide> {
        let parser = self
            .factory
            .create_parser(constraint.grammar()?)
            .map_err(|e| InferError::Constraint(e.to_string()))?;
        Ok(Guide {
            matcher: Matcher::new(Ok(parser)),
            eos: Arc::clone(&self.eos),
            trie_eos: self.trie_eos,
        })
    }
}

/// The parser of one answer held to a constraint.
pub(crate) struct Guide {
    matcher: Matcher,
    eos: Arc<[u32]>,
    trie_eos: u32,
}

impl Guide {
    /// Gives a logit of −∞ to every token that cannot come next.
    ///
    /// # Errors
    ///
    /// If the parser failed (it ran out of its limits, say).
    pub(crate) fn mask(&mut self, logits: &mut [f64]) -> Result<()> {
        let allowed = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| InferError::Constraint(e.to_string()))?;
        let may_end = allowed.is_allowed(self.trie_eos);
        for (id, l) in logits.iter_mut().enumerate() {
            let id = id as u32;
            let ok = if self.eos.contains(&id) {
                may_end
            } else {
                id < allowed.len() as u32 && allowed.is_allowed(id)
            };
            if !ok {
                *l = f64::NEG_INFINITY;
            }
        }
        Ok(())
    }

    /// Reads the token chosen: an end token reads nothing.
    ///
    /// # Errors
    ///
    /// If the token was not allowed.
    pub(crate) fn advance(&mut self, token: u32) -> Result<()> {
        if self.eos.contains(&token) {
            return Ok(());
        }
        self.matcher
            .consume_token(token)
            .map_err(|e| InferError::Constraint(e.to_string()))
    }
}
