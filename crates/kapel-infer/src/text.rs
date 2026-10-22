//! The text of the generated tokens, as it becomes final.

use kapel_tokenizers::Tokenizer;

/// The text of a sequence of generated tokens, given out as soon as no later
/// token can change it: the end of a character whose bytes are still coming
/// (decoded as U+FFFD for now), and the start of a stop sequence, wait.
#[derive(Clone, Debug)]
pub(crate) struct TextStream {
    ids: Vec<u32>,
    /// The text given out so far.
    emitted: String,
    stop: Vec<String>,
}

/// What a token added.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Added {
    /// The new final text.
    pub(crate) text: String,
    /// The stop sequence that ended the text, if one did: the text stops
    /// before it.
    pub(crate) stopped: Option<String>,
}

impl TextStream {
    pub(crate) fn new(stop: Vec<String>) -> Self {
        Self {
            ids: Vec::new(),
            emitted: String::new(),
            stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }

    /// Adds token `id`.
    pub(crate) fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Added {
        self.ids.push(id);
        let text = tokenizer.decode(&self.ids, true);
        // Decoding the whole sequence again keeps what depends on neighbours
        // (spaces, bytes split across tokens) right; what was given out stays.
        let Some(fresh) = text.strip_prefix(self.emitted.as_str()) else {
            return Added {
                text: String::new(),
                stopped: None,
            };
        };
        let fresh = fresh.to_string();
        // A stop sequence may begin in text already given out, up to its
        // length minus one byte before.
        let longest = self.stop.iter().map(String::len).max().unwrap_or(0);
        let mut from = self.emitted.len().saturating_sub(longest.saturating_sub(1));
        while !text.is_char_boundary(from) {
            from -= 1;
        }
        let found = self
            .stop
            .iter()
            .filter_map(|s| text[from..].find(s.as_str()).map(|at| (from + at, s)))
            .min_by_key(|&(at, _)| at);
        if let Some((at, stop)) = found {
            let text = text[self.emitted.len().min(at)..at].to_string();
            self.emitted.push_str(&text);
            return Added {
                text,
                stopped: Some(stop.clone()),
            };
        }
        let mut keep = fresh.len();
        // An unfinished character.
        while fresh[..keep].ends_with('\u{FFFD}') {
            keep -= '\u{FFFD}'.len_utf8();
        }
        // The longest end that could start a stop sequence.
        let held = self
            .stop
            .iter()
            .flat_map(|s| {
                (1..s.len())
                    .filter(|&n| s.is_char_boundary(n))
                    .map(|n| &s[..n])
            })
            .filter(|prefix| text.ends_with(prefix))
            .map(|prefix| prefix.len())
            .max()
            .unwrap_or(0);
        keep = keep.min(fresh.len().saturating_sub(held));
        while !fresh.is_char_boundary(keep) {
            keep -= 1;
        }
        let text = fresh[..keep].to_string();
        self.emitted.push_str(&text);
        Added {
            text,
            stopped: None,
        }
    }

    /// The text still held back, given out now that no token follows.
    pub(crate) fn finish(&mut self, tokenizer: &Tokenizer) -> String {
        let text = tokenizer.decode(&self.ids, true);
        let rest = text
            .strip_prefix(self.emitted.as_str())
            .unwrap_or_default()
            .to_string();
        self.emitted.push_str(&rest);
        rest
    }
}
