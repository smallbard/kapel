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
    /// The text of `ids[..read]`, final: no character is left unfinished.
    confirmed: String,
    /// The tokens decoded again with the new ones (`ids[prefix..]`), so that
    /// what depends on neighbours (a space a decoder takes off a first token,
    /// bytes split across tokens) comes out as in the whole text.
    prefix: usize,
    read: usize,
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
            confirmed: String::new(),
            prefix: 0,
            read: 0,
        }
    }

    /// The text of every token so far, as decoding them all gives it, from
    /// the last few: the cost of a token does not grow with the answer.
    fn text(&mut self, tokenizer: &Tokenizer) -> String {
        let before = tokenizer.decode(&self.ids[self.prefix..self.read], true);
        let after = tokenizer.decode(&self.ids[self.prefix..], true);
        let Some(tail) = after.strip_prefix(before.as_str()) else {
            // The last tokens changed what came before them: all again.
            return tokenizer.decode(&self.ids, true);
        };
        let text = format!("{}{tail}", self.confirmed);
        if !tail.is_empty() && !tail.ends_with('\u{FFFD}') {
            self.confirmed.clone_from(&text);
            self.prefix = self.read;
            self.read = self.ids.len();
        }
        text
    }

    /// Adds token `id`.
    pub(crate) fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Added {
        self.ids.push(id);
        let text = self.text(tokenizer);
        // What was given out stays.
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
        let text = self.text(tokenizer);
        let rest = text
            .strip_prefix(self.emitted.as_str())
            .unwrap_or_default()
            .to_string();
        self.emitted.push_str(&rest);
        rest
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;

    /// For every tokenizer of `data/parity` (byte-level, SentencePiece with
    /// byte fallback, Gemma's), the text given out token by token is the
    /// text of all the tokens, decoded at once.
    #[test]
    fn streamed_text_is_the_whole_text() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
        let Ok(dirs) = std::fs::read_dir(&root) else {
            return;
        };
        let mut checked = 0;
        for dir in dirs.flatten() {
            let Ok(cases) = std::fs::read_to_string(dir.path().join("tokenizer_cases.json")) else {
                continue;
            };
            let tokenizer = Tokenizer::from_file(dir.path().join("tokenizer.json")).unwrap();
            let cases: Value = serde_json::from_str(&cases).unwrap();
            for case in cases.as_array().unwrap() {
                let ids: Vec<u32> = case["ids_no_special"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                let mut stream = TextStream::new(Vec::new());
                let mut text = String::new();
                for &id in &ids {
                    text.push_str(&stream.push(&tokenizer, id).text);
                }
                text.push_str(&stream.finish(&tokenizer));
                assert_eq!(text, tokenizer.decode(&ids, true), "{:?}", dir.file_name());
                checked += 1;
            }
        }
        eprintln!("{checked} texts streamed");
    }
}
