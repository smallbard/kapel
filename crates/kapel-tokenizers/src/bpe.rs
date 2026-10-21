//! Byte-pair encoding: a piece of text to token ids.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Mutex;

use serde_json::Value;

use crate::{TokenizerError, bad, id_of};

/// The BPE model of `tokenizer.json`.
#[derive(Debug)]
pub(crate) struct Bpe {
    pub(crate) vocab: HashMap<String, u32>,
    /// For each mergeable pair of ids, its rank (lower merges first) and the id
    /// of the merged token.
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_fallback: bool,
    unk: Option<u32>,
    fuse_unk: bool,
    ignore_merges: bool,
    /// Pieces already encoded: in running text, most words come back.
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

impl Bpe {
    pub(crate) fn from_json(json: &Value) -> Result<Self, TokenizerError> {
        if json.get("type").and_then(Value::as_str) != Some("BPE") {
            return Err(bad("only BPE models are supported"));
        }
        let vocab: HashMap<String, u32> = json
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("BPE without a vocab"))?
            .iter()
            .map(|(k, v)| Ok((k.clone(), id_of(v, &format!("the id of {k:?}"))?)))
            .collect::<Result<_, _>>()?;
        let mut merges = HashMap::new();
        for (rank, merge) in json
            .get("merges")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            // Older files write "a b", newer ones ["a", "b"].
            let pair = match merge {
                Value::String(s) => s
                    .split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string())),
                Value::Array(parts) => match (
                    parts.first().and_then(Value::as_str),
                    parts.get(1).and_then(Value::as_str),
                ) {
                    (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                    _ => None,
                },
                _ => None,
            }
            .ok_or_else(|| bad(format!("merge {rank} is malformed")))?;
            // A merge whose parts or result are not in the vocabulary can never
            // apply; `tokenizers` refuses such files, Kapel skips the merge.
            let ids = (
                vocab.get(&pair.0),
                vocab.get(&pair.1),
                vocab.get(&format!("{}{}", pair.0, pair.1)),
            );
            if let (Some(&a), Some(&b), Some(&merged)) = ids {
                let rank = u32::try_from(rank).map_err(|_| bad("more than 2^32 merges"))?;
                merges.entry((a, b)).or_insert((rank, merged));
            }
        }
        let flag = |key: &str| json.get(key).and_then(Value::as_bool).unwrap_or(false);
        let unk = json
            .get("unk_token")
            .and_then(Value::as_str)
            .and_then(|t| vocab.get(t).copied());
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if json
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
            {
                return Err(bad(format!("BPE with {key} is not supported")));
            }
        }
        Ok(Self {
            vocab,
            merges,
            byte_fallback: flag("byte_fallback"),
            unk,
            fuse_unk: flag("fuse_unk"),
            ignore_merges: flag("ignore_merges"),
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// The ids of one piece.
    pub(crate) fn encode(&self, piece: &str) -> Vec<u32> {
        if let Some(ids) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(piece)
        {
            return ids.clone();
        }
        let ids = self.encode_uncached(piece);
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() < 100_000 {
            cache.insert(piece.to_string(), ids.clone());
        }
        ids
    }

    fn encode_uncached(&self, piece: &str) -> Vec<u32> {
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(piece)
        {
            return vec![id];
        }
        // The characters of the piece, as a linked list of symbols. The pair of
        // lowest rank merges first, the leftmost among equals, until no pair can:
        // a priority queue keeps this at O(n log n) for a piece of n characters.
        let mut symbols: Vec<Symbol> = piece
            .chars()
            .map(|c| {
                let text = c.to_string();
                Symbol {
                    id: self.vocab.get(&text).copied(),
                    text,
                    alive: true,
                }
            })
            .collect();
        let n = symbols.len();
        let mut next: Vec<Option<usize>> = (0..n).map(|i| (i + 1 < n).then_some(i + 1)).collect();
        let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let mut queue = BinaryHeap::new();
        let candidate =
            |symbols: &[Symbol], left: usize, right: usize| -> Option<Reverse<(u32, usize, u32)>> {
                let (a, b) = (symbols[left].id?, symbols[right].id?);
                self.merges
                    .get(&(a, b))
                    .map(|&(rank, merged)| Reverse((rank, left, merged)))
            };
        for i in 0..n.saturating_sub(1) {
            queue.extend(candidate(&symbols, i, i + 1));
        }
        while let Some(Reverse((rank, left, merged))) = queue.pop() {
            // Skip a candidate made stale by an earlier merge.
            let Some(right) = next[left] else { continue };
            if !symbols[left].alive
                || candidate(&symbols, left, right) != Some(Reverse((rank, left, merged)))
            {
                continue;
            }
            let text = std::mem::take(&mut symbols[right].text);
            symbols[left].text.push_str(&text);
            symbols[left].id = Some(merged);
            symbols[right].alive = false;
            next[left] = next[right];
            if let Some(after) = next[right] {
                prev[after] = Some(left);
                queue.extend(candidate(&symbols, left, after));
            }
            if let Some(before) = prev[left] {
                queue.extend(candidate(&symbols, before, left));
            }
        }
        let symbols: Vec<String> = symbols
            .into_iter()
            .filter(|s| s.alive)
            .map(|s| s.text)
            .collect();
        let mut ids = Vec::new();
        let mut previous_unk = false;
        for symbol in symbols {
            if let Some(&id) = self.vocab.get(&symbol) {
                ids.push(id);
                previous_unk = false;
                continue;
            }
            if self.byte_fallback {
                let bytes: Option<Vec<u32>> = symbol
                    .bytes()
                    .map(|b| self.vocab.get(&format!("<0x{b:02X}>")).copied())
                    .collect();
                if let Some(bytes) = bytes {
                    ids.extend(bytes);
                    previous_unk = false;
                    continue;
                }
            }
            if let Some(unk) = self.unk {
                if !(self.fuse_unk && previous_unk) {
                    ids.push(unk);
                }
                previous_unk = true;
            }
        }
        ids
    }
}

/// A symbol of a piece being merged.
#[derive(Debug)]
struct Symbol {
    text: String,
    id: Option<u32>,
    alive: bool,
}
