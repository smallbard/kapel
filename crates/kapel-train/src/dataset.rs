//! A prepared dataset: shards of tokens, the index of their documents, and
//! the manifest that lists them.
//!
//! `kapel prepare` writes a directory:
//!
//! - `dataset.json`, the manifest ([`Manifest`]): the tokenizer, the token
//!   that starts each document, the counts, and for each split (`train`,
//!   `val`) its shards with their sizes and SHA-256;
//! - `train_000000.bin`, … : the tokens, in llm.c's format (see
//!   [`crate::data::TokenFile`]), each document whole and preceded by the
//!   separator, as llm.c's FineWeb shards are;
//! - `train_000000.idx`, … : where each document starts in its shard;
//! - for conversations, `train_000000.mask`, … : which tokens the loss
//!   counts, those the assistant wrote ([`crate::chat`]).
//!
//! An index file is the 8 bytes `KAPELIDX`, a little-endian `u32` version
//! (1), four bytes of zeros, a `u64` count, then that many `u64`: the first
//! token of each document (its separator), in increasing order.
//!
//! A mask file is the 8 bytes `KAPELMSK`, a `u32` version (1), four bytes
//! of zeros, a `u64` count of tokens, then one bit per token, the first in
//! the lowest bit of the first byte: 1 for a token the loss counts as a
//! target.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data::DataError;

type Result<T> = std::result::Result<T, DataError>;

/// The name of the manifest in a dataset's directory.
pub const MANIFEST: &str = "dataset.json";

const INDEX_MAGIC: &[u8; 8] = b"KAPELIDX";

const MASK_MAGIC: &[u8; 8] = b"KAPELMSK";

/// What `dataset.json` holds.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Manifest {
    /// The format's version: 1.
    pub kapel_dataset: u32,
    /// The tokenizer the tokens come from.
    pub tokenizer: TokenizerInfo,
    /// The token that starts each document, and pads the rows.
    pub separator: u32,
    /// Whether the separator is written before each document: not for
    /// conversations, whose template begins them, where it only pads.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub separated: bool,
    /// For conversations, how they were made into tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatInfo>,
    /// The documents read, duplicates and empty ones included.
    pub documents: u64,
    /// The documents left out as exact duplicates of an earlier one.
    pub duplicates: u64,
    /// The documents left out as empty.
    pub empty: u64,
    /// One document in `val_every` goes to validation; 0 for none.
    pub val_every: u64,
    /// The files read, in order.
    pub inputs: Vec<InputInfo>,
    /// The training split.
    pub train: Split,
    /// The validation split, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<Split>,
}

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

/// How conversations were made into tokens.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ChatInfo {
    /// The SHA-256 of the chat template.
    pub template_sha256: String,
}

/// The tokenizer of a dataset.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct TokenizerInfo {
    /// The SHA-256 of its `tokenizer.json`.
    pub sha256: String,
    /// The size of its vocabulary.
    pub vocab: usize,
}

/// A file read.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct InputInfo {
    /// Its path, as given.
    pub path: String,
    /// The documents read from it.
    pub documents: u64,
}

/// The shards of one split.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct Split {
    /// Its tokens, separators included.
    pub tokens: u64,
    /// Its documents.
    pub documents: u64,
    /// Its tokens the loss counts, for conversations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained: Option<u64>,
    /// Its shards, in order.
    pub shards: Vec<ShardInfo>,
}

/// One shard.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ShardInfo {
    /// The file of tokens, relative to the manifest.
    pub file: String,
    /// The index of its documents, relative to the manifest.
    pub index: String,
    /// Its tokens.
    pub tokens: u64,
    /// Its documents.
    pub documents: u64,
    /// The SHA-256 of the file of tokens.
    pub sha256: String,
    /// The mask of its tokens the loss counts, relative to the manifest,
    /// for conversations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
}

impl Manifest {
    /// Reads `path`, a manifest.
    ///
    /// # Errors
    ///
    /// If it cannot be read or is not a manifest of version 1.
    pub fn read(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).map_err(|e| DataError(format!("{}: {e}", path.display())))?;
        Self::parse(&bytes).map_err(|e| DataError(format!("{}: {}", path.display(), e.0)))
    }

    /// A manifest from its bytes.
    ///
    /// # Errors
    ///
    /// If they are not a manifest of version 1.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|e| DataError(format!("not a manifest: {e}")))?;
        if manifest.kapel_dataset != 1 {
            return Err(DataError(format!(
                "a manifest of version {}, Kapel reads version 1",
                manifest.kapel_dataset
            )));
        }
        for shard in manifest
            .train
            .shards
            .iter()
            .chain(manifest.val.iter().flat_map(|s| &s.shards))
        {
            for name in [&shard.file, &shard.index].into_iter().chain(&shard.mask) {
                if !is_relative_name(name) {
                    return Err(DataError(format!(
                        "a shard named {name:?}: names stay inside the dataset"
                    )));
                }
            }
        }
        Ok(manifest)
    }

    /// The split named `name`, `train` or `val`.
    pub fn split(&self, name: &str) -> Option<&Split> {
        match name {
            "train" => Some(&self.train),
            "val" => self.val.as_ref(),
            _ => None,
        }
    }
}

/// Whether `name` is a plain relative path: no root, no `..`, no
/// backslash, as a manifest read from a server may not escape its
/// directory.
fn is_relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && !name.contains(':')
        && name
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != "..")
}

/// Writes the index of `starts` into `path`.
///
/// # Errors
///
/// If the file cannot be written.
pub fn write_index(path: &Path, starts: &[u64]) -> Result<()> {
    let mut bytes = Vec::with_capacity(24 + 8 * starts.len());
    bytes.extend_from_slice(INDEX_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(starts.len() as u64).to_le_bytes());
    for s in starts {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| DataError(format!("{}: {e}", path.display())))
}

/// Reads the index in `path`: the first token of each document, checked to
/// increase and to stay below `tokens`.
///
/// # Errors
///
/// If the file cannot be read or is not an index of such a shard.
pub fn read_index(path: &Path, tokens: usize) -> Result<Vec<u64>> {
    let fail = |m: &str| DataError(format!("{}: {m}", path.display()));
    let bytes = std::fs::read(path).map_err(|e| fail(&e.to_string()))?;
    if bytes.len() < 24 || &bytes[..8] != INDEX_MAGIC {
        return Err(fail("not an index of documents"));
    }
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
    if u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) != 1 {
        return Err(fail("an index of another version"));
    }
    let count = usize::try_from(word(16)).map_err(|_| fail("a count beyond memory"))?;
    if (bytes.len() - 24) / 8 != count || (bytes.len() - 24) % 8 != 0 {
        return Err(fail("its count is not its size"));
    }
    let starts: Vec<u64> = (0..count).map(|i| word(24 + 8 * i)).collect();
    let ordered = starts.windows(2).all(|w| w[0] < w[1]);
    if !ordered || starts.last().is_some_and(|&s| s >= tokens as u64) {
        return Err(fail("starts out of order, or beyond the shard"));
    }
    Ok(starts)
}

/// Writes the mask `bits` (one per token) into `path`.
///
/// # Errors
///
/// If the file cannot be written.
pub fn write_mask(path: &Path, bits: &[bool]) -> Result<()> {
    let mut bytes = Vec::with_capacity(24 + bits.len().div_ceil(8));
    bytes.extend_from_slice(MASK_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(bits.len() as u64).to_le_bytes());
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            byte |= u8::from(b) << i;
        }
        bytes.push(byte);
    }
    std::fs::write(path, bytes).map_err(|e| DataError(format!("{}: {e}", path.display())))
}

/// A mask read back: one bit per token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    bits: Vec<u8>,
    len: usize,
}

impl Mask {
    /// Whether the loss counts token `i`; `false` beyond the end.
    pub fn get(&self, i: usize) -> bool {
        i < self.len && self.bits[i / 8] >> (i % 8) & 1 == 1
    }

    /// Its tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has no token.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Reads the mask in `path`, checked to have `tokens` bits.
///
/// # Errors
///
/// If the file cannot be read or is not the mask of such a shard.
pub fn read_mask(path: &Path, tokens: usize) -> Result<Mask> {
    let fail = |m: &str| DataError(format!("{}: {m}", path.display()));
    let bytes = std::fs::read(path).map_err(|e| fail(&e.to_string()))?;
    if bytes.len() < 24 || &bytes[..8] != MASK_MAGIC {
        return Err(fail("not a mask of tokens"));
    }
    if u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) != 1 {
        return Err(fail("a mask of another version"));
    }
    let count = u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes"));
    if count != tokens as u64 || (bytes.len() - 24) != tokens.div_ceil(8) {
        return Err(fail("its count is not the shard's tokens"));
    }
    Ok(Mask {
        bits: bytes[24..].to_vec(),
        len: tokens,
    })
}

/// The SHA-256 of `bytes`, in hexadecimal.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_reads_back_and_a_bad_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("kapel-index-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.idx");
        write_index(&path, &[0, 5, 9]).unwrap();
        assert_eq!(read_index(&path, 10).unwrap(), vec![0, 5, 9]);
        assert!(read_index(&path, 9).is_err());
        write_index(&path, &[0, 5, 5]).unwrap();
        assert!(read_index(&path, 10).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_mask_reads_back_and_a_bad_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("kapel-mask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.mask");
        let bits: Vec<bool> = (0..19).map(|i| i % 3 == 1).collect();
        write_mask(&path, &bits).unwrap();
        let mask = read_mask(&path, 19).unwrap();
        assert_eq!((0..19).map(|i| mask.get(i)).collect::<Vec<_>>(), bits);
        assert!(!mask.get(19));
        assert!(read_mask(&path, 18).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn names_stay_inside_the_dataset() {
        for good in ["train_000000.bin", "a/b.idx"] {
            assert!(is_relative_name(good), "{good}");
        }
        for bad in [
            "",
            "/etc/passwd",
            "../x.bin",
            "a/../../x",
            "a//b",
            "C:x",
            "a\\b",
            "./a",
        ] {
            assert!(!is_relative_name(bad), "{bad}");
        }
    }
}
