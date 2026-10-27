//! Token files, and the batches read from them.
//!
//! The files are llm.c's shards: a header of 256 little-endian `i32` (the
//! magic number, the version, the number of tokens), then the tokens,
//! little-endian `u16` (magic 20240520, GPT-2's vocabulary) or `u32` (magic
//! 20240801, Llama 3's). They are mapped, not read. FineWeb as llm.c and
//! modded-nanogpt publish it is in this format, and so are the shards of
//! `kapel prepare` ([`crate::prepare`]), which add the index of their
//! documents.
//!
//! A [`Source`] is a sequence of shards: a dataset's directory or manifest
//! (local or at an `http(s)://` URL), or a pattern of llm.c files. Each
//! shard is cut into rows of at most `length` tokens:
//!
//! - [`Pack::Stream`]: the tokens one after the other, cut every `length`,
//!   documents running from one row into the next;
//! - [`Pack::Documents`]: whole documents, as many as fit, the rest of the row
//!   padded (ids of the separator, targets [`crate::IGNORE`]); a document
//!   longer than a row is cut into rows of its own;
//! - [`Pack::Padded`]: one document a row, the rest of a longer one left
//!   out, as TRL's batches without packing;
//! - [`Pack::Joined`]: the rows of [`Pack::Padded`] without their padding,
//!   the batch's documents one after the other in a single row, attention
//!   kept within each: the same examples a pass, only their own tokens
//!   computed (TRL's `padding_free`).
//!
//! Each row also reads the token after it, its last target. A shard of
//! conversations has a mask of the tokens the loss counts: the targets
//! outside it are [`crate::IGNORE`]. With `mask`, a
//! batch carries the document of each token ([`Tokens::segments`]): each
//! token attends to the earlier tokens of its own document only, and
//! positions start again at each document.
//!
//! A [`Loader`] reads the rows of one or more sources: each row is drawn from
//! a source with the probability of its weight, by a generator that depends
//! only on the seed and the row's rank. A source goes through its shards in
//! order, or, with `shuffle`, in an order drawn anew for each epoch, and
//! through each shard's rows in order or in an order drawn for that shard
//! and that epoch. The whole order thus depends on the seed alone, and the
//! [`Position`] (the rows read, and where each source is) is all a
//! checkpoint needs to read on exactly as before.
//!
//! [`Prefetched`] reads the batches ahead on a thread of its own, and a
//! remote source downloads its next shard while the current one is read.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::dataset::{MANIFEST, Manifest, Mask, read_index, read_mask, sha256_hex};

/// A file that cannot be read as tokens.
#[derive(Debug)]
pub struct DataError(pub String);

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DataError {}

type Result<T> = std::result::Result<T, DataError>;

fn fail(m: impl Into<String>) -> DataError {
    DataError(m.into())
}

const HEADER: usize = 256 * 4;

/// One file of tokens, mapped.
pub struct TokenFile {
    path: PathBuf,
    map: Mmap,
    wide: bool,
    len: usize,
}

impl TokenFile {
    /// Maps `path` and checks its header.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, has an unknown magic number, or holds
    /// fewer tokens than its header says.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let fail = |m: String| DataError(format!("{}: {m}", path.display()));
        let file = std::fs::File::open(&path).map_err(|e| fail(e.to_string()))?;
        // SAFETY: the file is only read; a file changed while mapped would
        // give other tokens, never memory unsafety for `u8` reads.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| fail(e.to_string()))?;
        if map.len() < HEADER {
            return Err(fail("shorter than its header".into()));
        }
        let word =
            |i: usize| i32::from_le_bytes(map[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        let wide = match (word(0), word(1)) {
            (20_240_520, 1) => false,
            (20_240_801, 7) => true,
            (magic, version) => {
                return Err(fail(format!(
                    "not a file of tokens of llm.c (magic {magic}, version {version})"
                )));
            }
        };
        let len = usize::try_from(word(2)).map_err(|_| fail("a negative count".into()))?;
        let size = if wide { 4 } else { 2 };
        if map.len() < HEADER + len * size {
            return Err(fail(format!(
                "{len} tokens announced, {} present",
                (map.len() - HEADER) / size
            )));
        }
        Ok(Self {
            path,
            map,
            wide,
            len,
        })
    }

    /// The number of tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no token.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Its path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The tokens `start..start + count`.
    ///
    /// # Panics
    ///
    /// If the range is beyond the file.
    pub fn tokens(&self, start: usize, count: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        self.extend(start, count, &mut out);
        out
    }

    /// Appends the tokens `start..start + count` to `out`.
    ///
    /// # Panics
    ///
    /// If the range is beyond the file.
    pub fn extend(&self, start: usize, count: usize, out: &mut Vec<u32>) {
        assert!(start + count <= self.len, "tokens beyond the file");
        let bytes = &self.map[HEADER..];
        if self.wide {
            out.extend(
                bytes[4 * start..4 * (start + count)]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b)),
            );
        } else {
            out.extend(
                bytes[2 * start..2 * (start + count)]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u32::from(u16::from_le_bytes(*b))),
            );
        }
    }

    /// The positions of `token`, in order.
    fn positions_of(&self, token: u32) -> Vec<u64> {
        let mut out = Vec::new();
        let bytes = &self.map[HEADER..];
        if self.wide {
            for (i, b) in bytes[..4 * self.len].as_chunks::<4>().0.iter().enumerate() {
                if u32::from_le_bytes(*b) == token {
                    out.push(i as u64);
                }
            }
        } else if let Ok(t) = u16::try_from(token) {
            for (i, b) in bytes[..2 * self.len].as_chunks::<2>().0.iter().enumerate() {
                if u16::from_le_bytes(*b) == t {
                    out.push(i as u64);
                }
            }
        }
        out
    }
}

/// The paths of `pattern`: a path, or a path whose file name holds `*`, the
/// files that match in the order of their names.
///
/// # Errors
///
/// If no file matches.
pub fn matching(pattern: &str) -> Result<Vec<PathBuf>> {
    let path = Path::new(pattern);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let paths: Vec<PathBuf> = if name.contains('*') {
        let dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let entries =
            std::fs::read_dir(dir).map_err(|e| fail(format!("{}: {e}", dir.display())))?;
        let mut found: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| crate::config::matches(&name, &n.to_string_lossy()))
            })
            .collect();
        found.sort();
        found
    } else {
        vec![path.to_path_buf()]
    };
    if paths.is_empty() {
        return Err(fail(format!("no file matches {pattern}")));
    }
    Ok(paths)
}

/// The files of `pattern` ([`matching`]), opened.
///
/// # Errors
///
/// If no file matches, or one cannot be opened.
pub fn files(pattern: &str) -> Result<Vec<TokenFile>> {
    matching(pattern)?.iter().map(TokenFile::open).collect()
}

/// How the tokens of a shard become rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Pack {
    /// One after the other, cut every `length` tokens.
    #[default]
    Stream,
    /// Whole documents, as many as fit, then padding.
    Documents,
    /// One document a row, cut to the row (its rest left out) or padded:
    /// the batches of `transformers` without packing.
    Padded,
    /// The documents of [`Pack::Padded`]'s rows one after the other in one
    /// row, without padding: a batch of `rows` documents is one row of their
    /// tokens, each attending to its own only.
    Joined,
}

/// How a [`Loader`] reads.
#[derive(Clone, Debug)]
pub struct Reading {
    /// The rows of a batch.
    pub rows: usize,
    /// The tokens of a row.
    pub length: usize,
    /// The seed of every order drawn.
    pub seed: u64,
    /// Shards and rows in an order drawn for each epoch.
    pub shuffle: bool,
    /// How rows are formed.
    pub pack: Pack,
    /// Give each token's document: attention within documents.
    pub mask: bool,
    /// The token that starts each document, for files without an index nor
    /// a manifest.
    pub separator: Option<u32>,
    /// Where remote shards are kept.
    pub cache: PathBuf,
}

impl Reading {
    /// Rows of `length` tokens, `rows` a batch, in order, as streams.
    pub fn new(rows: usize, length: usize) -> Self {
        Self {
            rows,
            length,
            seed: 0,
            shuffle: false,
            pack: Pack::Stream,
            mask: false,
            separator: None,
            cache: default_cache(),
        }
    }
}

/// Where remote datasets are kept: `$KAPEL_DATA_CACHE`, else
/// `~/.cache/kapel/data`.
pub fn default_cache() -> PathBuf {
    if let Some(dir) = std::env::var_os("KAPEL_DATA_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".cache/kapel/data")
}

/// One shard of a source, before it is opened.
#[derive(Clone, Debug)]
struct ShardMeta {
    /// Its file: a path, or a URL.
    file: String,
    /// Its index: a path or a URL, if any.
    index: Option<String>,
    /// The mask of its targets: a path or a URL, if any.
    mask: Option<String>,
    tokens: usize,
    sha256: Option<String>,
}

/// A sequence of shards.
#[derive(Clone, Debug)]
pub struct Source {
    name: String,
    shards: Vec<ShardMeta>,
    /// The token that starts each document, if known.
    separator: Option<u32>,
    /// The vocabulary of the tokenizer, if the manifest gives it.
    vocab: Option<usize>,
    /// Where its remote shards are kept, for a remote source.
    cache: Option<PathBuf>,
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

impl Source {
    /// The source `path` names: a dataset's directory, its `dataset.json`,
    /// the URL of either, or a pattern of llm.c files; `split` chooses
    /// among a manifest's splits (`train` or `val`).
    ///
    /// # Errors
    ///
    /// If nothing can be read there, or a manifest has no such split.
    pub fn open(path: &str, split: &str, cache: &Path) -> Result<Self> {
        if is_url(path) {
            let url = if path.ends_with(".json") {
                path.to_string()
            } else {
                format!("{}/{MANIFEST}", path.trim_end_matches('/'))
            };
            let base = url[..url.rfind('/').expect("a URL has a slash") + 1].to_string();
            let dir = cache.join(&sha256_hex(url.as_bytes())[..16]);
            let local = dir.join(MANIFEST);
            kapel_hub::download::fetch_url(&url, None, &local).map_err(|e| fail(e.to_string()))?;
            let manifest = Manifest::read(&local)?;
            return Self::from_manifest(
                path,
                &manifest,
                split,
                |n| format!("{base}{n}"),
                Some(dir),
            );
        }
        let p = Path::new(path);
        let manifest_path = if p.is_dir() {
            Some(p.join(MANIFEST))
        } else if path.ends_with(".json") {
            Some(p.to_path_buf())
        } else {
            None
        };
        if let Some(m) = manifest_path {
            let manifest = Manifest::read(&m)?;
            let dir = m.parent().unwrap_or(Path::new(".")).to_path_buf();
            return Self::from_manifest(
                path,
                &manifest,
                split,
                |n| dir.join(n).to_string_lossy().into_owned(),
                None,
            );
        }
        let shards = matching(path)?
            .into_iter()
            .map(|p| {
                let file = TokenFile::open(&p)?;
                let index = p.with_extension("idx");
                Ok(ShardMeta {
                    file: p.to_string_lossy().into_owned(),
                    index: index.exists().then(|| index.to_string_lossy().into_owned()),
                    mask: None,
                    tokens: file.len(),
                    sha256: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            name: path.to_string(),
            shards,
            separator: None,
            vocab: None,
            cache: None,
        })
    }

    fn from_manifest(
        name: &str,
        manifest: &Manifest,
        split: &str,
        locate: impl Fn(&str) -> String,
        cache: Option<PathBuf>,
    ) -> Result<Self> {
        let s = manifest
            .split(split)
            .ok_or_else(|| fail(format!("{name}: no split {split:?}")))?;
        if s.shards.is_empty() {
            return Err(fail(format!("{name}: the split {split:?} is empty")));
        }
        Ok(Self {
            name: name.to_string(),
            shards: s
                .shards
                .iter()
                .map(|sh| ShardMeta {
                    file: locate(&sh.file),
                    index: Some(locate(&sh.index)),
                    mask: sh.mask.as_deref().map(&locate),
                    tokens: sh.tokens as usize,
                    sha256: Some(sh.sha256.clone()),
                })
                .collect(),
            separator: Some(manifest.separator),
            vocab: Some(manifest.tokenizer.vocab),
            cache,
        })
    }

    /// Its name, as given.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its tokens.
    pub fn tokens(&self) -> u64 {
        self.shards.iter().map(|s| s.tokens as u64).sum()
    }

    /// The vocabulary of its tokenizer, when its manifest gives it.
    pub fn vocab(&self) -> Option<usize> {
        self.vocab
    }

    /// The local path of shard `i`'s file `what` (`file` or `index`),
    /// downloading it first for a remote source.
    fn local(&self, location: &str, sha256: Option<&str>) -> Result<PathBuf> {
        match &self.cache {
            Some(dir) if is_url(location) => {
                let name = location.rsplit('/').next().unwrap_or(location);
                let target = dir.join(name);
                kapel_hub::download::fetch_url(location, sha256, &target)
                    .map_err(|e| fail(e.to_string()))?;
                Ok(target)
            }
            _ => Ok(PathBuf::from(location)),
        }
    }

    /// Opens shard `i`: its tokens, and the start of each document when an
    /// index or a separator gives them and `starts` asks for them.
    fn open_shard(&self, i: usize, starts: bool, separator: Option<u32>) -> Result<OpenShard> {
        let meta = &self.shards[i];
        let file = TokenFile::open(self.local(&meta.file, meta.sha256.as_deref())?)?;
        if file.len() != meta.tokens {
            return Err(fail(format!(
                "{}: {} tokens, the dataset says {}",
                file.path().display(),
                file.len(),
                meta.tokens
            )));
        }
        let starts = if !starts {
            None
        } else if let Some(index) = &meta.index {
            Some(read_index(&self.local(index, None)?, file.len())?)
        } else {
            let separator = self.separator.or(separator).ok_or_else(|| {
                fail(format!(
                    "{}: no index of its documents; give data.separator, the token that starts each",
                    file.path().display()
                ))
            })?;
            Some(file.positions_of(separator))
        };
        let mask = match &meta.mask {
            Some(m) => Some(read_mask(&self.local(m, None)?, file.len())?),
            None => None,
        };
        Ok(OpenShard { file, starts, mask })
    }

    /// Starts downloading shard `i` on a thread, for a remote source.
    fn fetch_ahead(&self, i: usize) -> Option<JoinHandle<()>> {
        let meta = self.shards.get(i)?.clone();
        let dir = self.cache.clone()?;
        is_url(&meta.file).then(|| {
            std::thread::spawn(move || {
                for (location, sha) in [
                    (Some(&meta.file), meta.sha256.as_deref()),
                    (meta.index.as_ref(), None),
                    (meta.mask.as_ref(), None),
                ] {
                    if let Some(location) = location {
                        let name = location.rsplit('/').next().unwrap_or(location);
                        // An error comes back when the shard is opened.
                        let _ = kapel_hub::download::fetch_url(location, sha, &dir.join(name));
                    }
                }
            })
        })
    }

    /// Removes shard `i`'s downloaded files, for a remote source.
    fn evict(&self, i: usize) {
        if let (Some(dir), Some(meta)) = (&self.cache, self.shards.get(i)) {
            for location in std::iter::once(&meta.file)
                .chain(meta.index.as_ref())
                .chain(meta.mask.as_ref())
            {
                if is_url(location) {
                    let name = location.rsplit('/').next().unwrap_or(location);
                    let _ = std::fs::remove_file(dir.join(name));
                }
            }
        }
    }
}

struct OpenShard {
    file: TokenFile,
    starts: Option<Vec<u64>>,
    /// The tokens the loss counts as targets, if not all.
    mask: Option<Mask>,
}

/// A row of a shard: `len` tokens from `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    start: u64,
    len: u32,
}

/// The rows of a shard.
fn rows_of(shard: &OpenShard, length: usize, pack: Pack) -> Vec<Row> {
    let n = shard.file.len();
    match pack {
        Pack::Padded | Pack::Joined => {
            let starts = shard.starts.as_deref().unwrap_or(&[]);
            let ends = starts
                .iter()
                .skip(1)
                .copied()
                .chain(std::iter::once(n as u64));
            starts
                .iter()
                .zip(ends)
                .map(|(&s, e)| Row {
                    start: s,
                    len: (e - s).min(length as u64) as u32,
                })
                .collect()
        }
        Pack::Stream => (0..n.saturating_sub(1) / length)
            .map(|k| Row {
                start: (k * length) as u64,
                len: length as u32,
            })
            .collect(),
        Pack::Documents => {
            let starts = shard.starts.as_deref().unwrap_or(&[]);
            let mut rows = Vec::new();
            // The documents' ends: the next one's start, or the shard's end.
            let ends = starts
                .iter()
                .skip(1)
                .copied()
                .chain(std::iter::once(n as u64));
            let mut row: Option<(u64, u64)> = None;
            for (&s, e) in starts.iter().zip(ends) {
                if let Some((a, b)) = row {
                    if e - a <= length as u64 {
                        row = Some((a, e));
                        continue;
                    }
                    rows.push(Row {
                        start: a,
                        len: (b - a) as u32,
                    });
                }
                // A new row from this document: whole rows of its excess
                // first.
                let mut s = s;
                while e - s > length as u64 {
                    rows.push(Row {
                        start: s,
                        len: length as u32,
                    });
                    s += length as u64;
                }
                row = Some((s, e));
            }
            if let Some((a, b)) = row {
                rows.push(Row {
                    start: a,
                    len: (b - a) as u32,
                });
            }
            rows
        }
    }
}

/// A 64-bit mix of `values` (SplitMix64's finalizer, chained).
pub(crate) fn mix(values: &[u64]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for &v in values {
        h = splitmix(h ^ v);
    }
    h
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A permutation of `0..n` drawn from `seed` (Fisher–Yates).
pub(crate) fn permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = splitmix(state);
        let j = ((u128::from(state) * (i as u128 + 1)) >> 64) as usize;
        v.swap(i, j);
    }
    v
}

const SHARDS: u64 = 1;
const ROWS: u64 = 2;
const SOURCES: u64 = 3;

/// Where a source is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Cursor {
    /// The passes over the whole source done.
    pub epoch: u64,
    /// The rank of the current shard in this epoch's order.
    pub shard: usize,
    /// The rows of the current shard read.
    pub row: usize,
}

/// Where a [`Loader`] is.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Position {
    /// The rows read, from all sources.
    pub rows: u64,
    /// Where each source is.
    pub sources: Vec<Cursor>,
}

/// A batch read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokens {
    /// `rows × length` ids, row after row.
    pub ids: Vec<u32>,
    /// As many targets: each the token after, [`crate::IGNORE`] for padding.
    pub targets: Vec<u32>,
    /// With `mask` (and always with [`Pack::Joined`]), the document of each
    /// token: equal numbers for the tokens of one document of one row,
    /// different ones otherwise.
    pub segments: Option<Vec<u32>>,
}

impl Tokens {
    /// The lengths of the documents, in order: runs of one segment, or the
    /// whole batch without segments.
    pub fn documents(&self) -> Vec<usize> {
        let Some(segments) = &self.segments else {
            return vec![self.ids.len()];
        };
        let mut lengths = Vec::new();
        for (i, s) in segments.iter().enumerate() {
            if i == 0 || *s != segments[i - 1] {
                lengths.push(0);
            }
            *lengths.last_mut().expect("pushed") += 1;
        }
        lengths
    }
}

/// A source being read.
struct Reader {
    source: Source,
    index: u64,
    cursor: Cursor,
    /// The shard open, its rows, and their order.
    current: Option<(usize, OpenShard, Vec<Row>, Vec<u32>)>,
    ahead: Option<(usize, JoinHandle<()>)>,
}

impl Reader {
    /// The shard at rank `rank` of `epoch`'s order.
    fn shard_at(&self, reading: &Reading, epoch: u64, rank: usize) -> usize {
        let n = self.source.shards.len();
        if reading.shuffle {
            permutation(n, mix(&[reading.seed, SHARDS, self.index, epoch]))[rank] as usize
        } else {
            rank
        }
    }

    fn next_row(&mut self, reading: &Reading) -> Result<(Row, usize)> {
        let n = self.source.shards.len();
        let needs_starts = reading.mask || reading.pack != Pack::Stream;
        // Every shard may be visited once without a row before giving up.
        for _ in 0..=n {
            if self.current.is_none() {
                let shard = self.shard_at(reading, self.cursor.epoch, self.cursor.shard);
                if let Some((s, handle)) = self.ahead.take() {
                    if s == shard {
                        let _ = handle.join();
                    } else {
                        self.ahead = Some((s, handle));
                    }
                }
                let open = self
                    .source
                    .open_shard(shard, needs_starts, reading.separator)?;
                let rows = rows_of(&open, reading.length, reading.pack);
                let order = if reading.shuffle {
                    permutation(
                        rows.len(),
                        mix(&[
                            reading.seed,
                            ROWS,
                            self.index,
                            self.cursor.epoch,
                            shard as u64,
                        ]),
                    )
                } else {
                    (0..rows.len() as u32).collect()
                };
                // The next shard, downloaded meanwhile.
                let (epoch, rank) = if self.cursor.shard + 1 < n {
                    (self.cursor.epoch, self.cursor.shard + 1)
                } else {
                    (self.cursor.epoch + 1, 0)
                };
                let next = self.shard_at(reading, epoch, rank);
                if next != shard && self.ahead.is_none() {
                    self.ahead = self.source.fetch_ahead(next).map(|h| (next, h));
                }
                self.current = Some((shard, open, rows, order));
            }
            let (shard, _, rows, order) = self.current.as_ref().expect("a shard open");
            if self.cursor.row < rows.len() {
                let row = rows[order[self.cursor.row] as usize];
                self.cursor.row += 1;
                return Ok((row, *shard));
            }
            let done = *shard;
            self.current = None;
            self.cursor.row = 0;
            self.cursor.shard += 1;
            if self.cursor.shard == n {
                self.cursor.shard = 0;
                self.cursor.epoch += 1;
            }
            if n > 1 {
                self.source.evict(done);
            }
        }
        Err(fail(format!(
            "{}: no shard holds a row of {} tokens",
            self.source.name, reading.length
        )))
    }

    /// Appends the row's ids, targets and segments.
    fn append(&self, row: Row, reading: &Reading, out: &mut Tokens, segment: &mut u32) {
        let (_, open, _, _) = self.current.as_ref().expect("a shard open");
        let file = &open.file;
        let (start, len) = (row.start as usize, row.len as usize);
        // A padded or joined row is its document: nothing after it is a
        // target.
        let after = usize::from(!matches!(reading.pack, Pack::Padded | Pack::Joined));
        let available = (file.len() - start).min(len + after);
        let mut tokens = Vec::with_capacity(len + 1);
        file.extend(start, available, &mut tokens);
        out.ids.extend_from_slice(&tokens[..len]);
        match &open.mask {
            Some(mask) => out.targets.extend((1..available).map(|j| {
                if mask.get(start + j) {
                    tokens[j]
                } else {
                    crate::IGNORE
                }
            })),
            None => out.targets.extend_from_slice(&tokens[1..available]),
        }
        if available < len + 1 {
            out.targets.push(crate::IGNORE);
        }
        let pad = if reading.pack == Pack::Joined {
            0
        } else {
            reading.length - len
        };
        let filler = self.source.separator.or(reading.separator).unwrap_or(0);
        out.ids.extend(std::iter::repeat_n(filler, pad));
        out.targets.extend(std::iter::repeat_n(crate::IGNORE, pad));
        if let Some(segments) = &mut out.segments {
            let starts = open.starts.as_deref().unwrap_or(&[]);
            let first = starts.partition_point(|&s| s <= row.start);
            let mut next = starts[first..].iter().copied().peekable();
            *segment += 1;
            for p in row.start..row.start + len as u64 {
                if next.peek() == Some(&p) {
                    next.next();
                    if p != row.start {
                        *segment += 1;
                    }
                }
                segments.push(*segment);
            }
            if pad > 0 {
                *segment += 1;
                segments.extend(std::iter::repeat_n(*segment, pad));
            }
        }
    }
}

/// Batches read from weighted sources.
pub struct Loader {
    readers: Vec<Reader>,
    /// The cumulative weights, normalized to 1.
    cumulative: Vec<f64>,
    reading: Reading,
    rows: u64,
}

impl Loader {
    /// A loader of `reading`'s batches from `sources`, each with its
    /// weight, from the start.
    ///
    /// # Errors
    ///
    /// If there is no source, or a weight is not positive.
    pub fn new(sources: Vec<(Source, f64)>, reading: Reading) -> Result<Self> {
        if sources.is_empty() {
            return Err(fail("no source of data"));
        }
        if reading.rows == 0 || reading.length == 0 || reading.length > u32::MAX as usize {
            return Err(fail("rows and length must be positive"));
        }
        let total: f64 = sources.iter().map(|(_, w)| w).sum();
        if sources.iter().any(|(_, w)| !(*w > 0.0 && w.is_finite())) {
            return Err(fail("the weights of the sources must be positive"));
        }
        let mut acc = 0.0;
        let cumulative = sources
            .iter()
            .map(|(_, w)| {
                acc += w / total;
                acc
            })
            .collect();
        Ok(Self {
            readers: sources
                .into_iter()
                .enumerate()
                .map(|(i, (source, _))| Reader {
                    source,
                    index: i as u64,
                    cursor: Cursor::default(),
                    current: None,
                    ahead: None,
                })
                .collect(),
            cumulative,
            reading,
            rows: 0,
        })
    }

    /// The sources.
    pub fn sources(&self) -> impl Iterator<Item = &Source> {
        self.readers.iter().map(|r| &r.source)
    }

    /// Where the next batch starts.
    pub fn position(&self) -> Position {
        Position {
            rows: self.rows,
            sources: self.readers.iter().map(|r| r.cursor).collect(),
        }
    }

    /// Goes to `position`, one this loader's configuration gave.
    ///
    /// # Errors
    ///
    /// If it has another number of sources, or a cursor beyond them.
    pub fn seek(&mut self, position: &Position) -> Result<()> {
        if position.sources.len() != self.readers.len() {
            return Err(fail(format!(
                "a position in {} sources, the configuration has {}",
                position.sources.len(),
                self.readers.len()
            )));
        }
        for (reader, cursor) in self.readers.iter_mut().zip(&position.sources) {
            if cursor.shard >= reader.source.shards.len() {
                return Err(fail(format!(
                    "{}: a position in shard {}, of {}",
                    reader.source.name,
                    cursor.shard,
                    reader.source.shards.len()
                )));
            }
            reader.cursor = *cursor;
            reader.current = None;
        }
        self.rows = position.rows;
        Ok(())
    }

    /// The source of row `rank`.
    fn source_of(&self, rank: u64) -> usize {
        if self.readers.len() == 1 {
            return 0;
        }
        let u = (mix(&[self.reading.seed, SOURCES, rank]) >> 11) as f64 / (1u64 << 53) as f64;
        self.cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.readers.len() - 1)
    }

    /// The next batch.
    ///
    /// # Errors
    ///
    /// If a shard cannot be read.
    pub fn next_batch(&mut self) -> Result<Tokens> {
        let n = self.reading.rows * self.reading.length;
        let mut out = Tokens {
            ids: Vec::with_capacity(n),
            targets: Vec::with_capacity(n),
            segments: (self.reading.mask || self.reading.pack == Pack::Joined)
                .then(|| Vec::with_capacity(n)),
        };
        let mut segment = 0u32;
        for _ in 0..self.reading.rows {
            let s = self.source_of(self.rows);
            let reader = &mut self.readers[s];
            let (row, _) = reader.next_row(&self.reading)?;
            reader.append(row, &self.reading, &mut out, &mut segment);
            self.rows += 1;
        }
        Ok(out)
    }

    /// Reads ahead on a thread, `depth` batches.
    pub fn prefetch(self, depth: usize) -> Prefetched {
        let position = self.position();
        let (tx, rx) = sync_channel(depth.max(1));
        let mut loader = self;
        let handle = std::thread::spawn(move || {
            loop {
                let batch = loader.next_batch().map(|t| (t, loader.position()));
                let failed = batch.is_err();
                if tx.send(batch).is_err() || failed {
                    return;
                }
            }
        });
        Prefetched {
            rx: Some(rx),
            handle: Some(handle),
            position,
            waited: Duration::ZERO,
        }
    }
}

/// Batches read ahead by a thread.
pub struct Prefetched {
    rx: Option<Receiver<Result<(Tokens, Position)>>>,
    handle: Option<JoinHandle<()>>,
    position: Position,
    waited: Duration,
}

impl Prefetched {
    /// The next batch.
    ///
    /// # Errors
    ///
    /// Those of the reading.
    pub fn next_batch(&mut self) -> Result<Tokens> {
        let start = Instant::now();
        let got = self
            .rx
            .as_ref()
            .expect("a channel until dropped")
            .recv()
            .map_err(|_| fail("the thread reading the data stopped"))?;
        self.waited += start.elapsed();
        let (tokens, position) = got?;
        self.position = position;
        Ok(tokens)
    }

    /// Where the next batch given starts.
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// The time spent waiting for batches since the last call, which
    /// resets it.
    pub fn take_waited(&mut self) -> Duration {
        std::mem::take(&mut self.waited)
    }
}

impl Drop for Prefetched {
    fn drop(&mut self) {
        // The thread stops at its next send.
        drop(self.rx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A file of llm.c with `tokens`, in `u16`.
    pub(crate) fn write_file(path: &Path, tokens: &[u16]) {
        let tokens: Vec<u32> = tokens.iter().map(|&t| u32::from(t)).collect();
        std::fs::write(path, crate::prepare::token_file_bytes(&tokens, false)).unwrap();
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kapel-data-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn loader(pattern: &str, reading: Reading) -> Loader {
        let source = Source::open(pattern, "train", &reading.cache).unwrap();
        Loader::new(vec![(source, 1.0)], reading).unwrap()
    }

    #[test]
    fn streamed_rows_follow_each_other_and_resume_where_they_were() {
        let dir = temp("stream");
        write_file(&dir.join("t_000.bin"), &(0..30).collect::<Vec<u16>>());
        write_file(&dir.join("t_001.bin"), &(100..112).collect::<Vec<u16>>());
        let pattern = dir.join("t_*.bin").to_string_lossy().into_owned();
        let mut l = loader(&pattern, Reading::new(2, 5));
        let a = l.next_batch().unwrap();
        assert_eq!(a.ids, (0..10).collect::<Vec<u32>>());
        assert_eq!(a.targets, (1..11).collect::<Vec<u32>>());
        // The first file has (30 − 1) / 5 = 5 rows, the second 2.
        let b = l.next_batch().unwrap();
        assert_eq!(b.ids[0], 10);
        let saved = l.position();
        let c = l.next_batch().unwrap();
        assert_eq!(&c.ids[..5], &[20, 21, 22, 23, 24]);
        assert_eq!(&c.ids[5..], &[100, 101, 102, 103, 104]);
        let d = l.next_batch().unwrap();
        assert_eq!(d.ids[0], 105);
        assert_eq!(d.ids[5], 0, "back to the first file");
        let mut again = loader(&pattern, Reading::new(2, 5));
        again.seek(&saved).unwrap();
        assert_eq!(again.next_batch().unwrap(), c);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn other_files_are_refused() {
        let dir = temp("bad");
        std::fs::write(dir.join("x.bin"), vec![0u8; 2048]).unwrap();
        assert!(TokenFile::open(dir.join("x.bin")).is_err());
        write_file(&dir.join("short.bin"), &[1, 2, 3]);
        let mut bytes = std::fs::read(dir.join("short.bin")).unwrap();
        bytes.truncate(bytes.len() - 2);
        std::fs::write(dir.join("short.bin"), bytes).unwrap();
        assert!(TokenFile::open(dir.join("short.bin")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Documents `[9, 1, 2]`, `[9, 3]`, `[9, 4, 5, 6, 7, 8, 10]`, `[9, 11]`:
    /// the separator 9 starts each.
    fn documents(dir: &Path) -> String {
        let tokens = [9, 1, 2, 9, 3, 9, 4, 5, 6, 7, 8, 10, 9, 11];
        write_file(&dir.join("d.bin"), &tokens);
        dir.join("d.bin").to_string_lossy().into_owned()
    }

    #[test]
    fn whole_documents_fill_rows_and_long_ones_are_cut() {
        let dir = temp("documents");
        let path = documents(&dir);
        let mut reading = Reading::new(1, 6);
        reading.pack = Pack::Documents;
        reading.mask = true;
        reading.separator = Some(9);
        let mut l = loader(&path, reading);
        let ignore = crate::IGNORE;
        // [9 1 2 9 3] fits in 6; the next document does not.
        let a = l.next_batch().unwrap();
        assert_eq!(a.ids, vec![9, 1, 2, 9, 3, 9]);
        assert_eq!(a.targets, vec![1, 2, 9, 3, 9, ignore]);
        let s = a.segments.unwrap();
        assert_eq!(s[0], s[2]);
        assert_ne!(s[2], s[3]);
        assert_eq!(s[3], s[4]);
        assert_ne!(s[4], s[5], "the padding is a document of its own");
        // [9 4 5 6 7 8 10]: 7 tokens, a row of 6, then [10] with [9 11].
        let b = l.next_batch().unwrap();
        assert_eq!(b.ids, vec![9, 4, 5, 6, 7, 8]);
        assert_eq!(b.targets, vec![4, 5, 6, 7, 8, 10]);
        let c = l.next_batch().unwrap();
        assert_eq!(c.ids, vec![10, 9, 11, 9, 9, 9]);
        assert_eq!(c.targets, vec![9, 11, ignore, ignore, ignore, ignore]);
        let s = c.segments.unwrap();
        assert_ne!(s[0], s[1]);
        assert_eq!(s[1], s[2]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn padded_rows_hold_one_document_each() {
        let dir = temp("padded");
        let path = documents(&dir);
        let mut reading = Reading::new(4, 6);
        reading.pack = Pack::Padded;
        reading.separator = Some(9);
        let mut l = loader(&path, reading);
        let i = crate::IGNORE;
        let a = l.next_batch().unwrap();
        // [9 1 2], [9 3], [9 4 5 6 7 8] (its 10 left out), [9 11].
        assert_eq!(
            a.ids,
            vec![
                9, 1, 2, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 4, 5, 6, 7, 8, 9, 11, 9, 9, 9, 9
            ]
        );
        assert_eq!(
            a.targets,
            vec![
                1, 2, i, i, i, i, 3, i, i, i, i, i, 4, 5, 6, 7, 8, i, 11, i, i, i, i, i
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn streamed_rows_carry_their_documents() {
        let dir = temp("segments");
        let path = documents(&dir);
        let mut reading = Reading::new(2, 4);
        reading.mask = true;
        reading.separator = Some(9);
        let mut l = loader(&path, reading);
        let a = l.next_batch().unwrap();
        assert_eq!(a.ids, vec![9, 1, 2, 9, 3, 9, 4, 5]);
        let s = a.segments.unwrap();
        // Row 0: [9 1 2 | 9]; row 1: [3 | 9 4 5], a new segment per row.
        assert_eq!((s[0], s[1], s[2]), (s[0], s[0], s[0]));
        assert_ne!(s[3], s[2]);
        assert_ne!(s[4], s[3]);
        assert_ne!(s[5], s[4]);
        assert_eq!((s[6], s[7]), (s[5], s[5]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_same_seed_reads_the_same_rows_and_mixes_by_weight() {
        let dir = temp("mix");
        for (name, base) in [("a", 0u16), ("b", 1000)] {
            for k in 0..3u16 {
                let start = base + 100 * k;
                write_file(
                    &dir.join(format!("{name}_{k}.bin")),
                    &(start..start + 41).collect::<Vec<u16>>(),
                );
            }
        }
        let reading = |seed| {
            let mut r = Reading::new(4, 4);
            r.seed = seed;
            r.shuffle = true;
            r
        };
        let open = |seed| {
            let r = reading(seed);
            let a =
                Source::open(&dir.join("a_*.bin").to_string_lossy(), "train", &r.cache).unwrap();
            let b =
                Source::open(&dir.join("b_*.bin").to_string_lossy(), "train", &r.cache).unwrap();
            Loader::new(vec![(a, 3.0), (b, 1.0)], r).unwrap()
        };
        let read = |l: &mut Loader, n| {
            (0..n)
                .map(|_| l.next_batch().unwrap().ids)
                .collect::<Vec<_>>()
        };
        let (mut x, mut y, mut z) = (open(7), open(7), open(8));
        let first = read(&mut x, 50);
        assert_eq!(first, read(&mut y, 50));
        assert_ne!(first, read(&mut z, 50));
        let from_a = first
            .iter()
            .flatten()
            .step_by(4)
            .filter(|&&t| t < 1000)
            .count();
        let share = from_a as f64 / 200.0;
        assert!((0.65..0.85).contains(&share), "{share}");
        // 3 shards × 10 rows each per source: an epoch of a is 30 rows,
        // each row once.
        let mut l = open(7);
        let mut rows_a = Vec::new();
        while rows_a.len() < 30 {
            let ids = l.next_batch().unwrap().ids;
            rows_a.extend(ids.chunks(4).filter(|r| r[0] < 1000).map(|r| r[0]));
        }
        let mut epoch: Vec<u32> = rows_a[..30].to_vec();
        epoch.sort_unstable();
        epoch.dedup();
        assert_eq!(epoch.len(), 30, "every row of an epoch once");
        // Resumed anywhere, the same rows follow.
        let mut a = open(7);
        read(&mut a, 17);
        let saved = a.position();
        let rest = read(&mut a, 20);
        let mut b = open(7);
        b.seek(&saved).unwrap();
        assert_eq!(read(&mut b, 20), rest);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prefetched_batches_are_the_batches_and_say_where_they_are() {
        let dir = temp("prefetch");
        write_file(&dir.join("p.bin"), &(0..200).collect::<Vec<u16>>());
        let path = dir.join("p.bin").to_string_lossy().into_owned();
        let mut plain = loader(&path, Reading::new(2, 8));
        let mut ahead = loader(&path, Reading::new(2, 8)).prefetch(3);
        for _ in 0..30 {
            assert_eq!(ahead.next_batch().unwrap(), plain.next_batch().unwrap());
            assert_eq!(ahead.position(), &plain.position());
        }
        drop(ahead);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn permutations_are_permutations() {
        for n in [0, 1, 2, 10, 1000] {
            let mut p = permutation(n, 42);
            p.sort_unstable();
            assert_eq!(p, (0..n as u32).collect::<Vec<_>>());
        }
        assert_ne!(permutation(100, 1), permutation(100, 2));
    }
}
