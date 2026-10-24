//! Token files, and batches read from them in order.
//!
//! The files are llm.c's shards: a header of 256 little-endian `i32` (the
//! magic number, the version, the number of tokens), then the tokens,
//! little-endian `u16` (magic 20240520, GPT-2's vocabulary) or `u32` (magic
//! 20240801, Llama 3's). They are mapped, not read. FineWeb as llm.c and
//! modded-nanogpt publish it is in this format.
//!
//! A [`Loader`] reads batches one after the other: each takes the next
//! `rows × length + 1` tokens of the current file (the last one only as a
//! target) and moves on by `rows × length`; a file that has not enough left
//! hands over to the next, and the last to the first. Its [`Position`] is
//! part of a checkpoint: a run that resumes reads the tokens it would have
//! read.

use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

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
        assert!(start + count <= self.len, "tokens beyond the file");
        let bytes = &self.map[HEADER..];
        if self.wide {
            bytes[4 * start..4 * (start + count)]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        } else {
            bytes[2 * start..2 * (start + count)]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u32::from(u16::from_le_bytes(*b)))
                .collect()
        }
    }
}

/// The files of `pattern`: a path, or a path whose file name holds `*`, the
/// files that match taken in the order of their names.
///
/// # Errors
///
/// If no file matches, or one cannot be opened.
pub fn files(pattern: &str) -> Result<Vec<TokenFile>> {
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
            std::fs::read_dir(dir).map_err(|e| DataError(format!("{}: {e}", dir.display())))?;
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
        return Err(DataError(format!("no file matches {pattern}")));
    }
    paths.iter().map(TokenFile::open).collect()
}

/// Where a [`Loader`] is: the file, and the first token of the next batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Position {
    /// The file, by its rank in the sorted list.
    pub file: usize,
    /// The first token of the next batch in it.
    pub token: usize,
}

/// A batch read: token ids, and the token each should predict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokens {
    /// `rows × length` ids, row after row.
    pub ids: Vec<u32>,
    /// As many targets: the ids shifted by one.
    pub targets: Vec<u32>,
}

/// Batches read in order from files of tokens.
pub struct Loader {
    files: Vec<TokenFile>,
    rows: usize,
    length: usize,
    position: Position,
}

impl Loader {
    /// A loader of batches of `rows × length` tokens, from the start.
    ///
    /// # Errors
    ///
    /// If no file holds one batch.
    pub fn new(files: Vec<TokenFile>, rows: usize, length: usize) -> Result<Self> {
        let need = rows * length + 1;
        if !files.iter().any(|f| f.len() >= need) {
            return Err(DataError(format!(
                "no file holds a batch of {rows} × {length} + 1 tokens"
            )));
        }
        Ok(Self {
            files,
            rows,
            length,
            position: Position::default(),
        })
    }

    /// The files.
    pub fn files(&self) -> &[TokenFile] {
        &self.files
    }

    /// Where the next batch starts.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Goes to `position`, a position this loader gave.
    ///
    /// # Errors
    ///
    /// If it is beyond the files.
    pub fn seek(&mut self, position: Position) -> Result<()> {
        let fits = self
            .files
            .get(position.file)
            .is_some_and(|f| position.token <= f.len());
        if !fits {
            return Err(DataError(format!(
                "position {position:?} beyond the {} files",
                self.files.len()
            )));
        }
        self.position = position;
        Ok(())
    }

    /// The next batch.
    pub fn next_batch(&mut self) -> Tokens {
        let n = self.rows * self.length;
        loop {
            let p = self.position;
            let file = &self.files[p.file];
            if p.token + n < file.len() {
                let tokens = file.tokens(p.token, n + 1);
                self.position.token += n;
                return Tokens {
                    ids: tokens[..n].to_vec(),
                    targets: tokens[1..].to_vec(),
                };
            }
            self.position = Position {
                file: (p.file + 1) % self.files.len(),
                token: 0,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of llm.c with `tokens`, in `u16`.
    pub(crate) fn write_file(path: &Path, tokens: &[u16]) {
        let mut bytes = vec![0u8; HEADER];
        bytes[0..4].copy_from_slice(&20_240_520i32.to_le_bytes());
        bytes[4..8].copy_from_slice(&1i32.to_le_bytes());
        bytes[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn batches_follow_each_other_and_resume_where_they_were() {
        let dir = std::env::temp_dir().join(format!("kapel-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_file(&dir.join("t_000.bin"), &(0..30).collect::<Vec<u16>>());
        write_file(&dir.join("t_001.bin"), &(100..112).collect::<Vec<u16>>());
        let pattern = dir.join("t_*.bin").to_string_lossy().into_owned();
        let mut loader = Loader::new(files(&pattern).unwrap(), 2, 5).unwrap();
        let a = loader.next_batch();
        assert_eq!(a.ids, (0..10).collect::<Vec<u32>>());
        assert_eq!(a.targets, (1..11).collect::<Vec<u32>>());
        let b = loader.next_batch();
        assert_eq!(b.ids[0], 10);
        // 20 + 10 + 1 > 30: the second file, then back to the first.
        let saved = loader.position();
        let c = loader.next_batch();
        assert_eq!(c.ids[0], 100);
        let d = loader.next_batch();
        assert_eq!(d.ids[0], 0);
        let mut again = Loader::new(files(&pattern).unwrap(), 2, 5).unwrap();
        again.seek(saved).unwrap();
        assert_eq!(again.next_batch(), c);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn other_files_are_refused() {
        let dir = std::env::temp_dir().join(format!("kapel-data-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.bin"), vec![0u8; 2048]).unwrap();
        assert!(TokenFile::open(dir.join("x.bin")).is_err());
        write_file(&dir.join("short.bin"), &[1, 2, 3]);
        let mut bytes = std::fs::read(dir.join("short.bin")).unwrap();
        bytes.truncate(bytes.len() - 2);
        std::fs::write(dir.join("short.bin"), bytes).unwrap();
        assert!(TokenFile::open(dir.join("short.bin")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
