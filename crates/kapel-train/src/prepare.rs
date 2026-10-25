//! Preparing a corpus: documents read from Parquet, JSONL or text files,
//! tokenized on every core, exact duplicates left out, one document in `n`
//! set aside for validation, and the tokens written as shards with the
//! index of their documents ([`crate::dataset`]).
//!
//! The result depends on the inputs, the tokenizer and the options only:
//! the documents are read in order, cut into jobs tokenized by as many
//! threads as asked, and their tokens put back in order before anything is
//! decided, so the same inputs give the same files, bit for bit, whatever
//! the number of threads.
//!
//! A document is the text of one row (the column `text` of a Parquet file,
//! the field `text` of a JSONL line) or of one whole text file. It is
//! encoded as plain text ([`kapel_tokenizers::Tokenizer::encode_ordinary`]):
//! a document that quotes `<|endoftext|>` gets its characters, not the
//! token. Each document is written whole, preceded by the separator, in the
//! shard being filled; a shard is closed before the document that would
//! take it beyond its size.
//!
//! With `chat`, each row or line is an example of fine-tuning instead, a
//! conversation or a prompt and its completion ([`crate::chat`]), made into
//! tokens by the model's chat template; the shards then have a mask of the
//! tokens the loss counts, those of the assistant, and no separator: the
//! template begins each conversation. An example with nothing to learn is
//! left out as empty.

use std::collections::{BTreeMap, HashSet};
use std::io::BufRead as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use kapel_tokenizers::Tokenizer;
use sha2::Digest as _;

use crate::chat::{Encoder, Example};
use crate::data::DataError;
use crate::dataset::{
    ChatInfo, InputInfo, MANIFEST, Manifest, ShardInfo, Split, TokenizerInfo, sha256_hex,
    write_index, write_mask,
};

type Result<T> = std::result::Result<T, DataError>;

fn fail(m: impl Into<String>) -> DataError {
    DataError(m.into())
}

/// What [`prepare`] does.
#[derive(Clone, Debug)]
pub struct PrepareOptions {
    /// The files read, in order: paths, or patterns with `*` in their file
    /// name (the matches in the order of their names). `.parquet`,
    /// `.jsonl` (or `.ndjson`), and `.txt` or `.md` (one document per
    /// file).
    pub inputs: Vec<String>,
    /// A `tokenizer.json`, or a model directory that holds one.
    pub tokenizer: PathBuf,
    /// The directory written.
    pub output: PathBuf,
    /// The column or field of the text; for `chat`, that of the messages.
    pub field: String,
    /// Each row or line is an example of fine-tuning (see the module).
    pub chat: bool,
    /// The token that starts each document, by its text; `None` for the
    /// `bos_token` of the tokenizer's configuration, else its `eos_token`,
    /// else `<|endoftext|>`.
    pub separator: Option<String>,
    /// One document in `val_every` (the first, then every `val_every`-th of
    /// those kept) goes to validation; 0 for none.
    pub val_every: u64,
    /// The size of a shard, in tokens.
    pub shard_tokens: usize,
    /// The threads that tokenize; 0 for one per core.
    pub threads: usize,
    /// Leave out the documents whose text is that of an earlier one.
    pub dedup: bool,
    /// Stop after this many documents read.
    pub limit: Option<u64>,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            tokenizer: PathBuf::new(),
            output: PathBuf::new(),
            field: "text".into(),
            chat: false,
            separator: None,
            val_every: 0,
            shard_tokens: 100_000_000,
            threads: 0,
            dedup: true,
            limit: None,
        }
    }
}

/// Where a preparation is, for its progress.
#[derive(Clone, Copy, Debug, Default)]
pub struct Progress {
    /// The documents read.
    pub documents: u64,
    /// The bytes of text read.
    pub bytes: u64,
    /// The tokens written.
    pub tokens: u64,
    /// The seconds since the start.
    pub seconds: f64,
}

/// A document read: a text (`None` for a null value), or an example.
enum Doc {
    Text(Option<String>),
    Json(serde_json::Value),
}

/// Documents read, each with its input.
type Job = (u64, Vec<(usize, Doc)>);

/// A document tokenized: its input, its text's hash, its tokens, and for an
/// example the tokens the loss counts.
struct Done {
    input: usize,
    bytes: usize,
    hash: [u8; 16],
    tokens: Vec<u32>,
    mask: Option<Vec<bool>>,
}

/// What makes documents into tokens, on one thread.
enum Coder {
    Text(Tokenizer),
    Chat(Encoder, String),
}

/// The documents of one job per message, so many go through a channel at
/// once.
const JOB_DOCUMENTS: usize = 256;

/// Reads, tokenizes and writes the corpus, calling `progress` about once a
/// second.
///
/// # Errors
///
/// If an input cannot be read, the tokenizer cannot be loaded or has no such
/// separator, or a file cannot be written.
pub fn prepare(options: &PrepareOptions, mut progress: impl FnMut(&Progress)) -> Result<Manifest> {
    let start = Instant::now();
    let tokenizer_file = if options.tokenizer.is_dir() {
        options.tokenizer.join("tokenizer.json")
    } else {
        options.tokenizer.clone()
    };
    let tokenizer_bytes = std::fs::read(&tokenizer_file)
        .map_err(|e| fail(format!("{}: {e}", tokenizer_file.display())))?;
    let load = || {
        Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| fail(format!("{}: {e}", tokenizer_file.display())))
    };
    let tokenizer = load()?;
    let separator = separator(&tokenizer, &tokenizer_file, options.separator.as_deref())?;
    let tokenizer_dir = tokenizer_file
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let chat = if options.chat {
        let encoder = Encoder::from_dir(&tokenizer_dir)?;
        let template = encoder
            .template()
            .ok_or_else(|| fail(format!("{}: no chat template", tokenizer_dir.display())))?;
        Some(ChatInfo {
            template_sha256: sha256_hex(template.source().as_bytes()),
        })
    } else {
        None
    };
    let vocab = tokenizer.vocab_size();
    let mut paths = Vec::new();
    for pattern in &options.inputs {
        paths.extend(crate::data::matching(pattern)?);
    }
    if paths.is_empty() {
        return Err(fail("no input"));
    }
    for p in &paths {
        kind(p)?;
    }
    if options.shard_tokens == 0 || options.shard_tokens > i32::MAX as usize {
        return Err(fail(format!(
            "a shard of {} tokens: between 1 and {}",
            options.shard_tokens,
            i32::MAX
        )));
    }
    std::fs::create_dir_all(&options.output)
        .map_err(|e| fail(format!("{}: {e}", options.output.display())))?;
    let threads = if options.threads == 0 {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    } else {
        options.threads
    };
    // One tokenizer per thread: their caches would contend for one lock.
    let mut coders = Vec::with_capacity(threads);
    let mut first = Some(tokenizer);
    for _ in 0..threads {
        coders.push(if options.chat {
            Coder::Chat(Encoder::from_dir(&tokenizer_dir)?, options.field.clone())
        } else {
            Coder::Text(match first.take() {
                Some(t) => t,
                None => load()?,
            })
        });
    }
    let wide = vocab > 1 << 16;
    let separated = !options.chat;
    let mut train = SplitWriter::new(&options.output, "train", options.shard_tokens, wide);
    let mut val = SplitWriter::new(&options.output, "val", options.shard_tokens, wide);
    for split in [&mut train, &mut val] {
        split.separated = separated;
        split.masked = options.chat;
    }
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut counts = vec![0u64; paths.len()];
    let (mut documents, mut duplicates, mut empty, mut kept, mut bytes) = (0u64, 0, 0, 0u64, 0u64);
    let mut last_report = Instant::now();

    let (job_tx, job_rx) = sync_channel::<Result<Job>>(2 * threads);
    let (done_tx, done_rx) = sync_channel::<(u64, Result<Vec<Done>>)>(2 * threads);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let field = options.field.as_str();
    let limit = options.limit;
    let as_json = options.chat;
    let outcome: Result<()> = std::thread::scope(|scope| {
        let paths = &paths;
        // The channels' ends move in here: dropped on an error, they stop
        // the threads instead of letting them read the whole corpus.
        let (job_rx, done_rx) = (job_rx, done_rx);
        scope.spawn(move || read_all(paths, field, as_json, limit, &job_tx));
        for coder in coders {
            let (rx, tx) = (Arc::clone(&job_rx), done_tx.clone());
            scope.spawn(move || tokenize(&coder, &rx, &tx));
        }
        drop((done_tx, job_rx));
        let mut pending: BTreeMap<u64, Result<Vec<Done>>> = BTreeMap::new();
        let mut next = 0u64;
        let mut result = Ok(());
        for (seq, done) in done_rx.iter() {
            if result.is_err() {
                break;
            }
            pending.insert(seq, done);
            while let Some(done) = pending.remove(&next) {
                next += 1;
                if result.is_err() {
                    continue;
                }
                let done = match done {
                    Ok(d) => d,
                    Err(e) => {
                        result = Err(e);
                        continue;
                    }
                };
                for d in done {
                    documents += 1;
                    counts[d.input] += 1;
                    bytes += d.bytes as u64;
                    let nothing_learnt = d.mask.as_ref().is_some_and(|m| !m.contains(&true));
                    if d.tokens.is_empty() || nothing_learnt {
                        empty += 1;
                        continue;
                    }
                    if options.dedup && !seen.insert(d.hash) {
                        duplicates += 1;
                        continue;
                    }
                    let to_val = options.val_every > 0 && kept.is_multiple_of(options.val_every);
                    kept += 1;
                    let split = if to_val { &mut val } else { &mut train };
                    if let Err(e) = split.add(separator, &d.tokens, d.mask.as_deref()) {
                        result = Err(e);
                    }
                }
                if last_report.elapsed().as_secs_f64() >= 1.0 {
                    last_report = Instant::now();
                    progress(&Progress {
                        documents,
                        bytes,
                        tokens: train.split.tokens
                            + val.split.tokens
                            + train.pending()
                            + val.pending(),
                        seconds: start.elapsed().as_secs_f64(),
                    });
                }
            }
        }
        drop(done_rx);
        if result.is_ok() {
            // A reading error comes as a job past the last one.
            if let Some(Err(e)) = pending.remove(&u64::MAX) {
                result = Err(e);
            } else if !pending.is_empty() {
                result = Err(fail("a tokenizing thread stopped"));
            }
        }
        result
    });
    outcome?;
    train.finish()?;
    val.finish()?;
    let manifest = Manifest {
        kapel_dataset: 1,
        tokenizer: TokenizerInfo {
            sha256: sha256_hex(&tokenizer_bytes),
            vocab,
        },
        separator,
        separated,
        chat,
        documents,
        duplicates,
        empty,
        val_every: options.val_every,
        inputs: paths
            .iter()
            .zip(&counts)
            .map(|(p, &documents)| InputInfo {
                path: p.to_string_lossy().into_owned(),
                documents,
            })
            .collect(),
        train: train.split,
        val: (options.val_every > 0).then_some(val.split),
    };
    let json = serde_json::to_vec_pretty(&manifest).expect("a manifest serializes");
    let path = options.output.join(MANIFEST);
    std::fs::write(&path, json).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    progress(&Progress {
        documents,
        bytes,
        tokens: manifest.train.tokens + manifest.val.as_ref().map_or(0, |v| v.tokens),
        seconds: start.elapsed().as_secs_f64(),
    });
    Ok(manifest)
}

/// The id of the separator: the one named, or the configuration's
/// `bos_token`, `eos_token`, or `<|endoftext|>`.
fn separator(tokenizer: &Tokenizer, file: &Path, named: Option<&str>) -> Result<u32> {
    let id = |text: &str| {
        tokenizer
            .token_to_id(text)
            .ok_or_else(|| fail(format!("the tokenizer has no token {text:?}")))
    };
    if let Some(text) = named {
        return id(text);
    }
    let config = file
        .parent()
        .map(|d| d.join("tokenizer_config.json"))
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    for key in ["bos_token", "eos_token"] {
        let text = config.as_ref().and_then(|c| match c.get(key)? {
            serde_json::Value::String(s) => Some(s.clone()),
            v => v.get("content")?.as_str().map(str::to_string),
        });
        if let Some(found) = text.and_then(|t| tokenizer.token_to_id(&t)) {
            return Ok(found);
        }
    }
    id("<|endoftext|>").map_err(|_| {
        fail("the tokenizer names no beginning or end of text: give the separator (--separator)")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Parquet,
    Jsonl,
    Text,
}

fn kind(path: &Path) -> Result<Kind> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "parquet" => Ok(Kind::Parquet),
        "jsonl" | "ndjson" => Ok(Kind::Jsonl),
        "txt" | "md" => Ok(Kind::Text),
        _ => Err(fail(format!(
            "{}: not .parquet, .jsonl, .ndjson, .txt or .md",
            path.display()
        ))),
    }
}

/// Reads every input in order, sending jobs of documents: texts, or with
/// `as_json` the rows and lines as they are, examples.
fn read_all(
    paths: &[PathBuf],
    field: &str,
    as_json: bool,
    limit: Option<u64>,
    tx: &SyncSender<Result<Job>>,
) {
    let mut seq = 0u64;
    let mut job: Vec<(usize, Doc)> = Vec::with_capacity(JOB_DOCUMENTS);
    let mut read = 0u64;
    let mut send = |job: &mut Vec<(usize, Doc)>| -> bool {
        let full = std::mem::replace(job, Vec::with_capacity(JOB_DOCUMENTS));
        let ok = tx.send(Ok((seq, full))).is_ok();
        seq += 1;
        ok
    };
    for (input, path) in paths.iter().enumerate() {
        let mut push = |doc: Doc| -> bool {
            if limit.is_some_and(|l| read >= l) {
                return false;
            }
            read += 1;
            job.push((input, doc));
            job.len() < JOB_DOCUMENTS || send(&mut job)
        };
        let outcome = match (kind(path), as_json) {
            (Ok(Kind::Parquet), false) => {
                read_parquet(path, field, &mut |t: Option<String>| push(Doc::Text(t)))
            }
            (Ok(Kind::Jsonl), false) => {
                read_jsonl(path, field, &mut |t: Option<String>| push(Doc::Text(t)))
            }
            (Ok(Kind::Parquet), true) => read_parquet_rows(path, &mut |v| push(Doc::Json(v))),
            (Ok(Kind::Jsonl), true) => read_jsonl_values(path, &mut |v| push(Doc::Json(v))),
            (Ok(Kind::Text), false) => std::fs::read(path)
                .map(|b| {
                    push(Doc::Text(Some(String::from_utf8_lossy(&b).into_owned())));
                })
                .map_err(|e| fail(format!("{}: {e}", path.display()))),
            (Ok(Kind::Text), true) => Err(fail(format!(
                "{}: examples come from .parquet or .jsonl files",
                path.display()
            ))),
            (Err(e), _) => Err(e),
        };
        if let Err(e) = outcome {
            let _ = tx.send(Err(e));
            return;
        }
        if limit.is_some_and(|l| read >= l) {
            break;
        }
    }
    if !job.is_empty() {
        send(&mut job);
    }
}

/// The values of column `field` of a Parquet file, one row group at a time;
/// stops when `push` says so.
fn read_parquet(
    path: &Path,
    field: &str,
    push: &mut dyn FnMut(Option<String>) -> bool,
) -> Result<()> {
    use parquet::column::reader::ColumnReader;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let at = |e: &dyn std::fmt::Display| fail(format!("{}: {e}", path.display()));
    let file = std::fs::File::open(path).map_err(|e| at(&e))?;
    let reader = SerializedFileReader::new(file).map_err(|e| at(&e))?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let column = (0..schema.num_columns())
        .find(|&i| schema.column(i).path().string() == field)
        .ok_or_else(|| at(&format!("no column {field:?}")))?;
    let max_def = schema.column(column).max_def_level();
    for g in 0..reader.num_row_groups() {
        let group = reader.get_row_group(g).map_err(|e| at(&e))?;
        let rows = usize::try_from(group.metadata().num_rows()).unwrap_or(0);
        let ColumnReader::ByteArrayColumnReader(mut values) =
            group.get_column_reader(column).map_err(|e| at(&e))?
        else {
            return Err(at(&format!("the column {field:?} is not text")));
        };
        let mut read = 0;
        while read < rows {
            let want = (rows - read).min(4096);
            let mut def = Vec::new();
            let mut texts = Vec::new();
            let (records, _, _) = values
                .read_records(want, (max_def > 0).then_some(&mut def), None, &mut texts)
                .map_err(|e| at(&e))?;
            if records == 0 {
                break;
            }
            read += records;
            let mut texts = texts.into_iter();
            for r in 0..records {
                let present = max_def == 0 || def.get(r).is_some_and(|&d| d == max_def);
                let text = if present {
                    texts
                        .next()
                        .map(|b| String::from_utf8_lossy(b.data()).into_owned())
                } else {
                    None
                };
                if !push(text) {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// The records of the files `pattern` names (`*` in the file name), in
/// order: the rows of `.parquet` files and the lines of `.jsonl` files as
/// JSON values, `limit` at most. For the prompts and pairs that fine-tuning
/// reads whole.
///
/// # Errors
///
/// If a file cannot be read, or is neither Parquet nor JSONL.
pub fn read_records(pattern: &str, limit: Option<usize>) -> Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for path in crate::data::matching(pattern)? {
        let mut push = |v: serde_json::Value| {
            if limit.is_some_and(|l| out.len() >= l) {
                return false;
            }
            out.push(v);
            true
        };
        match kind(&path)? {
            Kind::Parquet => read_parquet_rows(&path, &mut push)?,
            Kind::Jsonl => read_jsonl_values(&path, &mut push)?,
            Kind::Text => {
                return Err(fail(format!(
                    "{}: records come from .parquet or .jsonl files",
                    path.display()
                )));
            }
        }
    }
    Ok(out)
}

/// Each row of a Parquet file as a JSON object; stops when `push` says so.
fn read_parquet_rows(path: &Path, push: &mut dyn FnMut(serde_json::Value) -> bool) -> Result<()> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let at = |e: &dyn std::fmt::Display| fail(format!("{}: {e}", path.display()));
    let file = std::fs::File::open(path).map_err(|e| at(&e))?;
    let reader = SerializedFileReader::new(file).map_err(|e| at(&e))?;
    for row in reader.get_row_iter(None).map_err(|e| at(&e))? {
        if !push(row.map_err(|e| at(&e))?.to_json_value()) {
            break;
        }
    }
    Ok(())
}

/// Each line of a JSONL file as a JSON value; stops when `push` says so.
fn read_jsonl_values(path: &Path, push: &mut dyn FnMut(serde_json::Value) -> bool) -> Result<()> {
    let file = std::fs::File::open(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| fail(format!("{}: {e}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str(&line)
            .map_err(|e| fail(format!("{}:{}: {e}", path.display(), n + 1)))?;
        if !push(value) {
            break;
        }
    }
    Ok(())
}

/// The field `field` of each line of a JSONL file; stops when `push` says so.
fn read_jsonl(
    path: &Path,
    field: &str,
    push: &mut dyn FnMut(Option<String>) -> bool,
) -> Result<()> {
    let file = std::fs::File::open(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| fail(format!("{}: {e}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let at = |m: String| fail(format!("{}:{}: {m}", path.display(), n + 1));
        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| at(e.to_string()))?;
        let text = match value.get(field) {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Null) => None,
            Some(_) => return Err(at(format!("the field {field:?} is not text"))),
            None => return Err(at(format!("no field {field:?}"))),
        };
        if !push(text) {
            break;
        }
    }
    Ok(())
}

/// A document made into tokens.
fn encode(coder: &Coder, input: usize, doc: Doc) -> Result<Done> {
    let at = |e: &dyn std::fmt::Display| fail(format!("a document of input {input}: {e}"));
    let digest = |bytes: &[u8]| -> [u8; 16] {
        let hash: [u8; 32] = sha2::Sha256::digest(bytes).into();
        hash[..16].try_into().expect("16 bytes")
    };
    match (coder, doc) {
        (Coder::Text(tokenizer), Doc::Text(text)) => {
            let text = text.unwrap_or_default();
            let tokens = tokenizer.encode_ordinary(&text).map_err(|e| at(&e))?;
            Ok(Done {
                input,
                bytes: text.len(),
                hash: digest(text.as_bytes()),
                tokens,
                mask: None,
            })
        }
        (Coder::Chat(encoder, field), Doc::Json(value)) => {
            let example = Example::from_json(&value, field).map_err(|e| at(&e))?;
            let (tokens, mask) = encoder.encode(&example).map_err(|e| at(&e))?;
            Ok(Done {
                input,
                bytes: example.bytes(),
                hash: digest(value.to_string().as_bytes()),
                tokens,
                mask: Some(mask),
            })
        }
        _ => Err(at(&"a document of the other kind")),
    }
}

/// Tokenizes jobs until there are none.
fn tokenize(
    coder: &Coder,
    jobs: &Mutex<Receiver<Result<Job>>>,
    done: &SyncSender<(u64, Result<Vec<Done>>)>,
) {
    loop {
        let job = {
            let rx = jobs.lock().unwrap_or_else(|e| e.into_inner());
            rx.recv()
        };
        let (seq, result) = match job {
            Err(_) => return,
            // A reading error ends the preparation: it takes the place of
            // the next job.
            Ok(Err(e)) => (u64::MAX, Err(e)),
            Ok(Ok((seq, docs))) => {
                let result = docs
                    .into_iter()
                    .map(|(input, doc)| encode(coder, input, doc))
                    .collect();
                (seq, result)
            }
        };
        if done.send((seq, result)).is_err() {
            return;
        }
    }
}

/// The shards of one split, written as they fill.
struct SplitWriter {
    dir: PathBuf,
    name: &'static str,
    shard_tokens: usize,
    wide: bool,
    /// Whether the separator goes before each document.
    separated: bool,
    /// Whether the shards have a mask.
    masked: bool,
    tokens: Vec<u32>,
    mask: Vec<bool>,
    starts: Vec<u64>,
    split: Split,
}

impl SplitWriter {
    fn new(dir: &Path, name: &'static str, shard_tokens: usize, wide: bool) -> Self {
        Self {
            dir: dir.to_path_buf(),
            name,
            shard_tokens,
            wide,
            separated: true,
            masked: false,
            tokens: Vec::new(),
            mask: Vec::new(),
            starts: Vec::new(),
            split: Split::default(),
        }
    }

    fn pending(&self) -> u64 {
        self.tokens.len() as u64
    }

    fn add(&mut self, separator: u32, document: &[u32], mask: Option<&[bool]>) -> Result<()> {
        let size = document.len() + usize::from(self.separated);
        if !self.tokens.is_empty() && self.tokens.len() + size > self.shard_tokens {
            self.flush()?;
        }
        if self.tokens.len() + size > i32::MAX as usize {
            return Err(fail("a document of more than 2³¹ tokens"));
        }
        self.starts.push(self.tokens.len() as u64);
        if self.separated {
            self.tokens.push(separator);
        }
        self.tokens.extend_from_slice(document);
        if self.masked {
            if self.separated {
                self.mask.push(false);
            }
            match mask {
                Some(m) => self.mask.extend_from_slice(m),
                None => self.mask.extend(std::iter::repeat_n(true, document.len())),
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.tokens.is_empty() {
            return Ok(());
        }
        let rank = self.split.shards.len();
        let file = format!("{}_{rank:06}.bin", self.name);
        let index = format!("{}_{rank:06}.idx", self.name);
        let bytes = token_file_bytes(&self.tokens, self.wide);
        let path = self.dir.join(&file);
        std::fs::write(&path, &bytes).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        write_index(&self.dir.join(&index), &self.starts)?;
        let mask = if self.masked {
            let name = format!("{}_{rank:06}.mask", self.name);
            write_mask(&self.dir.join(&name), &self.mask)?;
            let trained = self.mask.iter().filter(|&&b| b).count() as u64;
            *self.split.trained.get_or_insert(0) += trained;
            Some(name)
        } else {
            None
        };
        let (tokens, documents) = (self.tokens.len() as u64, self.starts.len() as u64);
        self.split.tokens += tokens;
        self.split.documents += documents;
        self.split.shards.push(ShardInfo {
            file,
            index,
            tokens,
            documents,
            sha256: sha256_hex(&bytes),
            mask,
        });
        self.tokens.clear();
        self.mask.clear();
        self.starts.clear();
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush()
    }
}

/// A file of tokens in llm.c's format: `u16` (magic 20240520, version 1) for
/// vocabularies of at most 65,536 tokens, `u32` (20240801, version 7)
/// beyond.
pub fn token_file_bytes(tokens: &[u32], wide: bool) -> Vec<u8> {
    let size = if wide { 4 } else { 2 };
    let mut bytes = vec![0u8; 1024];
    let (magic, version) = if wide {
        (20_240_801i32, 7i32)
    } else {
        (20_240_520, 1)
    };
    bytes[0..4].copy_from_slice(&magic.to_le_bytes());
    bytes[4..8].copy_from_slice(&version.to_le_bytes());
    bytes[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
    bytes.reserve(tokens.len() * size);
    for &t in tokens {
        if wide {
            bytes.extend_from_slice(&t.to_le_bytes());
        } else {
            bytes.extend_from_slice(&(t as u16).to_le_bytes());
        }
    }
    bytes
}
