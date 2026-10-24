//! `kapel prepare`: a corpus made into shards of tokens for `kapel train`.

use std::path::PathBuf;

use kapel_train::prepare::{PrepareOptions, prepare};

pub(crate) const USAGE: &str = "\
usage: kapel prepare INPUT... --tokenizer PATH --output DIR [options]

Reads the documents of each INPUT, in order: a .parquet file (a row each, its
column `text`), a .jsonl or .ndjson file (a line each, its field `text`), or a
.txt or .md file (one document). An INPUT may hold `*` in its file name. The
documents are tokenized on every core, the exact duplicates and the empty ones
left out, and the tokens written into DIR as shards in llm.c's format, each
document preceded by the separator, with the index of the documents of each
shard and dataset.json, which kapel train reads (data.train = \"DIR\"). The same
inputs give the same files, whatever the number of threads.

options:
  --tokenizer PATH         a tokenizer.json, or a model directory holding one
  --output DIR             the directory written
  --field NAME             the column or field of the text (default text)
  --separator TEXT         the token that starts each document (default the
                           tokenizer's bos_token, else its eos_token, else
                           <|endoftext|>)
  --val-every N            one document in N goes to the validation split
                           (default 0: none)
  --shard-tokens N         the tokens of a shard (default 100000000)
  --threads N              the threads that tokenize (default one per core)
  --keep-duplicates        keep the documents seen before
  --limit N                read N documents at most
  --quiet                  print nothing but errors
";

fn number<T: std::str::FromStr>(name: &str, value: Option<&String>) -> Result<T, String> {
    let v = value.ok_or(format!("{name} needs a number"))?;
    v.parse().map_err(|_| format!("{name}: not a number: {v}"))
}

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let mut options = PrepareOptions::default();
    let (mut tokenizer, mut output, mut quiet) = (None, None, false);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--tokenizer" => tokenizer = rest.next().map(PathBuf::from),
            "--output" => output = rest.next().map(PathBuf::from),
            "--field" => options.field = rest.next().ok_or("--field needs a name")?.clone(),
            "--separator" => {
                options.separator = Some(rest.next().ok_or("--separator needs a token")?.clone())
            }
            "--val-every" => options.val_every = number(arg, rest.next())?,
            "--shard-tokens" => options.shard_tokens = number(arg, rest.next())?,
            "--threads" => options.threads = number(arg, rest.next())?,
            "--limit" => options.limit = Some(number(arg, rest.next())?),
            "--keep-duplicates" => options.dedup = false,
            "--quiet" => quiet = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other => options.inputs.push(other.to_string()),
        }
    }
    options.tokenizer = tokenizer.ok_or(format!("--tokenizer is needed\n\n{USAGE}"))?;
    options.output = output.ok_or(format!("--output is needed\n\n{USAGE}"))?;
    if options.inputs.is_empty() {
        return Err(USAGE.to_string());
    }
    let manifest = prepare(&options, |p| {
        if !quiet {
            eprint!(
                "\rprepare: {} documents, {:.1} MB/s, {} tokens, {:.0} s   ",
                p.documents,
                p.bytes as f64 / 1e6 / p.seconds.max(1e-9),
                p.tokens,
                p.seconds
            );
        }
    })
    .map_err(|e| e.to_string())?;
    if !quiet {
        eprintln!();
        let val = manifest.val.as_ref();
        eprintln!(
            "prepare: {} documents read, {} duplicates and {} empty left out; train {} tokens in {} shards{}",
            manifest.documents,
            manifest.duplicates,
            manifest.empty,
            manifest.train.tokens,
            manifest.train.shards.len(),
            val.map(|v| format!(", val {} tokens in {} shards", v.tokens, v.shards.len()))
                .unwrap_or_default()
        );
    }
    Ok(())
}
