//! Preparing a corpus: Parquet, JSONL and text read, the same files whatever
//! the number of threads, duplicates and empty documents left out, one
//! document in `n` set aside, and the shards read back as documents.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kapel_tokenizers::Tokenizer;
use kapel_train::data::{Pack, Reading, Source, TokenFile};
use kapel_train::dataset::{Manifest, read_index};
use kapel_train::prepare::{PrepareOptions, prepare};

fn tokenizer_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gpt2")
}

/// Whether the tokenizer of `data/parity/gpt2` is there; without it (in
/// continuous integration) the test is skipped and says so, and with
/// `KAPEL_REQUIRE_PARITY=1` it fails instead.
fn have_tokenizer() -> bool {
    if tokenizer_dir().join("tokenizer.json").exists() {
        return true;
    }
    assert!(
        std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
        "data/parity/gpt2 is missing, and KAPEL_REQUIRE_PARITY=1"
    );
    let _ = writeln!(
        std::io::stderr(),
        "skipped: data/parity/gpt2 missing (see tools/parity/README.md)"
    );
    false
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kapel-prepare-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A Parquet file with a column `text`, `None` for a null.
fn parquet(path: &Path, texts: &[Option<&str>]) {
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    let schema = Arc::new(
        parquet::schema::parser::parse_message_type(
            "message doc { optional binary text (UTF8); optional int64 score; }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    // Two row groups, so reading goes on from one to the next.
    for part in texts.chunks(texts.len().div_ceil(2)) {
        let mut group = writer.next_row_group().unwrap();
        let mut column = 0;
        while let Some(mut c) = group.next_column().unwrap() {
            let def: Vec<i16> = part.iter().map(|t| i16::from(t.is_some())).collect();
            if column == 0 {
                let values: Vec<ByteArray> = part.iter().flatten().map(|t| (*t).into()).collect();
                c.typed::<ByteArrayType>()
                    .write_batch(&values, Some(&def), None)
                    .unwrap();
            } else {
                let values: Vec<i64> = part.iter().flatten().map(|t| t.len() as i64).collect();
                c.typed::<parquet::data_type::Int64Type>()
                    .write_batch(&values, Some(&def), None)
                    .unwrap();
            }
            c.close().unwrap();
            column += 1;
        }
        group.close().unwrap();
    }
    writer.close().unwrap();
}

const INJECTED: &str = "a text that quotes <|endoftext|> in its middle";

/// The inputs: 12 documents read, of which a duplicate across files, a
/// duplicate within one, an empty one and a null.
fn inputs(dir: &Path) -> Vec<String> {
    let lines = [
        r#"{"text": "The first document, about rivers.", "id": 1}"#,
        r#"{"text": "The second one speaks of mountains and of the snow on them."}"#,
        "",
        r#"{"text": "The first document, about rivers."}"#,
        r#"{"text": ""}"#,
        &format!(r#"{{"text": "{INJECTED}"}}"#),
        r#"{"text": "Une phrase en français, avec des accents : été, où, ça."}"#,
    ];
    std::fs::write(dir.join("a.jsonl"), lines.join("\n")).unwrap();
    parquet(
        &dir.join("b.parquet"),
        &[
            Some("A row of a Parquet file."),
            None,
            Some("The second one speaks of mountains and of the snow on them."),
            Some("Another row, the fourth, somewhat longer than the others so far."),
            Some("And the last row of the file."),
        ],
    );
    std::fs::write(
        dir.join("c.txt"),
        "A whole text file is one document.\nEven over lines.\n",
    )
    .unwrap();
    ["a.jsonl", "b.parquet", "c.txt"]
        .iter()
        .map(|n| dir.join(n).to_string_lossy().into_owned())
        .collect()
}

fn options(dir: &Path, out: &str, threads: usize) -> PrepareOptions {
    PrepareOptions {
        inputs: inputs(dir),
        tokenizer: tokenizer_dir(),
        output: dir.join(out),
        val_every: 3,
        shard_tokens: 40,
        threads,
        ..PrepareOptions::default()
    }
}

#[test]
fn the_same_inputs_give_the_same_files_whatever_the_threads() {
    if !have_tokenizer() {
        return;
    }
    let dir = temp("threads");
    let one = prepare(&options(&dir, "one", 1), |_| {}).unwrap();
    let many = prepare(&options(&dir, "many", 5), |_| {}).unwrap();
    assert_eq!(one.train, many.train);
    assert_eq!(one.val, many.val);
    for shard in one
        .train
        .shards
        .iter()
        .chain(&one.val.as_ref().unwrap().shards)
    {
        for name in [&shard.file, &shard.index] {
            assert_eq!(
                std::fs::read(dir.join("one").join(name)).unwrap(),
                std::fs::read(dir.join("many").join(name)).unwrap(),
                "{name}"
            );
        }
    }
    assert_eq!(one.documents, 12);
    assert_eq!(one.empty, 2, "the empty text and the null");
    assert_eq!(one.duplicates, 2);
    assert_eq!(
        one.inputs.iter().map(|i| i.documents).collect::<Vec<_>>(),
        [6, 5, 1]
    );
    let kept = one.documents - one.empty - one.duplicates;
    let val = one.val.as_ref().unwrap();
    assert_eq!(val.documents, kept.div_ceil(3));
    assert_eq!(one.train.documents + val.documents, kept);
    assert!(one.train.shards.len() > 2, "shards of at most 40 tokens");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_shards_hold_the_documents_whole_and_in_order() {
    if !have_tokenizer() {
        return;
    }
    let dir = temp("documents");
    let manifest = prepare(&options(&dir, "out", 3), |_| {}).unwrap();
    let out = dir.join("out");
    assert_eq!(Manifest::read(&out.join("dataset.json")).unwrap(), manifest);
    let tokenizer = Tokenizer::from_file(tokenizer_dir().join("tokenizer.json")).unwrap();
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap();
    assert_eq!(manifest.separator, eot, "GPT-2's bos_token");
    let mut texts = Vec::new();
    for split in [&manifest.train, manifest.val.as_ref().unwrap()] {
        for shard in &split.shards {
            let file = TokenFile::open(out.join(&shard.file)).unwrap();
            let starts = read_index(&out.join(&shard.index), file.len()).unwrap();
            assert_eq!(starts.len() as u64, shard.documents);
            assert_eq!(file.len() as u64, shard.tokens);
            assert!(
                file.len() <= 40 || starts.len() == 1,
                "a shard over its size holds one document"
            );
            let ends: Vec<u64> = starts
                .iter()
                .skip(1)
                .copied()
                .chain([file.len() as u64])
                .collect();
            for (&s, &e) in starts.iter().zip(&ends) {
                let tokens = file.tokens(s as usize, (e - s) as usize);
                assert_eq!(tokens[0], eot);
                assert!(!tokens[1..].contains(&eot), "a quoted separator is text");
                texts.push(tokenizer.decode(&tokens[1..], false));
            }
        }
    }
    assert!(texts.iter().any(|t| t == INJECTED));
    assert!(texts.iter().any(|t| t.contains("été, où, ça")));
    assert!(
        texts
            .iter()
            .any(|t| t == "A whole text file is one document.\nEven over lines.\n")
    );
    // Validation: the first kept document, then every third.
    let val = manifest.val.as_ref().unwrap();
    let first = TokenFile::open(out.join(&val.shards[0].file)).unwrap();
    let starts = read_index(&out.join(&val.shards[0].index), first.len()).unwrap();
    let end = starts.get(1).map_or(first.len(), |&e| e as usize);
    assert_eq!(
        tokenizer.decode(&first.tokens(1, end - 1), false),
        "The first document, about rivers."
    );
    // Read back for training: whole documents, each its own segment.
    let mut reading = Reading::new(2, 32);
    reading.pack = Pack::Documents;
    reading.mask = true;
    let source = Source::open(&out.to_string_lossy(), "train", &reading.cache).unwrap();
    assert_eq!(source.vocab(), Some(tokenizer.vocab_size()));
    let mut loader = kapel_train::data::Loader::new(vec![(source, 1.0)], reading).unwrap();
    let batch = loader.next_batch().unwrap();
    let segments = batch.segments.unwrap();
    for (i, &id) in batch.ids.iter().enumerate() {
        if id == eot && i % 32 != 0 && batch.targets[i] != kapel_train::IGNORE {
            assert_ne!(
                segments[i],
                segments[i - 1],
                "a document starts at its separator"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bad_inputs_are_errors() {
    if !have_tokenizer() {
        return;
    }
    let dir = temp("bad");
    std::fs::write(dir.join("x.jsonl"), "{\"body\": \"no text field\"}\n").unwrap();
    let mut o = PrepareOptions {
        inputs: vec![dir.join("x.jsonl").to_string_lossy().into_owned()],
        tokenizer: tokenizer_dir(),
        output: dir.join("out"),
        ..PrepareOptions::default()
    };
    let e = prepare(&o, |_| {}).unwrap_err().to_string();
    assert!(e.contains("x.jsonl:1") && e.contains("text"), "{e}");
    o.inputs = vec![dir.join("x.csv").to_string_lossy().into_owned()];
    assert!(prepare(&o, |_| {}).is_err());
    o.inputs = vec![dir.join("x.jsonl").to_string_lossy().into_owned()];
    o.separator = Some("<|no such token|>".into());
    assert!(prepare(&o, |_| {}).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Serves the files of `root` over HTTP on a port of its own, until the
/// test ends; `tamper` changes the bytes of the files whose name ends so.
fn serve(root: PathBuf, tamper: Option<&'static str>) -> String {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            // The rest of the request, up to its blank line.
            let mut header = String::new();
            while reader.read_line(&mut header).unwrap() > 2 {
                header.clear();
            }
            let path = line
                .split(' ')
                .nth(1)
                .unwrap_or("/")
                .trim_start_matches('/');
            match std::fs::read(root.join(path)) {
                Ok(mut body) => {
                    if tamper.is_some_and(|t| path.ends_with(t)) {
                        let last = body.len() - 1;
                        body[last] ^= 1;
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body);
                }
                Err(_) => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        }
    });
    format!("http://{address}")
}

#[test]
fn a_remote_dataset_reads_as_the_local_one_and_keeps_little_on_disk() {
    if !have_tokenizer() {
        return;
    }
    let dir = temp("remote");
    let manifest = prepare(&options(&dir, "out", 2), |_| {}).unwrap();
    let url = serve(dir.join("out"), None);
    let mut reading = Reading::new(1, 8);
    reading.pack = Pack::Documents;
    reading.mask = true;
    reading.cache = dir.join("cache");
    let local = Source::open(&dir.join("out").to_string_lossy(), "train", &reading.cache).unwrap();
    let remote = Source::open(&url, "train", &reading.cache).unwrap();
    assert_eq!(remote.tokens(), local.tokens());
    let mut a = kapel_train::data::Loader::new(vec![(local, 1.0)], reading.clone()).unwrap();
    let mut b = kapel_train::data::Loader::new(vec![(remote, 1.0)], reading.clone()).unwrap();
    for _ in 0..40 {
        assert_eq!(a.next_batch().unwrap(), b.next_batch().unwrap());
    }
    // The shards read are removed once read: at most the current one and
    // the next are kept.
    let cached: usize = std::fs::read_dir(&reading.cache)
        .unwrap()
        .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "bin")
        })
        .count();
    assert!(manifest.train.shards.len() > 3);
    assert!(cached <= 2, "{cached} shards kept");
    // A shard whose bytes are not those of the manifest is refused.
    let bad = serve(dir.join("out"), Some("train_000001.bin"));
    let mut r = reading.clone();
    r.cache = dir.join("cache-bad");
    let source = Source::open(&format!("{bad}/dataset.json"), "train", &r.cache).unwrap();
    let mut loader = kapel_train::data::Loader::new(vec![(source, 1.0)], r).unwrap();
    let error = (0..40)
        .find_map(|_| loader.next_batch().err())
        .expect("an error");
    assert!(error.to_string().contains("checksum"), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A tokenizer directory with a chat template: GPT-2's subset, each turn
/// `role: content`, the assistant's ended by `<|endoftext|>`.
fn chat_tokenizer(dir: &Path) -> PathBuf {
    let tok = dir.join("tokenizer");
    std::fs::create_dir_all(&tok).unwrap();
    std::fs::copy(
        tokenizer_dir().join("tokenizer.json"),
        tok.join("tokenizer.json"),
    )
    .unwrap();
    let template = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% if m.role == 'assistant' %}<|endoftext|>{% endif %}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
    let config = serde_json::json!({"chat_template": template, "eos_token": "<|endoftext|>"});
    std::fs::write(tok.join("tokenizer_config.json"), config.to_string()).unwrap();
    tok
}

/// A Parquet file of `prompt` and `completion` texts.
fn prompts_parquet(path: &Path, rows: &[(&str, &str)]) {
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    let schema = Arc::new(
        parquet::schema::parser::parse_message_type(
            "message doc { required binary prompt (UTF8); required binary completion (UTF8); }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut group = writer.next_row_group().unwrap();
    for column in 0..2 {
        let mut c = group.next_column().unwrap().unwrap();
        let values: Vec<ByteArray> = rows
            .iter()
            .map(|r| if column == 0 { r.0 } else { r.1 }.into())
            .collect();
        c.typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .unwrap();
        c.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
}

#[test]
fn conversations_keep_the_assistant_s_tokens_as_the_only_targets() {
    if !have_tokenizer() {
        return;
    }
    let dir = temp("chat");
    let tok = chat_tokenizer(&dir);
    let lines = [
        r#"{"messages": [{"role": "user", "content": "Where is the river?"}, {"role": "assistant", "content": "In the valley."}, {"role": "user", "content": "And the snow?"}, {"role": "assistant", "content": "On the mountains."}]}"#,
        r#"{"prompt": [{"role": "user", "content": "Say one."}], "completion": [{"role": "assistant", "content": "One."}]}"#,
        r#"{"messages": [{"role": "user", "content": "Where is the river?"}, {"role": "assistant", "content": "In the valley."}, {"role": "user", "content": "And the snow?"}, {"role": "assistant", "content": "On the mountains."}]}"#,
        r#"{"messages": [{"role": "user", "content": "Nothing to answer."}]}"#,
    ];
    std::fs::write(dir.join("chat.jsonl"), lines.join("\n")).unwrap();
    prompts_parquet(&dir.join("p.parquet"), &[("The sky is", " blue.")]);
    let options = PrepareOptions {
        inputs: vec![
            dir.join("chat.jsonl").to_string_lossy().into_owned(),
            dir.join("p.parquet").to_string_lossy().into_owned(),
        ],
        tokenizer: tok.clone(),
        output: dir.join("out"),
        field: "messages".into(),
        chat: true,
        threads: 2,
        ..PrepareOptions::default()
    };
    let manifest = prepare(&options, |_| {}).unwrap();
    assert_eq!(
        (manifest.documents, manifest.duplicates, manifest.empty),
        (5, 1, 1)
    );
    assert!(!manifest.separated && manifest.chat.is_some());
    let trained = manifest.train.trained.unwrap();
    // The learnt text, by the tokenizer: the answers with their end, each
    // encoded apart (the prompt ends with "assistant: ", its space included;
    // the template's newlines after a tag are trimmed).
    let tokenizer = Tokenizer::from_file(tok.join("tokenizer.json")).unwrap();
    let count = |t: &str| tokenizer.encode(t, false).unwrap().len() as u64;
    let expected = count("In the valley.<|endoftext|>")
        + count("On the mountains.<|endoftext|>")
        + count("One.<|endoftext|>")
        + count(" blue.")
        + 1;
    assert_eq!(trained, expected);
    // Read back as whole documents: the targets that count are those.
    let mut reading = Reading::new(1, 64);
    reading.pack = Pack::Documents;
    reading.mask = true;
    let source = Source::open(&dir.join("out").to_string_lossy(), "train", &reading.cache).unwrap();
    let mut loader = kapel_train::data::Loader::new(vec![(source, 1.0)], reading).unwrap();
    // In order: the second epoch starts with the first batch again.
    let first = loader.next_batch().unwrap();
    let mut batch = first.clone();
    let mut counted = 0u64;
    loop {
        counted += batch
            .targets
            .iter()
            .filter(|&&t| t != kapel_train::IGNORE)
            .count() as u64;
        batch = loader.next_batch().unwrap();
        if batch == first {
            break;
        }
    }
    assert_eq!(counted, trained);
    std::fs::remove_dir_all(&dir).unwrap();
}
