//! The speed of encoding and decoding, on the tokenizers of `data/parity` (see
//! `tools/parity`) or the `tokenizer.json` files given.
//!
//! `cargo run --release -p kapel-tokenizers --example encode [tokenizer.json...]`

use std::path::{Path, PathBuf};
use std::time::Instant;

/// About `bytes` of text: words, numbers, punctuation and some accents, drawn
/// with a fixed pseudo-random sequence.
fn text(bytes: usize) -> String {
    const WORDS: &[&str] = &[
        "the",
        "model",
        "reads",
        "tokens",
        "quickly",
        "and",
        "écrit",
        "naïve",
        "über",
        "a",
        "transformer",
        "2024",
        "3.14159",
        "(",
        ")",
        "—",
        "don't",
        "it's",
        "\n",
        "  ",
        "HTTP",
        "café",
        "日本語",
        "🙂",
        "x",
        "Hello",
        "world",
        "of",
        "in",
        "to",
    ];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut out = String::new();
    while out.len() < bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push_str(WORDS[(state % WORDS.len() as u64) as usize]);
        out.push(' ');
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
        for name in [
            "gpt2",
            "llama3-tokenizer",
            "qwen2",
            "mistral-instruct",
            "gemma2-it",
        ] {
            let path = root.join(name).join("tokenizer.json");
            if path.exists() {
                paths.push(path);
            }
        }
    }
    let text = text(4 << 20);
    for path in paths {
        let tokenizer = kapel_tokenizers::Tokenizer::from_file(&path)?;
        // Once with the cache of BPE empty, then again with it filled.
        let start = Instant::now();
        let ids = tokenizer.encode(&text, false)?;
        let cold = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let ids_again = tokenizer.encode(&text, false)?;
        let warm = start.elapsed().as_secs_f64();
        assert_eq!(ids, ids_again);
        let start = Instant::now();
        let decoded = tokenizer.decode(&ids, false);
        let decode = start.elapsed().as_secs_f64();
        let mb = text.len() as f64 / 1e6;
        println!(
            "{:20} {:>8} ids  encode {:7.1} MB/s (cache filled: {:7.1})  decode {:7.1} MB/s",
            path.parent()
                .unwrap_or(&path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            ids.len(),
            mb / cold,
            mb / warm,
            decoded.len() as f64 / 1e6 / decode
        );
    }
    Ok(())
}
