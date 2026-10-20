//! Any text through real tokenizers (those of `data/parity`, if present).
#![no_main]

use std::sync::OnceLock;

use kapel_tokenizers::Tokenizer;
use libfuzzer_sys::fuzz_target;

fn tokenizers() -> &'static Vec<Tokenizer> {
    static ALL: OnceLock<Vec<Tokenizer>> = OnceLock::new();
    ALL.get_or_init(|| {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/parity");
        ["gpt2", "llama", "qwen3", "gemma2", "llama3-tokenizer", "mistral-instruct"]
            .iter()
            .filter_map(|name| Tokenizer::from_file(format!("{root}/{name}/tokenizer.json")).ok())
            .collect()
    })
}

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    for tokenizer in tokenizers() {
        if let Ok(ids) = tokenizer.encode(text, true) {
            let _ = tokenizer.decode(&ids, false);
        }
    }
});
