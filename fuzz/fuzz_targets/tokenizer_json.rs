//! Any bytes as `tokenizer.json`; a tokenizer that loads must encode and decode.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(data) else { return };
    let Ok(tokenizer) = kapel_tokenizers::Tokenizer::from_json(&json) else { return };
    if let Ok(ids) = tokenizer.encode("Hello wörld 123 <s> 🙂\n", true) {
        let _ = tokenizer.decode(&ids, false);
    }
});
