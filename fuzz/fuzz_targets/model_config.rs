//! Any bytes as `config.json`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(data) {
        let _ = kapel_models::Config::from_json(&json);
    }
});
