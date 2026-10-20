//! Any text as a chat template, rendered with a fixed conversation.
#![no_main]

use std::collections::BTreeMap;

use kapel_tokenizers::{ChatTemplate, Message};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(source) = std::str::from_utf8(data) else { return };
    let template = ChatTemplate::new(source, BTreeMap::from([("bos_token".to_string(), "<s>".to_string())]));
    let messages = [Message::new("system", "Be brief."), Message::new("user", "Hi {{ 7 * 7 }}")];
    let _ = template.render(&messages, true);
});
