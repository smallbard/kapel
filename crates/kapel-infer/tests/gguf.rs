//! Models read from GGUF against the models of `transformers` they come from:
//! the files that `tools/parity/gguf_models_reference.py` downloads. Without
//! them, the tests are skipped and say so on stderr; with
//! `KAPEL_REQUIRE_PARITY=1`, they fail.

use std::path::PathBuf;

use kapel_core::DType;
use kapel_infer::Pipeline;
use kapel_models::Cache;
use kapel_tokenizers::Message;
use serde_json::Value;

/// The references found: each names a GGUF file and the original directory.
fn references() -> Vec<Value> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gguf-models");
    let found: Vec<Value> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| std::fs::read(entry.path()).ok())
        .filter_map(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|r| {
            PathBuf::from(r["gguf"].as_str().unwrap_or_default()).is_file()
                && PathBuf::from(r["directory"].as_str().unwrap_or_default()).is_dir()
        })
        .collect();
    if found.is_empty() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "{} is missing, and KAPEL_REQUIRE_PARITY=1",
            dir.display()
        );
        eprintln!(
            "skipped: {} missing (see tools/parity/README.md)",
            dir.display()
        );
    }
    found
}

fn pipelines(reference: &Value) -> (Pipeline<kapel_cpu::Cpu>, Pipeline<kapel_cpu::Cpu>) {
    let load = |key: &str| {
        Pipeline::load(
            reference[key].as_str().unwrap(),
            kapel_cpu::Cpu::new(),
            Some(DType::F32),
        )
        .unwrap()
    };
    (load("gguf"), load("directory"))
}

const TEXTS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog, 1234567 times.",
    "Les élèves ont étudié l'œuvre de Molière — « L'Avare », 1668 — en détail.\n\n\tFin.",
    "fn main() {\n    println!(\"{}\", 2 + 2); // 4\n}\n",
    "東京は日本の首都です。 Emoji: 🦀🚀  spaces   and\r\nnewlines",
];

#[test]
fn a_gguf_file_has_the_tokenizer_and_end_tokens_of_its_model() {
    for reference in references() {
        let (gguf, original) = pipelines(&reference);
        let name = reference["name"].as_str().unwrap();
        for text in TEXTS {
            for special in [false, true] {
                assert_eq!(
                    gguf.tokenizer().encode(text, special).unwrap(),
                    original.tokenizer().encode(text, special).unwrap(),
                    "{name}: {text:?}"
                );
            }
        }
        let mut gguf_eos = gguf.eos().to_vec();
        let mut original_eos = original.eos().to_vec();
        gguf_eos.sort_unstable();
        original_eos.sort_unstable();
        assert_eq!(gguf_eos, original_eos, "{name}: end tokens");
        if original.has_chat_template() {
            let chat = [Message::new("user", "Hello, who are you?")];
            assert_eq!(
                gguf.chat_prompt(&chat).unwrap(),
                original.chat_prompt(&chat).unwrap(),
                "{name}: chat template"
            );
        }
    }
}

#[test]
fn a_gguf_model_predicts_as_its_model() {
    for reference in references() {
        let (gguf, original) = pipelines(&reference);
        let name = reference["name"].as_str().unwrap();
        let ids = original.tokenizer().encode(&TEXTS.concat(), true).unwrap();
        let cpu = kapel_cpu::Cpu::new();
        let logits = |p: &Pipeline<kapel_cpu::Cpu>| {
            p.model()
                .extend(&cpu, &ids, &mut Cache::new())
                .unwrap()
                .to_f64_vec()
                .unwrap()
        };
        let (a, b) = (logits(&gguf), logits(&original));
        let vocab = a.len() / ids.len();
        let best = |row: &[f64]| {
            (0..row.len())
                .max_by(|&i, &j| row[i].total_cmp(&row[j]))
                .unwrap()
        };
        let agree = a
            .chunks(vocab)
            .zip(b.chunks(vocab))
            .filter(|(x, y)| best(x) == best(y))
            .count();
        // Q8_0 rounds the weights: the most likely tokens rarely change.
        assert!(
            agree * 100 >= 95 * ids.len(),
            "{name}: {agree} of {} predictions agree",
            ids.len()
        );
    }
}
