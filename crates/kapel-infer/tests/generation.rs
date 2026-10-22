//! Generation against `transformers`, on the processor: the greedy tokens of
//! SmolLM2-135M saved by `tools/parity/generation_reference.py`. Without them,
//! the tests are skipped and say so on stderr; with `KAPEL_REQUIRE_PARITY=1`,
//! they fail.

use std::path::PathBuf;

use kapel_core::DType;
use kapel_infer::{Finish, Pipeline, Request};
use serde_json::Value;

/// The reference of SmolLM2-135M, if it and the model are there.
fn reference() -> Option<Value> {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/parity/generation/smollm2-135m.json");
    let found = std::fs::read(&file)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|r| PathBuf::from(r["directory"].as_str().unwrap_or_default()).is_dir());
    if found.is_none() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "{} is missing, and KAPEL_REQUIRE_PARITY=1",
            file.display()
        );
        eprintln!(
            "skipped: {} missing (see tools/parity/README.md)",
            file.display()
        );
    }
    found
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn pipeline(reference: &Value) -> Pipeline<kapel_cpu::Cpu> {
    let dir = reference["directory"].as_str().unwrap();
    Pipeline::load(dir, kapel_cpu::Cpu::new(), Some(DType::F32)).unwrap()
}

#[test]
fn greedy_tokens_are_those_of_transformers() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    for case in reference["cases"].as_array().unwrap() {
        let want = ids(&case["f32"]);
        let request = Request {
            max_tokens: want.len(),
            ..Request::default()
        };
        let mut generation = pipeline.generate(&ids(&case["ids"]), &request).unwrap();
        let mut text = String::new();
        for piece in generation.by_ref() {
            text.push_str(&piece.unwrap().text);
        }
        assert_eq!(generation.tokens(), want.as_slice(), "{}", case["text"]);
        // The pieces' texts make the decoded answer.
        assert_eq!(text, pipeline.tokenizer().decode(&want, true));
    }
}

#[test]
fn a_stop_sequence_ends_the_text_before_it() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    let case = &reference["cases"][0];
    let full = pipeline.tokenizer().decode(&ids(&case["f32"]), true);
    // A word some way into the answer, spanning tokens or not.
    let at = full.char_indices().nth(full.chars().count() / 2).unwrap().0;
    let stop: String = full[at..].chars().take(6).collect();
    let before = &full[..full.find(&stop).unwrap()];
    let request = Request {
        stop: vec![stop.clone()],
        ..Request::default()
    };
    let mut generation = pipeline.generate(&ids(&case["ids"]), &request).unwrap();
    let text: String = generation.by_ref().map(|p| p.unwrap().text).collect();
    assert_eq!(text, before);
    assert_eq!(generation.finish(), Some(&Finish::Stop(stop)));
}

#[test]
fn the_same_seed_samples_the_same_answer() {
    let Some(reference) = reference() else { return };
    let pipeline = pipeline(&reference);
    let prompt = ids(&reference["cases"][0]["ids"]);
    let request = Request {
        sampling: kapel_infer::Sampling {
            temperature: 0.8,
            top_p: 0.95,
            seed: 42,
            ..kapel_infer::Sampling::greedy()
        },
        max_tokens: 40,
        stop: Vec::new(),
    };
    let tokens = || {
        let mut g = pipeline.generate(&prompt, &request).unwrap();
        g.by_ref().for_each(|p| drop(p.unwrap()));
        g.tokens().to_vec()
    };
    let first = tokens();
    assert_eq!(first.len(), 40);
    assert_eq!(tokens(), first);
}
