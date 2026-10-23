//! Speculative decoding on the processor gives the greedy tokens of the model
//! alone, whatever the proposer: the same model quantized (its guesses
//! sometimes wrong), the model itself (always right), the tokens of the
//! prompt. The greedy tokens of SmolLM2-135M saved by
//! `tools/parity/generation_reference.py`; without them, the tests are
//! skipped, or fail with `KAPEL_REQUIRE_PARITY=1`.

use std::path::PathBuf;

use kapel_core::DType;
use kapel_infer::{Draft, LoadOptions, Pipeline, Request};
use serde_json::Value;

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

#[test]
fn speculation_gives_the_greedy_tokens() {
    let Some(reference) = reference() else { return };
    let dir = reference["directory"].as_str().unwrap();
    let load = |quantize: Option<DType>| {
        let options = LoadOptions {
            dtype: Some(DType::F32),
            quantize,
            ..LoadOptions::default()
        };
        Pipeline::load_with(dir, kapel_cpu::Cpu::new(), &options).unwrap()
    };
    let model = load(None);
    let draft = load(Some(DType::int4(32).unwrap()));
    for case in reference["cases"].as_array().unwrap() {
        let want = ids(&case["f32"]);
        let prompt = ids(&case["ids"]);
        let request = Request {
            max_tokens: want.len(),
            ..Request::default()
        };
        let drafts = [
            Draft::Model {
                pipeline: &draft,
                tokens: 4,
            },
            Draft::Model {
                pipeline: &model,
                tokens: 3,
            },
            Draft::Ngram {
                tokens: 5,
                longest: 3,
                shortest: 1,
            },
        ];
        for (d, proposer) in drafts.into_iter().enumerate() {
            let mut speculation = model.speculate(&prompt, &request, proposer).unwrap();
            let mut text = String::new();
            for piece in speculation.by_ref() {
                text.push_str(&piece.unwrap().text);
            }
            assert_eq!(speculation.tokens(), &want[..], "proposer {d}");
            assert_eq!(text, model.tokenizer().decode(&want, true), "proposer {d}");
            let stats = speculation.stats();
            match d {
                0 => assert!(
                    stats.accepted > 0 && stats.accepted < stats.proposed,
                    "{stats:?}"
                ),
                1 => assert_eq!(stats.accepted, stats.proposed, "{stats:?}"),
                _ => {}
            }
            assert!(stats.passes < want.len() || d == 2, "{stats:?}");
        }
    }
}

/// The small MiMo of `tools/parity/multi_token_reference.py`, its own
/// multi-token layer proposing: the greedy tokens of the model alone.
#[test]
fn the_multi_token_layer_proposes_without_changing_the_tokens() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/mimo");
    if !dir.is_dir() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "data/parity/mimo is missing, and KAPEL_REQUIRE_PARITY=1"
        );
        return;
    }
    let model = Pipeline::load(&dir, kapel_cpu::Cpu::new(), Some(DType::F32)).unwrap();
    let request = Request {
        max_tokens: 40,
        ignore_eos: true,
        ..Request::default()
    };
    for text in ["The quick brown fox", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"] {
        let prompt = model.prompt(text).unwrap();
        let mut alone = model.generate(&prompt, &request).unwrap();
        for piece in alone.by_ref() {
            piece.unwrap();
        }
        let mut speculation = model
            .speculate(&prompt, &request, Draft::MultiToken)
            .unwrap();
        for piece in speculation.by_ref() {
            piece.unwrap();
        }
        assert_eq!(speculation.tokens(), alone.tokens(), "{text}");
        let stats = speculation.stats();
        assert!(stats.proposed > 0, "{stats:?}");
        assert!(stats.passes <= 40, "{stats:?}");
    }
    let base = load_smollm2();
    if let Some(base) = base {
        let error = base.speculate(&[1, 2], &request, Draft::MultiToken);
        assert!(error.is_err(), "a model without the layer");
    }
}

fn load_smollm2() -> Option<Pipeline<kapel_cpu::Cpu>> {
    let reference = reference()?;
    let dir = reference["directory"].as_str()?.to_string();
    Pipeline::load(dir, kapel_cpu::Cpu::new(), Some(DType::F32)).ok()
}
