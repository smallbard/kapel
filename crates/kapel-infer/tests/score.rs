//! The scores of `Scorer` against a full pass of the model, on SmolLM2-135M
//! (from `data/parity/generation/smollm2-135m.json`; skipped without it, as
//! in `generation.rs`).

use std::path::PathBuf;

use kapel_core::DType;
use kapel_infer::{Pipeline, Scorer};
use serde_json::Value;

fn pipeline() -> Option<Pipeline<kapel_cpu::Cpu>> {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/parity/generation/smollm2-135m.json");
    let dir = std::fs::read(&file)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|r| r["directory"].as_str().map(PathBuf::from))
        .filter(|d| d.is_dir());
    let Some(dir) = dir else {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "{} is missing, and KAPEL_REQUIRE_PARITY=1",
            file.display()
        );
        eprintln!("skipped: {} missing", file.display());
        return None;
    };
    Some(Pipeline::load(dir, kapel_cpu::Cpu::new(), Some(DType::F32)).unwrap())
}

/// The log-likelihood of `continuation` after `context`, from the logits of
/// one pass over the whole sequence.
fn full_pass(pipeline: &Pipeline<kapel_cpu::Cpu>, context: &[u32], continuation: &[u32]) -> f64 {
    let ids: Vec<u32> = context.iter().chain(continuation).copied().collect();
    let logits = pipeline
        .model()
        .forward(pipeline.backend(), &ids[..ids.len() - 1])
        .unwrap()
        .to_f64_vec()
        .unwrap();
    let vocab = logits.len() / (ids.len() - 1);
    (context.len() - 1..ids.len() - 1)
        .map(|p| {
            let row = &logits[p * vocab..(p + 1) * vocab];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
            row[ids[p + 1] as usize] - max - sum.ln()
        })
        .sum()
}

#[test]
fn scores_match_a_full_pass_whatever_the_cache_holds() {
    let Some(pipeline) = pipeline() else { return };
    let tokenizer = pipeline.tokenizer();
    let encode = |text: &str| tokenizer.encode(text, false).unwrap();
    let context = encode("The capital of France is the city of");
    let choices = [" Paris, on the Seine.", " Lyon", " Marseille, by the sea"];
    // Small pieces: the scored rows cross pieces.
    let mut scorer = Scorer::new(&pipeline, 3);
    for _round in 0..2 {
        for choice in choices {
            let continuation = encode(choice);
            let score = scorer.score(&context, &continuation).unwrap();
            let want = full_pass(&pipeline, &context, &continuation);
            assert!(
                (score.log_likelihood - want).abs() < 1e-3,
                "{choice}: {} against {want}",
                score.log_likelihood
            );
            assert_eq!(score.tokens, continuation.len());
        }
    }
    // A context of one token, and a sequence sharing nothing.
    let one = encode("Paris");
    let continuation = encode(" is a city");
    let score = scorer.score(&one[..1], &continuation).unwrap();
    assert!((score.log_likelihood - full_pass(&pipeline, &one[..1], &continuation)).abs() < 1e-3);
    // The most likely token is greedy.
    let logits = pipeline
        .model()
        .forward(pipeline.backend(), &context)
        .unwrap()
        .to_f64_vec()
        .unwrap();
    let vocab = logits.len() / context.len();
    let last = &logits[(context.len() - 1) * vocab..];
    let by_logit = |a: &usize, b: &usize| last[*a].total_cmp(&last[*b]);
    let best = (0..vocab).max_by(by_logit).unwrap() as u32;
    let worst = (0..vocab).min_by(by_logit).unwrap() as u32;
    assert!(scorer.score(&context, &[best]).unwrap().greedy);
    assert!(!scorer.score(&context, &[worst]).unwrap().greedy);
    assert!(scorer.score(&[], &[best]).is_err());
}
