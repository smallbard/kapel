//! Kapel's greedy generations against those of `transformers`, saved by
//! `tools/parity/generation_reference.py` in `data/parity/generation/`: on the
//! processor in `f32`, and with the feature `cuda`, on the card in `bf16`.
//!
//! For each prompt, the number of tokens generated alike, and at the first
//! difference, the gap between the logits of the two tokens in Kapel's own
//! computation, with the reference's tokens before it: a gap near zero is a
//! tie that rounding decides. In `bf16`, where such ties come early, both
//! generations are also compared to the `f32` one: Kapel's should stay with it
//! about as long as that of `transformers` does.
//!
//! `cargo run --release -p kapel-cli [--features cuda] --example generation_parity [NAMES…]`

use std::path::{Path, PathBuf};

use kapel_core::DType;
use kapel_infer::{Pipeline, Request};
use kapel_models::{Cache, Model};
use kapel_ops::{Backend, Transfer};
use serde_json::Value;

/// The gap between the logits of `ours` and `theirs` at the position after
/// `prefix`.
fn gap<B: Backend + Transfer>(
    backend: &B,
    model: &Model,
    prefix: &[u32],
    ours: u32,
    theirs: u32,
) -> f64 {
    let logits = model.step(backend, prefix, &mut Cache::new()).unwrap();
    let logits = backend.download(&logits).unwrap().to_f64_vec().unwrap();
    logits[ours as usize] - logits[theirs as usize]
}

fn compare<B: Backend + Transfer>(
    label: &str,
    pipeline: &Pipeline<B>,
    backend: &B,
    cases: &[Value],
) -> (usize, usize) {
    let (mut same_cases, mut all) = (0, 0);
    for (i, case) in cases.iter().enumerate() {
        let ids = |v: &Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect()
        };
        let Some(reference) = case.get(label).map(ids) else {
            continue;
        };
        let prompt = ids(&case["ids"]);
        let request = Request {
            max_tokens: reference.len().max(1),
            ..Request::default()
        };
        let mut generation = pipeline.generate(&prompt, &request).unwrap();
        for piece in generation.by_ref() {
            piece.unwrap();
        }
        let ours = generation.tokens();
        let common = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        let alike = common(ours, &reference);
        if label != "f32" {
            let exact = ids(&case["f32"]);
            println!(
                "  {label} {i}: with the f32 generation, {} tokens alike here, {} in transformers",
                common(ours, &exact),
                common(&reference, &exact)
            );
        }
        all += 1;
        let kind = if case["chat"].as_bool() == Some(true) {
            "chat"
        } else {
            "text"
        };
        if alike == reference.len() && ours.len() == reference.len() {
            same_cases += 1;
            println!("  {label} {kind} {i}: {alike} tokens, identical");
        } else if alike < ours.len().min(reference.len()) {
            let mut prefix = prompt.clone();
            prefix.extend_from_slice(&reference[..alike]);
            let g = gap(
                backend,
                pipeline.model(),
                &prefix,
                ours[alike],
                reference[alike],
            );
            println!(
                "  {label} {kind} {i}: {alike} of {} tokens alike, then {} instead of {} \
                 (logit gap {g:.2e})",
                reference.len(),
                ours[alike],
                reference[alike]
            );
        } else {
            println!(
                "  {label} {kind} {i}: {alike} tokens alike, then {} tokens here, {} there",
                ours.len(),
                reference.len()
            );
        }
    }
    (same_cases, all)
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/generation");
    let names: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("data/parity/generation: run tools/parity/generation_reference.py")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| {
            names.is_empty()
                || names
                    .iter()
                    .any(|n| p.file_stem().is_some_and(|s| s == n.as_str()))
        })
        .collect();
    files.sort();
    let (mut same, mut all) = (0, 0);
    for file in files {
        let reference: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        let dir = Path::new(reference["directory"].as_str().unwrap());
        let cases = reference["cases"].as_array().unwrap();
        println!(
            "{} ({}):",
            file.file_stem().unwrap().to_string_lossy(),
            reference["model"]
        );
        let cpu = kapel_cpu::Cpu::new();
        let pipeline = Pipeline::load(dir, cpu.clone(), Some(DType::F32)).unwrap();
        let (s, a) = compare("f32", &pipeline, &cpu, cases);
        same += s;
        all += a;
        drop(pipeline);
        #[cfg(feature = "cuda")]
        {
            let cuda = kapel_cuda::Cuda::new(0).unwrap();
            let pipeline = Pipeline::load(dir, cuda.clone(), Some(DType::BF16)).unwrap();
            let (s, a) = compare("bf16", &pipeline, &cuda, cases);
            same += s;
            all += a;
        }
    }
    println!("{same} of {all} generations identical");
}
