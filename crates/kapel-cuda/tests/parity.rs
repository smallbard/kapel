//! Models on the card: the logits of the small models of `data/parity` (see
//! `tools/parity/README.md`) against those of `transformers`, in `f32`, and in
//! `bf16` against the processor's own `bf16`. Without these files, the test is
//! skipped and says so on stderr; with `KAPEL_REQUIRE_PARITY=1`, it fails.

use std::path::PathBuf;

use kapel_core::{DType, Tensor};
use kapel_cuda::Cuda;
use kapel_hub::{Checkpoint, SafeTensors};
use kapel_models::Model;
use kapel_ops::Transfer;

/// The token ids and logits of each text of a reference.
fn texts(reference: &SafeTensors) -> Vec<(Vec<u32>, Vec<f32>)> {
    (0..)
        .map_while(|i| {
            let ids = reference.tensor(&format!("ids_{i}")).ok()?;
            let ids = ids
                .to_vec::<i32>()
                .unwrap()
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let logits = reference.tensor(&format!("logits_{i}")).unwrap();
            Some((ids, logits.to_vec::<f32>().unwrap()))
        })
        .collect()
}

fn logits(logits: &Tensor) -> Vec<f32> {
    logits
        .to_f64_vec()
        .unwrap()
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// The largest difference relative to the largest logit, over every text.
fn worst(pairs: impl Iterator<Item = (Vec<f32>, Vec<f32>)>) -> f64 {
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for (got, want) in pairs {
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max(f64::from((a - b).abs()));
            scale = scale.max(f64::from(b.abs()));
        }
    }
    worst / scale
}

#[test]
fn models_on_the_card_match_transformers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let cuda = Cuda::new(0).expect("a CUDA device");
    let cpu = kapel_cpu::Cpu::new();
    let mut checked = Vec::new();
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    dirs.sort_by_key(std::fs::DirEntry::file_name);
    for dir in dirs {
        let Ok(reference) = SafeTensors::open(dir.path().join("reference.safetensors")) else {
            continue;
        };
        let name = dir.file_name().to_string_lossy().into_owned();
        let model = Model::load(&Checkpoint::open(dir.path()).unwrap()).unwrap();
        let texts = texts(&reference);
        let on_card = model.upload(&cuda).unwrap();
        let card = |m: &Model, ids: &[u32]| {
            logits(&cuda.download(&m.forward(&cuda, ids).unwrap()).unwrap())
        };
        let f32 = worst(
            texts
                .iter()
                .map(|(ids, want)| (card(&on_card, ids), want.clone())),
        );
        let half_card = on_card.cast(&cuda, DType::BF16).unwrap();
        let half_cpu = model.cast(&cpu, DType::BF16).unwrap();
        let bf16_card = worst(
            texts
                .iter()
                .map(|(ids, want)| (card(&half_card, ids), want.clone())),
        );
        let bf16_cpu = worst(
            texts
                .iter()
                .map(|(ids, want)| (logits(&half_cpu.forward(&cpu, ids).unwrap()), want.clone())),
        );
        eprintln!("{name}: f32 {f32:.1e}, bf16 {bf16_card:.1e} (processor bf16 {bf16_cpu:.1e})");
        assert!(f32 < 2e-5, "{name}: f32 relative difference {f32:e}");
        // Rounded to bf16 at every operation, the card stays as close to
        // `transformers` as the processor does.
        assert!(
            bf16_card < 2.0 * bf16_cpu.max(1e-2),
            "{name}: bf16 relative difference {bf16_card:e}, processor {bf16_cpu:e}"
        );
        checked.push(name);
    }
    if checked.is_empty() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "no data/parity/*/reference.safetensors, and KAPEL_REQUIRE_PARITY=1"
        );
        eprintln!(
            "skipped: data/parity/*/reference.safetensors missing (see tools/parity/README.md)"
        );
    }
}
