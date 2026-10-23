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

/// The position of the largest of `row`.
fn best(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&i, &j| row[i].total_cmp(&row[j]))
        .unwrap_or(0)
}

/// Quantized models on the card (the GGUF files of `data/parity/gguf-models`,
/// and models of the Hub quantized to int8 and int4 at loading), against the
/// same model on the processor in `f32`: a prompt at once (products of many
/// rows, through dequantized pieces), then tokens one at a time (the products
/// that read the codes in place). The card computes in `bf16`: the most
/// likely tokens nearly always agree.
#[test]
fn quantized_models_on_the_card_predict_as_on_the_processor() {
    use kapel_models::Cache;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gguf-models");
    let cuda = Cuda::new(0).expect("a CUDA device");
    let cpu = kapel_cpu::Cpu::new();
    let mut models = Vec::new();
    for entry in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(reference) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let name = reference["name"].as_str().unwrap_or_default().to_string();
        let gguf = PathBuf::from(reference["gguf"].as_str().unwrap_or_default());
        if let Ok(file) = kapel_hub::Gguf::open(&gguf) {
            models.push((format!("{name} q8_0"), Model::from_gguf(&file).unwrap()));
        }
        let q4 = gguf.with_file_name(
            gguf.file_name()
                .unwrap()
                .to_string_lossy()
                .replace("Q8_0", "Q4_K_M"),
        );
        if let Ok(file) = kapel_hub::Gguf::open(&q4) {
            models.push((format!("{name} q4_k_m"), Model::from_gguf(&file).unwrap()));
        }
        let dir = PathBuf::from(reference["directory"].as_str().unwrap_or_default());
        if let Ok(checkpoint) = Checkpoint::open(&dir) {
            let model = Model::load(&checkpoint).unwrap();
            for dtype in [DType::int8(128).unwrap(), DType::int4(64).unwrap()] {
                models.push((
                    format!("{name} {dtype}"),
                    model.quantize(dtype, false).unwrap(),
                ));
            }
        }
    }
    if models.is_empty() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "no data/parity/gguf-models, and KAPEL_REQUIRE_PARITY=1"
        );
        eprintln!("skipped: data/parity/gguf-models missing (see tools/parity/README.md)");
        return;
    }
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + 97 * i).collect();
    let next: Vec<u32> = (0..16).map(|i| 2000 + 31 * i).collect();
    for (name, model) in models {
        let host = model.cast(&cpu, DType::F32).unwrap();
        let card = model
            .upload(&cuda)
            .unwrap()
            .cast(&cuda, DType::BF16)
            .unwrap();
        let (mut on_cpu, mut on_card) = (Cache::new(), Cache::new());
        let mut agree = 0;
        let mut total = 0;
        let mut compare = |a: Vec<f32>, b: Vec<f32>| {
            total += 1;
            agree += usize::from(best(&a) == best(&b));
        };
        compare(
            logits(&host.step(&cpu, &prompt, &mut on_cpu).unwrap()),
            logits(
                &cuda
                    .download(&card.step(&cuda, &prompt, &mut on_card).unwrap())
                    .unwrap(),
            ),
        );
        for &t in &next {
            compare(
                logits(&host.step(&cpu, &[t], &mut on_cpu).unwrap()),
                logits(
                    &cuda
                        .download(&card.step(&cuda, &[t], &mut on_card).unwrap())
                        .unwrap(),
                ),
            );
        }
        eprintln!("{name}: {agree} of {total} most likely tokens agree");
        assert!(agree * 10 >= 8 * total, "{name}: {agree} of {total} agree");
    }
}
