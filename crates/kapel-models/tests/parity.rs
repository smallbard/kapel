//! Kapel's logits against those of `transformers`, saved by
//! `tools/parity/reference.py` in `data/parity/<model>/`. Without these files
//! (in continuous integration), the test is skipped and says so on stderr; with
//! `KAPEL_REQUIRE_PARITY=1`, it fails instead.

use std::io::Write;
use std::path::PathBuf;

use kapel_hub::{Checkpoint, SafeTensors};
use kapel_models::{Cache, KvCache, Model};

/// The largest difference, relative to the largest logit, over every text.
fn worst(model: &Model, reference: &SafeTensors, cpu: &kapel_cpu::Cpu) -> f64 {
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for i in 0.. {
        let Ok(ids) = reference.tensor(&format!("ids_{i}")) else {
            break;
        };
        let ids: Vec<u32> = ids
            .to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let want = reference
            .tensor(&format!("logits_{i}"))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let got = model.forward(cpu, &ids).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max(f64::from((a - b).abs()));
            scale = scale.max(f64::from(b.abs()));
        }
    }
    worst / scale
}

/// The logits of `ids` computed with a cache, a prefill of the first half and
/// then one token at a time, against those of the whole sequence at once: the
/// largest difference relative to the largest logit.
fn cached_against_whole(model: &Model, ids: &[u32], cpu: &kapel_cpu::Cpu) -> f64 {
    let whole = model.forward(cpu, ids).unwrap().to_vec::<f32>().unwrap();
    let vocab = whole.len() / ids.len();
    let mut cache = Cache::new();
    let half = ids.len() / 2;
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    let mut compare = |row: usize, got: Vec<f32>| {
        for (a, b) in got.iter().zip(&whole[row * vocab..(row + 1) * vocab]) {
            worst = worst.max(f64::from((a - b).abs()));
            scale = scale.max(f64::from(b.abs()));
        }
    };
    let logits = |t: kapel_core::Tensor| t.to_vec::<f32>().unwrap();
    compare(
        half - 1,
        logits(model.step(cpu, &ids[..half], &mut cache).unwrap()),
    );
    for (i, &id) in ids.iter().enumerate().skip(half) {
        compare(i, logits(model.step(cpu, &[id], &mut cache).unwrap()));
    }
    assert_eq!(cache.len(), ids.len());
    worst / scale
}

#[test]
fn a_cache_gives_the_logits_of_the_whole_sequence() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let cpu = kapel_cpu::Cpu::new();
    for dir in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let Ok(reference) = SafeTensors::open(dir.path().join("reference.safetensors")) else {
            continue;
        };
        let name = dir.file_name().to_string_lossy().into_owned();
        let model = Model::load(&Checkpoint::open(dir.path()).unwrap()).unwrap();
        let ids: Vec<u32> = reference
            .tensor("ids_2")
            .unwrap()
            .to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let relative = cached_against_whole(&model, &ids, &cpu);
        assert!(relative < 1e-5, "{name}: relative difference {relative:e}");
    }
}

/// The largest difference between two lists of logits, relative to the
/// largest of the second.
fn relative(got: &[f32], want: &[f32]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (a, b)| m.max(f64::from((a - b).abs())));
    worst / scale
}

/// The texts of a reference, as token ids.
fn texts(reference: &SafeTensors) -> Vec<Vec<u32>> {
    (0..)
        .map_while(|i| reference.tensor(&format!("ids_{i}")).ok())
        .map(|ids| {
            ids.to_vec::<i32>()
                .unwrap()
                .into_iter()
                .map(|v| v as u32)
                .collect()
        })
        .collect()
}

/// Every model of `data/parity`, by name, with its reference.
fn models() -> Vec<(String, Model, SafeTensors)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let mut found = Vec::new();
    for dir in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let Ok(reference) = SafeTensors::open(dir.path().join("reference.safetensors")) else {
            continue;
        };
        let name = dir.file_name().to_string_lossy().into_owned();
        let model = Model::load(&Checkpoint::open(dir.path()).unwrap()).unwrap();
        found.push((name, model, reference));
    }
    found
}

/// Several sequences in one pass give the logits each gives alone: their
/// prompts together, then a token of each together, after caches of
/// different lengths.
#[test]
fn a_batch_gives_the_logits_of_its_sequences_alone() {
    let cpu = kapel_cpu::Cpu::new();
    for (name, model, reference) in models() {
        let texts = texts(&reference);
        let mut alone = Vec::new();
        for ids in &texts {
            let mut cache = Cache::new();
            let first = model
                .step(&cpu, ids, &mut cache)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let next = model
                .step(&cpu, &ids[..1], &mut cache)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            alone.push((first, next));
        }
        let mut cache = KvCache::with_pages(64);
        let ids: Vec<_> = texts.iter().map(|_| cache.add_sequence()).collect();
        let batch: Vec<_> = ids
            .iter()
            .zip(&texts)
            .map(|(&id, t)| (id, &t[..]))
            .collect();
        let first = model
            .step_batch(&cpu, &mut cache, &batch)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let batch: Vec<_> = ids
            .iter()
            .zip(&texts)
            .map(|(&id, t)| (id, &t[..1]))
            .collect();
        let next = model
            .step_batch(&cpu, &mut cache, &batch)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let vocab = first.len() / texts.len();
        for (s, (want_first, want_next)) in alone.iter().enumerate() {
            let rows = s * vocab..(s + 1) * vocab;
            let worst =
                relative(&first[rows.clone()], want_first).max(relative(&next[rows], want_next));
            assert!(
                worst < 1e-5,
                "{name}, sequence {s}: relative difference {worst:e}"
            );
        }
    }
}

/// Stacked projections ([`Model::fused`]) give the logits of the model.
#[test]
fn a_fused_model_gives_the_logits_of_the_model() {
    let cpu = kapel_cpu::Cpu::new();
    for (name, model, reference) in models() {
        let fused = model.fused(&cpu).unwrap();
        for ids in texts(&reference) {
            let want = model.forward(&cpu, &ids).unwrap().to_vec::<f32>().unwrap();
            let got = fused.forward(&cpu, &ids).unwrap().to_vec::<f32>().unwrap();
            let worst = relative(&got, &want);
            assert!(worst < 1e-5, "{name}: relative difference {worst:e}");
            let mut cache = Cache::new();
            let got = fused
                .step(&cpu, &ids, &mut cache)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let vocab = got.len();
            let worst = relative(&got, &want[want.len() - vocab..]);
            assert!(
                worst < 1e-5,
                "{name}, with a cache: relative difference {worst:e}"
            );
        }
    }
}

#[test]
fn logits_match_transformers_and_survive_a_save() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity");
    let cpu = kapel_cpu::Cpu::new();
    let mut checked = Vec::new();
    for dir in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let Ok(reference) = SafeTensors::open(dir.path().join("reference.safetensors")) else {
            continue;
        };
        let name = dir.file_name().to_string_lossy().into_owned();
        let model = Model::load(&Checkpoint::open(dir.path()).unwrap()).unwrap();
        let relative = worst(&model, &reference, &cpu);
        assert!(relative < 2e-5, "{name}: relative difference {relative:e}");
        // Saved and loaded again, the model gives the same logits.
        let copy = std::env::temp_dir().join(format!("kapel-parity-{name}-{}", std::process::id()));
        model.save(&copy).unwrap();
        let again = Model::load(&Checkpoint::open(&copy).unwrap()).unwrap();
        assert!(
            worst(&again, &reference, &cpu) < 2e-5,
            "{name}: after a save"
        );
        std::fs::remove_dir_all(&copy).unwrap();
        checked.push(name);
    }
    if checked.is_empty() {
        assert!(
            std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
            "no data/parity/*/reference.safetensors, and KAPEL_REQUIRE_PARITY=1 \
             (see tools/parity/README.md)"
        );
        // Straight to stderr, past the capture of the test harness.
        let _ = writeln!(
            std::io::stderr(),
            "skipped: data/parity/*/reference.safetensors missing (see tools/parity/README.md)"
        );
        return;
    }
    eprintln!("checked: {checked:?}");
}
