//! Kapel's logits against those of `transformers`, saved by
//! `tools/parity/reference.py` in `data/parity/<model>/`. Without these files
//! (in continuous integration), the test checks nothing.

use std::path::PathBuf;

use kapel_hub::{Checkpoint, SafeTensors};
use kapel_models::Model;

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
    eprintln!("checked: {checked:?}");
}
