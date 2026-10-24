//! The training loop: a run killed and resumed computes the bits of a run
//! that was not, whatever the optimizer; losses go down.

use std::path::{Path, PathBuf};

use kapel_train::config::TrainConfig;
use kapel_train::trainer::{RunOptions, run};

/// A file of tokens in llm.c's format.
fn tokens(path: &Path, count: usize, vocab: u16, seed: u64) {
    let mut rng = kapel_train::model::Rng::new(seed);
    let mut bytes = vec![0u8; 1024];
    bytes[0..4].copy_from_slice(&20_240_520i32.to_le_bytes());
    bytes[4..8].copy_from_slice(&1i32.to_le_bytes());
    bytes[8..12].copy_from_slice(&(count as i32).to_le_bytes());
    // A text with structure: each token often follows from the one before.
    let mut t = 0u16;
    for _ in 0..count {
        t = if rng.uniform() < 0.7 {
            (t * 3 + 1) % vocab
        } else {
            (rng.next_u64() % u64::from(vocab)) as u16
        };
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn setup(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kapel-trainer-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "gpt2", "n_embd": 32, "n_head": 2, "n_layer": 2,
            "vocab_size": 50, "n_positions": 32, "activation_function": "gelu_new"}"#,
    )
    .unwrap();
    tokens(&dir.join("train_000.bin"), 3000, 50, 1);
    tokens(&dir.join("train_001.bin"), 2000, 50, 2);
    tokens(&dir.join("val_000.bin"), 1000, 50, 3);
    dir
}

fn config(dir: &Path, run: &str, optimizer: &str, steps: usize) -> TrainConfig {
    let text = format!(
        r#"
        [model]
        config = "config.json"
        [data]
        train = "train_*.bin"
        val = "val_*.bin"
        [batch]
        rows = 4
        length = 16
        tokens = 128
        [optimizer]
        kind = "{optimizer}"
        lr = 3e-3
        weight_decay = 0.1
        [[optimizer.groups]]
        matches = "*.bias"
        lr_scale = 0.5
        [schedule]
        warmup = 2
        steps = {steps}
        final = 0.1
        [training]
        device = "cpu"
        dtype = "f32"
        clip = 0.5
        [eval]
        every = 3
        tokens = 256
        [checkpoint]
        dir = "{run}"
        every = 3
        keep = 2
        "#
    );
    let mut c = TrainConfig::parse(&text).unwrap();
    c.resolve(dir);
    c
}

const QUIET: RunOptions = RunOptions {
    resume: false,
    stop_after: None,
    quiet: true,
    profile: false,
};

#[test]
fn a_resumed_run_computes_the_bits_of_an_uninterrupted_one() {
    let dir = setup("resume");
    for optimizer in ["adamw", "muon", "adafactor"] {
        let whole = run(
            &config(&dir, &format!("{optimizer}-whole"), optimizer, 7),
            &QUIET,
        )
        .unwrap();
        let cut = config(&dir, &format!("{optimizer}-cut"), optimizer, 7);
        let first = run(
            &cut,
            &RunOptions {
                stop_after: Some(3),
                ..QUIET
            },
        )
        .unwrap();
        assert_eq!(first.steps, 3);
        let second = run(
            &cut,
            &RunOptions {
                resume: true,
                ..QUIET
            },
        )
        .unwrap();
        assert_eq!(second.steps, 7);
        assert_eq!(second.loss.to_bits(), whole.loss.to_bits(), "{optimizer}");
        assert_eq!(second.val_loss, whole.val_loss, "{optimizer}");
        for (name, w) in &whole.params {
            assert_eq!(
                w.to_f64_vec().unwrap(),
                second.params[name].to_f64_vec().unwrap(),
                "{optimizer}: {name}"
            );
        }
        // The log of the resumed run has every step once per part.
        let log = std::fs::read_to_string(dir.join(format!("{optimizer}-cut/log.jsonl"))).unwrap();
        let steps = log.lines().filter(|l| l.contains("\"lr\"")).count();
        assert_eq!(steps, 7);
        // Two checkpoints kept: steps 6 and 7.
        let kept = kapel_train::checkpoint::list(&dir.join(format!("{optimizer}-cut")));
        let kept: Vec<usize> = kept.iter().map(|(s, _)| *s).collect();
        assert_eq!(kept, [6, 7]);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Two weighted sources, shuffled, whole documents kept apart by the mask:
/// a run killed and resumed still reads the rows it would have read.
#[test]
fn a_resumed_run_over_mixed_shuffled_documents_computes_the_same_bits() {
    let dir = setup("mixed");
    let mixed = |run: &str| {
        let mut c = config(&dir, run, "adamw", 7);
        c.data.train = None;
        c.data.sources = ["train_000.bin", "train_001.bin"]
            .iter()
            .zip([2.0, 1.0])
            .map(|(n, weight)| kapel_train::config::SourceConfig {
                path: dir.join(n).to_string_lossy().into_owned(),
                weight,
            })
            .collect();
        c.data.seed = 5;
        c.data.shuffle = true;
        c.data.pack = kapel_train::data::Pack::Documents;
        c.data.mask = true;
        c.data.separator = Some(0);
        c.validate().unwrap();
        c
    };
    let whole = run(&mixed("whole"), &QUIET).unwrap();
    let again = run(&mixed("again"), &QUIET).unwrap();
    assert_eq!(
        whole.loss.to_bits(),
        again.loss.to_bits(),
        "the same seed, the same run"
    );
    let cut = mixed("cut");
    run(
        &cut,
        &RunOptions {
            stop_after: Some(3),
            ..QUIET
        },
    )
    .unwrap();
    let second = run(
        &cut,
        &RunOptions {
            resume: true,
            ..QUIET
        },
    )
    .unwrap();
    assert_eq!(second.loss.to_bits(), whole.loss.to_bits());
    assert_eq!(second.val_loss, whole.val_loss);
    for (name, w) in &whole.params {
        assert_eq!(
            w.to_f64_vec().unwrap(),
            second.params[name].to_f64_vec().unwrap(),
            "{name}"
        );
    }
    let mut other = mixed("other");
    other.data.seed = 6;
    let other = run(&other, &QUIET).unwrap();
    assert_ne!(
        other.loss.to_bits(),
        whole.loss.to_bits(),
        "another seed, other rows"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_loss_goes_down_and_the_checkpoint_is_a_model() {
    let dir = setup("learns");
    let summary = run(&config(&dir, "run", "adamw", 40), &QUIET).unwrap();
    let log = std::fs::read_to_string(dir.join("run/log.jsonl")).unwrap();
    let first: serde_json::Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    let first = first["loss"].as_f64().unwrap();
    assert!(first > 3.5, "a first loss of {first}, ln 50 = 3.9");
    assert!(summary.loss < first - 1.0, "{first} then {}", summary.loss);
    let (_, last) = kapel_train::checkpoint::list(&dir.join("run"))
        .pop()
        .unwrap();
    let model = kapel_models::Model::load(&kapel_hub::Checkpoint::open(&last).unwrap()).unwrap();
    assert_eq!(model.config().layers, 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The same on the card, in bf16: a resumed run computes the bits of an
/// uninterrupted one.
#[cfg(feature = "cuda")]
#[test]
fn a_resumed_run_on_the_card_computes_the_same_bits() {
    let dir = setup("resume-cuda");
    let on_card = |run: &str| {
        let mut c = config(&dir, run, "adamw", 7);
        c.training.device = "cuda".into();
        c.training.dtype = "bf16".into();
        c
    };
    let whole = run(&on_card("whole"), &QUIET).unwrap();
    let cut = on_card("cut");
    run(
        &cut,
        &RunOptions {
            stop_after: Some(3),
            ..QUIET
        },
    )
    .unwrap();
    let second = run(
        &cut,
        &RunOptions {
            resume: true,
            ..QUIET
        },
    )
    .unwrap();
    assert_eq!(second.loss.to_bits(), whole.loss.to_bits());
    for (name, w) in &whole.params {
        assert_eq!(
            w.to_f64_vec().unwrap(),
            second.params[name].to_f64_vec().unwrap(),
            "{name}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
