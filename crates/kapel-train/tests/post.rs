//! Post-training on the processor, with a small Qwen 3 and its chat
//! template: DPO starts at `ln 2` and learns the preference, measured on
//! pairs held out, with adapters or without, and resumes bit for bit; GRPO
//! makes a rewarded answer more likely; a task is evaluated along the run.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use kapel_train::config::TrainConfig;
use kapel_train::trainer::{RunOptions, run};
use serde_json::Value;

fn model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/qwen3")
}

/// Whether the small Qwen 3 of `data/parity/qwen3` is there; without it (in
/// continuous integration) the test is skipped and says so, and with
/// `KAPEL_REQUIRE_PARITY=1` it fails instead.
fn have_model() -> bool {
    if model().join("model.safetensors").exists() {
        return true;
    }
    assert!(
        std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
        "data/parity/qwen3 is missing, and KAPEL_REQUIRE_PARITY=1"
    );
    let _ = writeln!(
        std::io::stderr(),
        "skipped: data/parity/qwen3 missing (see tools/parity/README.md)"
    );
    false
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kapel-post-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const QUIET: RunOptions = RunOptions {
    resume: false,
    stop_after: None,
    quiet: true,
    profile: false,
};

fn log(dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(dir.join("run/log.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn steps(lines: &[Value]) -> Vec<&Value> {
    lines.iter().filter(|l| l.get("loss").is_some()).collect()
}

fn pairs(dir: &Path) {
    let lines: Vec<String> = (0..8)
        .map(|i| {
            serde_json::json!({
                "prompt": [{"role": "user", "content": format!("Pick one, number {i}.")}],
                "chosen": [{"role": "assistant", "content": "yes"}],
                "rejected": [{"role": "assistant", "content": "no"}],
            })
            .to_string()
        })
        .collect();
    std::fs::write(dir.join("pairs.jsonl"), lines.join("\n")).unwrap();
}

fn dpo_config(dir: &Path, lora: bool, steps: usize) -> TrainConfig {
    let model = model();
    // The output of this small model is too small for adapters alone to
    // move its answers far: it trains too, as `modules_to_save` would. The
    // reference is then a copy of the starting weights.
    let lora = if lora {
        "[lora]\nr = 4\nalpha = 8\n[[optimizer.groups]]\nmatches = \"lm_head.weight\"\ntrainable = true\n"
    } else {
        ""
    };
    let lr = 1e-2;
    let text = format!(
        r#"
        [model]
        config = "{}"
        init = "weights"
        [batch]
        rows = 4
        length = 48
        [optimizer]
        lr = {lr}
        [schedule]
        kind = "constant"
        steps = {steps}
        [training]
        device = "cpu"
        dtype = "f32"
        chunk = 64
        [eval]
        tokens = 0
        [checkpoint]
        dir = "run"
        every = 3
        keep = 0
        [dpo]
        train = "pairs.jsonl"
        val = "pairs.jsonl"
        beta = 0.5
        {lora}
        "#,
        model.display()
    );
    let mut c = TrainConfig::parse(&text).unwrap();
    c.resolve(dir);
    c
}

#[test]
fn dpo_with_adapters_starts_at_ln_2_and_learns_the_preference() {
    if !have_model() {
        return;
    }
    let dir = temp("dpo-lora");
    pairs(&dir);
    run(&dpo_config(&dir, true, 12), &QUIET).unwrap();
    let lines = log(&dir);
    let steps = steps(&lines);
    let first = steps[0]["loss"].as_f64().unwrap();
    assert!(
        (first - std::f64::consts::LN_2).abs() < 1e-9,
        "the model is its reference: ln 2, got {first}"
    );
    let last = steps.last().unwrap();
    assert_eq!(last["accuracy"].as_f64(), Some(1.0));
    assert!(last["margin"].as_f64().unwrap() > 0.5, "{last}");
    assert!(last["loss"].as_f64().unwrap() < 0.4, "{last}");
    // The pairs held out (here the same), measured at the end.
    let held = lines
        .iter()
        .find(|l| l.get("val_accuracy").is_some())
        .expect("an evaluation of the pairs held out");
    assert_eq!(held["step"], 12);
    assert_eq!(held["val_accuracy"].as_f64(), Some(1.0), "{held}");
    assert!(held["val_loss"].as_f64().unwrap() < 0.4, "{held}");
    assert!(held["val_margin"].as_f64().unwrap() > 0.5, "{held}");
    // The checkpoint is an adapter.
    let (_, checkpoint) = kapel_train::checkpoint::list(&dir.join("run"))
        .pop()
        .unwrap();
    assert!(checkpoint.join("adapter_config.json").exists());
    assert!(checkpoint.join("trainable.safetensors").exists());
    assert!(!checkpoint.join("model.safetensors").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Pairs joined in one row without padding train as padded ones: the same
/// losses and held-out measures but for the order of the sums, fewer tokens.
#[test]
fn joined_pairs_train_as_padded_ones() {
    if !have_model() {
        return;
    }
    let measure = |name: &str, joined: bool| -> Vec<Value> {
        let dir = temp(name);
        pairs(&dir);
        let mut c = dpo_config(&dir, true, 4);
        if joined {
            c.data.pack = kapel_train::data::Pack::Joined;
        }
        c.validate().unwrap();
        run(&c, &QUIET).unwrap();
        let lines = log(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        lines
    };
    let (padded, joined) = (measure("dpo-padded", false), measure("dpo-joined", true));
    assert_eq!(padded.len(), joined.len());
    for (a, b) in padded.iter().zip(&joined) {
        for key in ["loss", "accuracy", "margin", "val_loss", "val_margin"] {
            if let (Some(x), Some(y)) = (a.get(key), b.get(key)) {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                assert!(
                    (x - y).abs() <= 1e-5 * (1.0 + x.abs()),
                    "{key}: {x} and {y}"
                );
            }
        }
    }
    let tokens = |lines: &[Value]| steps(lines).last().unwrap()["tokens"].as_f64().unwrap();
    assert!(tokens(&joined) < tokens(&padded));
}

#[test]
fn dpo_without_adapters_resumes_bit_for_bit() {
    if !have_model() {
        return;
    }
    let dir = temp("dpo-full");
    pairs(&dir);
    let whole = run(&dpo_config(&dir, false, 6), &QUIET).unwrap();
    let first = steps(&log(&dir))[0]["loss"].as_f64().unwrap();
    assert!((first - std::f64::consts::LN_2).abs() < 1e-9, "{first}");
    std::fs::remove_dir_all(dir.join("run")).unwrap();
    let stopped = RunOptions {
        stop_after: Some(3),
        ..QUIET
    };
    run(&dpo_config(&dir, false, 6), &stopped).unwrap();
    let resumed = RunOptions {
        resume: true,
        ..QUIET
    };
    let again = run(&dpo_config(&dir, false, 6), &resumed).unwrap();
    assert_eq!(whole.loss.to_bits(), again.loss.to_bits());
    for (name, t) in &whole.params {
        assert_eq!(
            t.to_f64_vec().unwrap(),
            again.params[name].to_f64_vec().unwrap(),
            "{name}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The probability of `token` after `prompt` by a model of `params`.
fn probability(
    params: &std::collections::BTreeMap<String, kapel_core::Tensor>,
    prompt: &[u32],
    token: u32,
) -> f64 {
    let raw: Value =
        serde_json::from_slice(&std::fs::read(model().join("config.json")).unwrap()).unwrap();
    let m = kapel_models::Model::new(raw, params.clone()).unwrap();
    let logits = m.forward(&kapel_cpu::Cpu::new(), prompt).unwrap();
    let vocab = *logits.shape().last().unwrap();
    let all = logits.to_f64_vec().unwrap();
    let last = &all[all.len() - vocab..];
    let max = last.iter().copied().fold(f64::MIN, f64::max);
    let sum: f64 = last.iter().map(|l| (l - max).exp()).sum();
    (last[token as usize] - max).exp() / sum
}

#[test]
fn grpo_makes_a_rewarded_answer_more_likely_and_a_task_is_measured() {
    if !have_model() {
        return;
    }
    let dir = temp("grpo");
    let record = serde_json::json!({
        "prompt": [{"role": "user", "content": "Say a word."}],
        "answer": "unused",
    });
    std::fs::write(dir.join("prompts.jsonl"), record.to_string()).unwrap();
    // The answer rewarded: the third most likely first token.
    let encoder = kapel_train::chat::Encoder::from_dir(&model()).unwrap();
    let prompt = encoder
        .prompt(
            &kapel_train::chat::Turns::Messages(vec![kapel_tokenizers::Message::new(
                "user",
                "Say a word.",
            )]),
            &Default::default(),
        )
        .unwrap();
    let start = kapel_models::Model::load(&kapel_hub::Checkpoint::open(model()).unwrap()).unwrap();
    let before: std::collections::BTreeMap<String, kapel_core::Tensor> = start
        .names()
        .map(|n| (n.to_string(), start.tensor(n).unwrap().clone()))
        .collect();
    let logits = start.forward(&kapel_cpu::Cpu::new(), &prompt).unwrap();
    let vocab = *logits.shape().last().unwrap();
    let all = logits.to_f64_vec().unwrap();
    let mut ranked: Vec<usize> = (0..vocab).collect();
    let last = &all[all.len() - vocab..];
    ranked.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    let token = ranked[2] as u32;
    let text = encoder.tokenizer().decode(&[token], false);
    let pattern = format!("^{}", fancy_regex::escape(&text));
    let text = format!(
        r#"
        [model]
        config = "{}"
        init = "weights"
        [batch]
        rows = 8
        length = 32
        [optimizer]
        lr = 2e-2
        [schedule]
        kind = "constant"
        steps = 20
        [training]
        device = "cpu"
        dtype = "f32"
        chunk = 64
        [eval]
        every = 10
        tokens = 0
        [[eval.tasks]]
        name = "word"
        prompts = "prompts.jsonl"
        max_tokens = 1
        [[eval.tasks.rewards]]
        kind = "regex"
        pattern = {pattern:?}
        [checkpoint]
        dir = "run"
        every = 0
        [grpo]
        prompts = "prompts.jsonl"
        prompts_per_step = 1
        group = 8
        max_tokens = 1
        top_k = 4
        beta = 0.04
        [[grpo.rewards]]
        kind = "regex"
        pattern = {pattern:?}
        "#,
        model().display()
    );
    let mut config = TrainConfig::parse(&text).unwrap();
    config.resolve(&dir);
    let summary = run(&config, &QUIET).unwrap();
    let p0 = probability(&before, &prompt, token);
    let p1 = probability(&summary.params, &prompt, token);
    assert!(p1 > 3.0 * p0, "{p0} then {p1}");
    let lines = log(&dir);
    let steps = steps(&lines);
    assert!(
        steps
            .iter()
            .all(|s| s["reward"].is_number() && s["kl"].is_number())
    );
    // The task, answered greedily at steps 10 and 20: the rewarded token
    // ends up the most likely.
    let tasks: Vec<&Value> = lines.iter().filter(|l| l.get("task").is_some()).collect();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[1]["accuracy"].as_f64(), Some(1.0), "{:?}", tasks);
    std::fs::remove_dir_all(&dir).unwrap();
}
