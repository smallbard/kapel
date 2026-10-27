//! Several ranks from the command line, on the processor: two processes
//! started as torchrun starts them, joined by TCP, and two threads of one
//! process (`--ranks`), each computing the losses of one process that
//! accumulates their passes, bit for bit.

mod common;

use std::path::Path;
use std::process::Command;

use common::{kapel, tokens};

const RUN: &str = r#"
[model]
config = "config.json"
[data]
train = "train_*.bin"
[batch]
rows = 2
length = 16
tokens = 64
[optimizer]
kind = "adamw"
lr = 3e-3
[schedule]
steps = 4
[training]
device = "cpu"
dtype = "f32"
[eval]
every = 0
tokens = 0
[checkpoint]
every = 0
"#;

/// The loss of each step of a run's log.
fn losses(dir: &Path) -> Vec<u64> {
    std::fs::read_to_string(dir.join("log.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter_map(|v| v.get("loss").and_then(|l| l.as_f64()))
        .map(f64::to_bits)
        .collect()
}

#[test]
fn ranks_in_processes_and_threads_compute_the_losses_of_one() {
    let dir = std::env::temp_dir().join(format!("kapel-ranks-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "gpt2", "n_embd": 32, "n_head": 2, "n_layer": 2,
            "vocab_size": 50, "n_positions": 32, "activation_function": "gelu_new"}"#,
    )
    .unwrap();
    tokens(&dir.join("train_000.bin"), 3000, 1);
    std::fs::write(dir.join("train.toml"), RUN).unwrap();

    let (ok, out) = kapel(&["train", "train.toml", "--dir", "one", "--quiet"], &dir);
    assert!(ok, "{out}");
    let (ok, out) = kapel(
        &[
            "train",
            "train.toml",
            "--dir",
            "threads",
            "--ranks",
            "2",
            "--quiet",
        ],
        &dir,
    );
    assert!(ok, "{out}");
    // Two processes, as torchrun starts them, at once: rank 0 waits for the
    // other to connect.
    let port = 29_600 + (std::process::id() % 300) as u16;
    let spawned: Vec<_> = (0..2)
        .map(|rank| {
            Command::new(env!("CARGO_BIN_EXE_kapel"))
                .args(["train", "train.toml", "--dir", "processes", "--quiet"])
                .current_dir(&dir)
                .env("RANK", rank.to_string())
                .env("WORLD_SIZE", "2")
                .env("MASTER_ADDR", "127.0.0.1")
                .env("MASTER_PORT", port.to_string())
                .spawn()
                .unwrap()
        })
        .collect();
    for mut child in spawned {
        assert!(child.wait().unwrap().success());
    }
    let one = losses(&dir.join("one"));
    assert_eq!(one.len(), 4);
    assert_eq!(losses(&dir.join("threads")), one);
    assert_eq!(losses(&dir.join("processes")), one);
    std::fs::remove_dir_all(&dir).unwrap();
}

const LONG: &str = r#"
[model]
config = "config.json"
[data]
train = "train_*.bin"
[batch]
rows = 2
length = 16
tokens = 64
[optimizer]
kind = "adamw"
lr = 3e-3
[schedule]
steps = 300
[training]
device = "cpu"
dtype = "f32"
[eval]
every = 0
tokens = 0
[checkpoint]
every = 5
keep = 2
"#;

fn spawn(dir: &Path, port: u16, rank: usize, resume: bool) -> std::process::Child {
    let mut args = vec!["train", "long.toml", "--dir", "killed", "--quiet"];
    if resume {
        args.push("--resume");
    }
    Command::new(env!("CARGO_BIN_EXE_kapel"))
        .args(&args)
        .current_dir(dir)
        .env("RANK", rank.to_string())
        .env("WORLD_SIZE", "2")
        .env("MASTER_ADDR", "127.0.0.1")
        .env("MASTER_PORT", port.to_string())
        .spawn()
        .unwrap()
}

/// A rank killed stops the other, which does not wait for it; started again
/// with --resume, both go on from the last checkpoint to the losses of a run
/// that was never stopped.
#[test]
fn a_rank_killed_stops_the_other_and_the_run_resumes() {
    let dir = std::env::temp_dir().join(format!("kapel-killed-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "gpt2", "n_embd": 32, "n_head": 2, "n_layer": 2,
            "vocab_size": 50, "n_positions": 32, "activation_function": "gelu_new"}"#,
    )
    .unwrap();
    tokens(&dir.join("train_000.bin"), 3000, 1);
    std::fs::write(dir.join("long.toml"), LONG).unwrap();
    let (ok, out) = kapel(&["train", "long.toml", "--dir", "whole", "--quiet"], &dir);
    assert!(ok, "{out}");

    let port = 29_900 + (std::process::id() % 50) as u16;
    let (mut zero, mut one) = (spawn(&dir, port, 0, false), spawn(&dir, port, 1, false));
    let log = dir.join("killed/log.jsonl");
    let start = std::time::Instant::now();
    while std::fs::read_to_string(&log).map_or(0, |l| l.lines().count()) < 20 {
        assert!(start.elapsed().as_secs() < 120, "the run does not start");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    one.kill().unwrap();
    one.wait().unwrap();
    let status = zero.wait().unwrap();
    assert!(!status.success(), "rank 0 goes on without rank 1");
    let steps = std::fs::read_to_string(&log).unwrap().lines().count();
    assert!(steps < 300, "killed after the end: {steps} steps");

    let port = port + 50;
    let (mut zero, mut one) = (spawn(&dir, port, 0, true), spawn(&dir, port, 1, true));
    assert!(zero.wait().unwrap().success());
    assert!(one.wait().unwrap().success());
    let last = |d: &str| *losses(&dir.join(d)).last().unwrap();
    assert_eq!(last("killed"), last("whole"));
    std::fs::remove_dir_all(&dir).unwrap();
}
