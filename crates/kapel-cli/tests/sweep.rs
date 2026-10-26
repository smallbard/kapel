//! `kapel sweep` and `kapel fit` from end to end, on the processor: a grid
//! of three sizes at one budget runs, is noted, is not run again, and has
//! its best size fitted.

mod common;

use common::{kapel, tokens};

#[test]
fn a_grid_runs_once_and_its_best_size_is_fitted() {
    let dir = std::env::temp_dir().join(format!("kapel-sweep-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2,
            "vocab_size": 50, "hidden_act": "silu", "tie_word_embeddings": true}"#,
    )
    .unwrap();
    tokens(&dir.join("train_000.bin"), 20_000, 1);
    tokens(&dir.join("val_000.bin"), 1_000, 2);
    std::fs::write(
        dir.join("train.toml"),
        r#"
        [model]
        config = "config.json"
        [data]
        train = "train_*.bin"
        val = "val_*.bin"
        [batch]
        rows = 4
        length = 16
        [optimizer]
        kind = "adamw"
        lr = 3e-3
        [schedule]
        warmup = 2
        final = 0.1
        [training]
        device = "cpu"
        dtype = "f32"
        budget = 4e8
        [eval]
        every = 0
        tokens = 256
        [checkpoint]
        every = 5
        "#,
    )
    .unwrap();
    let mut sweep = String::from("[sweep]\nbase = \"train.toml\"\nparallel = 2\n");
    for h in [32, 48, 64] {
        sweep += &format!(
            "\n[[run]]\nname = \"h{h}\"\nmodel.set = {{ hidden_size = {h}, intermediate_size = {} }}\n",
            2 * h
        );
    }
    std::fs::write(dir.join("grid.toml"), sweep).unwrap();

    let (ok, plan) = kapel(&["sweep", "grid.toml", "--dry-run"], &dir);
    assert!(ok, "{plan}");
    assert_eq!(plan.lines().count(), 4, "{plan}");

    let (ok, out) = kapel(&["sweep", "grid.toml"], &dir);
    assert!(ok, "{out}");
    let table = std::fs::read_to_string(dir.join("grid-runs/results.tsv")).unwrap();
    let done: Vec<&str> = table.lines().skip(1).collect();
    assert_eq!(done.len(), 3, "{table}");
    assert!(
        done.iter().all(|l| l.split('\t').nth(1) == Some("done")),
        "{table}"
    );
    for h in [32, 48, 64] {
        assert!(dir.join(format!("grid-runs/h{h}/summary.json")).exists());
    }

    // Done runs are not run again.
    let (ok, again) = kapel(&["sweep", "grid.toml"], &dir);
    assert!(ok && again.contains("0 runs to do of 3"), "{again}");
    let table_again = std::fs::read_to_string(dir.join("grid-runs/results.tsv")).unwrap();
    assert_eq!(table, table_again);

    let (ok, fit) = kapel(
        &["fit", "grid-runs/results.tsv", "--loss", "val_loss"],
        &dir,
    );
    assert!(ok, "{fit}");
    assert!(
        fit.contains("3 runs") && fit.contains("budget 4.000e8"),
        "{fit}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
