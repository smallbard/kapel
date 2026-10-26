//! `kapel export` from end to end, on the processor: a model trained from
//! scratch, an adapter trained on it, and the adapter merged, each written
//! as the Hub holds it.

mod common;

use std::path::Path;

use common::{kapel, tokens};
use serde_json::Value;

/// The header of a safetensors file.
fn header(path: &Path) -> serde_json::Map<String, Value> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let Value::Object(mut map) = serde_json::from_slice(&bytes[8..8 + n]).unwrap() else {
        panic!("a header is an object")
    };
    map.remove("__metadata__");
    map
}

fn json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

const RUN: &str = r#"
[data]
train = "train_*.bin"
val = "val_*.bin"
[batch]
rows = 2
length = 16
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
tokens = 64
[checkpoint]
every = 2
"#;

#[test]
fn checkpoints_are_written_as_the_hub_holds_models() {
    let dir = std::env::temp_dir().join(format!("kapel-export-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2,
            "vocab_size": 50, "hidden_act": "silu", "tie_word_embeddings": false,
            "bos_token_id": 1, "eos_token_id": 2}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    tokens(&dir.join("train_000.bin"), 5_000, 1);
    tokens(&dir.join("val_000.bin"), 500, 2);
    std::fs::write(
        dir.join("scratch.toml"),
        format!("[model]\nconfig = \"config.json\"\n[checkpoint.x]\n{RUN}")
            .replace("[checkpoint.x]\n", "")
            .replace(
                "[checkpoint]\nevery = 2",
                "[checkpoint]\ndir = \"scratch\"\nevery = 2",
            ),
    )
    .unwrap();
    let (ok, out) = kapel(&["train", "scratch.toml", "--quiet"], &dir);
    assert!(ok, "{out}");

    // The model: bf16, its tokenizer, its card, no optimizer.
    let (ok, out) = kapel(&["export", "scratch", "--output", "model"], &dir);
    assert!(ok, "{out}");
    let model = dir.join("model");
    let mut names: Vec<String> = std::fs::read_dir(&model)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "README.md",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json"
        ]
    );
    assert_eq!(json(&model.join("config.json"))["torch_dtype"], "bfloat16");
    assert_eq!(
        json(&model.join("generation_config.json"))["eos_token_id"],
        2
    );
    let weights = header(&model.join("model.safetensors"));
    assert!(weights.contains_key("lm_head.weight"));
    assert!(
        weights.values().all(|t| t["dtype"] == "BF16"),
        "{weights:?}"
    );
    let card = std::fs::read_to_string(model.join("README.md")).unwrap();
    assert!(card.contains("library_name: transformers"), "{card}");
    assert!(card.contains("| Steps | 4 |"), "{card}");
    assert!(card.contains("validation loss"), "{card}");
    assert!(!card.contains(dir.to_str().unwrap()), "{card}");

    // An adapter trained on that model, read by PEFT.
    std::fs::write(
        dir.join("lora.toml"),
        format!(
            "[model]\nconfig = \"scratch/step-00000004\"\ninit = \"weights\"\n[lora]\nr = 4\nalpha = 8\n{RUN}"
        )
        .replace("[checkpoint]\nevery = 2", "[checkpoint]\ndir = \"lora\"\nevery = 2"),
    )
    .unwrap();
    let (ok, out) = kapel(&["train", "lora.toml", "--quiet"], &dir);
    assert!(ok, "{out}");
    let (ok, out) = kapel(
        &["export", "lora", "--output", "adapter", "--tokenizer", "."],
        &dir,
    );
    assert!(ok, "{out}");
    let adapter = dir.join("adapter");
    let config = json(&adapter.join("adapter_config.json"));
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["base_model_name_or_path"], "scratch/step-00000004");
    assert!(adapter.join("adapter_model.safetensors").exists());
    assert!(adapter.join("tokenizer.json").exists());
    assert!(!adapter.join("model.safetensors").exists());
    let card = std::fs::read_to_string(adapter.join("README.md")).unwrap();
    assert!(card.contains("library_name: peft"), "{card}");
    assert!(!card.contains(dir.to_str().unwrap()), "{card}");

    // The adapter merged: a model without factors.
    let (ok, out) = kapel(
        &[
            "export",
            "lora/step-00000004",
            "--output",
            "merged",
            "--merge",
        ],
        &dir,
    );
    assert!(ok, "{out}");
    let weights = header(&dir.join("merged/model.safetensors"));
    assert!(!weights.keys().any(|n| n.contains("lora")), "{weights:?}");
    assert!(
        weights.values().all(|t| t["dtype"] == "BF16"),
        "{weights:?}"
    );
    assert_eq!(
        json(&dir.join("merged/config.json"))["torch_dtype"],
        "bfloat16"
    );

    // Not into the checkpoint itself.
    let (ok, out) = kapel(
        &[
            "export",
            "scratch/step-00000004",
            "--output",
            "scratch/step-00000004",
        ],
        &dir,
    );
    assert!(!ok && out.contains("the checkpoint itself"), "{out}");
    std::fs::remove_dir_all(&dir).unwrap();
}
