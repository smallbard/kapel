//! The C example, `examples/c/generate.c`, compiled with the system's `cc`
//! against the library of this build and run on SmolLM2-135M-Instruct: it
//! prints the greedy answer that the Rust side gives.
//!
//! The model is read from the cache of the Hugging Face Hub
//! (`$HF_HOME/hub`, else `~/.cache/huggingface/hub`), or from
//! `$KAPEL_TEST_MODEL`. Without it, or without `cc`, the test is skipped and
//! says so; with `KAPEL_REQUIRE_PARITY=1`, it fails.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

use kapel_runtime::{GenerateOptions, LoadOptions, Model};

fn skip(why: &str) {
    assert!(
        std::env::var_os("KAPEL_REQUIRE_PARITY").is_none_or(|v| v != "1"),
        "{why}, and KAPEL_REQUIRE_PARITY=1"
    );
    eprintln!("skipped: {why}");
}

/// The snapshot of SmolLM2-135M-Instruct, if it is there.
fn model() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("KAPEL_TEST_MODEL") {
        return Some(PathBuf::from(dir));
    }
    let hub = match std::env::var_os("HF_HOME") {
        Some(home) => PathBuf::from(home).join("hub"),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub"),
    };
    let snapshots = hub.join("models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|d| d.join("model.safetensors").exists() && d.join("tokenizer.json").exists())
}

/// The directory of the shared library of this build, once built: the
/// parent of `deps`, where the test runs from. `cargo test` builds only the
/// Rust library; the shared one is built here, by the same Cargo, in the
/// same profile and with the same features.
fn library_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().unwrap().parent().unwrap().to_path_buf();
    let profile = dir.file_name().unwrap().to_str().unwrap();
    let profile = if profile == "debug" { "dev" } else { profile };
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut build = Command::new(cargo);
    build
        .args(["build", "-p", "kapel-capi", "--lib", "--profile", profile])
        .arg("--target-dir")
        .arg(dir.parent().unwrap());
    if cfg!(feature = "cuda") {
        build.args(["--features", "cuda"]);
    }
    assert!(
        build.status().unwrap().success(),
        "cargo build of libkapel failed"
    );
    dir
}

fn compile(out: &Path) -> bool {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lib = library_dir();
    let status = Command::new("cc")
        .args(["-std=c99", "-Wall", "-Wextra", "-Werror"])
        .arg(root.join("examples/c/generate.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&lib)
        .arg("-lkapel")
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .arg("-o")
        .arg(out)
        .status();
    match status {
        Ok(s) => {
            assert!(s.success(), "cc failed on generate.c");
            true
        }
        Err(e) => {
            skip(&format!("cc cannot run: {e}"));
            false
        }
    }
}

#[test]
fn the_c_example_answers_as_the_rust_side() {
    let Some(model) = model() else {
        skip("SmolLM2-135M-Instruct is not in the cache of the Hub");
        return;
    };
    let exe = std::env::temp_dir().join(format!("kapel-generate-{}", std::process::id()));
    if !compile(&exe) {
        return;
    }
    let question = "What is the capital of France?";
    let tokens = 16;
    let output = Command::new(&exe)
        .arg(&model)
        .arg(question)
        .arg(tokens.to_string())
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&exe);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");

    let rust = Model::load(model.to_str().unwrap(), &LoadOptions::default()).unwrap();
    let prompt = rust.chat_prompt(&[("user", question)]).unwrap();
    let options = GenerateOptions {
        temperature: 0.0,
        max_tokens: tokens,
        ..GenerateOptions::from(rust.default_request())
    };
    let outcome = rust.generate(&prompt, &options, |_| true).unwrap();
    assert!(!outcome.text.is_empty());
    assert_eq!(String::from_utf8_lossy(&output.stdout), outcome.text + "\n");
    let counts = format!(
        "[{} prompt tokens, {} tokens, ",
        prompt.len(),
        outcome.completion_tokens
    );
    assert!(stderr.contains(&counts), "{stderr}");
}
