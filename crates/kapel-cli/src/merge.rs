//! `kapel merge`: a LoRA adapter merged into its model.

use std::path::PathBuf;

use kapel_core::DType;

pub(crate) const USAGE: &str = "\
usage: kapel merge MODEL ADAPTER --output DIR [--dtype bf16|f16|f32]

Merges the LoRA adapter in the directory ADAPTER (adapter_config.json and
adapter_model.safetensors, as PEFT writes them, or a checkpoint of kapel train
with adapters) into MODEL (a model directory, or a repository of the Hub), and
writes the model into DIR: config.json and model.safetensors, which kapel run
and transformers load, with MODEL's tokenizer and other files. An adapter of
the embeddings of a model whose output shares their table unties them.

options:
  --output DIR             the directory written
  --dtype TYPE             the type of the float weights (default the model's)
  --revision REV           the revision of a Hub repository (default main)
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let (mut paths, mut output, mut dtype, mut revision) = (Vec::new(), None, None, "main");
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--output" => output = rest.next().map(PathBuf::from),
            "--revision" => revision = rest.next().ok_or("--revision needs a value")?,
            "--dtype" => {
                dtype = Some(match rest.next().map(String::as_str) {
                    Some("bf16") => DType::BF16,
                    Some("f16") => DType::F16,
                    Some("f32") => DType::F32,
                    other => return Err(format!("--dtype: bf16, f16 or f32, got {other:?}")),
                });
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other => paths.push(other.to_string()),
        }
    }
    let [model, adapter] = &paths[..] else {
        return Err(USAGE.to_string());
    };
    let output = output.ok_or(format!("--output is needed\n\n{USAGE}"))?;
    let base = crate::directory(model, revision)?;
    let merged = kapel_train::lora::merge_dirs(&base, std::path::Path::new(adapter), dtype)?;
    merged.save(&output).map_err(|e| e.to_string())?;
    // The tokenizer and the other files of the model, its weights apart.
    for entry in std::fs::read_dir(&base).map_err(|e| format!("{}: {e}", base.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let weights = name.ends_with(".safetensors")
            || name.ends_with(".bin")
            || name == "model.safetensors.index.json"
            || name == "config.json";
        if path.is_file() && !weights {
            std::fs::copy(&path, output.join(name))
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    eprintln!("merge: {} written", output.display());
    Ok(())
}
