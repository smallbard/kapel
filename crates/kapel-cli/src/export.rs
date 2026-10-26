//! `kapel export`: a checkpoint as a repository of the Hub holds it.

use std::path::{Path, PathBuf};

use kapel_core::DType;
use kapel_train::export::{ExportOptions, export};

pub(crate) const USAGE: &str = "\
usage: kapel export CHECKPOINT --output DIR [options]

Writes the checkpoint CHECKPOINT of kapel train (a step-NNNNNNNN directory, or
a run's directory for its last one) into DIR as a repository of the Hugging
Face Hub holds a model: model.safetensors in bf16 and config.json, which
transformers and kapel run load, without the optimizer's state; or, for a run
with LoRA adapters, adapter_model.safetensors and adapter_config.json, which
PEFT loads, naming the Hub's model the run started from. With them the
tokenizer's files and generation_config.json of that model, and README.md, a
model card with the last losses and the run's configuration, the paths of this
machine reduced to their file names. Publish DIR with `hf upload OWNER/NAME
DIR`.

options:
  --output DIR             the directory written
  --dtype TYPE             the type of the float weights: bf16 (default), f16
                           or f32
  --tokenizer MODEL        the tokenizer of MODEL (a directory, or a repository
                           of the Hub) instead of the run's model's: for a model
                           trained from a config.json alone
  --merge                  an adapter merged into its model, written as a model
";

pub(crate) fn execute(args: &[String]) -> Result<(), String> {
    let (mut path, mut output, mut options) = (None, None, ExportOptions::default());
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--output" => {
                output = Some(PathBuf::from(
                    rest.next().ok_or("--output needs a directory")?,
                ))
            }
            "--merge" => options.merge = true,
            "--tokenizer" => {
                let model = rest.next().ok_or("--tokenizer needs a model")?;
                options.tokenizer = Some(crate::directory(model, "main")?);
            }
            "--dtype" => {
                options.dtype = match rest.next().map(String::as_str) {
                    Some("bf16") => DType::BF16,
                    Some("f16") => DType::F16,
                    Some("f32") => DType::F32,
                    other => return Err(format!("--dtype: bf16, f16 or f32, got {other:?}")),
                };
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("one checkpoint, got {other} too")),
        }
    }
    let (Some(path), Some(output)) = (path, output) else {
        return Err(USAGE.to_string());
    };
    let done = export(Path::new(&path), &output, &options)?;
    let tokenizer = done
        .files
        .iter()
        .any(|f| f.starts_with("tokenizer") || f == "vocab.json");
    eprintln!(
        "export: {} written from {} as {}{}: {}",
        output.display(),
        done.checkpoint.display(),
        if done.adapter {
            "an adapter"
        } else {
            "a model"
        },
        done.base.map(|b| format!(" of {b}")).unwrap_or_default(),
        done.files.join(", ")
    );
    if !tokenizer {
        eprintln!("export: no tokenizer found for the model; give one with --tokenizer MODEL");
    }
    Ok(())
}
