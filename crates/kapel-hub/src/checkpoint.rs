//! A model directory in the layout of the Hugging Face Hub.
//!
//! The weights are one `model.safetensors`, or several shards listed by
//! `model.safetensors.index.json` (its `weight_map` gives the file of each
//! tensor). `config.json` describes the architecture, `tokenizer.json` the
//! tokenizer, and `generation_config.json` the default generation settings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kapel_core::Tensor;
use serde_json::Value;

use crate::error::{HubError, Result};
use crate::safetensors::{SafeTensors, TensorInfo};

/// The weights and files of a model directory.
pub struct Checkpoint {
    dir: PathBuf,
    shards: Vec<SafeTensors>,
    /// For each tensor, its shard.
    owner: BTreeMap<String, usize>,
}

/// Reads and parses a JSON file.
pub fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read(path).map_err(|e| HubError::io(path, e))?;
    serde_json::from_slice(&text)
        .map_err(|e| HubError::format(path, format!("not valid JSON: {e}")))
}

/// Whether `name` is the name of a file directly in a directory: not empty, not
/// `.` or `..`, without a separator (`/` or `\`), a drive (`:`) or a NUL.
/// Names read from files (a `weight_map`, a Hub listing) are checked with it
/// before they are joined to a directory, so that they cannot reach outside.
pub(crate) fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

impl Checkpoint {
    /// Opens the weights of a model directory.
    ///
    /// # Errors
    ///
    /// [`HubError::Io`] if no weights are found or a shard cannot be read,
    /// [`HubError::Format`] if a file is not valid.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<PathBuf> = if index.exists() {
            let map = read_json(&index)?;
            let mut names: Vec<String> = map
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| HubError::format(&index, "no weight_map object"))?
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            names.sort();
            names.dedup();
            if let Some(bad) = names.iter().find(|n| !is_plain_file_name(n)) {
                return Err(HubError::format(
                    &index,
                    format!("the shard {bad:?} is not a file of the model directory"),
                ));
            }
            names.into_iter().map(|n| dir.join(n)).collect()
        } else if dir.join("model.safetensors").exists() {
            vec![dir.join("model.safetensors")]
        } else {
            return Err(HubError::io(
                &dir,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no model.safetensors or index",
                ),
            ));
        };
        let mut shards = Vec::new();
        let mut owner = BTreeMap::new();
        for (i, file) in files.iter().enumerate() {
            let shard = SafeTensors::open(file)?;
            for name in shard.names() {
                owner.insert(name.to_string(), i);
            }
            shards.push(shard);
        }
        Ok(Self { dir, shards, owner })
    }

    /// The directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The names of all tensors, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.owner.keys().map(String::as_str)
    }

    /// The entry of a tensor.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.owner
            .get(name)
            .and_then(|&i| self.shards[i].info(name))
    }

    /// A tensor, sharing the memory of its file.
    ///
    /// # Errors
    ///
    /// [`HubError::Missing`] if no shard holds it.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let &i = self.owner.get(name).ok_or_else(|| HubError::Missing {
            name: name.to_string(),
        })?;
        self.shards[i].tensor(name)
    }

    /// `config.json`.
    ///
    /// # Errors
    ///
    /// If the file is missing or not valid JSON.
    pub fn config(&self) -> Result<Value> {
        read_json(&self.dir.join("config.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_outside_the_directory_are_rejected() {
        let dir = std::env::temp_dir().join(format!("kapel-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for shard in [
            "../model.safetensors",
            "/etc/passwd",
            "sub/model.safetensors",
            "sub\\model.safetensors",
            "C:model.safetensors",
            "..",
            "",
        ] {
            let index = serde_json::json!({ "weight_map": { "w": shard } });
            std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
            let error = Checkpoint::open(&dir).err().unwrap().to_string();
            assert!(
                error.contains("not a file of the model directory"),
                "{shard}: {error}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn plain_file_names() {
        for name in [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "a..b",
        ] {
            assert!(is_plain_file_name(name), "{name}");
        }
    }
}
