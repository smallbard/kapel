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
