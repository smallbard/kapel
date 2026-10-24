//! Checkpoints: everything a run needs to go on exactly where it was.
//!
//! A checkpoint is a directory `step-NNNNNNNN` of the run's directory:
//!
//! - `model.safetensors`, the parameters in `f32` under the names of
//!   `transformers`, and `config.json`: the directory is a model;
//! - `optimizer.safetensors`, the optimizer's tensors (its moments);
//! - `state.json`: the step, the optimizer's steps, the position in the
//!   training files, the tokens seen;
//! - `config.toml`, the configuration of the run.
//!
//! The tensors are copied to the host when the checkpoint is taken; the
//! files are then written by a thread of their own, under a temporary name,
//! and the directory renamed into place once complete, so a run killed
//! while writing leaves the previous checkpoints whole. The oldest are
//! removed beyond the number kept.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use kapel_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::data::Position;

/// What `state.json` holds.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct State {
    /// The steps done.
    pub step: usize,
    /// The optimizer's steps.
    pub optimizer_steps: u64,
    /// The optimizer's kind.
    pub optimizer: String,
    /// Where the training batches are.
    pub position: Position,
    /// The training tokens seen.
    pub tokens: u64,
}

/// A checkpoint in host memory.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// The state.
    pub state: State,
    /// The parameters, in `f32`.
    pub params: BTreeMap<String, Tensor>,
    /// The optimizer's tensors.
    pub optimizer: BTreeMap<String, Tensor>,
    /// The configuration of the model, `config.json`.
    pub model_config: serde_json::Value,
    /// The configuration of the run, as TOML.
    pub run_config: String,
}

fn io(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", path.display())
}

/// The name of the checkpoint of step `step`.
pub fn name(step: usize) -> String {
    format!("step-{step:08}")
}

impl Checkpoint {
    /// Writes the checkpoint into `dir/step-…`, through a temporary
    /// directory renamed once complete.
    ///
    /// # Errors
    ///
    /// If a file cannot be written.
    pub fn write(&self, dir: &Path) -> Result<PathBuf, String> {
        let target = dir.join(name(self.state.step));
        let temporary = dir.join(format!(".{}.partial", name(self.state.step)));
        if temporary.exists() {
            std::fs::remove_dir_all(&temporary).map_err(|e| io(&temporary, e))?;
        }
        std::fs::create_dir_all(&temporary).map_err(|e| io(&temporary, e))?;
        let save = |file: &str, tensors: &BTreeMap<String, Tensor>| -> Result<(), String> {
            let refs: Vec<(&str, &Tensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
            kapel_hub::safetensors::save(temporary.join(file), &refs, &metadata)
                .map_err(|e| e.to_string())
        };
        save("model.safetensors", &self.params)?;
        save("optimizer.safetensors", &self.optimizer)?;
        let json = serde_json::to_vec_pretty(&self.model_config).expect("JSON serializes");
        std::fs::write(temporary.join("config.json"), json).map_err(|e| io(&temporary, e))?;
        std::fs::write(temporary.join("config.toml"), &self.run_config)
            .map_err(|e| io(&temporary, e))?;
        // The state last: a directory with it is complete.
        let state = serde_json::to_vec_pretty(&self.state).expect("JSON serializes");
        std::fs::write(temporary.join("state.json"), state).map_err(|e| io(&temporary, e))?;
        if target.exists() {
            std::fs::remove_dir_all(&target).map_err(|e| io(&target, e))?;
        }
        std::fs::rename(&temporary, &target).map_err(|e| io(&target, e))?;
        Ok(target)
    }

    /// Reads the checkpoint in `path`.
    ///
    /// # Errors
    ///
    /// If a file is missing or unreadable.
    pub fn read(path: &Path) -> Result<Self, String> {
        let state: State = serde_json::from_slice(
            &std::fs::read(path.join("state.json")).map_err(|e| io(path, e))?,
        )
        .map_err(|e| io(path, e))?;
        let load = |file: &str| -> Result<BTreeMap<String, Tensor>, String> {
            let f = kapel_hub::safetensors::SafeTensors::open(path.join(file))
                .map_err(|e| e.to_string())?;
            f.names()
                .map(|n| {
                    let t = f.tensor(n).map_err(|e| e.to_string())?;
                    // A copy in memory: the files may be removed while the
                    // run goes on.
                    let t = Tensor::from_bytes(t.dtype(), t.shape(), &host_bytes(&t)?)
                        .map_err(|e| e.to_string())?;
                    Ok((n.to_string(), t))
                })
                .collect()
        };
        let model_config = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| io(path, e))?,
        )
        .map_err(|e| io(path, e))?;
        let run_config =
            std::fs::read_to_string(path.join("config.toml")).map_err(|e| io(path, e))?;
        Ok(Self {
            state,
            params: load("model.safetensors")?,
            optimizer: load("optimizer.safetensors")?,
            model_config,
            run_config,
        })
    }
}

/// The bytes of a host tensor, row-major: an owned buffer or a mapped file.
fn host_bytes(t: &Tensor) -> Result<Vec<u8>, String> {
    let t = t.contiguous().map_err(|e| e.to_string())?;
    let start = t.layout().offset() * t.dtype().block_bytes();
    let len = t
        .dtype()
        .storage_bytes(t.numel())
        .map_err(|e| e.to_string())?;
    t.storage()
        .host_bytes()
        .map(|b| b[start..start + len].to_vec())
        .ok_or_else(|| "a tensor not in host memory".to_string())
}

/// The complete checkpoints of `dir`, by step, oldest first.
pub fn list(dir: &Path) -> Vec<(usize, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(usize, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let step = name.strip_prefix("step-")?.parse().ok()?;
            e.path()
                .join("state.json")
                .exists()
                .then(|| (step, e.path()))
        })
        .collect();
    found.sort();
    found
}

/// The checkpoints written in the background, one at a time.
#[derive(Default)]
pub struct Writer {
    pending: Option<JoinHandle<Result<PathBuf, String>>>,
    keep: usize,
}

impl Writer {
    /// A writer that keeps the `keep` last checkpoints (all for 0).
    pub fn new(keep: usize) -> Self {
        Self {
            pending: None,
            keep,
        }
    }

    /// Writes `checkpoint` into `dir` on a thread, after the previous one is
    /// done.
    ///
    /// # Errors
    ///
    /// The error of the previous writing.
    pub fn write(&mut self, checkpoint: Checkpoint, dir: &Path) -> Result<(), String> {
        self.finish()?;
        let dir = dir.to_path_buf();
        let keep = self.keep;
        self.pending = Some(std::thread::spawn(move || {
            let path = checkpoint.write(&dir)?;
            if keep > 0 {
                let all = list(&dir);
                for (_, old) in all.iter().take(all.len().saturating_sub(keep)) {
                    std::fs::remove_dir_all(old).map_err(|e| io(old, e))?;
                }
            }
            Ok(path)
        }));
        Ok(())
    }

    /// Waits for the checkpoint being written.
    ///
    /// # Errors
    ///
    /// Its error.
    pub fn finish(&mut self) -> Result<(), String> {
        match self.pending.take() {
            Some(h) => h
                .join()
                .map_err(|_| "the thread writing a checkpoint panicked".to_string())?
                .map(drop),
            None => Ok(()),
        }
    }
}
