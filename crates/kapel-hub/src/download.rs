//! Downloading models from the Hugging Face Hub.
//!
//! Files land in a local cache, `<cache>/<owner>--<name>/<revision>/<file>`
//! (`~/.cache/kapel/hub` by default, or `KAPEL_CACHE`), and are downloaded once.
//! A file is written under a temporary name and renamed when complete; for files
//! stored with Git LFS (the weights), its SHA-256 is checked against the one the
//! Hub announces. A token in `HF_TOKEN` gives access to gated models.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{HubError, Result};

const ENDPOINT: &str = "https://huggingface.co";

/// The files of a model repository worth downloading to run it.
fn wanted(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name.ends_with(".json") && !name.contains('/')
        || name == "chat_template.jinja"
        || name == "tokenizer.model"
}

/// The default cache directory.
pub fn default_cache() -> PathBuf {
    if let Some(dir) = std::env::var_os("KAPEL_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".cache/kapel/hub")
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent(concat!("kapel/", env!("CARGO_PKG_VERSION")))
        .build()
}

fn request(agent: &ureq::Agent, url: &str) -> Result<ureq::Response> {
    let mut request = agent.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    request.call().map_err(|e| {
        let message = match &e {
            ureq::Error::Status(401 | 403, _) => {
                format!("{url}: access denied (a gated model needs HF_TOKEN)")
            }
            ureq::Error::Status(404, _) => format!("{url}: not found"),
            other => format!("{url}: {other}"),
        };
        HubError::io(url, std::io::Error::other(message))
    })
}

/// Downloads the files needed to run model `repo` at `revision` (`"main"` for the
/// latest), and returns the directory that holds them.
///
/// # Errors
///
/// If the Hub cannot be reached, the repository does not exist or is gated
/// without a token, or a file's checksum does not match.
pub fn snapshot(repo: &str, revision: &str, cache: &Path) -> Result<PathBuf> {
    let agent = agent();
    let url = format!("{ENDPOINT}/api/models/{repo}/revision/{revision}?blobs=true");
    let body = request(&agent, &url)?
        .into_string()
        .map_err(|e| HubError::io(&url, e))?;
    let info: Value = serde_json::from_str(&body)
        .map_err(|e| HubError::format(&url, format!("not valid JSON: {e}")))?;
    let dir = cache.join(repo.replace('/', "--")).join(revision);
    std::fs::create_dir_all(&dir).map_err(|e| HubError::io(&dir, e))?;
    for file in info
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(name) = file.get("rfilename").and_then(Value::as_str) else {
            continue;
        };
        if !wanted(name) {
            continue;
        }
        let sha256 = file
            .get("lfs")
            .and_then(|l| l.get("sha256"))
            .and_then(Value::as_str);
        fetch(&agent, repo, revision, name, sha256, &dir.join(name))?;
    }
    Ok(dir)
}

/// Downloads one file, unless it is already there.
fn fetch(
    agent: &ureq::Agent,
    repo: &str,
    revision: &str,
    name: &str,
    sha256: Option<&str>,
    target: &Path,
) -> Result<()> {
    if target.exists() {
        return Ok(());
    }
    let url = format!("{ENDPOINT}/{repo}/resolve/{revision}/{name}");
    let partial = target.with_extension("part");
    let mut reader = request(agent, &url)?.into_reader();
    let mut file = std::fs::File::create(&partial).map_err(|e| HubError::io(&partial, e))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = reader
            .read(&mut buffer)
            .map_err(|e| HubError::io(&url, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        file.write_all(&buffer[..n])
            .map_err(|e| HubError::io(&partial, e))?;
    }
    file.sync_all().map_err(|e| HubError::io(&partial, e))?;
    if let Some(expected) = sha256 {
        let got: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if got != expected {
            let _ = std::fs::remove_file(&partial);
            return Err(HubError::format(
                target,
                format!("checksum mismatch: {got}, expected {expected}"),
            ));
        }
    }
    std::fs::rename(&partial, target).map_err(|e| HubError::io(target, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_of_a_model_are_chosen() {
        for name in [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
        ] {
            assert!(wanted(name), "{name}");
        }
        for name in [
            "pytorch_model.bin",
            "README.md",
            "onnx/model.onnx",
            "onnx/config.json",
        ] {
            assert!(!wanted(name), "{name}");
        }
    }

    /// Downloads a tiny model from the Hub: `cargo test -p kapel-hub -- --ignored`.
    #[test]
    #[ignore = "needs the network"]
    fn a_tiny_model_downloads_and_loads() {
        let cache = std::env::temp_dir().join(format!("kapel-download-{}", std::process::id()));
        let dir = snapshot(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "main",
            &cache,
        )
        .unwrap();
        let checkpoint = crate::Checkpoint::open(&dir).unwrap();
        assert!(checkpoint.names().count() > 0);
        assert!(dir.join("tokenizer.json").exists());
        std::fs::remove_dir_all(&cache).unwrap();
    }
}
