//! Downloading models from the Hugging Face Hub.
//!
//! Files land in a local cache, `<cache>/<owner>--<name>/<commit>/<file>`
//! (`~/.cache/kapel/hub` by default, or `KAPEL_CACHE`), and are downloaded once.
//! The revision asked for (a branch, a tag or a commit) is resolved to its commit
//! by the Hub first, and every file is fetched at that commit: two revisions
//! never mix in one directory, and a branch that moves gets a new directory. A
//! file is written under a temporary name and renamed when complete; for files
//! stored with Git LFS (the weights), its SHA-256 is checked against the one the
//! Hub announces. A token in `HF_TOKEN` gives access to gated models.
//!
//! Only files at the top of the repository are fetched (the configurations, the
//! tokenizer, the chat template), with the weights [`Checkpoint`] reads:
//! the shards listed by `model.safetensors.index.json` if there is one, else
//! `model.safetensors`. Other copies of the weights (`consolidated.safetensors`,
//! subdirectories such as `onnx/`) are left on the Hub.
//!
//! [`Checkpoint`]: crate::Checkpoint

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::checkpoint::{is_plain_file_name, read_json};
use crate::error::{HubError, Result};

const ENDPOINT: &str = "https://huggingface.co";
/// The longest wait for a connection to the Hub.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// The longest wait for the next bytes of an answer: a stalled download fails
/// instead of hanging.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

const INDEX: &str = "model.safetensors.index.json";

/// The files of a model repository worth downloading to run it, weights aside:
/// [`weights`] chooses among those that [`is_weight`].
fn wanted(name: &str) -> bool {
    is_plain_file_name(name)
        && (name.ends_with(".json")
            || name == "chat_template.jinja"
            || name == "tokenizer.model"
            || is_weight(name))
}

fn is_weight(name: &str) -> bool {
    name.ends_with(".safetensors")
}

/// The weight files to download, among the top-level files `available`: the
/// shards of the index if there is one (`index`, its contents), else
/// `model.safetensors`.
///
/// # Errors
///
/// If the index lists a shard that is not a plain file name.
fn weights<'a>(available: &[&'a str], index: Option<&'a Value>) -> Result<Vec<&'a str>> {
    if let Some(index) = index {
        let mut shards: Vec<&str> = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| HubError::format(INDEX, "no weight_map object"))?
            .values()
            .filter_map(Value::as_str)
            .collect();
        shards.sort_unstable();
        shards.dedup();
        if let Some(bad) = shards.iter().find(|n| !is_plain_file_name(n)) {
            return Err(HubError::format(
                INDEX,
                format!("the shard {bad:?} is not a file of the model directory"),
            ));
        }
        return Ok(shards);
    }
    Ok(available
        .iter()
        .copied()
        .filter(|&n| n == "model.safetensors")
        .collect())
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
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
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
/// latest), and returns the directory that holds them, named after the commit
/// `revision` resolves to.
///
/// # Errors
///
/// If the Hub cannot be reached, the repository does not exist or is gated
/// without a token, its answer is malformed, or a file's checksum does not
/// match.
pub fn snapshot(repo: &str, revision: &str, cache: &Path) -> Result<PathBuf> {
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() > 2 || !parts.iter().all(|p| is_plain_file_name(p)) {
        return Err(HubError::format(
            repo,
            "a repository is named <owner>/<name> or <name>",
        ));
    }
    let agent = agent();
    let url = format!("{ENDPOINT}/api/models/{repo}/revision/{revision}?blobs=true");
    let body = request(&agent, &url)?
        .into_string()
        .map_err(|e| HubError::io(&url, e))?;
    let info: Value = serde_json::from_str(&body)
        .map_err(|e| HubError::format(&url, format!("not valid JSON: {e}")))?;
    let commit = info
        .get("sha")
        .and_then(Value::as_str)
        .filter(|sha| {
            !sha.is_empty() && sha.len() <= 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())
        })
        .ok_or_else(|| HubError::format(&url, "no commit sha in the answer"))?;
    let dir = cache.join(parts.join("--")).join(commit);
    std::fs::create_dir_all(&dir).map_err(|e| HubError::io(&dir, e))?;
    // The top-level files, with the SHA-256 of those stored with Git LFS.
    let files: BTreeMap<&str, Option<&str>> = info
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|file| {
            let name = file.get("rfilename").and_then(Value::as_str)?;
            let sha256 = file
                .get("lfs")
                .and_then(|l| l.get("sha256"))
                .and_then(Value::as_str);
            wanted(name).then_some((name, sha256))
        })
        .collect();
    for (&name, &sha256) in files.iter().filter(|(n, _)| !is_weight(n)) {
        fetch(&agent, repo, commit, name, sha256, &dir.join(name))?;
    }
    let index = if files.contains_key(INDEX) {
        Some(read_json(&dir.join(INDEX))?)
    } else {
        None
    };
    let available: Vec<&str> = files.keys().copied().collect();
    for name in weights(&available, index.as_ref())? {
        let sha256 = files.get(name).copied().flatten();
        fetch(&agent, repo, commit, name, sha256, &dir.join(name))?;
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
    let mut partial = target.as_os_str().to_owned();
    partial.push(".part");
    let partial = PathBuf::from(partial);
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
            "consolidated.safetensors",
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "tokenizer.model",
        ] {
            assert!(wanted(name), "{name}");
        }
        for name in [
            "pytorch_model.bin",
            "README.md",
            "onnx/model.onnx",
            "onnx/config.json",
            "onnx/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "../config.json",
            "/etc/config.json",
            "..\\config.json",
            "C:config.json",
            "",
        ] {
            assert!(!wanted(name), "{name}");
        }
    }

    #[test]
    fn one_copy_of_the_weights_is_chosen() {
        let single = [
            "consolidated.safetensors",
            "model.safetensors",
            "config.json",
        ];
        assert_eq!(weights(&single, None).unwrap(), ["model.safetensors"]);
        assert!(
            weights(&["consolidated.safetensors"], None)
                .unwrap()
                .is_empty()
        );
        let index = serde_json::json!({ "weight_map": {
            "a": "model-00002-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
            "c": "model-00001-of-00002.safetensors",
        }});
        let sharded = [
            "consolidated.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        assert_eq!(
            weights(&sharded, Some(&index)).unwrap(),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        for bad in ["../x.safetensors", "/x.safetensors", "a/x.safetensors"] {
            let index = serde_json::json!({ "weight_map": { "a": bad } });
            assert!(weights(&sharded, Some(&index)).is_err(), "{bad}");
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
