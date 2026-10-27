//! The safetensors format: a little-endian `u64` header length, a JSON header,
//! then the tensors' bytes.
//!
//! The header maps each tensor name to its data type, shape and byte range
//! (relative to the end of the header), plus an optional `__metadata__` map of
//! strings. Tensors read from a file share its memory mapping: nothing is copied
//! until a backend needs it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kapel_core::{DType, Layout, Tensor};
use serde_json::{Map, Value};

use crate::bytes::{Bytes, FileStorage};
use crate::error::{HubError, Result};

/// The largest header accepted, as in the reference implementation: 100 MB.
const MAX_HEADER: u64 = 100_000_000;

/// A tensor's entry in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// The data type, as written in the file (`"F32"`, `"BF16"`, `"I64"`…).
    pub dtype_name: String,
    /// The data type, if Kapel has it.
    pub dtype: Option<DType>,
    /// The shape.
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A safetensors file, mapped in memory.
pub struct SafeTensors {
    path: PathBuf,
    file: Arc<Bytes>,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

/// The Kapel type of a safetensors type name.
fn dtype_of(name: &str) -> Option<DType> {
    Some(match name {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "I8" => DType::I8,
        "U8" => DType::U8,
        "I32" => DType::I32,
        "U32" => DType::U32,
        _ => return None,
    })
}

/// The safetensors type name of a Kapel type.
fn name_of(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::F64 => "F64",
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F8E4M3 => "F8_E4M3",
        DType::F8E5M2 => "F8_E5M2",
        DType::I8 => "I8",
        DType::U8 => "U8",
        DType::I32 => "I32",
        DType::U32 => "U32",
        _ => return None,
    })
}

/// The size in bytes of one element of a safetensors type, Kapel's or not.
fn element_size(name: &str) -> Option<usize> {
    match name {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

impl SafeTensors {
    /// Opens and maps a file, and reads its header.
    ///
    /// # Errors
    ///
    /// [`HubError::Io`] if the file cannot be read, [`HubError::Format`] if its
    /// header is not valid.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::parse(path.to_path_buf(), Bytes::map(path)?)
    }

    /// Reads a safetensors image held in memory.
    ///
    /// # Errors
    ///
    /// [`HubError::Format`] if the header is not valid.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(PathBuf::from("<memory>"), Bytes::Owned(bytes))
    }

    fn parse(path: PathBuf, file: Bytes) -> Result<Self> {
        let bytes = file.as_slice();
        let bad = |message: String| HubError::format(&path, message);
        let header_len = bytes
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(|| bad("shorter than the 8 bytes of the header length".into()))?;
        if header_len > MAX_HEADER {
            return Err(bad(format!(
                "a header of {header_len} bytes is larger than the {MAX_HEADER} allowed"
            )));
        }
        let data_start = usize::try_from(header_len)
            .ok()
            .and_then(|n| n.checked_add(8))
            .filter(|&n| n <= bytes.len())
            .ok_or_else(|| {
                bad(format!(
                    "a header of {header_len} bytes does not fit in the file"
                ))
            })?;
        let header: Map<String, Value> = serde_json::from_slice(&bytes[8..data_start])
            .map_err(|e| bad(format!("the header is not a JSON object: {e}")))?;
        let data_len = bytes.len() - data_start;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, entry) in header {
            if name == "__metadata__" {
                for (key, value) in entry.as_object().into_iter().flatten() {
                    if let Some(value) = value.as_str() {
                        metadata.insert(key.clone(), value.to_string());
                    }
                }
                continue;
            }
            let field = |key: &str| {
                entry
                    .get(key)
                    .ok_or_else(|| bad(format!("tensor {name:?} has no {key:?}")))
            };
            let dtype_name = field("dtype")?
                .as_str()
                .ok_or_else(|| bad(format!("tensor {name:?}: dtype is not a string")))?
                .to_string();
            let numbers = |key: &str| -> Result<Vec<usize>> {
                field(key)?
                    .as_array()
                    .ok_or_else(|| bad(format!("tensor {name:?}: {key} is not an array")))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| {
                                bad(format!("tensor {name:?}: {key} holds a non-integer"))
                            })
                    })
                    .collect()
            };
            let shape = numbers("shape")?;
            let offsets = numbers("data_offsets")?;
            let [start, end] = offsets[..] else {
                return Err(bad(format!(
                    "tensor {name:?}: data_offsets needs 2 numbers"
                )));
            };
            let size = element_size(&dtype_name)
                .ok_or_else(|| bad(format!("tensor {name:?}: unknown dtype {dtype_name}")))?;
            let count = shape.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n));
            if start > end
                || end > data_len
                || count.and_then(|c| c.checked_mul(size)) != Some(end - start)
            {
                return Err(bad(format!(
                    "tensor {name:?}: the byte range {start}..{end} does not match shape {shape:?} of {dtype_name}"
                )));
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype: dtype_of(&dtype_name),
                    dtype_name,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            path,
            file: Arc::new(file),
            data_start,
            tensors,
            metadata,
        })
    }

    /// The names of the tensors, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The entry of a tensor.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// The `__metadata__` map.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// A tensor, sharing the file's memory.
    ///
    /// # Errors
    ///
    /// [`HubError::Missing`] if there is no such tensor, [`HubError::Format`] if
    /// its type is not one Kapel has (such as `I64` or `BOOL`).
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| HubError::Missing {
            name: name.to_string(),
        })?;
        let dtype = info.dtype.ok_or_else(|| {
            HubError::format(
                &self.path,
                format!(
                    "tensor {name:?} has type {}, which Kapel does not read",
                    info.dtype_name
                ),
            )
        })?;
        let storage = FileStorage {
            file: Arc::clone(&self.file),
            start: self.data_start + info.start,
            len: info.end - info.start,
        };
        Ok(Tensor::from_storage(
            Arc::new(storage),
            Layout::contiguous(&info.shape)?,
            dtype,
        )?)
    }
}

/// Writes tensors to a safetensors file. Tensors may be views: each is written
/// in row-major order. The header is padded to a multiple of 8 bytes, so that
/// the data starts aligned. The file is written under a temporary name and
/// renamed into place (see [`crate::write_atomic`]): `path` may be the file the
/// tensors are mapped from.
///
/// # Errors
///
/// [`HubError::Io`] if the file cannot be written; [`HubError::Format`] for a
/// quantized tensor, which safetensors cannot hold; a tensor error if a tensor
/// is not in host memory.
pub fn save(
    path: impl AsRef<Path>,
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let path = path.as_ref();
    let bytes = image(path, tensors, metadata)?;
    // Through a temporary file renamed into place: the file replaced may be the
    // one the tensors are mapped from.
    crate::write_atomic(path, |file| file.write_all(&bytes))
}

/// [`save`]'s image in memory, which [`SafeTensors::from_bytes`] reads back.
///
/// # Errors
///
/// As [`save`], but for writing.
pub fn to_bytes(
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    image(Path::new("<memory>"), tensors, metadata)
}

fn image(
    path: &Path,
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let mut header = Map::new();
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".into(),
            Value::Object(
                metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect(),
            ),
        );
    }
    let mut data = Vec::new();
    for (name, tensor) in tensors {
        let dtype_name = name_of(tensor.dtype()).ok_or_else(|| {
            HubError::format(
                path,
                format!(
                    "tensor {name:?}: safetensors cannot hold {}",
                    tensor.dtype()
                ),
            )
        })?;
        let start = data.len();
        let contiguous = tensor.contiguous()?;
        let bytes = contiguous
            .storage()
            .host_bytes()
            .ok_or(kapel_core::Error::WrongDevice {
                op: "save",
                expected: kapel_core::Device::Cpu,
                got: tensor.device(),
            })?;
        let size = tensor.dtype().block_bytes();
        let from = contiguous.layout().offset() * size;
        data.extend_from_slice(&bytes[from..from + tensor.numel() * size]);
        header.insert(
            (*name).to_string(),
            serde_json::json!({ "dtype": dtype_name, "shape": tensor.shape(), "data_offsets": [start, data.len()] }),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).expect("a JSON object serializes");
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + json.len() + data.len());
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensors_survive_a_round_trip() {
        let dir = std::env::temp_dir().join(format!("kapel-safetensors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t = a.transpose(0, 1).unwrap();
        let ids = Tensor::from_slice(&[7u32, 8], &[2]).unwrap();
        let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        save(&path, &[("a", &a), ("a.t", &t), ("ids", &ids)], &metadata).unwrap();
        let file = SafeTensors::open(&path).unwrap();
        assert_eq!(file.names().collect::<Vec<_>>(), vec!["a", "a.t", "ids"]);
        assert_eq!(file.metadata(), &metadata);
        assert_eq!(
            file.tensor("a").unwrap().to_vec::<f32>().unwrap(),
            a.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            file.tensor("a.t").unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(
            file.tensor("ids").unwrap().to_vec::<u32>().unwrap(),
            vec![7, 8]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Saving over the file a tensor is mapped from leaves that tensor intact.
    #[test]
    fn a_file_can_be_saved_over_itself() {
        let dir =
            std::env::temp_dir().join(format!("kapel-safetensors-self-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let values: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let a = Tensor::from_slice(&values, &[64, 64]).unwrap();
        save(&path, &[("a", &a)], &BTreeMap::new()).unwrap();
        let file = SafeTensors::open(&path).unwrap();
        let mapped = file.tensor("a").unwrap();
        let small = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        save(&path, &[("a", &mapped), ("b", &small)], &BTreeMap::new()).unwrap();
        assert_eq!(mapped.to_vec::<f32>().unwrap(), values);
        let again = SafeTensors::open(&path).unwrap();
        assert_eq!(again.tensor("a").unwrap().to_vec::<f32>().unwrap(), values);
        assert_eq!(again.names().count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_bad_header_is_rejected_with_its_reason() {
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend(b"{}");
        let error = SafeTensors::from_bytes(bytes).err().unwrap().to_string();
        assert!(error.contains("does not fit"), "{error}");
        let header = br#"{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0u8; 8]);
        let error = SafeTensors::from_bytes(bytes).err().unwrap().to_string();
        assert!(error.contains("does not match shape"), "{error}");
    }

    #[test]
    fn unknown_types_are_listed_but_not_read() {
        let header = br#"{"pos":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0u8; 16]);
        let file = SafeTensors::from_bytes(bytes).unwrap();
        assert_eq!(file.info("pos").unwrap().dtype_name, "I64");
        assert!(file.tensor("pos").is_err());
    }
}
