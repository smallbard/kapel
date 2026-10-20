//! GGUF, the format of llama.cpp: metadata and tensors in one file.
//!
//! The file holds a magic `GGUF`, a version (2 or 3), the number of tensors and
//! of metadata entries, the entries (typed key-value pairs), the tensor
//! descriptions (name, dimensions, type, offset), then the tensor data, aligned
//! on `general.alignment` bytes (32 by default). Dimensions are listed from the
//! fastest-varying; Kapel's shapes list them from the slowest, so they are
//! reversed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kapel_core::{DType, Layout, QuantFormat, Tensor};

use crate::bytes::{Bytes, FileStorage};
use crate::error::{HubError, Result};

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Any unsigned integer type.
    Uint(u64),
    /// Any signed integer type.
    Int(i64),
    /// `f32` or `f64`.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A UTF-8 string.
    String(String),
    /// An array of values.
    Array(Vec<Value>),
}

/// A tensor's description.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufTensorInfo {
    /// The shape, slowest dimension first.
    pub shape: Vec<usize>,
    /// The ggml type number.
    pub ggml_type: u32,
    /// The Kapel type, if Kapel reads it.
    pub dtype: Option<DType>,
    offset: usize,
}

/// A GGUF file, mapped in memory.
pub struct Gguf {
    path: PathBuf,
    file: Arc<Bytes>,
    data_start: usize,
    /// The metadata, by key.
    pub metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, GgufTensorInfo>,
}

/// The Kapel type of a ggml type number.
fn dtype_of(ggml: u32) -> Option<DType> {
    Some(match ggml {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Quantized(QuantFormat::Q4_0),
        3 => DType::Quantized(QuantFormat::Q4_1),
        6 => DType::Quantized(QuantFormat::Q5_0),
        7 => DType::Quantized(QuantFormat::Q5_1),
        8 => DType::Quantized(QuantFormat::Q8_0),
        10 => DType::Quantized(QuantFormat::Q2K),
        11 => DType::Quantized(QuantFormat::Q3K),
        12 => DType::Quantized(QuantFormat::Q4K),
        13 => DType::Quantized(QuantFormat::Q5K),
        14 => DType::Quantized(QuantFormat::Q6K),
        24 => DType::I8,
        26 => DType::I32,
        28 => DType::F64,
        30 => DType::BF16,
        _ => return None,
    })
}

/// A cursor over the header.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    path: &'a Path,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| {
                HubError::format(self.path, format!("truncated header at byte {}", self.at))
            })?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn size(&mut self) -> Result<usize> {
        let n = self.u64()?;
        usize::try_from(n)
            .ok()
            .filter(|&n| n <= self.bytes.len())
            .ok_or_else(|| {
                HubError::format(
                    self.path,
                    format!("a length of {n} does not fit in the file"),
                )
            })
    }

    fn string(&mut self) -> Result<String> {
        let n = self.size()?;
        let path = self.path;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| HubError::format(path, "a string is not UTF-8"))
    }

    fn value(&mut self, kind: u32, depth: usize) -> Result<Value> {
        let path = self.path;
        Ok(match kind {
            0 => Value::Uint(u64::from(self.take(1)?[0])),
            1 => Value::Int(i64::from(self.take(1)?[0] as i8)),
            2 => Value::Uint(u64::from(u16::from_le_bytes(
                self.take(2)?.try_into().expect("2"),
            ))),
            3 => Value::Int(i64::from(i16::from_le_bytes(
                self.take(2)?.try_into().expect("2"),
            ))),
            4 => Value::Uint(u64::from(self.u32()?)),
            5 => Value::Int(i64::from(self.u32()? as i32)),
            6 => Value::Float(f64::from(f32::from_bits(self.u32()?))),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth > 8 {
                    return Err(HubError::format(path, "arrays nested too deep"));
                }
                let inner = self.u32()?;
                let count = self.size()?;
                (0..count)
                    .map(|_| self.value(inner, depth + 1))
                    .collect::<Result<_>>()
                    .map(Value::Array)?
            }
            10 => Value::Uint(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            other => {
                return Err(HubError::format(
                    path,
                    format!("unknown metadata type {other}"),
                ));
            }
        })
    }
}

impl Gguf {
    /// Opens and maps a GGUF file, and reads its header.
    ///
    /// # Errors
    ///
    /// [`HubError::Io`] if it cannot be read, [`HubError::Format`] if it is not
    /// a valid GGUF file of version 2 or 3.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::parse(path, Bytes::map(path)?)
    }

    /// Reads a GGUF image held in memory.
    ///
    /// # Errors
    ///
    /// [`HubError::Format`] if it is not a valid GGUF file of version 2 or 3.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Path::new("<memory>"), Bytes::Owned(bytes))
    }

    fn parse(path: &Path, file: Bytes) -> Result<Self> {
        let bytes = file.as_slice();
        let mut r = Reader { bytes, at: 0, path };
        if r.take(4)? != b"GGUF" {
            return Err(HubError::format(path, "not a GGUF file (bad magic)"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(HubError::format(
                path,
                format!("GGUF version {version} is not supported"),
            ));
        }
        let (tensor_count, kv_count) = (r.size()?, r.size()?);
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let kind = r.u32()?;
            metadata.insert(key, r.value(kind, 0)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = r.string()?;
            let dims = r.u32()? as usize;
            if dims > 8 {
                return Err(HubError::format(
                    path,
                    format!("tensor {name:?} has {dims} dimensions"),
                ));
            }
            let mut shape = (0..dims).map(|_| r.size()).collect::<Result<Vec<_>>>()?;
            shape.reverse();
            let ggml_type = r.u32()?;
            let offset = r.size()?;
            tensors.insert(
                name,
                GgufTensorInfo {
                    shape,
                    ggml_type,
                    dtype: dtype_of(ggml_type),
                    offset,
                },
            );
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Uint(a)) if *a > 0 => *a as usize,
            _ => 32,
        };
        let data_start = r.at.div_ceil(alignment) * alignment;
        let gguf = Self {
            path: path.to_path_buf(),
            file: Arc::new(file),
            data_start,
            metadata,
            tensors,
        };
        for name in gguf.tensors.keys() {
            gguf.range(name)?;
        }
        Ok(gguf)
    }

    /// The names of the tensors, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The description of a tensor.
    pub fn info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// The type and byte range of a tensor, checked against the file.
    fn range(&self, name: &str) -> Result<(DType, usize, usize)> {
        let info = self.tensors.get(name).ok_or_else(|| HubError::Missing {
            name: name.to_string(),
        })?;
        let dtype = info.dtype.ok_or_else(|| {
            HubError::format(
                &self.path,
                format!(
                    "tensor {name:?} has ggml type {}, which Kapel does not read",
                    info.ggml_type
                ),
            )
        })?;
        let count = info.shape.iter().try_fold(1usize, |a, &n| a.checked_mul(n));
        let len = count
            .ok_or_else(|| HubError::format(&self.path, format!("tensor {name:?} is too large")))?;
        let bytes = dtype.storage_bytes(len)?;
        let start = self.data_start + info.offset;
        if start
            .checked_add(bytes)
            .is_none_or(|end| end > self.file.as_slice().len())
        {
            return Err(HubError::format(
                &self.path,
                format!("tensor {name:?} reaches beyond the file"),
            ));
        }
        Ok((dtype, start, bytes))
    }

    /// A tensor, sharing the file's memory. Quantized tensors keep their blocks.
    ///
    /// # Errors
    ///
    /// [`HubError::Missing`], or [`HubError::Format`] for a type Kapel does not
    /// read (such as the IQ formats).
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let (dtype, start, len) = self.range(name)?;
        let shape = &self.tensors[name].shape;
        let storage = FileStorage {
            file: Arc::clone(&self.file),
            start,
            len,
        };
        Ok(Tensor::from_storage(
            Arc::new(storage),
            Layout::contiguous(shape)?,
            dtype,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafeTensors;

    /// The file written by `tools/parity/gguf_reference.py`; without it (in
    /// continuous integration), the test checks nothing.
    #[test]
    fn every_format_dequantizes_as_llama_cpp_does() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gguf");
        let Ok(gguf) = Gguf::open(dir.join("quants.gguf")) else {
            return;
        };
        let expected = SafeTensors::open(dir.join("expected.safetensors")).unwrap();
        assert_eq!(gguf.metadata.get("test.answer"), Some(&Value::Uint(42)));
        assert_eq!(
            gguf.metadata.get("test.list"),
            Some(&Value::Array(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3)
            ]))
        );
        let mut checked = 0;
        for name in expected.names() {
            let got = gguf.tensor(name).unwrap();
            let want = expected.tensor(name).unwrap();
            assert_eq!(got.shape(), want.shape(), "{name}");
            let (got, want) = (got.to_f64_vec().unwrap(), want.to_vec::<f32>().unwrap());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                let b = f64::from(*b);
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1e-3),
                    "{name}[{i}]: {a} instead of {b}"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 11);
    }
}
