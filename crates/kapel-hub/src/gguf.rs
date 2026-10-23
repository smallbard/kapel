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

/// The alignment of the tensor data when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: usize = 32;
/// The largest `general.alignment` accepted: 64 KiB, far above what llama.cpp
/// writes (32), and small enough that the padding cannot overflow.
pub const MAX_ALIGNMENT: u64 = 1 << 16;

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
            None => DEFAULT_ALIGNMENT,
            Some(Value::Uint(a)) if a.is_power_of_two() && *a <= MAX_ALIGNMENT => {
                usize::try_from(*a).expect("at most 2^16")
            }
            Some(other) => {
                return Err(HubError::format(
                    path,
                    format!(
                        "general.alignment is {other:?}: a power of two of at most \
                         {MAX_ALIGNMENT} is expected"
                    ),
                ));
            }
        };
        let data_start =
            r.at.checked_next_multiple_of(alignment)
                .ok_or_else(|| HubError::format(path, "the tensor data starts beyond the file"))?;
        let gguf = Self {
            path: path.to_path_buf(),
            file: Arc::new(file),
            data_start,
            metadata,
            tensors,
        };
        // Every tensor Kapel reads must lie in the file. A tensor of a type Kapel
        // does not read (its size is unknown) only has its start checked: it
        // fails when asked for, not the whole file.
        for (name, info) in &gguf.tensors {
            if info.dtype.is_some() {
                gguf.range(name)?;
            } else {
                gguf.start(name, info)?;
            }
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

    /// Where the data of a tensor starts, checked to lie in the file.
    fn start(&self, name: &str, info: &GgufTensorInfo) -> Result<usize> {
        self.data_start
            .checked_add(info.offset)
            .filter(|&start| start <= self.file.as_slice().len())
            .ok_or_else(|| {
                HubError::format(
                    &self.path,
                    format!("tensor {name:?} starts beyond the file"),
                )
            })
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
        let start = self.start(name, info)?;
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

/// A metadata value to write, with its GGUF type: llama.cpp reads each key
/// as one type (`block_count` as `u32`, `rope.freq_base` as `f32`, …).
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    /// `u8`.
    U8(u8),
    /// `i8`.
    I8(i8),
    /// `u16`.
    U16(u16),
    /// `i16`.
    I16(i16),
    /// `u32`.
    U32(u32),
    /// `i32`.
    I32(i32),
    /// `f32`.
    F32(f32),
    /// A boolean.
    Bool(bool),
    /// A UTF-8 string.
    String(String),
    /// An array of values of one type, that of its first. An empty array is
    /// one of strings.
    Array(Vec<Entry>),
    /// `u64`.
    U64(u64),
    /// `i64`.
    I64(i64),
    /// `f64`.
    F64(f64),
}

impl Entry {
    /// An array of strings.
    pub fn strings<S: Into<String>>(values: impl IntoIterator<Item = S>) -> Self {
        Self::Array(values.into_iter().map(|s| Self::String(s.into())).collect())
    }

    /// The GGUF type number.
    fn kind(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    fn write(&self, out: &mut Vec<u8>) -> std::result::Result<(), String> {
        match self {
            Self::U8(v) => out.push(*v),
            Self::I8(v) => out.extend(v.to_le_bytes()),
            Self::U16(v) => out.extend(v.to_le_bytes()),
            Self::I16(v) => out.extend(v.to_le_bytes()),
            Self::U32(v) => out.extend(v.to_le_bytes()),
            Self::I32(v) => out.extend(v.to_le_bytes()),
            Self::F32(v) => out.extend(v.to_le_bytes()),
            Self::Bool(v) => out.push(u8::from(*v)),
            Self::String(s) => {
                out.extend((s.len() as u64).to_le_bytes());
                out.extend(s.as_bytes());
            }
            Self::Array(values) => {
                let kind = values.first().map_or(8, Self::kind);
                if values.iter().any(|v| v.kind() != kind) {
                    return Err("an array mixes types".into());
                }
                out.extend(kind.to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for v in values {
                    v.write(out)?;
                }
            }
            Self::U64(v) => out.extend(v.to_le_bytes()),
            Self::I64(v) => out.extend(v.to_le_bytes()),
            Self::F64(v) => out.extend(v.to_le_bytes()),
        }
        Ok(())
    }
}

/// The ggml type number of a Kapel type, if GGUF holds it.
pub fn ggml_type(dtype: DType) -> Option<u32> {
    (0..=30).find(|&t| dtype_of(t) == Some(dtype))
}

/// The bytes of a tensor in host memory, in row-major order: borrowed when
/// it is contiguous, copied otherwise. A quantized tensor must be
/// contiguous, from a whole block.
pub(crate) fn host_bytes(tensor: &Tensor) -> Result<std::borrow::Cow<'_, [u8]>> {
    let dtype = tensor.dtype();
    if !tensor.is_contiguous() {
        if dtype.is_quantized() {
            return Err(HubError::Tensor(kapel_core::Error::BlockMisaligned {
                dtype,
                elements: tensor.numel(),
            }));
        }
        let copy = tensor.contiguous()?;
        return Ok(std::borrow::Cow::Owned(host_bytes(&copy)?.into_owned()));
    }
    let bytes = tensor
        .storage()
        .host_bytes()
        .ok_or(kapel_core::Error::WrongDevice {
            op: "host_bytes",
            expected: kapel_core::Device::Cpu,
            got: tensor.device(),
        })?;
    let from = dtype.storage_bytes(tensor.layout().offset())?;
    let len = dtype.storage_bytes(tensor.numel())?;
    Ok(std::borrow::Cow::Borrowed(&bytes[from..from + len]))
}

/// Writes a GGUF file (version 3): `metadata` in its order, then the
/// descriptions of `tensors` and their data, each aligned on 32 bytes. The
/// tensors must be in host memory; views are written in row-major order,
/// quantized tensors as their blocks. The file is written under a temporary
/// name and renamed into place ([`crate::write_atomic`]), the data streamed
/// tensor by tensor.
///
/// # Errors
///
/// [`HubError::Format`] for a type GGUF does not hold, an array that mixes
/// types or a key given twice; [`HubError::Io`] if the file cannot be
/// written; a tensor error for a tensor outside host memory.
pub fn write(
    path: impl AsRef<Path>,
    metadata: &[(&str, Entry)],
    tensors: &[(&str, &Tensor)],
) -> Result<()> {
    let path = path.as_ref();
    let string = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend((metadata.len() as u64).to_le_bytes());
    let mut keys = std::collections::BTreeSet::new();
    for (key, value) in metadata {
        if !keys.insert(*key) {
            return Err(HubError::format(path, format!("key {key:?} given twice")));
        }
        if *key == "general.alignment" && *value != Entry::U32(DEFAULT_ALIGNMENT as u32) {
            return Err(HubError::format(
                path,
                "only the default alignment (32) is written",
            ));
        }
        string(&mut header, key);
        header.extend(value.kind().to_le_bytes());
        value
            .write(&mut header)
            .map_err(|e| HubError::format(path, format!("{key}: {e}")))?;
    }
    let mut offset = 0usize;
    let mut sizes = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let kind = ggml_type(tensor.dtype()).ok_or_else(|| {
            HubError::format(
                path,
                format!("tensor {name:?}: GGUF cannot hold {}", tensor.dtype()),
            )
        })?;
        string(&mut header, name);
        header.extend((tensor.rank() as u32).to_le_bytes());
        for &d in tensor.shape().iter().rev() {
            header.extend((d as u64).to_le_bytes());
        }
        header.extend(kind.to_le_bytes());
        header.extend((offset as u64).to_le_bytes());
        let size = tensor.dtype().storage_bytes(tensor.numel())?;
        sizes.push(size);
        offset = (offset + size).next_multiple_of(DEFAULT_ALIGNMENT);
    }
    header.resize(header.len().next_multiple_of(DEFAULT_ALIGNMENT), 0);
    // The bytes of every tensor, checked before the file is opened.
    for (_, tensor) in tensors {
        host_bytes(tensor)?;
    }
    let mut failure = None;
    crate::write_atomic(path, |file| {
        file.write_all(&header)?;
        let padding = [0u8; DEFAULT_ALIGNMENT];
        for ((_, tensor), &size) in tensors.iter().zip(&sizes) {
            let bytes = match host_bytes(tensor) {
                Ok(bytes) => bytes,
                Err(e) => {
                    failure = Some(e);
                    return Err(std::io::Error::other("a tensor cannot be read"));
                }
            };
            file.write_all(&bytes)?;
            file.write_all(&padding[..size.next_multiple_of(DEFAULT_ALIGNMENT) - size])?;
        }
        Ok(())
    })
    .map_err(|e| failure.take().unwrap_or(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafeTensors;

    /// A tensor of a synthetic file: name, shape (fastest dimension first, as
    /// GGUF writes it), ggml type, offset.
    type Entry<'a> = (&'a str, &'a [u64], u32, u64);

    /// A GGUF file with `metadata` (`u32` values) and `tensors`, followed by
    /// `data` at the 32-byte boundary.
    fn gguf(metadata: &[(&str, u32, &[u8])], tensors: &[Entry<'_>], data: &[u8]) -> Vec<u8> {
        let string = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((metadata.len() as u64).to_le_bytes());
        for (key, kind, value) in metadata {
            string(&mut out, key);
            out.extend(kind.to_le_bytes());
            out.extend(*value);
        }
        for (name, dims, ggml_type, offset) in tensors {
            string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for d in *dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(ggml_type.to_le_bytes());
            out.extend(offset.to_le_bytes());
        }
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        out.extend(data);
        out
    }

    /// Two `f32` tensors, `[2, 3]` and `[2]`.
    fn valid() -> Vec<u8> {
        let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        gguf(
            &[("general.alignment", 4, &32u32.to_le_bytes())],
            &[("a", &[3, 2], 0, 0), ("b", &[2], 0, 24)],
            &data,
        )
    }

    fn error_of(bytes: Vec<u8>) -> String {
        match Gguf::from_bytes(bytes) {
            Ok(_) => panic!("the file was accepted"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn a_minimal_file_is_read() {
        let file = Gguf::from_bytes(valid()).unwrap();
        assert_eq!(file.names().collect::<Vec<_>>(), ["a", "b"]);
        let a = file.tensor("a").unwrap();
        assert_eq!(a.shape(), [2, 3]);
        assert_eq!(a.to_vec::<f32>().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            file.tensor("b").unwrap().to_vec::<f32>().unwrap(),
            [6.0, 7.0]
        );
    }

    #[test]
    fn a_bad_alignment_is_an_error() {
        for alignment in [u64::MAX, 0, 3, 48, MAX_ALIGNMENT * 2] {
            let bytes = gguf(
                &[("general.alignment", 10, &alignment.to_le_bytes())],
                &[("a", &[1], 0, 0)],
                &[0; 4],
            );
            let error = error_of(bytes);
            assert!(error.contains("general.alignment"), "{alignment}: {error}");
        }
        let error = error_of(gguf(
            &[("general.alignment", 8, b"\x01\0\0\0\0\0\0\0x")],
            &[],
            &[],
        ));
        assert!(error.contains("general.alignment"), "{error}");
        // The largest alignment accepted, with a header much shorter.
        let bytes = gguf(
            &[("general.alignment", 10, &MAX_ALIGNMENT.to_le_bytes())],
            &[("a", &[1], 0, 0)],
            &[],
        );
        let mut padded = bytes.clone();
        padded.resize(1 << 16, 0);
        padded.extend(1.5f32.to_le_bytes());
        let file = Gguf::from_bytes(padded).unwrap();
        assert_eq!(file.tensor("a").unwrap().to_vec::<f32>().unwrap(), [1.5]);
        assert!(error_of(bytes).contains("beyond the file"));
    }

    #[test]
    fn offsets_beyond_the_file_are_errors() {
        let data = [0u8; 8];
        for offset in [8, 9, 1 << 40, u64::MAX, u64::MAX - 31] {
            let error = error_of(gguf(&[], &[("a", &[2], 0, offset)], &data));
            assert!(
                error.contains("beyond the file") || error.contains("does not fit"),
                "{offset}: {error}"
            );
        }
        // Sizes whose product overflows.
        let error = error_of(gguf(&[], &[("a", &[u64::from(u32::MAX); 3], 0, 0)], &data));
        assert!(error.contains("does not fit"), "{error}");
    }

    #[test]
    fn every_truncation_is_an_error() {
        let bytes = valid();
        for len in 0..bytes.len() {
            let error = error_of(bytes[..len].to_vec());
            assert!(
                ["truncated", "does not fit", "beyond the file"]
                    .iter()
                    .any(|m| error.contains(m)),
                "{len}: {error}"
            );
        }
    }

    #[test]
    fn a_tensor_of_an_unknown_type_fails_alone() {
        // ggml type 16 (IQ2_XXS) is not read by Kapel; its size is not checked.
        let data = [0u8; 8];
        let file = Gguf::from_bytes(gguf(
            &[],
            &[("a", &[2], 0, 0), ("iq", &[32, 4], 16, 8)],
            &data,
        ))
        .unwrap();
        assert_eq!(file.info("iq").unwrap().dtype, None);
        assert_eq!(file.tensor("a").unwrap().numel(), 2);
        let error = file.tensor("iq").err().unwrap().to_string();
        assert!(error.contains("does not read"), "{error}");
        // Its start must still lie in the file.
        let error = error_of(gguf(&[], &[("iq", &[32], 16, 9)], &data));
        assert!(error.contains("starts beyond the file"), "{error}");
    }

    #[test]
    fn a_written_file_reads_back() {
        use crate::gguf::Entry as W;
        let dir = std::env::temp_dir().join(format!("kapel-gguf-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t = a.transpose(0, 1).unwrap();
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) / 7.0).collect();
        let q4 = DType::Quantized(QuantFormat::Q4_0);
        let codes = kapel_core::quantize(q4, &values).unwrap();
        let q = Tensor::from_bytes(q4, &[2, 32], &codes).unwrap();
        let metadata = [
            ("general.architecture", W::String("llama".into())),
            ("llama.block_count", W::U32(2)),
            ("llama.rope.freq_base", W::F32(500_000.0)),
            ("x.i8", W::I8(-3)),
            ("x.u64", W::U64(1 << 40)),
            ("x.bool", W::Bool(true)),
            ("x.f64", W::F64(0.25)),
            ("x.list", W::Array(vec![W::I32(1), W::I32(-2)])),
            ("x.words", W::strings(["a", "b c"])),
        ];
        write(&path, &metadata, &[("a", &a), ("t", &t), ("q", &q)]).unwrap();
        let file = Gguf::open(&path).unwrap();
        assert_eq!(file.metadata["llama.block_count"], Value::Uint(2));
        assert_eq!(
            file.metadata["llama.rope.freq_base"],
            Value::Float(500_000.0)
        );
        assert_eq!(file.metadata["x.i8"], Value::Int(-3));
        assert_eq!(file.metadata["x.u64"], Value::Uint(1 << 40));
        assert_eq!(file.metadata["x.bool"], Value::Bool(true));
        assert_eq!(file.metadata["x.f64"], Value::Float(0.25));
        assert_eq!(
            file.metadata["x.list"],
            Value::Array(vec![Value::Int(1), Value::Int(-2)])
        );
        assert_eq!(
            file.metadata["x.words"],
            Value::Array(vec![Value::String("a".into()), Value::String("b c".into())])
        );
        assert_eq!(
            file.tensor("a").unwrap().to_vec::<f32>().unwrap(),
            a.to_vec::<f32>().unwrap()
        );
        let t_read = file.tensor("t").unwrap();
        assert_eq!(t_read.shape(), [3, 2]);
        assert_eq!(
            t_read.to_vec::<f32>().unwrap(),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let q_read = file.tensor("q").unwrap();
        assert_eq!(q_read.dtype(), q4);
        assert_eq!(q_read.to_f64_vec().unwrap(), q.to_f64_vec().unwrap());
        // A key given twice, or an array of two types, is refused.
        let twice = [("k", W::U8(1)), ("k", W::U8(2))];
        assert!(write(&path, &twice, &[]).is_err());
        let mixed = [("k", W::Array(vec![W::U8(1), W::I8(2)]))];
        assert!(write(&path, &mixed, &[]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Kapel's quantization gives the bytes of llama.cpp's (`gguf-py`'s) for
    /// the GGUF formats it writes.
    #[test]
    fn quantization_gives_the_bytes_of_llama_cpp() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gguf");
        let Ok(gguf) = Gguf::open(dir.join("quants.gguf")) else {
            crate::parity_skipped("data/parity/gguf/quants.gguf");
            return;
        };
        let Ok(source) = gguf.tensor("source") else {
            crate::parity_skipped("data/parity/gguf/quants.gguf of the current gguf_reference.py");
            return;
        };
        let source = source.to_vec::<f32>().unwrap();
        for name in ["q8_0", "q4_0", "q4_1"] {
            let want = gguf.tensor(&format!("quantized_{name}")).unwrap();
            let bytes = host_bytes(&want).unwrap();
            let got = kapel_core::quantize(want.dtype(), &source).unwrap();
            assert_eq!(got.len(), bytes.len(), "{name}");
            let differ = got.iter().zip(bytes.iter()).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{name}: {differ} bytes differ");
        }
    }

    /// The file written by `tools/parity/gguf_reference.py`; without it (in
    /// continuous integration), the test is skipped, or fails with
    /// `KAPEL_REQUIRE_PARITY=1`.
    #[test]
    fn every_format_dequantizes_as_llama_cpp_does() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/parity/gguf");
        let Ok(gguf) = Gguf::open(dir.join("quants.gguf")) else {
            crate::parity_skipped("data/parity/gguf/quants.gguf");
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
