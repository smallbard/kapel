# 8. Model files: kapel-hub

`kapel-hub` turns files on disk, or on the Hugging Face Hub, into tensors. It reads safetensors and GGUF, opens sharded checkpoints, downloads repositories into a cache, and writes safetensors and GGUF back. Every reader treats its file as untrusted: sizes are checked before anything is allocated, and arithmetic on offsets is checked for overflow. It depends only on `kapel-core`.

## 8.1 Zero-copy reading

Model weights are large, so Kapel does not copy them. A file is memory-mapped, and every tensor of that file is a view of the mapping:

```rust
enum Bytes { Mapped(memmap2::Mmap), Owned(Vec<u8>) }   // Owned serves tests and fuzzing

struct FileStorage { file: Arc<Bytes>, start: usize, len: usize }   // implements Storage
```

`FileStorage` reports `Device::Cpu` and exposes its bytes through `host_bytes`, so the CPU backend computes directly on the file's pages, which the operating system reads when first touched. The soundness of the mapping rests on one assumption, written in its `SAFETY` comment and in `SECURITY.md`: model files are not modified while in use. Kapel itself never truncates a file it may have mapped; it writes a new file and renames it (8.5).

## 8.2 safetensors

A safetensors file is an 8-byte little-endian header length, a JSON header, and the data. The header maps each tensor name to its type, shape and byte range.

The parser checks, in order: that the header length is at most 100 MB (the reference implementation's limit) and fits in the file; that the header is a JSON object; that each entry has a string type, a shape and exactly two offsets; that `start ≤ end ≤ data length`; and that the element count times the element size, computed with checked multiplication, equals the byte range.

Types map to `DType` for F64, F32, F16, BF16, the two F8 formats, I8, U8, I32 and U32. Types Kapel does not compute with (BOOL, I16, U16, I64, U64) are still parsed and listed, so a file containing them opens; asking for one of those tensors returns an error.

## 8.3 GGUF

GGUF, the format of llama.cpp, stores metadata and tensors in one file: a magic number, a version (2 or 3), the counts, typed key-value pairs, tensor descriptions, and data aligned to `general.alignment`.

| Check | Rule |
| --- | --- |
| Lengths and counts | every u64 length or count must be at most the file length, so a hostile header cannot request a huge allocation |
| Metadata arrays | at most 8 levels of nesting |
| Tensor rank | at most 8 dimensions |
| Alignment | a power of two, at most 2¹⁶; the data start and every offset use checked arithmetic |
| Tensor ranges | a tensor of a known type must lie entirely inside the file |

Dimensions are stored fastest first and reversed into Kapel's shape order. The ggml types Kapel reads are F32, F16, BF16, F64, I8, I32, and the quantized Q4\_0, Q4\_1, Q5\_0, Q5\_1, Q8\_0 and Q2\_K to Q6\_K. Other types, such as the IQ formats, are listed but not read: "fails when asked for, not the whole file". Quantized tensors keep their blocks as stored, sharing the mapping. A parity test compares the dequantization of eleven tensors with llama.cpp's own `gguf-py`.

## 8.4 Checkpoints and the Hub

A checkpoint is a model directory: `config.json`, `tokenizer.json`, often `generation_config.json`, and either `model.safetensors` or shards listed in `model.safetensors.index.json`. `Checkpoint::open` opens every shard named in the index, after checking that each name is a plain file name (no separator, no `..`, no drive letter), so an index cannot make Kapel open a file outside the directory. Tensors are then found by name through the shards' own headers.

`download::snapshot(repo, revision, cache)` fetches a repository from the Hub:

1. Validate the repository name, then ask the Hub API which commit the revision points to.
2. Use `cache/<owner>--<name>/<commit>/` as the directory. A branch that moves gets a new directory, so files of two revisions never mix.
3. Download the top-level configuration and tokenizer files, then only the weights the checkpoint will read: the shards of the index, or `model.safetensors`. Duplicate formats such as `consolidated.safetensors` or `onnx/` are skipped.
4. Stream each file to a `.part` file, compute its SHA-256 and compare it with the Hub's checksum for large files, then rename it into place.

Once every file is there, `refs/<revision>` in the repository's directory records the commit, as the Hub's own cache does. When the Hub cannot be reached (a transport error: no network, a timeout), the commit recorded for the revision is used without asking, so a model downloaded once runs offline; `KAPEL_OFFLINE=1` or `HF_HUB_OFFLINE=1` skips the Hub altogether. A refusal of the Hub (404, 401) stays an error: the revision may have been deleted or its access withdrawn. A recorded commit must be hexadecimal, and a revision that cannot name a file (`refs/pr/1`) is not recorded.

The cache root is `$KAPEL_CACHE`, or `~/.cache/kapel/hub`. A gated model needs `HF_TOKEN`. Connections time out after 30 s, and a transfer that stalls for 60 s fails rather than hanging.

## 8.5 Writing safely

`safetensors::save` writes any host tensors, contiguous or not, padding the header so the data starts on an 8-byte boundary. Like every write in Kapel, it goes through `write_atomic`: the content is written to a temporary file in the same directory, flushed to disk, and renamed over the target. A process that has the old file mapped keeps reading the old content. This is what makes `Model::save` into the directory the model was loaded from safe; on Windows, where a mapped file cannot be replaced, it returns an error instead.

`gguf::write(path, metadata, tensors)` writes a GGUF file of version 3: the metadata in the order given, each value an `Entry` that carries its GGUF type (llama.cpp reads each key as one type: `block_count` as `u32`, `rope.freq_base` as `f32`), then the tensor descriptions and their data, each aligned on 32 bytes. Views are written in row-major order, quantized tensors as their blocks (`ggml_type` gives the ggml number of a Kapel type), and the data is streamed tensor by tensor through `write_atomic`. A test writes metadata of every kind, a view and a quantized tensor, and reads them back; `kapel quantize` uses it to write models llama.cpp reads (chapter 13).

Errors from this crate are `HubError`: `Io` and `Format` carry the path of the file, `Missing` the name of the tensor, and `Tensor` wraps a `kapel-core` error.
