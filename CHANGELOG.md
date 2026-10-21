# Changelog

All notable changes to Kapel are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

Until 1.0, the public API is unstable: any minor version may break it, and the
changelog says how.

## [Unreleased]

### Added

- Cargo workspace, crate layout, lints and formatting rules.
- Continuous integration: format, clippy, tests on Linux (x86 and ARM), macOS and
  Windows, minimum supported Rust version, license checks.
- `kapel-core`: data types, shapes and strided layouts, devices, typed errors, host
  storage with a buffer pool, and tensors that share storage across views.
- `kapel-ops`: the operation catalogue (element-wise, matrix products, reductions,
  softmax, normalizations, attention, RoPE and its scalings, row gather and scatter,
  casts, quantized products), with input checks shared by every backend.
- `kapel-ref`: the reference backend in `f64`, its gradients checked by finite
  differences, and the conformity suite for other backends.
- `kapel-cpu`: the CPU backend, ported from Smallem's `Fast` (blocked products with
  AVX-512 and AVX2 micro-kernels, deterministic thread pool, vectorized exponential);
  the whole catalogue in `f32` and `bf16`, conforming to the reference.
- `kapel-ops`: exact GELU (`erf`) and capped attention scores (Gemma 2).
- `kapel-hub`: safetensors (memory-mapped reading, writing, sharded checkpoints), GGUF
  (metadata and tensors, every common quantization), downloads from the Hugging Face
  Hub with checksums.
- `kapel-tokenizers`: `tokenizer.json` BPE tokenizers and chat templates, matching
  `tokenizers` and `transformers`.
- `kapel-models`: Llama, Mistral, Qwen 2 and 3, Gemma 2, Phi and GPT-2 described by
  configuration, with logits matching `transformers`; saving models it reads back.
- `kapel-core`: decoders of the GGUF Q5 and K formats.
- `kapel-cuda` (feature `cuda`): the CUDA backend, with kernels written in Rust and
  compiled to PTX by cuda-oxide; the whole catalogue in `f32`, conforming to the
  reference and giving the same bits on every run.
- `kapel-cuda`: matrix products as cutile-rs tile kernels, in `f32`, `bf16` and `f16`
  with an `f32` accumulator, reading strided and broadcast operands in place.
- `kapel-cuda`: attention as cutile-rs tile kernels in the manner of FlashAttention 2,
  forward and backward, with causal and sliding-window masks, additive masks, capped
  scores and grouped keys, deterministic.
- `kapel-cuda`: tile sizes and compiler hints tuned per kernel and head width on an
  RTX 3090: bf16 products reach 69 Top/s at `4096³`, attention 44 to 51 Top/s forward
  and 37 to 41 backward.

### Security

- Fuzz targets for every reader of untrusted files, and a weekly fuzzing and Miri job.
- Bounded work: safetensors header size, GGUF nesting, configuration sizes, chat
  template fuel; byte-pair encoding is now O(n log n).
- A configuration announcing a huge number of layers no longer allocates without
  limit (found by fuzzing).
- Regexes that give up now reject the text instead of silently cutting it wrongly.
- `SECURITY.md`: what Kapel guarantees, and how to report a vulnerability.
