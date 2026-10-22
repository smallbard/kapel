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
- `kapel-ops`: `Contiguous`, a bit-exact row-major copy of a strided view, in any
  non-quantized type; the attention of `kapel-models` uses it, and runs on any backend.
- `kapel-cuda`: the whole catalogue in `bf16` and `f16`, and conversions between `f32`,
  `bf16` and `f16` on the device, rounding exactly as the host does.
- `kapel-models`: `Model::upload` and `Model::cast`, to run a model on a backend's
  device and in a given type; the small parity models run on the card.
- `kapel-cuda`: rotary angles computed on the device in `f64`, and normalizations with
  one warp per row: no copy back to the host per layer.
- `kapel-ops`: `Concat`, tensors joined along a dimension, bit for bit.
- `kapel-models`: a cache of keys and values (`Cache`) and `Model::step`, which
  computes only the new positions after it and the logits of the last.
- `kapel-infer`: the generation engine. `Pipeline` loads a model directory onto a
  backend; `generate` gives the tokens after a prompt one at a time, with their final
  text, and stops on the model's end tokens, stop sequences or a length. Greedy
  decoding and sampling as `transformers` does them (repetition penalty,
  temperature, top-k, top-p, min-p), with a seeded generator. Chat prompts through
  the model's template.
- `kapel-cli`: `kapel run MODEL`, a conversation or a single answer, on the
  processor or the card, with a model from a directory or the Hub.
- `tools/parity/generation_reference.py` and the example `generation_parity`: greedy
  generations against `transformers`.

### Changed

- `kapel-ops`: `Operation::infer_backward` checks the gradient's shape, type and
  device; every backward of every backend starts with it.
- `kapel-ops`: `Error::BroadcastIncompatible` names its operation, and
  `shape::broadcast` takes the operation's name.
- `kapel-ops`: quantized products need `K` to be a multiple of the block size;
  attention rejects `D = 0`; RoPE checks its frequency parameters.
- `kapel-ops`: attention rows whose visible keys are all masked with `−∞` give
  zeros and no gradient on every backend; finite masks such as `f32::MIN` take part
  in the softmax. Norm weights and attention masks may be in any float type.
- `kapel-cpu`: `Isa` can only be obtained for instructions the processor has, and
  `Matrix` checks its extent at construction (both could cause undefined behaviour
  from safe code).
- `kapel-cpu`: sums and means are blocked and pairwise (16M elements: 64 ms to
  0.8 ms, and far more accurate); attention is also split over query blocks.
- `kapel-cuda`: one backend per device, shared process-wide; shape tables are
  cached on the device; a failed allocation empties the pool and retries.
- `kapel-hub`: the download cache is keyed by commit; only top-level files are
  downloaded; saving writes a temporary file and renames it.
- `kapel-tokenizers`: regexes built once and added tokens found with
  Aho-Corasick (encoding 1.4 to 8 times faster); `strftime_now` uses local time.
- Parity tests say when their data is missing, and fail if
  `KAPEL_REQUIRE_PARITY=1`. The CUDA job rebuilds the PTX and compares it, and
  runs on pull requests from this repository.

### Fixed

- `kapel-cpu`: rotary embeddings accept negative `i32` positions, as the contract and
  the reference do.
- Empty dimensions no longer panic in the reference products.
- `kapel-cuda`: `Max` backward could write outside its buffer; kernels writing
  through raw pointers check their bounds; positions beyond 32 bits are refused.
- `kapel-cpu`: the `bf16` rounding of products kept NaN as infinity or zero.
- The chat template test no longer depends on today's date.

### Security

- Fuzz targets for every reader of untrusted files, and a weekly fuzzing and Miri job.
- Bounded work: safetensors header size, GGUF nesting, configuration sizes, chat
  template fuel; byte-pair encoding is now O(n log n).
- A configuration announcing a huge number of layers no longer allocates without
  limit (found by fuzzing).
- Regexes that give up now reject the text instead of silently cutting it wrongly.
- `SECURITY.md`: what Kapel guarantees, and how to report a vulnerability.
- GGUF: alignment and offsets are checked (a file could panic the reader); a
  tensor of an unknown type no longer rejects the whole file.
- Configurations: sizes are bounded and their products checked.
- Chat templates: output is capped at 1 MiB.
- Downloads: file names from the Hub and shard names from an index cannot leave
  their directory; connections time out.
- Saving a model over the files it was loaded from no longer truncates mapped
  memory.
