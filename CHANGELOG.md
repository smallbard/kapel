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
- `kapel-cpu`: products by quantized weights that read the codes in place, for Kapel's
  int8 and int4 and GGUF's Q4_0 to Q8_0 and Q2_K to Q6_K: AVX-512 kernels decode each
  block in registers, with scales and minimums taken out of the sums, and read the
  weights at 52 to 54 GB/s in generation on a Ryzen 7 7700. For many rows (the
  prefill), tiles of the weights are dequantized, then multiplied by blocks.
- `kapel-cpu`: activations in bytes (`Cpu::with_byte_activations`): the rows of `x`
  quantized by blocks of 32, the products summed in integers with AVX-512 VNNI.
- `kapel-ops`: `QuantizedMatmul` takes several weights, their rows stacked;
  `GatherRows` takes quantized tables and gives their rows in `f32`.
- `kapel-core`: `quantize`, the reference quantization to Kapel's int8 and int4 (with
  the scale of least squared error) and GGUF's Q8_0 and Q4_0.
- `kapel-models`: `Model::from_gguf` (Llama and Mistral, Qwen 2 and 3), the RoPE
  scaling of Llama 3 recovered from its factors; `Model::quantize`, with the output
  table optionally; `Model::extend`, the logits of every new position after a cache.
- `kapel-infer`: GGUF files, with their tokenizer, chat template and end tokens
  (`Pipeline::load_gguf`); `LoadOptions` and `Pipeline::load_with`, to quantize at
  loading.
- `kapel-cli`: `kapel run` reads GGUF files, and takes `--quantize`,
  `--quantize-output` and `--byte-activations`; the examples `profile` (where the time
  of a token goes) and `perplexity` (as `llama-perplexity`).
- `tools/parity/gguf_models_reference.py` and `kapel-infer/tests/gguf.rs`: models read
  from GGUF against the models they come from.
- `kapel-ops`: `PagedAttention`, attention of new rows over a cache of keys and values
  kept by pages, with a page table per sequence, sliding windows and capped scores;
  the traits `Write` (zeros, rows written in place, copies from the host) and `Record`
  (a pass recorded once and replayed), both in `Backend`; `Glu`, a gated activation
  of the two halves of its input; `ArgMax`, the index of the largest value, NaN
  ignored. On every backend, in the conformity suite.
- `kapel-cuda`: CUDA graphs. A pass is captured once, its buffers kept with the graph,
  and replayed (`Record`); its rotary angles and small index tensors are kept too.
- `kapel-cuda`: paged attention in `f32`, `bf16`, `f16` and `f8` caches, split by
  chunks of keys and combined; long prompts go through the tiled attention.
- `kapel-cuda`: products by quantized weights on the card: one warp per weight row,
  reading the codes in place, with kernels of their own for int8, int4, Q8_0, Q4_K and
  Q6_K; with many rows, pieces dequantized and multiplied by tiles. Quantized
  embedding tables give their rows on the card.
- `kapel-cuda`: element-wise operations, normalizations, RoPE, `Glu` and row writes in
  `bf16` and `f16` without a pass through `f32`; conversions to and from `f8`; product
  tiles chosen by the number of rows, and a kernel that reads the weights `[N, K]`
  as they are stored.
- `kapel-models`: `KvCache`, the cache by pages of 16 positions shared by sequences
  (`add_sequence`, `remove_sequence`, `truncate`), in any float type including `f8`;
  `Model::step_batch` and `extend_batch`, a step of many sequences at once;
  `record_step` and `replay_step`; `Model::fused`, the projections of keys, queries
  and values, and of the gated MLP, stacked into one product.
- `kapel-infer`: `Batch` (`Pipeline::batch`), generation of many requests together, a
  step for all at each pass, with recorded passes and greedy choice on the device;
  `Request::ignore_eos`; `LoadOptions::fuse` and `cache_dtype`.
- `kapel-cli`: `kapel run --kv-cache`; the examples `throughput` (as
  `vllm bench throughput`), and `--kv-cache` for `perplexity`. `kapel-cuda`: the
  examples `decode` and `quantized`.
- `kapel-core`: `Tensor::storage_mut` and `Storage::as_any_mut`; `Layout::reshape`
  reshapes strided views when their strides allow it.

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
- `kapel-cuda`: decoding Llama 3.2 3B in `bf16` on an RTX 3090 went from 16 to 105
  tokens per second, and 64 sequences together reach 3330 tokens per second (90% of
  vLLM); a prompt of 4096 tokens takes 0.76 s.
- `kapel-cuda`: small uploads keep a copy on the host, read for their indices; the
  tests fill new buffers with NaN (`Cuda::poison_buffers`).
- `kapel-cuda`: one backend per device, shared process-wide; shape tables are
  cached on the device; a failed allocation empties the pool and retries.
- `kapel-hub`: the download cache is keyed by commit; only top-level files are
  downloaded; saving writes a temporary file and renames it.
- `kapel-cpu`: the pool publishes its work without a lock, on separate cache lines
  (2.5 µs to 0.7 µs per call); the exponential vectorizes (3.7 times faster); binary
  operations on inputs of one shape skip the offsets; `bf16`, `f16` and quantized
  tables are gathered row by row; outputs are computed in pooled buffers.
- `kapel-core`: `Tensor::to_f64_vec` decodes contiguous floats without a call per
  element (the logits of a large vocabulary, at every token).
- The conformity suite compares an output narrower than the type under test at its
  own tolerance.
- `tools/parity/generation_reference.py` reads Llama 3.2 and Mistral from copies of
  their gated repositories: no token needed.
- `kapel-tokenizers`: regexes built once and added tokens found with
  Aho-Corasick (encoding 1.4 to 8 times faster); `strftime_now` uses local time.
- Parity tests say when their data is missing, and fail if
  `KAPEL_REQUIRE_PARITY=1`. The CUDA job rebuilds the PTX and compares it, and
  runs on pull requests from this repository.

### Fixed

- `kapel-cli`: on the processor, `kapel run` converts the weights to `f32` by default,
  as its help says; it kept `bf16`, forty times slower.
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
