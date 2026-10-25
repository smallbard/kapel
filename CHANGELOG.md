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
- `kapel-serve`: the server of OpenAI's API (`serve`, `Server`):
  `/v1/chat/completions` and `/v1/completions`, whole or streamed (server-sent
  events, `stream_options.include_usage`), `n` answers, stop sequences, OpenAI's
  and vLLM's sampling settings, `response_format` (`json_object`, `json_schema`)
  and vLLM's `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`;
  `/v1/models`, `/health`, `/metrics` (Prometheus: requests, tokens, latency of
  the first token and between tokens, use of the cache, suspensions, shared
  prompt positions). A closed connection cancels its request; limits on the
  body, the positions of the model, the answers in flight (429) and the time
  of a request; one line of JSON per request on the standard error. The
  example `load`, a Poisson load as Smallem's simulator of serving drew it.
- `kapel-cli`: `kapel serve MODEL`, the cache sized from the card's free memory
  (`--gpu-memory-utilization`) or given (`--cache-memory`).
- `kapel-infer`: `Engine`, continuous batching: requests added and cancelled at
  any time, prompts computed by pieces within a budget of tokens per pass,
  admission by free pages, the request admitted last suspended when they run
  out and computed again later; recorded decoding passes for a few sizes of
  batch. `Pipeline::prepare` and `Engine::add_prepared`, to get a request ready
  on another thread. `Batch` is an engine whose requests all come at once.
- `kapel-infer`: answers held to a form, `Constraint` (JSON schema, JSON object,
  regular expression, Lark grammar, list of texts), through llguidance;
  `Request::constraint`; `Finish::Error`, a request failed on its own.
- `kapel-infer`: OpenAI's `frequency_penalty` and `presence_penalty` in
  `Sampling`.
- `kapel-models`: prefix sharing (`KvCache::with_prefix_sharing`,
  `add_sequence_sharing`): sequences take the pages of the longest beginning of
  their prompt the cache holds, counted by references, and idle shared pages
  are taken back least recently used first; a sequence cut inside a shared
  page gets its own copy. `KvCache::reserve_padding`.
- `kapel-models`: a recorded step serves any batch of as many sequences or
  fewer (`record_step` takes its rows and pages; padding rows write lines of
  their own); `Recorded`, what `record_step` did, the step computed even when
  the backend fails to record it. `Config::max_positions`.
- `kapel-infer`: speculative decoding, `Pipeline::speculate` and `Draft`: a
  smaller model of the same tokenizer, prompt lookup (the tokens that followed
  the last ones where they last appeared), or the model's multi-token prediction
  layer proposes; the exact rule of speculative sampling checks the proposals in
  one pass of the model (checked by a chi-square test), so greedy answers do not
  change and sampled ones keep their distribution. `SpeculationStats`. Qwen3 8B
  with Qwen3 0.6B proposing: 1.72 times faster on an RTX 3090, the same answers.
- `kapel-models`: MiMo (`model_type` `mimo`), Qwen 2 with a multi-token
  prediction layer, and `Model::multi_token`, the layer as vLLM runs it (1.53
  times faster for MiMo-7B-RL, 76 % of its proposals accepted).
  `Model::extend_last`, `Model::extend_hidden`, `Model::into_fused` (stacks the
  projections of a model without holding two copies of its weights);
  `Config::multi_token_layers`.
- `kapel-cli`: `kapel run --draft MODEL`, `--prompt-lookup`, `--multi-token` and
  `--speculate K`; the example `speculative`, decoding alone against each
  proposer.
- `tools/parity/multi_token_reference.py`: a small MiMo and the outputs of its
  multi-token layer.
- `kapel-tokenizers`: `Tokenizer::token_bytes`, the bytes of a token in the
  middle of a text, and `Tokenizer::is_special`.
- `kapel-cuda`: `Cuda::memory`, the free and total memory of the device.
- `tools/serve/openai_client.py`: the server through OpenAI's official Python
  client. A `Dockerfile` for the server on NVIDIA cards.
- `kapel-quant`: quantizing models from calibration texts: rounding to the nearest,
  GPTQ, AWQ, and AWQ then GPTQ, to Kapel's int8 and int4 and to GGUF's Q8_0, Q4_0 and
  Q4_1, block by block on the processor or the card.
- `kapel-core`: `block_scale`, `block_code`, `block_value` and `encode_block`, the
  quantization of a block with a scale chosen elsewhere (by GPTQ or AWQ), giving the
  bytes llama.cpp gives.
- `kapel-hub`: writing GGUF files (`gguf::write`, `Entry`, `ggml_type`).
- `kapel-models`: `Model::to_gguf`, a model with llama.cpp's names and layout;
  `Model::embed`, `Model::block`, `Model::tensor`, `Model::names`,
  `Model::replace` and `Site`, to compute and change a model one block at a time.
- `kapel-infer`: `save_gguf`, a model and its tokenizer written to a GGUF file that
  llama.cpp reads; `Scorer`, the log-likelihood of continuations after contexts,
  sharing the cache between sequences that begin alike, its rows decoded and scored on
  every core (WikiText on a card 2.4 times faster).
- `kapel-cli`: `kapel quantize`, to GGUF or to a model directory; `kapel eval`, the
  tasks of lm-evaluation-harness (perplexity, HellaSwag, ARC, MMLU) replayed by Kapel,
  tokenized and scored as lm-evaluation-harness does.
- `tools/eval/export_tasks.py`: the requests of lm-evaluation-harness's tasks,
  written once for `kapel eval`.
- `kapel-hub`: a model downloaded once runs offline: the commit of each revision is
  recorded (`refs/<revision>`) and used when the Hub cannot be reached, or when
  `KAPEL_OFFLINE` or `HF_HUB_OFFLINE` is `1`.
- `kapel-runtime`: a model loaded on whichever device, behind one type, for the
  bindings.
- `kapel-capi`: the C API, `libkapel` and `include/kapel.h`, with an example in C.
- `docs/guide`: *The Kapel Guide*, for users: installing, a first model, serving,
  quantizing, evaluating, integrating from Python, C and Rust, a reference and
  troubleshooting. The book gains chapters on quantization and evaluation (13) and
  on the bindings (14).
- `kapel-cli`: the example `bench`, the speeds and perplexity Kapel tracks;
  `tools/bench/run.sh` measures a fixed list of setups and `tools/bench/compare.py`
  compares them with a baseline. The workflow `measures` runs them on the
  self-hosted machine at every push to `main`, the workflow `release` builds
  binaries and wheels for a tag, and CI builds the book and the guide.
- `kapel-python`: the Python module `kapel` (PyO3, stable ABI from Python 3.9): load,
  generate, chat, serve. CI builds both and tests them on Linux, macOS and Windows.
- `kapel-train`: reverse-mode differentiation on any backend (`Tape`, `Var`): values
  let go as soon as no derivative needs them, each derivative keeping only what it
  reads; activation checkpointing (`Tape::checkpoint`); gradients of slices joined
  without padding; gradients of table rows kept as rows for sparse leaves;
  derivatives written by hand (`Tape::custom`).
- `kapel-train`: losses (`cross_entropy`, and `linear_cross_entropy`, which never
  forms the logits of a large vocabulary whole), AdamW dense and by rows, the
  decoders of `kapel-models` on a tape (`Decoder`, `Parameters`, `init`), and a
  backend that times each operation (`profile::Profiled`).
- `kapel-ops`: `CrossEntropy`, the loss of each row of logits with its gradient
  written by hand, on the reference, the processor and the card.
- `kapel-cuda`: the bytes held by live tensors and their peak
  (`Cuda::memory_in_use`, `Cuda::peak_memory`, `Cuda::reset_peak_memory`).
- `kapel-models`: `Model::new`, `Model::with_weights` and `Model::raw_config`, to
  make a model of trained weights.
- The examples `gradients` and `steps` of `kapel-train`, and
  `tools/train/gradients.py`, which compares Kapel's gradients and peak memory
  with PyTorch's.
- The Kapel Book: chapter 15, training.
- `kapel train CONFIG.toml`: training runs described by a TOML file, checked
  before they start (`kapel_train::config`): batches read in order from llm.c's
  token files (`data`), gradient accumulation, clipping of the global norm,
  warmup then cosine, constant or WSD schedules (`schedule`), evaluations, the
  log of every step in `log.jsonl` with the model's FLOP utilization (`flops`)
  and the card's peak memory; `--resume` goes on from the last checkpoint bit
  for bit, `--steps` stops a run, `--profile` times each operation.
- `kapel-train`: checkpoints written in the background and rotated, each a model
  directory with the optimizer's state, the position in the data and the
  configuration (`checkpoint`); Muon and Adafactor next to AdamW, and groups of
  parameters with learning rates, decay and freezing of their own.
- `kapel-ops`: `Update::adamw`, one AdamW step in place, on the reference, the
  processor and the card; `Matmul` takes an optional bias, added in the product.
- `kapel-cuda`: `Cuda::name`, and `Cuda::keep_attention_for_backward`, which keeps
  each attention's output and statistics for its backward pass.
- `recipes/gpt2-124m`: GPT-2 124M on FineWeb with llm.c's settings.
- The examples `training_gemm` and `elementwise` of `kapel-cuda`.
- `kapel prepare`: Parquet, JSONL and text files tokenized on every core into
  shards of llm.c's format with the index of their documents and a manifest
  (`kapel_train::prepare`, `kapel_train::dataset`); exact duplicates and empty
  documents left out, one document in `n` set aside for validation; the same
  files whatever the number of threads.
- `kapel-train`: training data from several weighted sources, shuffled by epoch
  from a seed, cut as a stream or packed as whole documents, with attention
  kept within documents and positions from each document's start; batches read
  ahead by a thread; datasets read from `http(s)://` URLs, each shard downloaded
  while the one before is read, checked and removed once read. The position in
  every source is part of a checkpoint. `data_ms` in the log.
- `kapel-ops`: `Attention` takes the documents of its keys (`[B, T]` integers,
  where each document starts) in place of a mask; the card skips the keys
  before each document instead of reading a mask of `T²` values.
- `kapel-tokenizers`: `Tokenizer::encode_ordinary`, a text encoded without
  reading its added tokens.
- `kapel-hub`: `download::fetch_url`, a file downloaded and checked.
- The example `loader` of `kapel-train`.
- `kapel prepare --chat`: conversations and prompt-completion examples (TRL's
  forms, from JSONL or Parquet) made into tokens by the model's chat template,
  with a mask of the assistant's tokens beside each shard; the loss counts
  them only (`kapel_train::chat`). `data.pack = "padded"`: one document a row.
- `kapel-train`: LoRA adapters on every projection and on the embeddings, drawn
  as PEFT draws them, merged into the model, saved and read in PEFT's format;
  QLoRA, the frozen projections quantized; runs with adapters save only what
  they train (`kapel_train::lora`).
- `kapel-train`: DPO, with the likelihood term of RPO (`[dpo]`,
  `kapel_train::preference`); GRPO and Dr. GRPO, the answers written by the
  inference engine from the weights being trained, scored by rewards of four
  kinds or of one's own (`[grpo]`, `kapel_train::grpo`,
  `kapel_train::rewards`); the losses of tokens without forming the logits
  whole (`linear_token_losses`).
- `kapel-train`: the loss of a text of control and the accuracy of tasks whose
  answers a program checks, measured along a run (`eval.control`,
  `[[eval.tasks]]`); `schedule.kind = "linear"`, as `transformers`; a model of
  the Hub as `model.config`.
- `kapel merge`: a LoRA adapter merged into its model.
- `kapel-infer`: `Pipeline::with_model`, a pipeline of a model already in
  memory; `kapel-tokenizers`: `ChatTemplate::render_with`, more variables for
  the template (`enable_thinking`).
- `kapel-ops`: `Cast` decodes quantized types to float ones.
- `tools/train`: `peft_parity.py`, `trl_sft.py` and `gsm8k.py`; the examples
  `lora` and `tasks` of `kapel-train`.

### Changed

- `kapel-cuda`: GPT-2 124M trains at 1.16 times llm.c's time on an RTX 3090,
  from 2.3 times: the gradient of a weight (`xᵀ · g`) has a kernel of its own;
  element-wise operations on 16-bit values go eight values a thread in 128-bit
  loads and stores; the backward passes of element-wise functions, norms and
  sums read 16-bit inputs in place; sums over the first or the last dimensions
  need no table of offsets, and long ones go in as many stages as they need;
  `CrossEntropy` takes a block per row; a product's bias is added in its kernel.
- `kapel-train`: the fused loss pads the vocabulary to a multiple of 64 for its
  products, and forms 4096 rows of logits at once by default; gradients stay
  views instead of being made contiguous.
- `kapel-cuda`: sums of more than 128 terms are computed in two stages, segments
  of adjacent terms for all results at once, then each result's segments, in a
  fixed order: faster, and more precise over millions of terms (a sum of 12.8
  million values was off by 5e-5). The weight gradients of norms use them.
- `kapel-cuda`: cuda-oxide's crates (`cuda-core`, `cuda-bindings`, `cutile`) come from
  crates.io (0.4.0) instead of a Git revision: every crate but the bindings can be
  published.
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
- `kapel-cuda`: paged attention cuts the positions into chunks sized by the
  sequences and heads of a pass, no longer by its queries: the positions of one
  sequence give the same bits computed together (the proposals of speculative
  decoding) as one at a time.
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

- `kapel-cuda`: the backward pass of a product by quantized weights runs on
  the card, the weights dequantized there; it went through the host.
- `kapel-tokenizers`: a `tokenizer.json` whose model has no type but merges is
  read as a BPE, as `tokenizers` does; GPT-2's file on the Hub is so.
- `kapel-hub`: `HF_TOKEN` is sent to the Hub only.

- `kapel-models`: GPT-2's first checkpoints (`openai-community/gpt2`) load: their
  weights get the names of `GPT2LMHeadModel`, and the causal masks stored among
  them are left out.
- `kapel-cuda`: products by Q4_0 and Q4_1 weights have kernels of their own; they
  went through the generic kernel, seven times slower (Llama 3.2 3B in Q4_0: 28 to
  204 tokens per second).
- `kapel-cuda`: a recorded pass keeps the tables its kernels read; once the
  table cache had let them go, a replay read freed memory. The table cache
  evicts the least recently used, and memory short at an allocation frees the
  pool before the tables.
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
