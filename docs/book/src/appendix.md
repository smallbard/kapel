# Appendix

## Glossary

| Term | Meaning in Kapel |
| --- | --- |
| Activations in bytes | Quantizing the input of a quantized product too, by blocks of 32, so that the product is a sum of products of bytes; chosen explicitly, as it changes the result a little |
| Backend | A type implementing `Run` for every operation of the catalogue: `Reference`, `Cpu` or `Cuda` |
| Batch | Several sequences computed in the same pass, one row per new token of each, so the weights are read once for all of them (`Model::step_batch`, `kapel_infer::Batch`) |
| Block (quantized) | The unit a quantized type stores: a scale, sometimes a minimum, and packed codes for 32 or 256 elements |
| Broadcast | Repeating a tensor along dimensions of size 1 or new leading dimensions, by giving them stride 0 |
| Cache (KV) | The keys and values of the positions already computed, per layer, so each new token only computes itself; in Kapel, kept by pages (`KvCache`) |
| Cache by pages | A cache whose lines are handed out by pages of 16 positions from a common reserve, each sequence keeping the list of its pages: nothing is copied as a sequence grows, and its pages go back when it ends |
| Conformity | Agreement of a backend with the reference within the tolerance of the data type, forward and backward |
| Contiguous | Stored in row-major order without gaps |
| CUDA graph | A pass recorded once and replayed in one launch, its kernels and buffers fixed; the `Record` trait, and `Model::record_step` |
| Decoding | Generating one token at a time after the prompt, bound by memory bandwidth |
| Determinism | The same bits whatever the number of threads, achieved by splitting work by shape |
| f8 | The 8-bit floats E4M3 and E5M2, used by Kapel for caches of keys and values half the size of bf16 |
| Fusing | Stacking the projections that read the same input into one weight, one product instead of several (`Model::fused`) |
| GGUF | llama.cpp's file format: a model's configuration, tokenizer and weights, quantized or not, in one file |
| GQA | Grouped-query attention: several query heads share one key and value head |
| `infer` | The validation and shape computation shared by every backend for an operation |
| `Isa` | A proof that the processor has the instructions a code path uses |
| Micro-kernel | The innermost routine of the matrix product, holding a block of C in registers |
| Packing | Copying a block of an operand into the contiguous order the micro-kernel reads |
| Paged attention | Attention of new rows over a cache by pages, each row knowing its sequence and how many positions it sees (`PagedAttention`) |
| Parity | Agreement with the Python libraries `transformers`, `tokenizers` or llama.cpp on real models and files |
| Perplexity | The exponential of the mean negative log-likelihood of a text: how a quantization is judged |
| Prefill | Computing the whole prompt at once, before decoding |
| PTX | NVIDIA's intermediate assembly; the thread kernels are checked in as PTX |
| Reference | The `f64` backend that defines correct results |
| RoPE | Rotary position embeddings: pairs of components rotated by an angle proportional to the position |
| Softcap | A smooth bound on scores or logits, `cap · tanh(x / cap)`, used by Gemma 2 |
| Stride | The distance in the buffer between two consecutive elements of a dimension |
| Thread kernel / tile kernel | A CUDA kernel with one thread per element, written for cuda-oxide / a kernel over tiles of a matrix, written for cutile-rs |
| Throughput | Tokens generated per second for a batch of requests, compared with vLLM's `vllm bench throughput` |
| View | A tensor sharing another's storage with a different layout; on the GPU, the `View` struct passed to kernels |

## Where things are

| Topic | Files |
| --- | --- |
| Tensors, types, layouts | `crates/kapel-core/src/{tensor,dtype,float,layout,storage}.rs` |
| Errors | `crates/kapel-core/src/error.rs` |
| Operation traits and catalogue | `crates/kapel-ops/src/lib.rs` and one file per family |
| Paged attention, writes in place, recording | `crates/kapel-ops/src/paged.rs`, `crates/kapel-ref/src/paged.rs`, `crates/kapel-cpu/src/paged.rs`, `crates/kapel-cuda/src/paged.rs` |
| Reference and conformity suite | `crates/kapel-ref/src/{lib,conformity,golden}.rs` |
| CPU threads, product, kernels | `crates/kapel-cpu/src/{pool,gemm,kernels,math,isa}.rs` |
| CPU attention | `crates/kapel-cpu/src/attention.rs` |
| CPU quantized products | `crates/kapel-cpu/src/{quantized,bytes,matmul}.rs` |
| CUDA device, launches, signatures | `crates/kapel-cuda/src/{device,signatures}.rs` |
| CUDA products and attention | `crates/kapel-cuda/src/{tile,attention}.rs` |
| CUDA graphs | `crates/kapel-cuda/src/graph.rs`, `crates/kapel-cuda/tests/recorded.rs` |
| CUDA quantized products | `crates/kapel-cuda/src/quantized.rs` |
| CUDA benchmarks | `crates/kapel-cuda/examples/{prefill,decode,quantized}.rs` |
| CUDA thread kernels and ABI | `kernels/src/lib.rs`, `crates/kapel-cuda-abi/src/lib.rs` |
| safetensors, GGUF, checkpoints, Hub | `crates/kapel-hub/src/{safetensors,gguf,checkpoint,download,atomic}.rs` |
| Tokenizers and chat templates | `crates/kapel-tokenizers/src/{lib,bpe,text,decode,chat}.rs` |
| Configurations and forward pass | `crates/kapel-models/src/{config,model}.rs` |
| Cache by pages | `crates/kapel-models/src/cache.rs` |
| Models from GGUF | `crates/kapel-models/src/gguf.rs`, `crates/kapel-infer/src/gguf.rs` (tokenizer) |
| Generation and the command | `crates/kapel-infer/src/`, `crates/kapel-cli/src/` |
| Batches | `crates/kapel-infer/src/batch.rs` |
| Profiling, perplexity and throughput | `crates/kapel-cli/examples/{profile,perplexity,throughput}.rs` |
| Parity references | `tools/parity/*.py`, data in `data/parity/` |
| Fuzzing | `fuzz/fuzz_targets/`, `fuzz/regressions/` |
| Rules and plans | `CONTRIBUTING.md`, `SECURITY.md`, `PLAN-*.md` (in French), `CHANGELOG.md` |
