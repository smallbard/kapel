# 13. Working on Kapel

This chapter is the practical companion to the rest of the book: how to build and test Kapel, how correctness and speed are verified, what continuous integration checks, and where the project goes next.

## 13.1 Building and testing

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

These four commands are the checklist of `CONTRIBUTING.md` before any pull request. The workspace builds on stable Rust 1.89 or later, edition 2024. The CUDA backend needs the `cuda` feature, an NVIDIA GPU and CUDA 13.2 or later:

```
cargo test -p kapel-cuda --features cuda
```

Rebuilding the thread kernels after changing `kernels/src/lib.rs` needs the nightly toolchain pinned in `kernels/rust-toolchain.toml` and `cargo oxide`: run `kernels/build.sh`, and commit the regenerated `crates/kapel-cuda/kernels.ptx`.

## 13.2 The layers of testing

| Layer | What it proves | Where |
| --- | --- | --- |
| Unit tests | each module's own rules: float codecs on every bit pattern, layouts, BPE, sampling | next to the code |
| Finite differences | the reference's gradients are derivatives | `kapel-ref` |
| Conformity | each backend matches the reference, forward and backward, within written tolerances | `kapel-ref/src/conformity.rs`, called by each backend |
| Determinism | the same bits with 1, 2, 3, 8 or 16 threads, and on every GPU run | CPU and CUDA tests |
| Parity | the same tokens, renderings, logits and generations as `transformers`, `tokenizers` and llama.cpp | `tests/parity.rs`, `tools/parity/` |
| Fuzzing | no panic, hang or out-of-bounds read on any input | `fuzz/` |

Parity tests need reference data produced by the Python scripts in `tools/parity/` and stored in `data/parity/`, which is not in git. Without it they print that they were skipped; with `KAPEL_REQUIRE_PARITY=1` they fail instead, which is what a machine that has the data should set.

## 13.3 Measuring performance

The rule "every optimization comes with a measurement" is applied with example programs that live next to the code they measure:

| Crate | Examples |
| --- | --- |
| `kapel-cpu` | `gemm`, `gemm_few_rows`, `gemm_bf16`, `attention`, `reduce`, `softmax` |
| `kapel-cuda` | `gemm`, `attention`, `reductions`, `prefill` |
| `kapel-tokenizers` | `encode` |

Run them with `cargo run --release -p <crate> --example <name>`, on an idle machine. They report the best of several runs after a warm-up, in Gop/s, GB/s or tokens per second. The CUDA examples count operations conventionally, for instance two operations per multiply-add, and only the visible half of causal attention scores.

## 13.4 Continuous integration

| Job | When | What |
| --- | --- | --- |
| `format`, `clippy` | every push and pull request | `cargo fmt --check`; clippy with warnings as errors |
| `test` | every push and pull request | tests and docs on Linux x86, Linux ARM, macOS and Windows |
| `msrv` | every push and pull request | `cargo check` with Rust 1.89 |
| `licenses` | every push and pull request | `cargo deny`: licenses, advisories, sources |
| `cuda` | main, on demand, and pull requests from the repository itself | rebuild and compare the PTX, clippy and tests with the `cuda` feature, on a self-hosted RTX 3090 |
| `fuzz`, `miri` | every Monday | one hour per fuzz target; Miri on the crates with `unsafe` code |

The GPU job never runs code from forks, because anyone can open a pull request and the runner is a real machine; it also runs with read-only permissions and no stored credentials.

## 13.5 Contributing

The rules of chapter 1 are also the review checklist:

- A new operation comes with its reference definition, its gradient, conformity cases with a written tolerance, and implementations in each backend (section 4.5).
- Work is split by shape, never by thread count, and a test checks it bit for bit.
- User input never panics; a test asserts the exact error.
- An optimization comes with a benchmark and its before and after numbers.
- Every `unsafe` block has a `// SAFETY:` comment.
- Code, documentation and messages are in English, and `CHANGELOG.md` gets a line under *Unreleased*.

## 13.6 The road ahead

The foundation plan is complete, and the first inference step, correct generation, landed on 21 October 2026. The rest of the inference plan and the whole training plan are ahead:

| Step | Goal | Main pieces |
| --- | --- | --- |
| I1 | The best single-sequence generation on CPU, x86 and ARM | quantized formats read in place (int8/int4, Q4\_0, Q4\_K, Q5\_K, Q6\_K, Q8\_0), memory-bandwidth-bound decoding, NEON kernels |
| I2 | Batched generation on a GPU | a paged cache of 16-position pages, Flash-Decoding, kernels for few and many rows |
| I3 | A production server | continuous batching, shared prefixes, an OpenAI-compatible API, constrained output |
| I4 | Faster without changing the distribution | speculative decoding, tensor parallelism over two to four GPUs |
| I5 | Quantizing one's own models | GPTQ and AWQ, quantized writing, perplexity and task evaluation |
| I6 | Being used | a C API, a Python module, guides, published benchmarks |
| T0–T6 | Bit-reproducible training | a gradient tape, a single-GPU loop with AdamW and Muon, data pipelines, multi-GPU and multi-node, SFT, LoRA, DPO and GRPO, experiment tools for scaling laws |

The targets are concrete: on the CPU, 90 % of llama.cpp's speed; on the GPU, 70 % of vLLM's; for training, nanoGPT, torchtitan and TRL. Each step has its validation criterion written in the plan, and each will add a chapter's worth of material to this book.
