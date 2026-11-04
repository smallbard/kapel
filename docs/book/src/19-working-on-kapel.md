# 19. Working on Kapel

This chapter is the practical companion to the rest of the book: how to build and test Kapel, how correctness and speed are verified, what continuous integration checks, and where the project goes next.

## 19.1 Building and testing

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

## 19.2 The layers of testing

| Layer | What it proves | Where |
| --- | --- | --- |
| Unit tests | each module's own rules: float codecs on every bit pattern, layouts, BPE, sampling | next to the code |
| Finite differences | the reference's gradients are derivatives | `kapel-ref` |
| Conformity | each backend matches the reference, forward and backward, within written tolerances | `kapel-ref/src/conformity.rs`, called by each backend |
| Determinism | the same bits with 1, 2, 3, 8 or 16 threads, and on every GPU run | CPU and CUDA tests |
| Parity | the same tokens, renderings, logits and generations as `transformers`, `tokenizers` and llama.cpp | `tests/parity.rs`, `tools/parity/` |
| Fuzzing | no panic, hang or out-of-bounds read on any input | `fuzz/` |

The server has a check of its own: `tools/serve/openai_client.py` drives a running server with OpenAI's official Python client, unchanged, through every feature the server offers (chapter 12).

Parity tests need reference data produced by the Python scripts in `tools/parity/` and stored in `data/parity/`, which is not in git. Without it they print that they were skipped; with `KAPEL_REQUIRE_PARITY=1` they fail instead, which is what a machine that has the data should set.

## 19.3 Measuring performance

The rule "every optimization comes with a measurement" is applied with example programs that live next to the code they measure:

| Crate | Examples |
| --- | --- |
| `kapel-cpu` | `gemm`, `gemm_few_rows`, `gemm_bf16`, `attention`, `reduce`, `softmax` |
| `kapel-cuda` | `gemm`, `training_gemm` (the products of a training step of GPT-2, each form), `elementwise` (bias, sums and conversions of a step, in GB/s), `attention`, `reductions`, `prefill`, `decode`, `quantized` |
| `kapel-tokenizers` | `encode` |
| `kapel-cli` | `throughput`, `perplexity`, `profile`, `speculative` (decoding alone against each proposer), `bench` (the tracked measures) |
| `kapel-serve` | `load`: a Poisson load on a server, the latencies of the first token and between tokens |
| `kapel-train` | `gradients`: the loss, gradients, time and peak memory of one batch, against PyTorch with `tools/train/gradients.py`, and the time of each operation with `--profile`; `steps`: a few steps of AdamW |

Run them with `cargo run --release -p <crate> --example <name>`, on an idle machine. They report the best of several runs after a warm-up, in Gop/s, GB/s or tokens per second. The CUDA examples count operations conventionally, for instance two operations per multiply-add, and only the visible half of causal attention scores.

**Tracked measures.** A few numbers are watched on every change, so that a regression is seen when it lands rather than months later. The example `bench` of `kapel-cli` measures one setup (a model, a device, a type): the speed of a prompt of 512 random tokens computed at once, the speed of decoding 128 tokens of one sequence (greedy, past the end tokens, the first token aside), on request a batch of sequences of 128 + 256 tokens, and the perplexity of eight chunks of wikitext-2 as `llama-perplexity` computes it. Each speed is the best of three runs after one that warms up. `tools/bench/run.sh` runs the fixed list of setups, three on the processor and three on the card, into one JSON report, and `tools/bench/compare.py` compares it with the baseline in `tools/bench/baselines/`, measured on the same machine. A speed more than 10 % below the baseline fails, and so does a perplexity that moved by more than 0.01 %: since results do not depend on threads, a perplexity changes only when the arithmetic does, and the change that did it must say why and update the baseline. The recipes of training are tracked the same way (`tools/bench/recipes.py`, the suite `train` of `run.sh`): each runs a few steps (`kapel train --steps N --dir …`) in a copy of the repository's layout where its data are prepared once and kept, and its report gives the mean speed of the steps after the first, the peak memory, and the loss of the last step. A speed 10 % lower or a memory 10 % larger fails, and so does a loss that moved: a run of the same configuration computes the same bits. The workflow `measures` runs them on the self-hosted machine at every push to `main` and every Monday; the guide publishes the numbers and their method.

## 19.4 Continuous integration

| Job | When | What |
| --- | --- | --- |
| `format`, `clippy` | every push and pull request | `cargo fmt --check`; clippy with warnings as errors |
| `test` | every push and pull request | tests and docs on Linux x86, Linux ARM, macOS and Windows (the Python module aside) |
| `bindings` | every push and pull request | the C library and its example, the Python wheel and its tests, on Linux, macOS and Windows, with SmolLM2-135M-Instruct |
| `msrv` | every push and pull request | `cargo check` with Rust 1.89 |
| `licenses` | every push and pull request | `cargo deny`: licenses, advisories, sources |
| `cuda` | main, on demand, and pull requests from the repository itself | rebuild and compare the PTX, clippy and tests with the `cuda` feature, on a self-hosted RTX 3090 |
| `docs` | every push and pull request | the book and the guide build with mdBook |
| `measures` | main, every Monday, on demand | the tracked measures on the self-hosted machine, against their baseline |
| `fuzz`, `miri` | every Monday | one hour per fuzz target; Miri on the crates with `unsafe` code |

The GPU job never runs code from forks, because anyone can open a pull request and the runner is a real machine; it also runs with read-only permissions and no stored credentials.

## 19.5 Contributing

The rules of chapter 1 are also the review checklist:

- A new operation comes with its reference definition, its gradient, conformity cases with a written tolerance, and implementations in each backend (section 4.5).
- Work is split by shape, never by thread count, and a test checks it bit for bit.
- User input never panics; a test asserts the exact error.
- An optimization comes with a benchmark and its before and after numbers.
- Every `unsafe` block has a `// SAFETY:` comment.
- Code, documentation and messages are in English, and `CHANGELOG.md` gets a line under *Unreleased*.

## 19.6 The road ahead

The foundation is complete. The first inference step, correct generation, landed on 21 October 2026; the second, the CPU, on 22 October, but for its ARM kernels and a cache of keys and values in int8; and the third, the card, the same day. Its targets were 70 % of vLLM's throughput for batches, met at 90 %, and 90 % of the card's bandwidth for one sequence, missed at 81 %: the products alone reach 92 %, and the rest of a token goes to the small kernels between them. The fourth, the server, landed on 23 October 2026. Its targets were a first token within 500 ms for 99 % of the requests up to 80 % of the maximal throughput, met up to 90 % (83 ms there, 63 ms at 80 %), and OpenAI's official clients working unchanged, met. The validation needs no other server: the load is drawn as Smallem's simulator drew it, and the maximal throughput is the server's own. The fifth, speculative decoding, landed the same day, without its two other parts: tensor parallelism over several GPUs was left out, and the tables kept off the card waits for a model with a memory of n-grams that Kapel can read (the published ones are a mixture of experts far too large for one card, or architectures of their own). Its target, greedy Qwen3 8B 1.5 times faster with Qwen3 0.6B proposing and the same answers, was met at 1.72 times. The sixth, quantizing one's own models, landed the same day. Its targets were perplexities within 0.5 % of lm-evaluation-harness's, met at 0.001 %, and an int4 model that loses less than the same quantization by llama.cpp: by GPTQ, Llama 3.2 1B in Q4\_0 loses 0.41 of perplexity, against 0.74 for llama.cpp with its importance matrix and 1.31 without (chapter 13). The seventh, being used, landed the same day but for its first releases. Training began on 24 October 2026 with its first step, the differentiation; its targets were the gradients of GPT-2 124M equal to PyTorch's within the tolerance of bf16, met (16.9 % from PyTorch's f32 gradients, against 16.6 % for PyTorch's own in bf16), and a peak memory at most 10 % above PyTorch's, met at 47 % of it (chapter 15). The second, training on one card, landed the same day; its target, GPT-2 124M to the speedrun's loss in at most 1.2 times llm.c's time, was measured on the time of a step with llm.c's settings, 1.16 times, and on the losses of a shorter run, a whole run taking some 45 hours on the card (chapter 15). The third, the data, landed the same day: loading a batch takes 0.003 ms of a step of GPT-2 124M on one card (chapter 15). The fourth, post-training, too: Qwen 3 0.6B fine-tuned with LoRA on GSM8K answers it within 1 point of TRL with the same settings (48.3 % against 47.5 %), and GRPO raises its accuracy from 61.2 % to 69.3 % in 200 steps (chapter 16). The fifth, the tools of experiments, landed on 25 October 2026: Smallem's curve at 10¹⁶ operations, run again with one command, finds its lowest point within 5 %: two seeds fitted together put it at 14.8 M parameters and 0.894 nats per byte, against Smallem's 15.3 M and 0.897, the size 3.3 % off and the loss 0.3 %; each seed alone moves it by some 6 % (14.4 and 15.3 M), as much as it differs from Smallem's (chapter 17). The sixth, being used, landed on 26 October 2026: *The Kapel Guide* tells how to pre-train, fine-tune, resume and publish, and each recipe of `recipes/`, run again from it, finds its expected results: GPT-2 124M's validation loss at step 250, Llama 3.2 1B's architecture trained with Adafactor in 16.1 GiB, Qwen 3 0.6B fine-tuned on GSM8K (50.0 % with every weight, 48.3 % with LoRA), its DPO on UltraFeedback within the noise of TRL's (64.8 % of held-out pairs against 65.1 %), and its GRPO from 61.2 % to 69.3 %, the same numbers as in chapter 16; `kapel export` writes a run as the Hub holds models, which `transformers` and PEFT load (chapters 15 and 16). A seventh, larger and faster on one card, landed on 27 October 2026: AdamW in 8 bits follows AdamW within 0.03 % of GPT-2's validation loss and within a point of GSM8K, and trains Qwen 3 1.7B with all its weights on the card; fine-tuning without padding gives the same results in 5 minutes 24 instead of 7 minutes 20, faster than TRL; and with its state in the processor's memory, Llama 3.2 3B trains with all its weights by AdamW on 24 GB (chapters 15 and 16). The step that needs several cards is split: what one card can check came first and landed on 27 October 2026, four ranks computing the bits of one rank that accumulates their passes, with or without their state shared out (ZeRO 1 and 2, FSDP), and tensor parallelism the gradients of the whole model within `f32`'s rounding (chapter 15); only their speed, with NCCL, waits for rented cards. What is ahead:

| Goal | Main pieces |
| --- | --- |
| The best single-sequence generation on CPU, x86 and ARM | done on x86 (chapter 6); left: NEON kernels and `sdot`, a cache of keys and values in int8 |
| Batched generation on a GPU | done (chapters 7, 10 and 11): a cache by pages of 16 positions, paged attention, CUDA graphs, products by quantized weights on the card, batches |
| A production server | done (chapters 10, 11 and 12): continuous batching, shared prefixes, an OpenAI-compatible API with streaming and cancellation, constrained output, metrics, logs, a container image |
| Faster without changing the distribution | done (chapters 7, 10 and 11): speculative decoding with a smaller model, prompt lookup or a multi-token layer (MiMo); left: tensor parallelism over two to four GPUs, tables kept off the card |
| Quantizing one's own models | done (chapter 13): GPTQ and AWQ, writing GGUF, perplexity and tasks of lm-evaluation-harness replayed; left: FineWeb-Edu, a faster AWQ |
| Being used | done but for the releases (chapter 14): a C API, a Python module, *The Kapel Guide* (`docs/guide`), measures published and tracked in CI; left: the first releases on crates.io and PyPI, prepared |
| Differentiation | done (chapter 15): a tape that keeps only what the gradients need, checkpointing, a fused cross-entropy over a large vocabulary, gradients by rows and AdamW by rows, the decoders of chapter 10 on a tape |
| Training on one card | done (chapter 15): `kapel train` and its TOML configuration, accumulation, clipping, schedules, evaluations, AdamW fused, Muon, Adafactor, groups of parameters, checkpoints that resume bit for bit, the model's FLOP utilization; GPT-2 124M at 1.16 times llm.c's time; left: 8-bit floats, for a card that has them |
| The data | done (chapter 15): `kapel prepare`, weighted sources shuffled from a seed, rows packed by documents, batches read ahead from disk or HTTP, the position in every source in the checkpoint |
| Several cards, several machines | what one card can check done (chapter 15): collectives (threads, TCP), data parallelism, ZeRO 1 and 2, FSDP, tensor parallelism, checkpoints any number of ranks resumes, a rank that stops stopping the others; left, on rented cards: NCCL, communications overlapped with computation, Slurm, tensor parallelism in `kapel train` |
| Post-training | done (chapter 16): conversations and their masks, LoRA and QLoRA in PEFT's format, `kapel merge`, DPO, GRPO and its rewards, tasks along a run; left: tokens added to a vocabulary |
| The tools of experiments | done (chapter 17): budgets of operations, `kapel sweep` and `kapel fit`, the laws of Chinchilla, logs for TensorBoard and Weights & Biases, the operations of each call in the profiler |
| Being used | done (chapters 15, 16 and 19): a guide to training, recipes with their expected results and costs, `kapel export` for the Hub, the recipes measured in CI |
| Larger and faster on one card | done (chapters 15 and 16): AdamW in 8 bits, fine-tuning and DPO without padding, the optimizer's state in the processor's memory |

The targets are concrete: on the CPU, 90 % of llama.cpp's speed; on the GPU, 70 % of vLLM's; for training, nanoGPT, torchtitan and TRL. Each step has its validation criterion, measured against them, and each adds a chapter's worth of material to this book.
