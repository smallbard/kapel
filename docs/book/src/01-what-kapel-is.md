# 1. What Kapel is

Kapel is a Rust framework for running, serving and training language models, from half a billion to a few billion parameters, on a CPU or a consumer GPU. It reads the models and files of the Hugging Face ecosystem (`config.json`, safetensors, `tokenizer.json`, chat templates, GGUF) and aims to produce the same logits and the same tokens as the Python library `transformers`.

## Where it comes from

Kapel succeeds Smallem, an educational project that built a language model in Rust and CUDA from scratch: autograd, CPU and CUDA backends, pre-training, scaling laws, quantized inference. Smallem was written to be read; Kapel is written to be used, on other people's models and other people's machines. Code is not copied wholesale: it is taken over module by module, with its API reviewed, its errors typed and its documentation in English. The fast CPU product, the deterministic thread pool and the vectorized exponential, for instance, come from Smallem's `Fast` backend.

## The rules

Five rules shape almost every design decision in the code. They are stated in `CONTRIBUTING.md` and in the plan documents, and they come back in every chapter of this book.

1. **Every fast backend is checked against the reference.** The reference backend computes each operation in `f64`, as plainly as possible. A new operation comes with its reference definition, its gradient, and a conformity test with a written tolerance (chapter 5).
2. **Results do not depend on the number of threads.** Work is split according to tensor shapes, never according to the hardware. Two runs give the same bits, and tests check it with 1, 2, 3 and 8 threads (chapter 6).
3. **The library does not panic on user input.** Wrong shapes, unsupported types, corrupt files, hostile templates: each returns a typed error. `panic!` is reserved for broken internal invariants (chapters 3 and 15).
4. **Every optimization comes with a measurement**, kept as a benchmark program next to the code it measures (chapter 17).
5. **Code that can be audited**: few dependencies, short modules, documented kernels, and a `// SAFETY:` comment on every `unsafe` block.

The development hardware is a Ryzen 7 7700 (Zen 4, with AVX-512 and its bf16 extension) and an RTX 3090 (Ampere, 24 GB). Several tuning constants in the code are named after it, such as the block sizes `Blocks::ZEN4`.

## Three plans

The work is organised in three plans, each a sequence of steps with a goal and a validation criterion. The plan documents at the root of the repository are written in French.

| Plan | File | Goal | Steps | State on 24 October 2026 |
| --- | --- | --- | --- | --- |
| Foundation | `PLAN-SOCLE.md` | Tensors, backends, model files, tooling | S0–S5 | All six steps done |
| Inference | `PLAN-INFERENCE.md` | Load, quantize and serve small models on CPU and GPU | I0–I6 | I0 (correct generation), I1 (the CPU, but for ARM), I2 (the card), I3 (serving) I4 (speculative decoding, without several GPUs nor off-card tables) and I5 (quantizing one's own models) done on 23 October 2026; I6 (being used) done but for its first releases |
| Training | `PLAN-ENTRAINEMENT.md` | Bit-reproducible training up to a few billion parameters, on one card (the steps that need several are set aside for now) | T0–T6 | T0 (differentiation) done on 24 October 2026 |

The inference plan targets one sequence at a time on a CPU first. Smallem showed that a small int4 model generates faster on a Ryzen than on an RTX 3090: token-by-token generation waits on memory, and a well-programmed CPU uses nearly all of its bandwidth. Batched generation on a GPU, with a paged cache, came after, then a server that computes the requests of many clients together.

## What exists today

- A tensor type with every common float, integer and quantized type (chapter 3).
- A catalogue of 23 operations with forward and backward passes, shared input checks, and three backends: reference, CPU and CUDA (chapters 4 to 7).
- Readers for safetensors, GGUF, sharded checkpoints and Hub downloads (chapter 8); models read whole from GGUF files, or quantized at loading (chapter 10).
- Tokenizers and chat templates that match `transformers` (chapter 9).
- Model families Llama, Mistral, Qwen 2 and 3, MiMo, Gemma 2, Phi and GPT-2, checked logit for logit against `transformers` (chapter 10).
- A generation engine and the `kapel run` command (chapter 11), which on a Ryzen 7 7700 generates with a quantized Llama 3.2 1B at 90 % of llama.cpp's speed (chapter 6), and generation by batches over a cache by pages, which on an RTX 3090 reaches 90 % of vLLM's throughput (chapters 7, 10 and 11).
- A server of OpenAI's API, `kapel serve` (chapter 12): requests that come and go computed together, streamed answers, answers held to a JSON schema or a grammar, Prometheus metrics. On an RTX 3090 it serves Llama 3.2 3B at about 20 requests per second, and at 80 % of that load 99 % of the requests get their first token within 63 ms.
- Speculative decoding (chapter 11): a smaller model, the tokens of the sequence, or the model's own multi-token layer proposes, and the model checks several tokens per pass without changing its answers. On an RTX 3090, Qwen3 8B generates 1.72 times faster with Qwen3 0.6B proposing, and MiMo-7B 1.53 times faster with its own layer.
- Quantization with data, by GPTQ and AWQ, and the `kapel quantize` command, which writes GGUF files that llama.cpp reads (chapter 13). Llama 3.2 1B in Q4\_0 by GPTQ loses 44 % less perplexity than llama.cpp's quantization with its importance matrix.
- `kapel eval`, which replays tasks of lm-evaluation-harness (WikiText, HellaSwag, ARC, MMLU) and gives its perplexities within 0.001 % and its accuracies exactly (chapter 13).
- A C library with a stable header, and a Python module (chapter 14).
- A tape of gradients for training (chapter 15): the decoders of chapter 10 train with the same weights and operations. GPT-2 124M computes its gradients on an RTX 3090 for a batch of 16 × 1024 tokens with less than half of PyTorch's peak memory, and as close to PyTorch's f32 gradients as PyTorch's own in bf16.
- Fuzzing of every reader of untrusted files, and continuous integration on four platforms plus a GPU runner (chapters 16 and 17).
