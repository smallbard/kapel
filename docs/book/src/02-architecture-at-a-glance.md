# 2. Architecture at a glance

Kapel is a Cargo workspace of seventeen crates arranged in layers: a tensor type at the bottom, a catalogue of operations above it, interchangeable backends that implement the catalogue, and above them the crates that read files, turn text into tokens, run models and generate text. Each layer depends only on the layers below it.

## The layers

![Kapel's crates by layer: the operation catalogue is the contract between models and backends](architecture.svg)

Models sit above the catalogue, never above a particular backend: swapping the reference for the CPU or the GPU changes no model code.

| Layer | Crate | Role | Depends on (Kapel crates) |
| --- | --- | --- | --- |
| Data | `kapel-core` | Tensors, data types, layouts, storage, devices, errors | none |
| Contract | `kapel-ops` | The operation catalogue: shapes, types and checks shared by every backend | core |
| Backends | `kapel-ref` | The reference: every operation in `f64`, and the conformity suite | core, ops |
|  | `kapel-cpu` | The fast CPU backend: AVX-512, AVX2 and portable code | core, ops |
|  | `kapel-cuda` | The CUDA backend, behind the `cuda` feature | core, ops, cuda-abi |
|  | `kapel-cuda-abi` | Types shared by host code and device kernels | none |
| Files | `kapel-hub` | safetensors, GGUF, checkpoints, Hub downloads | core |
| Text | `kapel-tokenizers` | `tokenizer.json`, BPE, decoding, chat templates | none |
| Models | `kapel-models` | Configurations and the transformer forward pass | core, ops, hub |
| Applications | `kapel-infer` | Generation: pipeline, engine of many requests, sampling, constraints, streaming text | core, ops, hub, models, tokenizers |
|  | `kapel-serve` | The server of OpenAI's API: HTTP, streams, metrics | core, ops, models, infer, tokenizers |
|  | `kapel-quant` | Quantizing with calibration texts: GPTQ, AWQ | core, ops, models |
|  | `kapel-cli` | The `kapel` command: `run`, `serve`, `quantize`, `eval`, `train` | all of the above |
| Bindings | `kapel-runtime` | A model on whichever device, behind one type | core, ops, cpu, cuda, hub, models, tokenizers, infer, serve |
|  | `kapel-capi` | The C library `libkapel` and its header `kapel.h` | runtime |
|  | `kapel-python` | The Python module `kapel` (PyO3) | runtime |
|  | `kapel-train` | Training: the tape of gradients, losses, optimizers, the decoders on a tape, the training loop and its checkpoints, the preparation and reading of the data (chapter 15); conversations, LoRA and QLoRA, DPO, GRPO and its rewards, tasks evaluated along a run (chapter 16); budgets of operations, grids of runs and the laws of scale fitted to them (chapter 17) | core, ops, hub, models, tokenizers, infer, cpu (cuda with its feature) |

Outside the workspace, `kernels/` holds the simple CUDA kernels written in Rust for the nightly compiler. They are compiled once to PTX and the result, `crates/kapel-cuda/kernels.ptx`, is checked in, so the published crates build on stable Rust. `fuzz/` holds the fuzzing targets, `tools/parity/` the Python scripts that produce reference outputs from `transformers`, and `tools/eval/` the script that exports the tasks of lm-evaluation-harness for `kapel eval`.

Two dependency choices are deliberate. `kapel-tokenizers` depends on no other Kapel crate, so it can be used alone. And no crate above the backends names a backend: models and the generation engine are generic over `B: Backend`, so the same model code runs on the reference, the CPU or the GPU. Only the applications at the top choose one: `kapel-cli` from its `--device` option, and `kapel-runtime`, which holds a pipeline of either backend behind one type, for the bindings, whose callers choose at run time (chapter 14).

## One trait, three backends

The central idea is the split between *what* an operation computes and *how*. `kapel-ops` defines each operation as a small struct with a documented mathematical definition and an `infer` method that checks inputs and computes the output's shape and type. A backend implements `Run<Op>` for each operation: a `forward` and a `backward`. The trait `Backend` is simply the set of all twenty-three `Run` implementations, plus two traits for generation, `Write`, which writes rows into a cache in place, and `Record`, which records a pass once and replays it (a CUDA graph on the card), and one for training, `Update`, the optimizers' steps in place.

Because every backend calls the same `infer` first, every backend accepts and rejects the same calls with the same errors. Because the reference defines the numbers, the CPU and CUDA backends can be tested operation by operation against it, forward and backward, with tolerances that depend only on the data type.

## The life of a request

Follow `kapel run Qwen/Qwen3-0.6B --prompt "Hello"` from the command line to the screen.

1. **Find the files.** `kapel-hub` resolves the repository on the Hugging Face Hub, downloads the configuration, tokenizer and weights into a cache keyed by commit, and verifies checksums. A local directory skips this step.
2. **Open the checkpoint.** `Checkpoint::open` maps the safetensors files into memory. No weight is copied yet: tensors point into the mapped file.
3. **Build the model.** `Model::load` reads `config.json` into a `Config` and looks up each weight by name. `Model::upload` moves the weights to the backend's device, and `Model::cast` converts them, for instance to bf16 on a GPU.
4. **Prepare the text.** `Tokenizer::from_file` reads `tokenizer.json`; `ChatTemplate::from_dir` reads the Jinja chat template. The user's message is rendered by the template, then encoded into token ids.
5. **Prefill.** `Model::step` runs the whole prompt at once through every layer and writes each layer's keys and values into a `Cache`, kept by pages of 16 positions. It returns the logits of the last position.
6. **Decode.** A token is chosen from the logits, greedily or by sampling with the filters of `generation_config.json`; greedy choice runs on the backend (`ArgMax`), so only the token comes back. It is fed back one position at a time, each step reading the cache through `PagedAttention`. On a backend that records, the step is recorded on the first token and replayed for the next ones.
7. **Stream the text.** Each token is decoded to text. Incomplete UTF-8 characters, and text that might be the start of a stop sequence, wait until they are settled, so the text that reaches the screen is never taken back.

Every step of the forward pass is an operation from the catalogue, run by the chosen backend. The rest of the book walks through these layers from the bottom up.

## The life of a served request

`kapel serve` puts the same pipeline behind HTTP (chapter 12). Two kinds of threads share it: the HTTP threads of tokio, and one thread that owns the engine and is the only one to touch the device.

1. **Read.** An axum handler receives `POST /v1/chat/completions`, reads the JSON, checks the model's name and the limits, and turns the OpenAI fields into a `Request` (sampling, stop sequences, constraint).
2. **Prepare.** On a blocking thread of tokio, the chat template renders the messages, the tokenizer encodes them, and `Pipeline::prepare` compiles the constraint, if any. Nothing of this waits for the engine.
3. **Submit.** The prepared request, with the sending end of a channel, goes to the engine's thread.
4. **Serve.** The engine's thread admits it between two passes (`Engine::add_prepared`) once the cache has pages for it. Each pass computes a token of every request that decodes, and pieces of new prompts (chapter 11).
5. **Answer.** Each token goes back through its request's channel; the handler sends it to the client as a server-sent event, or gathers the whole answer. A client that closes its connection closes the channel, and the engine cancels the request before its next pass.
