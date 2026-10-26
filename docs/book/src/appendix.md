# Appendix

## Glossary

| Term | Meaning in Kapel |
| --- | --- |
| Acceptance rate | The share of the proposals of speculative decoding that the model accepts (`SpeculationStats::acceptance`) |
| Activations in bytes | Quantizing the input of a quantized product too, by blocks of 32, so that the product is a sum of products of bytes; chosen explicitly, as it changes the result a little |
| AWQ | Activation-aware weight quantization: input channels with large activations scaled up in the weights (and down in the operation before) so that rounding spares them, then blocks clipped (`kapel-quant`, chapter 13) |
| Activation checkpointing | Keeping only the inputs of a region of the forward pass and computing it again during the backward pass, memory traded for time (`Tape::checkpoint`) |
| Backend | A type implementing `Run` for every operation of the catalogue: `Reference`, `Cpu` or `Cuda` |
| Batch | Several sequences computed in the same pass, one row per new token of each, so the weights are read once for all of them (`Model::step_batch`, `kapel_infer::Batch`) |
| Block (quantized) | The unit a quantized type stores: a scale, sometimes a minimum, and packed codes for 32 or 256 elements |
| Bootstrap | Fitting a law again to draws of its points with replacement, for an interval of each constant (`scaling::bootstrap`) |
| Broadcast | Repeating a tensor along dimensions of size 1 or new leading dimensions, by giving them stride 0 |
| Budget (of a run) | The operations a run may spend, `training.budget`: its steps are the budget over the operations of a step (chapter 17) |
| Cache (KV) | The keys and values of the positions already computed, per layer, so each new token only computes itself; in Kapel, kept by pages (`KvCache`) |
| Cache by pages | A cache whose lines are handed out by pages of 16 positions from a common reserve, each sequence keeping the list of its pages: nothing is copied as a sequence grows, and its pages go back when it ends |
| Calibration | Texts run through a model to measure the inputs of its projections before quantizing it with data |
| Checkpoint (training) | A directory of a run with everything it needs to go on exactly: the parameters, the optimizer's moments, the position in the data, the configuration (`kapel_train::checkpoint`) |
| Conformity | Agreement of a backend with the reference within the tolerance of the data type, forward and backward |
| Constraint | The form an answer must take: a JSON schema, any JSON object, a regular expression, a Lark grammar, a list of texts; the tokens that cannot continue it get a logit of −∞ (`kapel_infer::Constraint`); OpenAI calls it a structured output |
| Contiguous | Stored in row-major order without gaps |
| Continuous batching | Serving requests that come and go in the same passes: each pass computes a token of every request in progress and pieces of new prompts, a request entering as soon as there is room and leaving as soon as it ends (`kapel_infer::Engine`) |
| CUDA graph | A pass recorded once and replayed in one launch, its kernels and buffers fixed; the `Record` trait, and `Model::record_step` |
| Cross-entropy (fused) | The loss of each row of logits, `ln Σ eˣ − x[target]`, with its gradient `softmax − onehot` written by hand, the softmax never formed (`CrossEntropy`); over a large vocabulary, by pieces of rows without the logits whole (`linear_cross_entropy`) |
| Decoding | Generating one token at a time after the prompt, bound by memory bandwidth |
| Determinism | The same bits whatever the number of threads, achieved by splitting work by shape |
| f8 | The 8-bit floats E4M3 and E5M2, used by Kapel for caches of keys and values half the size of bf16 |
| Fusing | Stacking the projections that read the same input into one weight, one product instead of several (`Model::fused`) |
| GGUF | llama.cpp's file format: a model's configuration, tokenizer and weights, quantized or not, in one file |
| GPTQ | Quantization column by column, the error of each column carried onto those not yet rounded through the inverse Hessian (`kapel-quant`, chapter 13) |
| GQA | Grouped-query attention: several query heads share one key and value head |
| Grid (sweep) | Runs of one base configuration, each with its changes, run by `kapel sweep` and noted in `results.tsv` (chapter 17) |
| Hessian (of a projection) | `H = Σ x xᵀ` over the inputs `x` of a projection: a change `δ` of a row of weights changes the outputs by `δ H δᵀ` |
| `infer` | The validation and shape computation shared by every backend for an operation |
| `Isa` | A proof that the processor has the instructions a code path uses |
| IsoFLOP curve | The losses of models of several sizes trained for one budget of operations: its lowest point is the best size for that budget (`scaling::isoflops`) |
| llguidance | The library of structured outputs that compiles constraints into a parser over the bytes of the tokens, and gives the set of tokens allowed next |
| Leaf | A value on a tape whose gradient is wanted: a parameter (`Tape::leaf`); a constant receives none |
| Log-likelihood | The sum of the logarithms of the probabilities of a continuation's tokens after a context; what `kapel eval` ranks answers by (`Scorer`) |
| MFU | Model FLOP utilization: the operations a step's definition asks for, over its time and the card's peak (`kapel_train::flops`) |
| Micro-kernel | The innermost routine of the matrix product, holding a block of C in registers |
| Model card | The `README.md` of a model on the Hub: what it is, how it was trained, how to load it; `kapel export` writes one from a run's configuration and log (`kapel_train::export`) |
| Muon | An optimizer of matrices: a momentum made orthogonal by Newton-Schulz's iterations, scaled to the size of an AdamW step (`kapel_train::optim::Muon`) |
| Multi-token prediction | A layer trained with the model to predict the token after the next from the model's hidden state and the next token (MiMo, DeepSeek-V3); a proposer that costs one block (`Model::multi_token`) |
| Packing | Copying a block of an operand into the contiguous order the micro-kernel reads |
| Padding rows | Rows added to a smaller batch so that it fits a recorded pass of more rows; each writes a scratch line of its own and is read by no one |
| Padding (of a vocabulary) | Rows of zeros added to an output table up to a multiple of 64 for the products, the logits of the padding not read |
| Paged attention | Attention of new rows over a cache by pages, each row knowing its sequence and how many positions it sees (`PagedAttention`) |
| Parity | Agreement with the Python libraries `transformers`, `tokenizers` or llama.cpp on real models and files |
| Perplexity | The exponential of the mean negative log-likelihood of a text: how a quantization is judged |
| Prefill | Computing the whole prompt at once, before decoding |
| Prefix sharing | Sequences that begin with the same tokens sharing the cache pages of that beginning, indexed by the tokens they hold and counted by references (`KvCache::with_prefix_sharing`) |
| Prompt lookup | Proposing the tokens that followed the last ones where they last appeared in the sequence: speculative decoding without a second model (`Draft::Ngram`) |
| Proposer (draft) | What guesses the next tokens in speculative decoding: a smaller model, prompt lookup, or a multi-token layer (`Draft`) |
| PTX | NVIDIA's intermediate assembly; the thread kernels are checked in as PTX |
| Recipe | A configuration of `kapel train` in `recipes/`, with the commands that prepare its data, its expected results and its cost, run again in CI for a few steps |
| Reference | The `f64` backend that defines correct results |
| RoPE | Rotary position embeddings: pairs of components rotated by an angle proportional to the position |
| Rounding to the nearest (RTN) | Quantizing each block alone, without data, as `llama-quantize` does |
| Server-sent events | The streaming format of OpenAI's API over HTTP: one `data:` line per chunk of the answer, then `data: [DONE]` |
| Site | A group of projections that read one input (`Attention`, `Output`, `Mlp`, `Down`), as `Model::block` shows it |
| Softcap | A smooth bound on scores or logits, `cap · tanh(x / cap)`, used by Gemma 2 |
| Sparse gradient | The gradient of a table of which few rows were read, kept as those rows and what to add to them (`Grad::Rows`), and the AdamW that updates only them |
| Speculative decoding | A proposer guesses several tokens, the model checks them in one pass, accepting each by the rule that keeps its distribution exactly (`Pipeline::speculate`) |
| Stable ABI (abi3) | CPython's interface that does not change between versions: one wheel of the Python module serves every Python from 3.9 |
| Stride | The distance in the buffer between two consecutive elements of a dimension |
| Suspension (preemption) | Taking the pages of the request admitted last when the cache runs out, to compute its tokens again later |
| Tape | The record of a computation, run backwards to get the gradients of a loss (`kapel_train::Tape`) |
| Thread kernel / tile kernel | A CUDA kernel with one thread per element, written for cuda-oxide / a kernel over tiles of a matrix, written for cutile-rs |
| Throughput | Tokens generated per second for a batch of requests, compared with vLLM's `vllm bench throughput` |
| Time to first token | From the arrival of a request at the server to its first token: what a user waits before the answer starts |
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
| Batches and the engine of many requests | `crates/kapel-infer/src/{engine,batch}.rs`, `crates/kapel-infer/tests/engine.rs` |
| Constraints | `crates/kapel-infer/src/constraint.rs` |
| Speculative decoding | `crates/kapel-infer/src/{speculative,multi_token}.rs`, `crates/kapel-infer/tests/speculative.rs`, `crates/kapel-cli/examples/speculative.rs` |
| MiMo's multi-token layer, its reference | `crates/kapel-models/src/model.rs` (`multi_token`), `tools/parity/multi_token_reference.py` |
| The server | `crates/kapel-serve/src/{lib,api,server,worker,metrics,log}.rs`, `crates/kapel-serve/tests/server.rs` |
| Load on a server, OpenAI's client | `crates/kapel-serve/examples/load.rs`, `tools/serve/openai_client.py` |
| Container image | `Dockerfile`, `.dockerignore` |
| Profiling, perplexity and throughput | `crates/kapel-cli/examples/{profile,perplexity,throughput}.rs` |
| Quantizing with data | `crates/kapel-quant/src/{lib,gptq,awq,linalg}.rs`, `crates/kapel-cli/src/quantize.rs` |
| Block-by-block passes | `crates/kapel-models/src/model.rs` (`embed`, `block`, `Site`) |
| Writing GGUF | `crates/kapel-hub/src/gguf.rs` (`write`), `crates/kapel-models/src/gguf.rs` (`to_gguf`), `crates/kapel-infer/src/gguf.rs` (`save_gguf`) |
| Evaluation | `crates/kapel-infer/src/score.rs`, `crates/kapel-cli/src/eval.rs`, `tools/eval/` |
| The tape, losses, optimizers | `crates/kapel-train/src/{tape,loss,optim}.rs`, `crates/kapel-train/tests/{tape,optim}.rs` |
| Decoders on a tape | `crates/kapel-train/src/model.rs`, `crates/kapel-train/tests/{model,cuda}.rs` |
| The fused cross-entropy | `crates/kapel-ops/src/loss.rs`, `crates/kapel-{ref,cpu}/src/loss.rs`, `crates/kapel-cuda/src/ops.rs`, `kernels/src/lib.rs` (`cross_entropy_rows`) |
| The training loop, its configuration and checkpoints | `crates/kapel-train/src/{trainer/mod,config,schedule,checkpoint,flops}.rs`, `crates/kapel-cli/src/train.rs`, `crates/kapel-train/tests/trainer.rs`, `recipes/gpt2-124m/` |
| Preparing and reading the data | `crates/kapel-train/src/{prepare,dataset,data}.rs`, `crates/kapel-cli/src/prepare.rs`, `crates/kapel-train/tests/prepare.rs`, `crates/kapel-train/examples/loader.rs` |
| Optimizers' steps in place | `crates/kapel-ops/src/update.rs`, `crates/kapel-{ref,cpu}/src/update.rs`, `crates/kapel-cuda/src/ops.rs` (`adamw`) |
| Training measured against PyTorch, the time of each operation | `crates/kapel-train/examples/{gradients,steps}.rs`, `tools/train/gradients.py`, `crates/kapel-train/src/profile.rs` |
| Conversations and their masks | `crates/kapel-train/src/chat.rs`, `crates/kapel-train/tests/prepare.rs` |
| LoRA and QLoRA, PEFT's files, merging | `crates/kapel-train/src/lora.rs`, `crates/kapel-cli/src/merge.rs`, `crates/kapel-train/tests/lora.rs`, `crates/kapel-train/examples/lora.rs`, `tools/train/peft_parity.py` |
| DPO, GRPO, rewards, tasks | `crates/kapel-train/src/{preference,grpo,rewards}.rs`, `crates/kapel-train/src/trainer/post.rs`, `crates/kapel-train/tests/post.rs`, `crates/kapel-train/examples/tasks.rs` |
| Budgets of operations, grids of runs, laws of scale | `crates/kapel-train/src/{flops,sweep,scaling}.rs`, `crates/kapel-cli/src/sweep.rs`, `crates/kapel-train/tests/flops.rs`, `recipes/scaling-smallem/`, `tools/train/smallem_tokens.py` |
| Logs for TensorBoard and Weights & Biases | `crates/kapel-train/src/tensorboard.rs`, `tools/train/wandb_log.py` |
| Post-training measured against TRL | `tools/train/{trl_sft,trl_dpo,gsm8k,ultrafeedback}.py`, `recipes/{sft-gsm8k,sft-lora-gsm8k,dpo-ultrafeedback,grpo-gsm8k}/` |
| Exporting a run for the Hub | `crates/kapel-train/src/export.rs`, `crates/kapel-cli/src/export.rs`, `crates/kapel-cli/tests/export.rs` |
| The recipes, their data, their measures in CI | `recipes/`, `recipes/README.md`, `tools/train/fineweb_gpt2.py`, `tools/bench/recipes.py`, `docs/guide/src/recipes.md` |
| One model type for the bindings | `crates/kapel-runtime/src/lib.rs` |
| The C API | `crates/kapel-capi/include/kapel.h`, `crates/kapel-capi/src/lib.rs`, `crates/kapel-capi/examples/c/` |
| The Python module | `crates/kapel-python/src/lib.rs`, `crates/kapel-python/kapel.pyi`, `crates/kapel-python/tests/` |
| Parity references | `tools/parity/*.py`, data in `data/parity/` |
| Fuzzing | `fuzz/fuzz_targets/`, `fuzz/regressions/` |
| Rules and plans | `CONTRIBUTING.md`, `SECURITY.md`, `PLAN-*.md` (in French), `CHANGELOG.md` |
