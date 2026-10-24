# Your first model

## Ask a question

```sh
kapel run Qwen/Qwen2.5-0.5B-Instruct --prompt "What is the capital of France? Answer in one sentence."
```

```
kapel: fetching Qwen/Qwen2.5-0.5B-Instruct from the Hub…
The capital of France is Paris.
```

The first time, Kapel downloads the model from the Hugging Face Hub (about 1 GB for this one); the next times, it reads it from its cache. The prompt goes through the model's chat template, as a message of the user, and the answer is printed as it is generated.

Without `--prompt`, `kapel run` holds a conversation: type a message, press Enter, read the answer; an empty line ends it. `--system "You are a pirate."` adds a system message first.

## Which models

`MODEL` is any of:

- **A repository of the Hub**, `owner/name`: `Qwen/Qwen3-1.7B`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` (gated: see [Installing Kapel](installation.md#where-kapel-keeps-models)). `--revision` takes a branch, tag or commit.
- **A model directory** on disk, as `transformers` saves it: `config.json`, `model.safetensors` (or shards), `tokenizer.json`.
- **A GGUF file** of llama.cpp, for example one of the `-GGUF` repositories of the Hub, downloaded by hand: `kapel run Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

Kapel reads the architectures Llama, Mistral, Qwen 2, Qwen 3, MiMo, Gemma 2, Phi and GPT-2 from model directories, and Llama, Mistral, Qwen 2 and Qwen 3 from GGUF files, in the quantized types Q4\_0 to Q8\_0 and Q2\_K to Q6\_K. Another architecture is refused with an error that names it.

Qwen 3 models think before they answer, in a `<think>` block; that is the model, not Kapel. For short answers, pick an instruct model that does not, or ask Qwen 3 to skip it with `/no_think` at the end of your message.

## Go faster

On the processor, generating is limited by how fast the weights are read from memory. Smaller weights go faster:

```sh
kapel run Qwen/Qwen2.5-1.5B-Instruct --quantize int8      # weights in 8 bits, almost lossless
kapel run Qwen/Qwen2.5-1.5B-Instruct --quantize int4      # 4 bits, faster still, loses a little
kapel run Llama-3.2-1B-Instruct-Q4_K_M.gguf               # already quantized
```

`--stats` prints the speed of the prompt and of the answer after each answer. `--byte-activations` also rounds the activations to bytes inside the quantized products, as llama.cpp does: faster, and the answers can change slightly.

On an NVIDIA card (Kapel built with `--features cuda`):

```sh
kapel run Qwen/Qwen3-8B --device cuda
```

Weights are in bf16 on the card by default (`--dtype` changes it), and `--kv-cache f8e4m3` halves the memory of the cache of keys and values, for long conversations.

**Speculative decoding** makes a large model faster without changing its answers: a small model of the same family proposes a few tokens, and the large one checks them in one pass.

```sh
kapel run Qwen/Qwen3-8B --device cuda --draft Qwen/Qwen3-0.6B
```

On an RTX 3090 this is 1.7 times faster. `--prompt-lookup` proposes from the conversation itself (for answers that copy their prompt: code edits, summaries), and `--multi-token` uses the multi-token layer of MiMo models.

## Choose how tokens are drawn

By default Kapel samples as the model's `generation_config.json` says, as `transformers` does. `--greedy` always takes the most likely token; `--temperature`, `--top-k`, `--top-p`, `--min-p` and `--repetition-penalty` change the sampling; `--seed` fixes the draws, so the same command gives the same answer. `--max-tokens` limits the answer (512 by default), and `--stop TEXT` ends it before `TEXT`.

`kapel --help` lists every option.
