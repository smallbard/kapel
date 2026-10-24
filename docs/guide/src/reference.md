# Reference

## Commands

| Command | Does | Help |
| --- | --- | --- |
| `kapel run MODEL` | answers a prompt, or holds a conversation | `kapel --help` |
| `kapel serve MODEL` | serves the model behind OpenAI's API | `kapel --help` |
| `kapel quantize MODEL --output PATH` | quantizes and writes a model | `kapel quantize --help` |
| `kapel eval MODEL --tasks FILE` | scores tasks of lm-evaluation-harness | `kapel eval --help` |
| `kapel prepare INPUT... --tokenizer PATH --output DIR` | tokenizes Parquet, JSONL or text files into shards for training, the exact duplicates left out (*The Kapel Book*, section 15.13) | `kapel prepare --help` |
| `kapel train CONFIG.toml` | trains a model as a TOML file describes it (`recipes/gpt2-124m/train.toml`; *The Kapel Book*, chapter 15) | `kapel train --help` |

## Loading options

Every command that loads a model takes these; the bindings take them in the same words.

| Option | Values | Default |
| --- | --- | --- |
| `--device` | `cpu`, `cuda`, `cuda:N` | `cpu` |
| `--dtype` | `f32`, `bf16`, `f16`: the type of the float weights | `f32` on the processor, `bf16` on a card |
| `--quantize` | `int8`, `int4`, `int8/G`, `int4/G`, `q8_0`, `q4_0` | none |
| `--quantize-output` | also quantize the output table | no |
| `--kv-cache` | `f32`, `bf16`, `f16`, `f8e4m3`, `f8e5m2`: the type of the cache | the type of the keys |
| `--byte-activations` | on the processor, round activations to bytes in quantized products | no |
| `--revision` | a branch, tag or commit of a Hub repository | `main` |

## Models

| | From a model directory or the Hub | From a GGUF file |
| --- | --- | --- |
| Architectures | Llama, Mistral, Qwen 2, Qwen 3, MiMo, Gemma 2, Phi, GPT-2 | Llama, Mistral, Qwen 2, Qwen 3 |
| Weights | safetensors, one file or shards; f32, bf16, f16, and Kapel's int8 and int4 | F32, F16, BF16, Q4\_0, Q4\_1, Q5\_0, Q5\_1, Q8\_0, Q2\_K to Q6\_K |
| Tokenizer | `tokenizer.json` (byte-level BPE), chat template from `tokenizer_config.json` or `chat_template.jinja` | from the file's metadata |

## Environment variables

| Variable | Effect |
| --- | --- |
| `KAPEL_CACHE` | where models from the Hub are kept (default `~/.cache/kapel/hub`) |
| `HF_TOKEN` | the Hub token for gated models (sent to the Hub only) |
| `KAPEL_DATA_CACHE` | where `kapel train` keeps the shards of a remote dataset while it reads them (default `~/.cache/kapel/data`) |
| `KAPEL_OFFLINE=1`, `HF_HUB_OFFLINE=1` | never contact the Hub; use the models already downloaded |
| `KAPEL_THREADS` | the threads computing on the processor (default one per physical core) |
| `CUDA_HOME` | the CUDA toolkit, when building with the `cuda` feature |

## Measured performance

See [Measures](measures.md): the numbers, how they are measured, and how to measure your own machine.
