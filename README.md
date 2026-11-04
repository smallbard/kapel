# Kapel

A Rust framework for training and serving language models.

Kapel succeeds Smallem, an educational project that built an LLM in Rust and CUDA
from scratch: autograd, CPU and CUDA backends, pre-training, scaling laws,
post-training, quantized inference, an n-gram memory. Smallem was written to be
read; Kapel is written to be used, on other people's models and on other machines.

## What Kapel keeps from Smallem

- **A reference backend**, plain and in `f64`, against which every fast backend is
  tested, operation by operation and step by step.
- **Results that do not depend on the number of threads**: work is split according
  to shapes, never according to the hardware. Two runs give the same bits.
- **Measure before optimizing**: every optimization is justified by a profile, and
  its gain measured.
- **Code that can be audited**: few dependencies, short modules, documented kernels.

## What Kapel adds

- The models and formats of the ecosystem: Hugging Face configurations and weights,
  `tokenizer.json`, GGUF.
- A stable public API, typed errors, documentation in English.
- Several cards, several processor architectures.
- A server compatible with OpenAI's API.
- Continuous integration, with its tests on the card and its performance measures.

## Trying it

```
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B
```

`kapel run` downloads the model from the Hugging Face Hub (or reads a local
directory, or a llama.cpp GGUF file), then holds a conversation: one message a
line, an empty line to end. `--prompt TEXT` answers once, `--device cuda` computes
on the card (Kapel built with `--features cuda`), and `kapel --help` gives the other
options.

On the processor, a quantized model generates several times faster, since
generation is bound by reading the weights:

```
cargo run --release -p kapel-cli -- run Llama-3.2-1B-Instruct-Q4_K_M.gguf
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B --quantize int8
```

A GGUF file is read as it is (Q4_0 to Q8_0, Q2_K to Q6_K); `--quantize int8|int4`
quantizes a Hub model at loading, and `--quantize-output` its output table too.
`--byte-activations` also quantizes the activations to bytes, as llama.cpp does:
faster, and the results change a little.

On the card, Kapel keeps the keys and values in pages and replays each decoding
step as a CUDA graph. `--kv-cache bf16|f16|f8e4m3|f8e5m2` chooses the cache's type;
in `f8`, it takes half the space. The `throughput` example generates for several
requests at once, like `vllm bench throughput`:

```
cargo run --release -p kapel-cli --features cuda -- run Qwen/Qwen3-0.6B --device cuda
cargo run --release -p kapel-cli --features cuda --example throughput -- \
    MODEL_DIRECTORY --device cuda --sequences 64 --input 128 --output 256
```

On an RTX 3090, Llama 3.2 3B in bf16 generates 105 tokens/s for a single sequence,
and 3330 tokens/s for 64 sequences together, 90 % of vLLM's throughput.

`kapel run --draft SMALL_MODEL` decodes speculatively: the small model proposes a
few tokens, the large one checks them in a single pass, and the answer does not
change. On an RTX 3090, Qwen3 8B generates 1.7 times faster with Qwen3 0.6B as the
proposer; `--multi-token` uses MiMo's multi-token prediction layer, and
`--prompt-lookup` the tokens already written.

`kapel serve` serves a model behind an API compatible with OpenAI's
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streamed or not, with
constrained output: JSON schema, regular expression, grammar, choice), the requests
computed together:

```
cargo run --release -p kapel-cli --features cuda -- serve Qwen/Qwen3-0.6B --device cuda
curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

OpenAI's clients use it unchanged (`base_url`). `/metrics` gives the measures in
Prometheus's format, `/health` the server's state; the `Dockerfile` builds its image
for NVIDIA cards. On an RTX 3090, Llama 3.2 3B serves about 20 requests a second
(4300 tokens/s), and at 80 % of that throughput, the first token arrives within
70 ms for 99 % of the requests.

## Roadmap

Kapel is built in three stages:

1. **The foundation**: tensors, backends, reading models and tokenizers, tooling.
   Done.
2. **Inference**: loading, quantizing and serving models, on the processor and on
   the card. Done, but for the ARM kernels and the first releases on crates.io and
   PyPI.
3. **Training**: pre-training and fine-tuning models, bit-reproducible, on a single
   card as well as on several machines of several cards each. Done on one card:
   `kapel train` (AdamW, AdamW in 8 bits, Muon, Adafactor), LoRA and QLoRA, DPO,
   GRPO, scaling-law sweeps, `kapel export` to the Hub. Several ranks already
   compute the same bits as one (data parallelism, ZeRO 1 and 2, FSDP); what is
   left is their speed across cards: NCCL, communication overlapped with
   computation, Slurm.

*The Kapel Book* (`docs/book`) describes how Kapel works, and *The Kapel Guide*
(`docs/guide`) how to use it.

## License

MIT, see [`LICENSE`](LICENSE).
