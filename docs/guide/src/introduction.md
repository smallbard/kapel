# Introduction

Kapel runs language models of half a billion to a few billion parameters on your own machine, on its processor or on an NVIDIA card. It reads the models of the Hugging Face Hub and the GGUF files of llama.cpp, gives the same tokens as the Python library `transformers`, and serves them behind an HTTP API that OpenAI's clients use unchanged.

This guide is for people who want to use Kapel: run a model, serve it, quantize it, measure it, or call it from a program. It takes you from nothing to a served model in about ten minutes, most of them spent downloading.

| You want to | Read |
| --- | --- |
| Install the `kapel` command | [Installing Kapel](installation.md) |
| Talk to a model in the terminal | [Your first model](first-model.md) |
| Give applications an OpenAI-compatible endpoint | [Serving a model](serving.md) |
| Make a model smaller and faster | [Quantizing a model](quantization.md) |
| Know what a model or a quantization is worth | [Evaluating a model](evaluation.md) |
| Use Kapel from Python, C or Rust | [Integrating Kapel](integration.md) |
| Look up an option | [Reference](reference.md) |

**What Kapel is good at.** One user or many, on modest hardware. On a Ryzen 7 7700, a quantized Llama 3.2 1B generates at 90 % of llama.cpp's speed; on an RTX 3090, Kapel serves Llama 3.2 3B at about 20 requests per second, 90 % of vLLM's throughput for batches. Results do not depend on the number of threads: the same command gives the same answer twice.

**What it is not (yet).** Kapel runs dense transformer models: Llama, Mistral, Qwen 2 and 3, Gemma 2, Phi, GPT-2 and MiMo. It does not run mixtures of experts, models that read images or sound, or one model over several cards, and it does not train models.

To understand how Kapel works inside, or to change it, read *The Kapel Book* (`docs/book`) instead.
