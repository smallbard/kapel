# Quantizing a model

Quantizing stores a model's weights in 8 or 4 bits instead of 16. The model takes two to four times less memory, and generates faster where reading the weights is what limits it: on the processor, and for one user on a card. It also answers a little worse; how much depends on how the weights are rounded.

## At loading, without a file

The quickest way: quantize as the model is loaded.

```sh
kapel run Qwen/Qwen2.5-1.5B-Instruct --quantize int8
kapel serve Qwen/Qwen2.5-1.5B-Instruct --quantize int4
```

Each block of weights is rounded to the nearest value of the format. int8 is almost lossless; int4 loses a little more. Nothing is written: the model is quantized again at each start, which takes a few seconds.

| Format | Bits per weight | Notes |
| --- | --- | --- |
| `int8` (`int8/G`) | 8, plus a scale per group of 128 (or `G`) | Kapel's own; almost lossless |
| `int4` (`int4/G`) | 4, plus a scale per group of 64 (or `G`) | Kapel's own |
| `q8_0` | 8.5 | llama.cpp's |
| `q4_0` | 4.5 | llama.cpp's |

`--quantize-output` also quantizes the output table (the embeddings, for models that tie them), which is large in models with a big vocabulary.

## With data, into a file: kapel quantize

Rounding each weight alone wastes accuracy. `kapel quantize` measures how the model actually uses its weights on a sample of text, and rounds where it costs least. Then it writes the model once, to a GGUF file that Kapel and llama.cpp read, or to a model directory.

```sh
kapel quantize meta-llama/Llama-3.2-1B --format q4_0 --method gptq \
    --calibration wiki.train.raw --output llama-3.2-1b-q4_0.gguf
kapel run llama-3.2-1b-q4_0.gguf
```

The **calibration text** is any plain text file representative of what the model will read: a few megabytes of Wikipedia (`wiki.train.raw` of WikiText-2, say), of code for a code model, of your documents. Kapel draws `--samples` windows (128) of `--length` tokens (512) from it.

| `--method` | What it does | Time for a 1B model on an RTX 3090 |
| --- | --- | --- |
| `rtn` | rounds each block to the nearest, without data (the default; same bytes as llama.cpp's `llama-quantize`) | seconds |
| `gptq` | rounds column by column and corrects the next columns for each error, by the statistics of the calibration (Frantar et al., 2022) | 4 minutes |
| `awq` | scales up the weights of the channels with large activations so that rounding spares them, and clips outliers (Lin et al., 2023) | 47 minutes |
| `awq+gptq` | both | 50 minutes |

**Use `gptq`.** On Llama 3.2 1B in Q4\_0, the increase in perplexity over the original model on WikiText-2 was:

| Quantization | Perplexity | Increase |
| --- | --- | --- |
| original, f16 | 10.19 | — |
| llama.cpp's `llama-quantize` | 11.50 | +1.31 |
| llama.cpp with an importance matrix | 10.93 | +0.74 |
| **Kapel, `gptq`** | **10.61** | **+0.41** |
| Kapel, `awq` | 11.01 | +0.82 |

`--device cuda` computes the calibration on the card (much faster than the processor). The formats are `q4_0`, `q4_1` and `q8_0` (GGUF), `int8` and `int4` (Kapel's, written to a model directory), and `f16`, `bf16` or `f32` to convert a model to GGUF without quantizing. `--quantize-output` rounds the output table too. `kapel quantize --help` lists the rest.

GGUF files are written for Llama, Mistral, Qwen 2 and Qwen 3. GPTQ and AWQ work on the models of Llama's family (Llama, Mistral, Qwen, Gemma); other models are rounded.

## Check what you lost

Measure the quantized model against the original before using it: see [Evaluating a model](evaluation.md).
