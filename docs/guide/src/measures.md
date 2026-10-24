# Measures

The numbers below are measured by the repository's own programs, on one machine: an AMD Ryzen 7 7700 (8 cores, AVX-512, DDR5) and an NVIDIA RTX 3090 (24 GB). Every change to Kapel's main branch measures the first table again on that machine and fails if a speed drops by more than 10 % or a perplexity moves (the workflow `measures`), so they stay true.

## Tracked on every change

`tools/bench/run.sh all` (the example `bench` of `kapel-cli`, on each setup), 23 October 2026:

| Setup | Prompt (tokens/s) | Decoding one sequence (tokens/s) | Batch of 64 (tokens/s) | Perplexity |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct, processor, f32 | 812 | 26.1 | | 15.21 |
| Qwen2.5-0.5B-Instruct, processor, Q4\_0 | 704 | 150.8 | | 18.54 |
| Llama-3.2-1B-Instruct, processor, Q4\_0 | 333 | 70.3 | | 17.02 |
| Qwen2.5-0.5B-Instruct, card, bf16 | 40,851 | 437.1 | | 15.20 |
| Llama-3.2-3B-Instruct, card, bf16 | 8,646 | 113.2 | 3,492 | 10.93 |
| Llama-3.2-3B-Instruct, card, Q4\_0 | 4,411 | 228.4 | | 11.71 |

**Method.** *Prompt*: 512 random tokens computed at once. *Decoding*: 128 tokens generated one at a time after a prompt of 32, greedy and past the end tokens, the first token (which comes with the prompt) aside. *Batch*: 64 sequences of 128 random tokens each continued by 256 tokens, all in one batch. Each speed is the best of three runs after one that warms up. *Perplexity*: eight chunks of 512 tokens of wikitext-2's test set, each scored on its second half, as llama.cpp's `llama-perplexity` does. On the processor, weights and activations are in f32 unless quantized; Q4\_0 is rounding at loading, the output table included; on the card, weights are in bf16. Random prompts make the speeds independent of the text; perplexity, which depends only on the arithmetic, shows when the numbers change.

To measure your own machine:

```sh
tools/bench/run.sh cpu mine.json            # or cuda, or all
python3 tools/bench/compare.py tools/bench/baselines/ryzen7-7700-rtx3090.json mine.json
```

## Against other engines

Measured when each feature landed, with the commands of `tools/parity/README.md`:

| Measure | Kapel | Other |
| --- | --- | --- |
| Llama 3.2 1B Q4\_K\_M, processor, decoding one sequence | 57.4 tokens/s | llama.cpp 62.6 |
| Llama 3.2 3B bf16, card, 64 sequences of 128 + 256 tokens | 3,328 tokens/s | vLLM 0.32: 3,710 |
| Llama 3.2 3B bf16, card, one sequence | 105 tokens/s | vLLM 0.32: 108.5 |
| Qwen3 8B bf16, card, greedy, Qwen3 0.6B proposing | 85.6 tokens/s, 1.72 times alone | |
| `kapel serve`, Llama 3.2 3B, card | 20 requests/s; 99 % of first tokens within 63 ms at 80 % load | |
| WikiText perplexity of SmolLM2-135M against lm-evaluation-harness | 21.42494 | 21.42474 |
| Llama 3.2 1B in Q4\_0, perplexity lost | +0.41 (GPTQ) | llama.cpp: +0.74 with an importance matrix, +1.31 without |

*The Kapel Book* gives the details of each: chapters 6 and 7 for the backends, 11 and 12 for generation and serving, 13 for quantization and evaluation.
