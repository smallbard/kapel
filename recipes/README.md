# Recipes

Each directory holds a configuration of `kapel train` (and what it reads), with
the commands that prepare its data in its header. The results below were
measured on one RTX 3090 (24 GB) and a Ryzen 7 7700, by running each recipe as
its header says; the same commands on the same card give the same numbers, the
times within a few percent. The guide's page [Recipes](../docs/guide/src/recipes.md)
explains them.

| Recipe | What it trains | Expected | Time | Memory |
| --- | --- | --- | --- | --- |
| `gpt2-124m` | GPT-2 124M from scratch on FineWeb, llm.c's settings | validation loss 6.170 at step 250 (training 6.207) | 43 min for 250 steps; the whole run, 19,073 steps (10 B tokens), about 53 h | 7.9 GiB |
| `llama-1b` | Llama 3.2 1B's architecture from scratch on FineWeb, Adafactor | validation loss 8.210 at step 50 (training 8.122) | 1 h 36 for 50 steps; the whole run, 1,300 steps (one pass over 698 M tokens), about 41 h | 16.1 GiB |
| `sft-gsm8k` | Qwen 3 0.6B fine-tuned on GSM8K, every weight | GSM8K 50.0 %; loss 0.408 (mean of steps 401–450) | 7 min | 21.3 GiB |
| `sft-lora-gsm8k` | the same with LoRA | GSM8K 48.3 % (TRL: 47.5 %); loss 0.419 | 7 min | 13.3 GiB |
| `dpo-ultrafeedback` | Qwen 3 0.6B by DPO with LoRA on 4,000 pairs of UltraFeedback | 500 held-out pairs at step 500: loss 0.627, accuracy 64.8 %, margin 0.47 (TRL: 0.622, 65.1 %, 0.49) | 26 min | 13.3 GiB |
| `grpo-gsm8k` | Qwen 3 0.6B by GRPO with LoRA on GSM8K's problems | GSM8K 64.9, 67.1, 68.4, 69.3 % at steps 50 to 200 (61.2 % before) | 1 h 34 | 11.5 GiB |
| `scaling-smallem` | Smallem's five sizes at 10¹⁶ operations (`kapel sweep`) | lowest point 14.4 M parameters at 0.894 nats per byte (seed 43: 15.3 M) | 1 h 13 | 0.5–1.5 GiB a run |

Qwen 3 0.6B answers 61.2 % of GSM8K's test without thinking before any
training: fine-tuning on GSM8K's own answers, which reason briefly, lowers it;
GRPO raises it.

`scaling-smallem` reads Smallem's tokens, which are not published.

To publish what a recipe trained: `kapel export RUN_DIR --output DIR`, then
`hf upload OWNER/NAME DIR`.

Each recipe's speed, peak memory and loss on a few steps are tracked by the
workflow `measures` (`tools/bench/recipes.py`).
