# Recipes

The repository's `recipes/` directory holds runs that are known to work: a configuration each, the commands that prepare its data in its header, and the results it gives. Each was run from this page on one RTX 3090 (24 GB) with a Ryzen 7 7700; the same commands on the same card give the same numbers, the times within a few percent. The workflow `measures` runs a few steps of each on every change to Kapel, and fails if a speed drops, a memory grows or a loss moves.

The Python tools need `huggingface_hub`, `pyarrow` and `transformers` (`pip install huggingface_hub pyarrow transformers`); `hf` comes with `huggingface_hub`. Commands run from the repository's root.

| Recipe | Expected | Time | Memory |
| --- | --- | --- | --- |
| [`gpt2-124m`](#gpt-2-124m) | validation loss 6.170 at step 250 | 43 min | 7.9 GiB |
| [`llama-1b`](#llama-1b) | validation loss 8.210 at step 50 | 1 h 36 | 16.1 GiB |
| [`sft-gsm8k`](#supervised-fine-tuning-on-gsm8k) | GSM8K 50.0 % | 5 min 24 | 15.9 GiB |
| [`sft-lora-gsm8k`](#supervised-fine-tuning-on-gsm8k) | GSM8K 48.0 % | 4 min 25 | 7.9 GiB |
| [`dpo-ultrafeedback`](#dpo-on-ultrafeedback) | held-out pairs: loss 0.629, accuracy 65.0 % | 14 min | 11.5 GiB |
| [`grpo-gsm8k`](#grpo-on-gsm8k) | GSM8K from 61.2 % to 69.3 % | 1 h 34 | 11.5 GiB |
| [`scaling-smallem`](experiments.md#the-recipe) | lowest point 14.4 M parameters | 1 h 13 | |

## GPT-2 124M

GPT-2 from scratch on FineWeb in GPT-2's tokens, with llm.c's settings: AdamW at 6e-4 after 700 steps of warmup, steps of 524,288 tokens (32 passes of 16 × 1,024).

```sh
python tools/train/fineweb_gpt2.py data/fineweb --train 2       # 600 MB
kapel train recipes/gpt2-124m/train.toml --steps 250
```

Expected: the validation loss 6.170 at step 250, on the first 10 million tokens of FineWeb's validation, the training loss 6.207; 52,400 tokens a second (63 % of the card's peak in bf16), 7.9 GiB, 43 minutes. The whole run, 19,073 steps over 10 billion tokens (`python tools/train/fineweb_gpt2.py data/fineweb --train 100`, 20 GB), would take about 53 hours: with these settings, the validation loss of modded-nanogpt's speedrun, 3.28, takes about that many tokens.

## Llama 1B

Llama 3.2 1B's architecture and tokenizer, from scratch, on 698 million tokens of FineWeb that `kapel prepare` makes from one of its files. AdamW's moments do not fit beside 1.24 billion parameters on 24 GB: the recipe uses Adafactor, and recomputes the blocks' activations in the backward pass.

```sh
hf download HuggingFaceFW/fineweb sample/10BT/000_00000.parquet --repo-type dataset --local-dir data/fineweb-parquet   # 2.1 GB
kapel prepare data/fineweb-parquet/sample/10BT/000_00000.parquet \
    --tokenizer unsloth/Llama-3.2-1B --output data/fineweb-llama3 --val-every 1000                          # 5 min, 2.7 GB
kapel train recipes/llama-1b/train.toml --steps 50
```

Expected: the validation loss 8.210 at step 50 (12.16 at the first step), the training loss 8.122; 4,630 tokens a second (54 % of the card's peak), 113 seconds a step, 16.1 GiB, 1 hour 36 minutes with the evaluation. Each step is 32 passes of 4 rows of 2,048 tokens; 8 rows a pass reach 18.2 GiB and run out of memory after a few steps. The losses jump during the warmup (gradient norms up to 100 before clipping) and settle as it goes on. The whole run, 1,300 steps, one pass over the file's tokens, would take about 41 hours.

## Supervised fine-tuning on GSM8K

Qwen 3 0.6B taught GSM8K's 7,473 worked answers, one pass, with TRL's settings: every weight (`sft-gsm8k`, AdamW at 1e-5) or LoRA adapters of rank 16 on every projection (`sft-lora-gsm8k`, 2e-4). The 16 examples of a step are laid one after the other in a single row (`pack = "joined"`), so no padding is computed.

```sh
python tools/train/gsm8k.py recipes/sft-gsm8k/gsm8k
kapel prepare recipes/sft-gsm8k/gsm8k/train.jsonl --chat --tokenizer Qwen/Qwen3-0.6B --output recipes/sft-gsm8k/data
kapel train recipes/sft-gsm8k/train.toml
```

(`sft-lora-gsm8k` the same, its directory in place of `sft-gsm8k`.) Expected, the accuracy on GSM8K's 1,319 test problems answered greedily at the end:

| | Every weight | LoRA | TRL, LoRA |
| --- | --- | --- | --- |
| GSM8K | 50.0 % | 48.0 % | 47.5 % |
| Loss, mean of steps 401–450 | 0.408 | 0.419 | 0.422 |
| Time, memory | 5 min 24 s, 15.9 GiB | 4 min 25 s, 7.9 GiB | 6 min 22 s, 2.5 GiB |

With `pack = "padded"`, one example a row of 512 tokens as TRL pads without packing, the same runs take 7 minutes, 21.3 and 13.3 GiB, for 50.0 % and 48.3 %: the greedy answers of two runs part ways at the first token that differs, by about a point.

Qwen 3 0.6B answers 61.2 % of them before, without thinking: GSM8K's answers reason in few words, and learning to imitate them costs accuracy. That is what the next two recipes are for.

## DPO on UltraFeedback

Qwen 3 0.6B trained by DPO with LoRA on 4,000 pairs of UltraFeedback (an answer GPT-4 scored higher against a lower one), β = 0.1, 8 pairs a step, one pass; 500 pairs held out are measured every 100 steps.

```sh
python tools/train/ultrafeedback.py recipes/dpo-ultrafeedback/ultrafeedback
kapel train recipes/dpo-ultrafeedback/train.toml
```

Expected, on the 500 held-out pairs, against TRL with the same settings (`tools/train/trl_dpo.py`):

| Step | Kapel: loss, accuracy, margin | TRL |
| --- | --- | --- |
| 0 | 0.693, 0 %, 0 | 0.693, 0 %, 0 |
| 100 | 0.658, 64.2 %, 0.51 | 0.664, 63.5 %, 0.41 |
| 200 | 0.650, 64.2 %, 0.51 | 0.657, 63.3 %, 0.45 |
| 300 | 0.635, 63.6 %, 0.43 | 0.643, 65.1 %, 0.45 |
| 400 | 0.634, 65.4 %, 0.48 | 0.622, 65.9 %, 0.47 |
| 500 | 0.629, 65.0 %, 0.46 | 0.622, 65.1 %, 0.49 |

Each step is two passes of 4 pairs, their 8 answers of at most 1,024 tokens one after the other in a row (`pack = "joined"`): 1.3 seconds a step, 11.5 GiB, 14 minutes; TRL took 28. Padded to 1,024 tokens each, the same run takes 26 minutes and 13.3 GiB, for 0.627 and 64.8 % at the end. The batches come in another order than TRL's, so the two runs agree within the noise of 500 pairs (a point of accuracy is 5 pairs), not exactly.

## GRPO on GSM8K

Qwen 3 0.6B trained by GRPO with LoRA on GSM8K's problems: each step, 8 answers to each of 16 problems, written by Kapel's engine from the weights being trained, rewarded 1 for the right final number; the test's accuracy every 50 steps.

```sh
python tools/train/gsm8k.py recipes/grpo-gsm8k/gsm8k
kapel train recipes/grpo-gsm8k/train.toml
```

Expected: the accuracy from 61.2 % to 64.9, 67.1, 68.4 and 69.3 % at steps 50, 100, 150 and 200, the mean reward from 0.635 (steps 1–50) to 0.703 (steps 151–200); 1 hour 34 minutes, 11.5 GiB. DeepSeekMath's GRPO raised its 7B model by 5.3 points on GSM8K.

## Publishing what a recipe made

```sh
kapel export recipes/grpo-gsm8k/run --output qwen3-0.6b-grpo-gsm8k           # an adapter, for PEFT
kapel export recipes/grpo-gsm8k/run --merge --output qwen3-0.6b-grpo-gsm8k   # or a whole model
hf upload my-name/qwen3-0.6b-grpo-gsm8k qwen3-0.6b-grpo-gsm8k
```

See [Publishing a model](publishing.md).
