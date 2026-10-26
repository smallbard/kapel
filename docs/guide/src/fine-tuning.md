# Fine-tuning a published model

A published model already writes well; fine-tuning teaches it a task, a format or a preference, with thousands of examples rather than billions of tokens. The run starts from the model's weights:

```toml
[model]
config = "Qwen/Qwen3-0.6B"    # a model of the Hub, or a model directory
init = "weights"
```

Three methods, by what the examples say:

| The examples are | Method | Section |
| --- | --- | --- |
| Conversations to imitate | supervised fine-tuning | [`[data]`](#conversations-supervised-fine-tuning) |
| Two answers to a prompt, the better one first | DPO | [`[dpo]`](#preferences-dpo) |
| Prompts, and a program that scores answers | GRPO | [`[grpo]`](#rewards-grpo) |

Any of them trains every weight, or LoRA adapters beside a frozen model.

## Conversations: supervised fine-tuning

The examples are JSON Lines or Parquet in TRL's two forms: whole conversations, every answer of the assistant learnt,

```json
{"messages": [{"role": "user", "content": "What is 6 × 7?"}, {"role": "assistant", "content": "42"}]}
```

or a prompt and its completion, the completion learnt:

```json
{"prompt": [{"role": "user", "content": "What is 6 × 7?"}], "completion": [{"role": "assistant", "content": "42"}]}
```

`kapel prepare --chat` writes them in the model's chat template, with a mask of the tokens the loss counts:

```sh
kapel prepare train.jsonl --chat --tokenizer Qwen/Qwen3-0.6B --output data
```

```toml
[data]
train = "data"
pack = "padded"      # one conversation a row, as TRL without packing; "documents" packs several
shuffle = true
```

A full fine-tuning takes a learning rate around 1e-5; with LoRA, 1e-4 to 2e-4. One to three passes over the examples (`schedule.steps` × `batch.rows` over their number) are usual; more and the model learns the examples by heart.

**Measuring along the way.** `eval.control` names a text whose loss is measured as the validation's, to see how much of what the model knew it forgets. `[[eval.tasks]]` measures the accuracy of answers a program checks:

```toml
[[eval.tasks]]
name = "gsm8k"
prompts = "gsm8k/test.jsonl"     # {"prompt": [...], "answer": "... #### 72"}
max_tokens = 512
template = { enable_thinking = false }
[[eval.tasks.rewards]]
kind = "math"                    # the answer's last number against the record's
```

## LoRA and QLoRA

`[lora]` freezes the model and trains a pair of small matrices beside each projection: a fraction of the memory of the optimizer, a few megabytes for the result.

```toml
[lora]
r = 16
alpha = 32
targets = ["all-linear"]     # or ["q_proj", "v_proj"], by the projections' names
quantize = "int4"            # QLoRA: the frozen model in 4 bits ("q4_0", "int4", "q8_0", "int8")
```

`embedding = true` adds an adapter to the table of embeddings. The checkpoints of the run are adapters in PEFT's format. `kapel merge MODEL ADAPTER --output DIR` merges one into its model, and `kapel export RUN --merge --output DIR` does it from the run ([Publishing a model](publishing.md)).

## Preferences: DPO

The examples are pairs, as TRL reads them: a prompt and two answers, the preferred one first,

```json
{"prompt": [{"role": "user", "content": "..."}], "chosen": [{"role": "assistant", "content": "..."}], "rejected": [{"role": "assistant", "content": "..."}]}
```

or two whole conversations, `chosen` and `rejected`, whose common beginning is the prompt. No preparation: the run reads the files.

```toml
[dpo]
train = "pairs/train.jsonl"
val = "pairs/test.jsonl"     # pairs held out: their loss, accuracy and margin every eval.every steps
beta = 0.1                   # how far from the starting model the run may go
nll = 0.0                    # the weight of the likelihood of the preferred answers (TRL's rpo_alpha)
```

`batch.rows` is twice the pairs of a pass. Each step logs the share of pairs whose preferred answer the model now prefers (`accuracy`), the rewards of both answers and their `margin`. When both answers' likelihoods fall while their margin grows, the model drifts towards answers neither of them is; `nll = 1.0` holds the preferred ones up.

## Rewards: GRPO

The examples are prompts, with what a program needs to score an answer:

```json
{"prompt": [{"role": "user", "content": "Natalia sold clips to 48 friends..."}], "answer": "... #### 72"}
```

Each step, the model writes a group of answers to each prompt, with Kapel's inference engine and the weights being trained; the program scores them, and the answers better than their group's mean are made more likely.

```toml
[grpo]
prompts = "gsm8k/prompts.jsonl"
prompts_per_step = 16
group = 8                    # answers to each prompt
max_tokens = 512
temperature = 1.0
beta = 0.04                  # the weight of the divergence from the starting model
loss = "grpo"                # or "dr_grpo", "token"
template = { enable_thinking = false }
[[grpo.rewards]]
kind = "math"
```

The rewards, summed with their `weight`: `math` (the answer's last number, or its `\boxed{}`, against the record's `answer`), `exact` (the answer is the record's text), `regex` (the answer matches a `pattern`), and `command`, a program of your own:

```toml
[[grpo.rewards]]
kind = "command"
command = ["python3", "score.py"]
weight = 0.5
```

It is started once per batch of answers, reads one JSON object a line, `{"completion": "...", "record": {...}}`, and writes one number a line.

`training.cache_gib` is the memory the answers' keys and values may take (4 GiB by default); a step's log shows the mean reward, its spread, the answers' length and the share that ended before `max_tokens`. If the rewards of a group are all equal, it teaches nothing: prompts the model always or never solves are wasted.

## The recipes

`sft-gsm8k` (every weight) and `sft-lora-gsm8k` fine-tune Qwen 3 0.6B on GSM8K; `dpo-ultrafeedback` trains it by DPO on UltraFeedback's pairs; `grpo-gsm8k` by GRPO on GSM8K's problems. Their commands, expected results and costs are in [Recipes](recipes.md).
