# Training a model

`kapel train` trains a model as a TOML file describes it: which model, what it learns from, how big a step is, the optimizer and its schedule, what is measured along the way, and where the checkpoints go. One file is one run, and the file is saved with every checkpoint, so a run can always be told and done again.

```sh
kapel train recipes/sft-lora-gsm8k/train.toml
```

Training needs Kapel built with `--features cuda` and an NVIDIA card for anything but toy models: the processor (`training.device = "cpu"`) computes the same numbers, slowly. Several ranks share a step's passes (`--ranks`, or processes started by torchrun), their numbers checked on one machine; NCCL, the speed of several cards, is not there yet ([Long runs](resuming.md#several-cards)).

## What it can do

| You want to | Read | Recipe |
| --- | --- | --- |
| Train a model from scratch on your text | [Pre-training](pretraining.md) | `gpt2-124m`, `llama-1b` |
| Teach a published model a task or a format | [Fine-tuning a published model](fine-tuning.md) | `sft-gsm8k`, `sft-lora-gsm8k` |
| Make it prefer some answers to others | [Fine-tuning](fine-tuning.md#preferences-dpo) | `dpo-ultrafeedback` |
| Make it find right answers by trying | [Fine-tuning](fine-tuning.md#rewards-grpo) | `grpo-gsm8k` |
| Stop and go on, train for days | [Long runs](resuming.md) | |
| Put the result on the Hugging Face Hub | [Publishing a model](publishing.md) | |
| Find the best size for a budget | [Grids and laws of scale](experiments.md) | `scaling-smallem` |

Each recipe of the repository's `recipes/` directory is a configuration with the commands that prepare its data, the results it should give and what it costs: [Recipes](recipes.md). Starting from the nearest recipe is the quickest way to a run of your own.

## A configuration

The recipe `gpt2-124m`, GPT-2 trained from scratch with llm.c's settings:

```toml
[model]
config = "config.json"        # the architecture: a config.json, a model directory, or owner/name on the Hub
init = "scratch"              # new weights; "weights" starts from the model's own

[data]
train = "../../data/fineweb/fineweb_train_*.bin"
val = "../../data/fineweb/fineweb_val_*.bin"

[batch]
rows = 16                     # sequences in one pass on the card
length = 1024                 # tokens in each
tokens = 524288               # tokens in a step: 32 passes accumulated

[optimizer]
kind = "adamw"
lr = 6e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1

[schedule]
kind = "cosine"               # warmup, then a cosine down to final × lr
warmup = 700
steps = 19073
final = 0.0

[training]
device = "cuda"
dtype = "bf16"                # the computation; weights and optimizer stay in f32
clip = 1.0

[eval]
every = 250                   # the validation loss every 250 steps
tokens = 10485760

[checkpoint]
dir = "run"                   # the run's directory, next to this file
every = 1000
keep = 3
```

Relative paths are taken from the file's directory. An unknown field, or a value out of its range, stops the run before it starts, with the field's name: a typo costs a second, not an hour. Every field is in the [configuration reference](training-config.md).

## What a run writes

Into its directory (`checkpoint.dir`, or `kapel train --dir DIR`):

- **`log.jsonl`**: a line for every step, with its loss, learning rate, gradient norm, tokens, time, speed, the model FLOP utilization (`mfu`, the share of the card's peak the model's operations use) and peak memory, plus what the method shows (DPO's accuracy, GRPO's rewards); a line for every evaluation (`val_loss`, `control_loss`, a task's `accuracy`).
- **`step-NNNNNNNN/`**: the checkpoints. Each is a model directory that `transformers` and `kapel run` load, with the optimizer's state and the position in the data beside it; with LoRA, an adapter PEFT loads.
- **`summary.json`** at the end: parameters, operations, steps, tokens, losses, speed, memory.
- **`tensorboard/`** with `log.tensorboard = true`: the same numbers for TensorBoard (`tensorboard --logdir RUN`), which Weights & Biases also reads (`wandb sync --sync-tensorboard RUN/tensorboard`, or `tools/train/wandb_log.py RUN`).

The terminal shows a line every `log.every` steps:

```
step    250/19073 | loss 6.207476 | norm 0.6029 | lr 2.14e-4 | 10005.6 ms | 52399 tok/s | MFU 63.1 % | 7.93 GiB
step    250/19073 | val loss 6.169836
```

`kapel train --profile` adds, at the end, the time of each kind of operation and the rate of the products, to see where a step's time goes.
