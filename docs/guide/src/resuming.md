# Long runs

## Resuming

A run killed, by a crash, a reboot or Ctrl-C, goes on from its last checkpoint:

```sh
kapel train recipes/gpt2-124m/train.toml --resume
```

The checkpoint holds the weights, the optimizer's moments, its step, and the position in every source of data; the run that resumes computes exactly what the run that was not stopped would have computed, to the bit, on the processor as on the card. Without a checkpoint in the run's directory, `--resume` starts from the beginning, so the same command both starts and resumes a run: a script, or a scheduler that restarts the job, can always use it.

Checkpoints are written every `checkpoint.every` steps by a thread of their own while training goes on, under a temporary name renamed once complete: a run killed while writing leaves the previous ones whole. The `checkpoint.keep` most recent are kept (3 by default; 0 keeps all).

The work since the last checkpoint is lost, and writing one costs the run only the copy of its tensors off the card. `kapel train --steps N` stops after step N as if killed there, to split a long run into jobs of a fixed length:

```sh
kapel train run.toml --resume --steps 5000     # the first job
kapel train run.toml --resume --steps 10000    # the next one goes on from 5000
```

**Changing the configuration.** A resumed run reads its configuration file again: the evaluations, the checkpoints and the log can change between two jobs. So can the schedule, to train longer than planned; the learning rate then follows the new schedule from the step reached. The optimizer's kind cannot change, nor the model.

**Another directory.** `kapel train CONFIG --dir DIR` writes the run into DIR instead of `checkpoint.dir`: the same recipe run twice side by side.

## How long

The terminal and `log.jsonl` give the speed of each step, in tokens per second. The time of a run is its tokens over that speed: `batch.tokens × schedule.steps / tokens_per_s`. The [recipes](recipes.md) give theirs on an RTX 3090.

## Several cards

Kapel trains on one card. Data, tensor and pipeline parallelism over several cards, and over several machines, are not done yet. Until they are, on one card:

- With AdamW, which keeps 16 bytes a parameter (the weights, their gradients and two moments in `f32`), a model of up to about 0.8 billion parameters trains with all its weights on 24 GB: Qwen 3 0.6B (0.75 billion with its table) takes 21.3 GiB in the recipe `sft-gsm8k`. Beyond, Adafactor keeps almost nothing for its moments: Llama 3.2 1B's architecture (1.24 billion) trains with it in 16.1 GiB, with `training.checkpoint_activations = true` (the recipe `llama-1b`).
- A larger model is fine-tuned with LoRA, or QLoRA (`lora.quantize = "int4"`): Qwen 3 8B, its projections in int4, adapters of rank 16 on all of them and `training.checkpoint_activations = true`, fine-tunes on GSM8K in 8.1 GiB, 9.7 seconds a step of 8,192 tokens.
- Several runs at once on one card, for small models, with [`kapel sweep`](experiments.md).
