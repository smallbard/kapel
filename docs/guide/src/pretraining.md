# Pre-training

Pre-training draws new weights and trains them on plain text: the model learns to continue it. It is the costly part of making a model, counted in billions of tokens, and the part where the speed of a step matters most.

## The tokens

`kapel train` reads tokens, not text. Two kinds of files:

- **Your own text**, made into tokens by `kapel prepare`, from Parquet files (a row each, its column `text`), JSON Lines (a line each, its field `text`) or text files:

  ```sh
  kapel prepare corpus/*.parquet --tokenizer unsloth/Llama-3.2-1B --output data/corpus --val-every 1000
  ```

  The documents are tokenized on every core, exact duplicates left out, and written as shards with an index of their documents; one in 1,000 goes to the validation split. The configuration then names the directory: `data.train = "data/corpus"`, and `data.val = "data/corpus"` for its validation split. A directory on a web server works too (`https://…/dataset.json`): the shards are fetched as they are read.

- **llm.c's files** (`*.bin` with its header), such as the shards of FineWeb in GPT-2's tokens that `tools/train/fineweb_gpt2.py` downloads: `data.train = "fineweb/fineweb_train_*.bin"`.

Several sources mix with weights, each row drawn from one of them:

```toml
[[data.sources]]
path = "data/fineweb"
weight = 0.9
[[data.sources]]
path = "data/code"
weight = 0.1
```

`data.shuffle = true` reads the shards and their rows in an order drawn from `data.seed`; otherwise in order. `data.pack` sets how documents fill the rows: `"stream"` (the default) cuts the tokens every `batch.length`, documents running across rows; `"documents"` puts whole documents in each row and pads the rest; `data.mask = true` keeps each token's attention within its own document.

## The model

`model.config` gives the architecture: a `config.json` of `transformers`, a model directory, or a model of the Hub (`unsloth/Llama-3.2-1B`), whose configuration and tokenizer are used and its weights ignored with `init = "scratch"`. Kapel trains Llama, Mistral, Qwen 2 and 3, Gemma 2, Phi and GPT-2. `model.set` changes fields of the configuration, to try another size without writing another file:

```toml
[model]
config = "unsloth/Llama-3.2-1B"
init = "scratch"
set = { num_hidden_layers = 8, hidden_size = 1024, intermediate_size = 4096 }
```

New weights are drawn from a normal law of deviation `model.std` (0.02), from `model.seed`.

## A step

A step is `batch.tokens` tokens: `batch.rows` sequences of `batch.length` tokens at once on the card, the passes accumulated until the step's tokens are seen, then one update. More rows go faster until the card is full; the step's size, not the rows, is what the learning rate depends on. When the card runs out of memory:

1. `training.checkpoint_activations = true` recomputes each block during the backward pass instead of keeping its activations: a forward pass more (about a third more computation), far less memory.
2. Fewer `batch.rows`, the same `batch.tokens`: the same training, more passes.
3. `training.chunk` lower (4,096 rows of logits by default) for a large vocabulary.

## The optimizer

- **AdamW** (`kind = "adamw"`), the usual: `lr`, `beta1`, `beta2`, `eps`, `weight_decay`.
- **Muon** (`kind = "muon"`): the update of each matrix of the blocks made orthogonal, AdamW for the tables and the vectors. Its update is scaled to the size of AdamW's, so one learning rate serves both; it reaches a loss in fewer steps, and keeps one moment instead of two.
- **Adafactor** (`kind = "adafactor"`): moments factored by rows and columns, for the least memory.

`[[optimizer.groups]]` gives some parameters settings of their own, the first group that matches a name applying: `decay = false` for the embeddings or the norms, `lr_scale`, `trainable = false` to freeze them.

```toml
[[optimizer.groups]]
matches = "*.bias"
decay = false
```

The learning rate follows `[schedule]`: `"cosine"` (warmup, then down to `final` × `lr`), `"wsd"` (warmup, stable, then a decay over the last `decay` share of the run), `"linear"` or `"constant"`.

## How many steps

`schedule.steps` sets them; or `training.budget` sets the computation, in operations (`1e18`), and the run takes the steps it pays for, its schedule fitted to them. Chinchilla's rule is about 20 tokens a parameter: 2.5 billion tokens for a model of 124 million parameters. [Grids and laws of scale](experiments.md) finds the best size for a budget by measuring.

## Watching it

The loss should fall fast, then slowly; the validation loss should follow it. Signs of trouble:

- **The loss rises, or `nan`**: the learning rate is too high for the batch, or the warmup too short. The gradient norm (`norm`) climbing before is the warning.
- **The model FLOP utilization is low** (`MFU`, under 40 % for a model of 100 million parameters or more on a recent card): the rows are too few to fill the card, or the data cannot keep up (`data_ms` in `log.jsonl`, the time waiting for a batch). `kapel train --profile` shows where the time goes.

## The recipes

- **`gpt2-124m`**: GPT-2 124M on FineWeb in GPT-2's tokens, with llm.c's settings. Its first 250 steps take 43 minutes on an RTX 3090; the whole run, 10 billion tokens, about 53 hours.
- **`llama-1b`**: Llama 3.2 1B's architecture and tokenizer from scratch on FineWeb prepared by `kapel prepare`.

Their commands, expected losses and costs are in [Recipes](recipes.md).
