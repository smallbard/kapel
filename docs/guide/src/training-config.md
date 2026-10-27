# The configuration of a run

Every field of the TOML file `kapel train` reads, by table. Relative paths are taken from the file's directory; an unknown field is refused.

## `[model]`

| Field | Default | Meaning |
| --- | --- | --- |
| `config` | (required) | The architecture: a `config.json`, a model directory, or a model of the Hub (`owner/name`), downloaded once |
| `init` | `"scratch"` | `"scratch"`: new weights; `"weights"`: the model directory's |
| `std` | 0.02 | The deviation of new weights |
| `seed` | 1 | The seed of new weights |
| `set` | | Fields of `config.json` replaced, for a model from scratch (`{ hidden_size = 384 }`) |

## `[data]`: tokens, for pre-training and supervised fine-tuning

| Field | Default | Meaning |
| --- | --- | --- |
| `train` | | A directory of `kapel prepare` (local, or `https://…`), or llm.c's files (`dir/train_*.bin`) |
| `sources` | | Instead of `train`, several, each `{ path, weight }`, the rows drawn from each by its weight |
| `val` | | The validation: a dataset's `val` split, or files |
| `shuffle` | false | Read shards and rows in an order drawn for each epoch |
| `seed` | 0 | The seed of the order and the mixing |
| `pack` | `"stream"` | `"stream"`: cut every `batch.length`; `"documents"`: whole documents, then padding; `"padded"`: one document a row; `"joined"`: the documents of `"padded"` in one row without padding (fine-tuning, DPO) |
| `mask` | false | Each token attends to its own document only |
| `separator` | | The token that starts each document, for files without an index (50256 for GPT-2's) |
| `prefetch` | 4 | Batches read ahead |
| `cache` | | Where remote shards are kept (default `$KAPEL_DATA_CACHE`, or `~/.cache/kapel/data`) |

## `[batch]`

| Field | Default | Meaning |
| --- | --- | --- |
| `rows` | 16 | Sequences in one pass |
| `length` | 1024 | Their tokens |
| `tokens` | one pass | Tokens in a step, a multiple of `rows × length`: passes accumulated |

## `[optimizer]`

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | `"adamw"` | `"adamw"`, `"adamw8bit"` (its moments in 8 bits), `"muon"` (the blocks' matrices; AdamW for the rest) or `"adafactor"` |
| `lr` | 6e-4 | The peak learning rate |
| `beta1`, `beta2`, `eps` | 0.9, 0.95, 1e-8 | AdamW's |
| `weight_decay` | 0 | Decoupled weight decay |
| `momentum` | 0.95 | Muon's |
| `muon_lr_scale` | 1 | Muon's learning rate as a factor of `lr` |
| `newton_schulz` | `"default"` | Muon's iterations: `"default"` (ten) or `"short"` (four) |

`[[optimizer.groups]]`, the first that matches a parameter applying: `matches` (a name, `*` for any text), `lr_scale`, `decay`, `trainable`, `sparse` (its gradient kept as rows, for large tables).

## `[schedule]`

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | `"cosine"` | `"cosine"`, `"wsd"` (warmup, stable, decay), `"linear"` or `"constant"` |
| `steps` | (required without `training.budget`) | The steps of the run |
| `warmup` | 0 | Steps of linear warmup from zero |
| `final` | 0 | The learning rate at the end, a fraction of `lr` |
| `decay` | 0.2 | For `"wsd"`, the share of the run that decays |

## `[training]`

| Field | Default | Meaning |
| --- | --- | --- |
| `device` | `"cuda"` | `"cpu"`, `"cuda"`, or `"cuda:1"` for another card |
| `dtype` | `"bf16"` | The computation: `"bf16"`, `"f16"` or `"f32"`; weights and optimizer stay in `f32` |
| `clip` | 1 | The norm the gradient is clipped to; 0 for none |
| `checkpoint_activations` | false | Recompute each block in the backward pass: less memory |
| `shard` | `"none"` | With several ranks: `"optimizer"` keeps each parameter's optimizer state on one rank (ZeRO 1), `"gradients"` its summed gradient too (ZeRO 2) |
| `offload` | false | The trained parameters in `f32`, their gradients and the optimizer's state in the processor's memory, the card holding the parameters in the computation's type only: larger models, slower steps |
| `chunk` | 4096 | Rows of logits the loss forms at once |
| `cache_gib` | 4 | Memory for the keys and values when the model writes answers (GRPO, tasks) |
| `budget` | | The run's computation in operations (`1e18`): the steps it pays for |

## `[eval]`

| Field | Default | Meaning |
| --- | --- | --- |
| `every` | 250 | Steps between evaluations; 0: at the end only |
| `tokens` | 10,485,760 | Validation tokens read |
| `control` | | A text whose loss is measured as the validation's, for what a fine-tuning forgets |

`[[eval.tasks]]`, accuracies measured on answers written greedily: `name`, `prompts` (JSONL or Parquet, a `prompt` each), `limit`, `max_tokens` (512), `template` (the chat template's variables, `{ enable_thinking = false }`), and `[[eval.tasks.rewards]]` as GRPO's.

## `[checkpoint]` and `[log]`

| Field | Default | Meaning |
| --- | --- | --- |
| `checkpoint.dir` | `"run"` | The run's directory (`kapel train --dir` replaces it) |
| `checkpoint.every` | 1000 | Steps between checkpoints; 0: at the end only |
| `checkpoint.keep` | 3 | Checkpoints kept; 0: all |
| `log.every` | | Steps between lines on the terminal (`log.jsonl` has every step) |
| `log.peak_tflops` | from the card's name | The card's peak, for the model FLOP utilization |
| `log.tensorboard` | false | Also write TensorBoard's events |

## `[lora]`

| Field | Default | Meaning |
| --- | --- | --- |
| `r` | 16 | The rank of the adapters |
| `alpha` | 32 | Their scale is `alpha / r` |
| `targets` | `["all-linear"]` | The projections adapted, by name (`q_proj`, `c_attn`…) |
| `embedding` | false | An adapter on the embeddings too |
| `rslora` | false | The scale `alpha / √r` |
| `quantize` | | QLoRA: the frozen projections in `"q4_0"`, `"int4"`, `"q8_0"` or `"int8"` |

## `[dpo]`

| Field | Default | Meaning |
| --- | --- | --- |
| `train` | (required) | The pairs, JSONL or Parquet |
| `val` | | Pairs held out, measured every `eval.every` steps |
| `beta` | 0.1 | How far from the reference the model may go |
| `nll` | 0 | The weight of the preferred answers' likelihood (TRL's `rpo_alpha`) |
| `shuffle` | true | Read the pairs in an order drawn for each epoch |
| `limit` | | The pairs read at most |

## `[grpo]`

| Field | Default | Meaning |
| --- | --- | --- |
| `prompts` | (required) | The prompts, JSONL or Parquet, with what the rewards read |
| `prompts_per_step` | 8 | Prompts in a step |
| `group` | 8 | Answers to each |
| `max_tokens` | 256 | The longest answer |
| `temperature`, `top_p`, `top_k` | 1, 1, all | How answers are drawn |
| `beta` | 0.04 | The weight of the divergence from the reference; 0: none |
| `epsilon`, `epsilon_high` | 0.2, `epsilon` | The bounds of the ratio |
| `loss` | `"grpo"` | `"grpo"`, `"dr_grpo"` or `"token"` |
| `scale_rewards` | true but for `"dr_grpo"` | Divide the advantages by their group's deviation |
| `iterations` | 1 | Optimizer steps on each set of answers |
| `shuffle` | true | Read the prompts in an order drawn for each epoch |
| `template` | | The chat template's variables |

`[[grpo.rewards]]`, summed: `kind` (`"math"`, `"exact"`, `"regex"`, `"command"`), `weight` (1), `field` (`"answer"`, the record's field for `math` and `exact`), `pattern` (for `regex`), `command` (for `command`).
