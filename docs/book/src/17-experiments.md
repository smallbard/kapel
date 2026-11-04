# 17. Experiments: kapel-train

For an hour of computation on a card, which model should be trained, and what loss can be predicted before it starts? Smallem answered by measuring: it counted the operations of a step, trained families of sizes at fixed budgets, and fitted laws to the losses. This chapter gives the same tools to any model Kapel trains. A budget of operations sets a run's steps; `kapel sweep` runs a grid of runs, several at once on one card, and starts again where it stopped; `kapel fit` fits the laws of scale to their table. The log goes to TensorBoard and Weights & Biases as well as to JSONL, and the profiler counts the operations of each kind of call.

## 17.1 Counting the computation

`flops.rs` counts the operations a step's definition asks for (section 15.8): the products by the weights of each block and by the output table, and the attention's whole score matrix, as llm.c and Smallem count it; three forward passes for a step. It knows the blocks of every architecture Kapel trains: GPT-2's (one product for the queries, keys and values), the gated MLP of Llama and its descendants (three matrices instead of two), keys and values shared by groups of heads, and the layers whose attention reads a sliding window, which count the keys of their window instead of all of them (Gemma 2, Mistral). With frozen weights (LoRA), a product by a weight costs two forward passes instead of three (section 16.2).

**Checked against a step.** `Profiled<B>` (section 15.7) now counts the operations of the calls whose cost is in their products: a `Matmul`, `M · K · N` multiply-adds from the shapes of its inputs, and as many again for each gradient its backward call gives; a `QuantizedMatmul`; an `Attention`, its scores and its mix of values over the keys each query reads. The test `a_step_asks_for_the_operations_predicted` runs the forward and backward passes of a step of GPT-2, of Llama with grouped keys and every bias, of Qwen 3, and of Gemma 2 with a window of 5 keys on every other layer, and finds the count of `flops.rs` to the operation, but for one thing: the backward pass of the attention computes the scores again, since it keeps their statistics rather than the probabilities. That is half a forward attention more, which the count leaves out, as llm.c's and Smallem's do; Smallem's test found the same. For Smallem's model of width 384, the count is 7.924 × 10¹¹ operations a step of 16 × 512 tokens, Smallem's own.

**Budgets.** `training.budget = 1e16` sets the steps of a run of tokens: the budget over the operations of a step, rounded, at least one, and `schedule.steps` may be left out (`trainer::with_budget`). The schedule of learning rates follows the steps paid for: a point of a grid is a whole run, its learning rate coming down to the end, not a long run stopped on the way. A warmup longer than the steps is refused. The budget applies to runs of tokens only: a step of DPO or GRPO costs what its answers make it.

**A family of sizes.** `model.set` replaces fields of the model's `config.json` for a model trained from scratch: one file describes a family, each run of a grid giving its width, heads and layers (`set = { hidden_size = 384, num_hidden_layers = 6 }`). The run's checkpoints keep the configuration as it was trained.

**Every bias.** Smallem's models have a bias in each projection: the queries, keys, values and output of the attention, the gate, up and down projections of the MLP. In `transformers`, a Llama has them with `attention_bias` and `mlp_bias`. Kapel read the first for the queries, keys and values only, and not the second: `attention_bias` now gives the output projection its bias too, as `transformers` does for Llama, Gemma and Qwen 3, and `mlp_bias` the three of the MLP, stacked with the gate's and up's weights when those are (section 10.4). The decoders on a tape and `init` have them too. A tiny Llama with every bias, drawn by `transformers`, gives Kapel's logits within 3 × 10⁻⁷ of its own; the test `the_decoders_give_the_logits_of_inference` checks the decoder on a tape against inference with biases that are not zero.

## 17.2 Grids of runs

`kapel sweep SWEEP.toml` runs a grid of trainings (`sweep.rs`). The file names a base configuration and lists the runs, each a part of a configuration laid over the base: tables merged key by key, other values replaced.

```toml
[sweep]
base = "train.toml"        # the configuration every run starts from
dir = "runs"               # each run in runs/NAME, the table in runs/results.tsv
parallel = 2               # runs at once on the card
bytes_per_token = 3.897    # the losses also in nats per byte

[[run]]
name = "c1e16-e384"
model.set = { hidden_size = 384, intermediate_size = 1024, num_attention_heads = 6, num_key_value_heads = 6, num_hidden_layers = 6 }
optimizer.lr = 2.17e-3
training.budget = 1e16
```

Every run's configuration is built and checked when the file is read: a mistake in the fifth run stops the grid before the first starts, not hours later. `--dry-run` prints each run's parameters (all of them, and but the tables of tokens), steps, tokens and operations, without running any: the size of a grid before paying for it.

**Each run** is a `kapel train` of its own, a process on the card: a run that fails, out of memory or otherwise, fails alone. Its configuration, the base and its part with the paths made absolute, is written in its directory as `train.toml`, its output in `train.out`; `parallel` runs share the card, Smallem's two at a time. When a run ends, `summary.json` in its directory gives its parameters, operations, steps, tokens, losses, speed and peak memory, and a line of `results.tsv` notes it:

```text
name  status  parameters  non_embedding  budget  flops  steps  tokens  val_loss  nats_per_byte  train_loss  tokens_per_s  seconds
```

**Starting again.** Every run starts with `--resume`: a run interrupted, by a stop of the grid or of the machine, goes on from its last checkpoint (section 15.11), and the same command goes on with the grid. A run in the table is not run again; a run that ended without being noted is noted from its summary; a run that failed is noted `failed`, and `--retry` runs it again. The last line of a name is the one that counts. `--only NAME` runs some of the grid.

## 17.3 Laws of scale

`kapel fit RESULTS.tsv...` fits Smallem's laws to a table (`scaling.rs`, Smallem's ported):

- **The curves at constant computation** (Hoffmann et al., 2022, their second method). The runs of one budget, three at least, lie on a parabola of the loss in `ln N`: its lowest point is the best size for that budget, `N_opt`, and its tokens `D_opt`, the budget over the operations by token of that size, interpolated between the runs. Power laws through the optima give `N_opt`, `D_opt` and the loss against the budget, and predict larger budgets (`--predict C`).
- **The formula of the loss** (their third), `L(N, D) = E + A / N^α + B / D^β`, fitted to every run: a Huber loss of threshold 10⁻³ on `ln L`, written with a log-sum-exp as the paper does, minimized by BFGS from the paper's grid of 4,500 starts, shared among the cores. The losses are **summed** over the runs: Besiroglu et al. (2024) found that Chinchilla averaged them, which stopped L-BFGS early and made its intervals fifty times too narrow. 200 draws of the runs with replacement (a bootstrap) give each constant an interval, and each prediction. `--check NAME` leaves a run out of the fits and compares its loss to what they predict.

The loss is the validation's, or in nats per byte when the table has them (`sweep.bytes_per_token`): a loss per token depends on the tokenizer, a loss per byte of text does not. `--non-embedding` counts the parameters but the tables, as Kaplan et al. did; Hoffmann et al. count them all, and Pearce and Song (2024) traced the difference of the two papers' exponents to that choice.

**Checked on Smallem's results.** Smallem's 21 runs of its grids of scale, their table put in Kapel's columns, give Smallem's numbers: optima of 4.28, 8.11, 15.29 and 27.15 M parameters for 10¹⁵, 3 × 10¹⁵, 10¹⁶ and 3 × 10¹⁶ operations, `N_opt ∝ C^0.541` and `D_opt ∝ C^0.470`, `L = 0.754 + 243 / N^0.481 + 8,840 / D^0.650`, and for 1.5 × 10¹⁷ operations 65.7 M parameters and a loss of 0.782 by the curves, 76.6 M and 0.821 by the formula. The bootstrap draws differently from Smallem's, and its interval differs in the third decimal ([0.804, 0.840] against [0.804, 0.839]). The tests fit synthetic runs: the formula's exponents found again within 0.05 under 0.1 % of noise, the optima of parabolas whose lowest points follow a known power law found exactly.

## 17.4 Logs

Every step of a run is a line of `log.jsonl` in its directory (section 15.8), every evaluation another. With `log.tensorboard = true`, the same lines go to TensorBoard too, a file `events.out.tfevents.*` in `tensorboard/` of the run, each number of a line a scalar under its name at the line's step (`tensorboard.rs`). The file is written by hand: TFRecords (a length, its masked CRC-32C, the bytes, theirs), each an `Event` of TensorFlow's protocol buffers, a few fields of which are written. TensorBoard 2's own reader reads it back, every scalar at its step.

Weights & Biases reads the same file (`wandb sync --sync-tensorboard RUN/tensorboard`), or `tools/train/wandb_log.py RUN`, which logs each line of `log.jsonl` at its step with the run's configuration as the run's config. Both are offline by default: nothing leaves the machine until `wandb sync`, or `--online`.

## 17.5 The profiler

`kapel train --profile` (section 15.7) gives, beside the time of each kind of call, the rate of those whose operations it counts, in Top/s: where a step's time goes, and how far each product is from the card's peak. A product that reaches a tenth of the card's peak is the next thing to look at. The profiler wraps any backend, the processor's as the card's.

## 17.6 Measured: Smallem's grid again

The validation: Smallem's curve at 10¹⁶ operations, run again by Kapel with one command, finds its lowest point within 5 %. `recipes/scaling-smallem/` holds it: Smallem's `modern` architecture as a Llama with every bias and the table as output, its five widths (256 to 512, heads of 64, as many layers as heads), its learning rates (`0.2002 × C^-0.761`, as Smallem fitted them), Muon with its four iterations of Newton-Schulz (`optimizer.newton_schulz = "short"`, SAMuon's coefficients), its batch of 16 × 512 tokens, its cosine to a tenth after 200 steps of warmup, its weight decay of 0.1 with the embedding table spared, and its tokens, BPE of 8,192 on FineWeb-Edu (`tools/train/smallem_tokens.py` gives them llm.c's header). The steps and the parameters are Smallem's to the unit.

```sh
python tools/train/smallem_tokens.py SMALLEM/data/bpe data/smallem
kapel sweep recipes/scaling-smallem/sweep.toml
kapel fit recipes/scaling-smallem/runs/results.tsv
```

The grid ran in 1 h 13 min, two runs at a time on the card, 8,109 seconds of runs (Smallem's five took 19,815, on a card it shared). It ran a second time with another seed, for the weights and for the order of the data (`sweep-seed43.toml`), to measure what a run's chance alone moves. The final validation losses, in nats per byte:

| Width | Parameters | Steps | Smallem | Kapel, seed 42 | Kapel, seed 43 |
| --- | --- | --- | --- | --- | --- |
| 256 | 5.27 M | 32,254 | 0.9298 | 0.9250 | 0.9254 |
| 320 | 8.80 M | 19,531 | 0.9055 | 0.9014 | 0.9020 |
| 384 | 13.79 M | 12,620 | 0.8974 | 0.8935 | 0.8960 |
| 448 | 20.62 M | 8,552 | 0.9000 | 0.8997 | 0.8961 |
| 512 | 29.44 M | 6,057 | 0.9092 | 0.9093 | 0.9064 |
| Lowest point | | | 15.29 M, 0.8971 | 14.36 M, 0.8943 | 15.28 M, 0.8945 |

```sh
kapel fit recipes/scaling-smallem/runs/results.tsv recipes/scaling-smallem/runs-seed43/results.tsv
```

**The validation is met.** The first seed put the lowest point 6.1 % below Smallem's, the second 0.07 %; the two seeds differ from each other by as much as the first from Smallem, and the ten runs fitted together put it at **14.79 M parameters and 0.8944 nats per byte**, 3.3 % below Smallem's 15.29 M. The size of the lowest point of five runs moves by some 6 % with the seed alone: a parabola through points 0.002 to 0.004 apart (a run's loss moves by about 0.002 from one seed to the other) is flat near its bottom, and a few thousandths at its ends move it. A single grid finds Smallem's within the noise of a grid, and a lowest point within 5 % asks for more than one.

The loss differs more steadily. Kapel's runs end 0.003 below Smallem's on average, the smaller models a little more (0.004 to 0.005 in both seeds, against 0 to 0.004 for the larger); its lowest loss is 0.3 % below Smallem's. What differs between the two programs, besides the seeds: the order of the rows (Kapel reads a permutation of the rows of the corpus, Smallem draws its windows), the validation tokens (Kapel's first 1.6 million, Smallem's 200 batches), RoPE's pairs (side by side in Smallem, halves in Kapel; the same model up to a permutation, not the same draws) and the rounding of the products in bf16. The losses of the step 1000 were already Smallem's within 0.02 (4.531 against 4.527 for the width 256). Five or ten runs of one budget fit the formula of section 17.3 to nothing: its intervals span orders of magnitude, as they should.
