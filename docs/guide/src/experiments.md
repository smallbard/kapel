# Grids and laws of scale

For a given computation, a small model trained on many tokens and a large one on few reach different losses; somewhere between is the best size. Hoffmann et al. (Chinchilla, 2022) found it by training families of sizes at fixed budgets of operations, and Kapel has the same tools.

## A grid

`kapel sweep GRID.toml` runs a list of trainings that differ by a few fields of one base configuration:

```toml
[sweep]
base = "train.toml"          # every run starts from it
dir = "runs"                 # each run in runs/NAME, the results in runs/results.tsv
parallel = 2                 # runs at once on the card, for small models
bytes_per_token = 3.897      # the losses also in nats per byte of text

[[run]]
name = "c1e16-e384"
model.set = { hidden_size = 384, intermediate_size = 1024, num_attention_heads = 6, num_key_value_heads = 6, num_hidden_layers = 6 }
optimizer.lr = 2.17e-3
training.budget = 1e16       # the steps this computation pays for

[[run]]
name = "c1e16-e448"
# ...
```

`--dry-run` prints each run's parameters, steps, tokens and operations without running any. Each run is a `kapel train` of its own, its output in `runs/NAME/train.out`; one that fails, fails alone and is noted `failed` (`--retry` runs it again). Every run starts with `--resume`, so after a stop the same command goes on where the grid was, and a run already in the table is not run again. `--only NAME` runs some of the grid.

## The laws

`kapel fit runs/results.tsv` fits two of Chinchilla's methods to the table:

- **The curves at constant computation**: the runs of one budget, three at least, lie on a parabola of the loss against the logarithm of the parameters; its lowest point is the best size for that budget. Power laws through the lowest points of several budgets predict a larger one: `--predict 1e18`.
- **The formula of the loss**, `L(N, D) = E + A / N^α + B / D^β`, fitted to every run, with intervals from 200 draws of the runs (`--bootstrap`).

`--check NAME` leaves a run out of the fits and compares its loss to their prediction. `--non-embedding` counts the parameters without the embedding tables, as Kaplan et al. (2020) did.

## The recipe

`recipes/scaling-smallem` runs again Smallem's curve at 10¹⁶ operations, five sizes from 5 to 29 million parameters, in 1 hour 13 minutes on an RTX 3090, two runs at a time:

```sh
python tools/train/smallem_tokens.py SMALLEM/data/bpe data/smallem
kapel sweep recipes/scaling-smallem/sweep.toml
kapel fit recipes/scaling-smallem/runs/results.tsv
```

Its lowest point is 14.4 million parameters at 0.894 nats per byte; with another seed (`sweep-seed43.toml`), 15.3 million. Smallem's was 15.3 million at 0.897. That the seed alone moves the best size by 6 % is worth knowing before reading much into one grid: the bottom of the parabola is flat. *The Kapel Book*, chapter 17, gives the details.
