# 15. Training: kapel-train

Every operation of the catalogue has had a backward pass since chapter 4: `Run::backward` turns the gradient of an output into the gradients of the inputs, and the conformity suite checks it on every backend. This chapter is about putting those derivatives to work. `kapel-train` records a computation on a tape, runs it backwards to get the gradient of a loss with respect to the parameters, and moves the parameters with an optimizer. The decoders of chapter 10 train on it without a second implementation: the same weights, by the same names, through the same operations.

Sections 15.1 to 15.7 are step T0 of the training plan, the differentiation; sections 15.8 to 15.12 are step T1, a training run on one card as fast as llm.c, with everything around it; section 15.13 is step T2, the data. Post-training and the tools of experiments come in the steps after them.

## 15.1 The tape

A `Tape<B>` (`crates/kapel-train/src/tape.rs`) is a list of nodes, in the order they were computed. Each node holds its value, its shape and type, whether a leaf reaches it, and, for the output of an operation, its inputs and its derivative. A node always comes after its inputs, so the list is already sorted: `backward` reads it from the loss down to the first node, gives each node the sum of what its consumers sent it, and calls its derivative.

```rust
let tape = Tape::new(backend.clone());
let w = tape.leaf(weight);              // a parameter: its gradient is wanted
let x = tape.constant(inputs);          // an input: no gradient
let y = x.linear(&w, None)?.unary(UnaryKind::Silu)?;
let loss = cross_entropy(&y, &targets)?;
let mut grads = tape.backward(&loss)?;
let g = grads.dense(&w)?;               // the gradient of `w`, of its shape
```

A `Var<B>` is a handle on a node, with the operations as methods: the element-wise ones, views (`reshape`, `transpose`, `permute`, `narrow`, `broadcast_to`), products (`matmul`, `linear`, `quantized_matmul`), reductions, normalizations, `attention`, `rope`, `gather_rows` and `scatter_add_rows`. `Var::apply(op, inputs)` records any operation of the catalogue, with the backend's own derivative. Three kinds of nodes start a computation:

- `leaf`: a parameter, whose gradient `backward` returns;
- `sparse_leaf`: a parameter of which few rows are read, whose gradient may stay rows (section 15.4);
- `constant`: an input or a frozen weight. A computation that no leaf reaches records no derivative at all, so frozen parts of a model cost no memory beyond their own values.

`Tape::custom(inputs, value, derivative)` records an operation computed outside the tape, with its derivative written by hand: any closure `FnOnce(&B, Tensor, &[bool]) -> Result<Vec<Option<Grad>>>`, the flags saying which inputs need a gradient. The fused loss of section 15.3 is one.

The tape works on any backend that is `Backend + Transfer + Clone` (the trait `Compute`): the processor, the card, and the reference, on which every gradient of the tests is checked against finite differences in `f64`.

## 15.2 Keeping only what the gradients need

Memory is what limits training on one card: the activations kept for the backward pass usually take more than the model. The tape keeps as little as it can.

**Counted handles.** A `Var` is a counted handle: when its last clone is dropped, the node lets its value go. A value lives on only if a derivative saved it, and each derivative saves only what it reads. A sum, a scale, a cast, a view or a join saves nothing; a product saves its two factors; a rotation (RoPE) saves only its positions, since the derivative of a rotation does not read what it rotated; a gather saves only its indices. The derivative of a quantized product needs only the shape of its input: it receives a broadcast zero in its place. `Tape::held` counts the nodes that still hold their value, and a test checks that a value saved by no derivative goes with its handle.

**Spent derivatives.** `backward` takes each derivative out of its node before running it, and drops it after: the tensors it saved go as soon as the gradient has passed through them. The tape is spent after one backward pass.

**Slices.** The gradient of a slice (`narrow`) reaches its source as a `Grad::Slice`, not padded with zeros. When the slices of one source do not overlap, as when the queries, keys and values are cut from one product, they are joined by one concatenation, the holes filled with zeros; only overlapping slices are padded and added.

**Checkpointing.** `Tape::checkpoint(inputs, f)` computes a region `f` on a tape of its own where every input is a constant, so that nothing is saved, and records one node with the region's output. Its derivative computes the region again, this time on a tape whose inputs are leaves, and runs that tape backwards. Only the region's inputs are kept: for a block of a decoder, its input hidden states (the parameters are leaves, kept anyway). `f` must give the same result each time; it has no randomness of its own. The tests check that checkpointing changes no gradient, by a single bit, on the processor and on the card.

## 15.3 Fused operations

Some operations are cheaper when their derivative is written by hand than when it is composed of the derivatives of their parts:

| Operation | Instead of | What the derivative keeps |
| --- | --- | --- |
| `Attention` | products, a mask, a softmax | `q`, `k`, `v`; the card recomputes the statistics of the softmax |
| `RmsNorm`, `LayerNorm` | a mean, a square root, a division | the input |
| `Glu` (SwiGLU) | a SiLU and a product | the stacked gate and up projection, one tensor instead of three |
| `CrossEntropy` | a log-softmax, a gather, a one-hot | the logits |

The first three were in the catalogue already. `CrossEntropy` (`kapel-ops/src/loss.rs`) was added for training: for logits `x [N, V]` and targets `t [N]`, `y[n] = ln Σⱼ e^(x[n, j]) − x[n, t[n]]`, and `y[n] = 0` where the target is `CrossEntropy::IGNORE` (all bits set: `u32::MAX`, or −1 as an `i32`), the positions that count in no loss. Its gradient is `g[n] · (softmax(x[n]) − onehot(t[n]))`, in the type of the logits. On the card, one warp computes a row: each lane keeps a running maximum and the sum of exponentials over its columns, in one reading, then the lanes are combined; the backward pass reads the row again and writes its gradient in place. The losses are in `f32` whatever the type of the logits (`f64` for logits in `f64`).

**The loss of a large vocabulary.** GPT-2 predicts among 50,257 tokens: the logits of a batch of 16 × 1024 positions take 1.6 GB in bf16, and twice that in f32. `linear_cross_entropy(h, w, targets, chunk)` never forms them whole. It takes the final hidden states `h [N, C]` and the output table `w [V, C]`, and for each piece of `chunk` rows computes the logits, their loss by `CrossEntropy`, and at once their gradient, then `gh = g · w` for the piece and `gw += gᵀ · h` (in `f32`). The logits of a piece are let go before the next one is computed. The gradients, formed during the forward pass, are kept in the node, and the backward pass only scales them by the loss's own gradient. The memory of the vocabulary is that of `chunk` rows of logits, 4096 by default, instead of `N`.

The table is padded with rows of zeros up to a multiple of 64 for the products: GPT-2's 50,257 columns make rows of an odd length, which slowed the products down eightfold on the card (8.6 Top/s against 70.6 for 50,304 columns). `CrossEntropy` reads the logits through a view of the true columns. On the card, its gradient comes in the layout of those logits, the padding columns written zero, so that the products that follow read it whole.

## 15.4 Sparse gradients

The gradient of a table read through `gather_rows` is zero but for the rows read. The tape gives it as `Grad::Rows { indices, values }`: the rows read, as many times as they were, and what to add to each. Several reads of one table join their lists. For a `sparse_leaf`, the gradient stays rows; for any other node, it becomes dense by `ScatterAddRows`, and so does a sparse leaf's when another use of it sends a dense part (GPT-2's embedding table, which is also its output table).

The optimizer follows: for a gradient by rows, AdamW updates only those rows of the parameter and of its two moments, after summing the rows that came several times, and writes them in place (`Write::write_rows`). This is the AdamW by rows of Smallem's memory tables of n-grams. Rows that were not read keep their values and moments; the bias correction is that of the global step. A test checks that, on a first step without weight decay, it gives exactly the update of the dense AdamW.

## 15.5 The decoders on a tape

`kapel_train::model` (`crates/kapel-train/src/model.rs`) trains the architectures of chapter 10 but Phi's parallel blocks, Gemma 2's capped logits and multi-token layers: GPT-2, Llama, Mistral, Qwen 2 and 3, Gemma 2's sandwich blocks.

- `Parameters` holds the weights by their names in the checkpoint, each with its options: `decay` (weight decay, for matrices and tables, not for norms and biases, as nanoGPT does), `sparse`, `trainable` and `lr_scale`. `Parameters::from_model` takes a loaded `Model`'s; `init(config, std, seed)` makes new ones, with GPT-2's initialization for every architecture (normal of standard deviation 0.02, `0.02 / √(2 · layers)` for the projections that add to the residual stream, zeros for biases, ones for norms), each parameter from a seed of its own drawn from its name, so that the order of the parameters does not matter. `Parameters::to_model` gives a `Model` again, in the type asked, which `Model::save` writes for `transformers` and the Hub.
- `Parameters::leaves(tape)` puts them on a tape; `Leaves::gradients` takes their gradients out, by name.
- `Decoder::loss(leaves, batch, options)` computes the mean cross-entropy of a `Batch`: `rows` sequences of the same length, their token ids and their targets. `Options` says the type of the computation (the parameters, kept in `f32`, are converted to it for the pass, and their gradients come back in `f32`), whether each block is checkpointed, and the chunk of the loss.

The pass is the inference pass of chapter 10, batched: the hidden states are `[rows · length, hidden]` for the products, and `[rows, heads, length, head width]` views for the attention. The embedding rows are gathered from the `f32` parameter itself, then converted, so that its gradient stays rows. Two things change: the query, key and value projections of Llama's blocks are joined into one product, and so are the gate and up projections, followed by `Glu`. Joining costs a copy of the weights per step and saves activations. The test `the_decoders_give_the_logits_of_inference` checks that a batch of two sequences gives the logits `Model::forward` gives each of them, for GPT-2, Llama and Qwen 3; `the_gradients_of_the_decoders_match_finite_differences` checks every parameter's gradient in `f64` on the reference.

GPT-2's first checkpoints (`openai-community/gpt2`) name their weights after `GPT2Model`, without `transformer.`, and keep the causal masks of the attention among them. `Model::load` now gives them the names of `GPT2LMHeadModel` and leaves the masks out, for inference as for training.

## 15.6 Measured

The validation of step T0 is GPT-2 124M on an RTX 3090, with a batch of 16 × 1024 tokens of wikitext-2: its gradients equal PyTorch's within the tolerance of bf16, and the peak memory is at most 10 % above PyTorch's. The reference is nanoGPT's model (its attention by `scaled_dot_product_attention`), with GPT-2's weights, the model in `f32` and the pass under autocast to bf16, as nanoGPT trains; `transformers`' own GPT-2 runs out of memory at this batch, since it forms the logits whole in bf16 then in f32. `tools/train/gradients.py` runs PyTorch and compares the gradients; the example `gradients` of `kapel-train` runs Kapel.

| | PyTorch bf16 | Kapel bf16 | Kapel bf16, checkpointed | Kapel f32 |
| --- | --- | --- | --- | --- |
| Loss | 3.40436 | 3.40170 | 3.40170 | 3.39517 (PyTorch f32: 3.39517) |
| Gradients' distance to PyTorch f32 | 16.6 % | 16.9 % | | 0.058 % |
| Peak memory | 13.54 GiB | 6.41 GiB | 2.57 GiB | 10.84 GiB |
| Forward and backward | 0.28 s | 0.55 s | 0.64 s | 5.75 s |

These are the measures at the end of step T0; section 15.12 has those of step T1. The distance is the norm of the difference of all gradients over the norm of PyTorch's in f32. In bf16, PyTorch's own gradients are 16.6 % from its f32 ones, and the worst tensor for both is the final norm's weight: that is the precision of bf16 on this model, and Kapel's gradients are as close. Kapel's peak memory is less than half of PyTorch's: the activations kept are those of the derivatives only, and the logits are never whole. With every block checkpointed, it is a fifth.

The example `steps` trains the model from GPT-2's initialization with AdamW at 6 × 10⁻⁴: the loss goes from 11.02 to 7.34 in 30 steps, 0.57 s each.

Two kernels of the card were found on the way, both doing a reduction with one thread for each result, adding its terms in sequence. That was slow and, over millions of terms, imprecise: a sum of 12.8 million values was off by 5 × 10⁻⁵. Sums of more than 128 terms now go in two stages (`reduce_sum_segments`, then `reduce_sum` over the segments): adjacent threads read adjacent values, and the order of the additions is still fixed. The backward pass of a layer norm over 16,384 rows, whose weight gradients are such sums, went from 11 ms to 1 ms. The other, the softmax, no longer runs in training: `CrossEntropy` replaced it.

What was left of the gap to PyTorch is the subject of section 15.12.

## 15.7 Seeing where the time goes

`kapel_train::profile::Profiled<B>` wraps a backend and times each call by operation and direction (`"matmul"`, `"attention backward"`), waiting for the device before and after each. `report()` lists them, the longest first. The example `gradients` prints it with `--profile`; it is how the reductions above, and a scatter of the first loss that took three quarters of a step, were found.

The card's backend counts the bytes held by its live tensors: `Cuda::memory_in_use`, and `Cuda::peak_memory` since the last `reset_peak_memory`, the equivalents of `torch.cuda.memory_allocated` and `max_memory_allocated`. Buffers kept for reuse are not counted.

## 15.8 The training loop

`kapel train CONFIG.toml` runs a training described by one TOML file (`kapel_train::config::TrainConfig`): the model (a `config.json`, new weights or a directory's), the token files, the batch, the optimizer and its groups of parameters, the schedule of learning rates, the computation, the evaluations, the checkpoints and the log. Every field but the model's configuration, the training files and the number of steps has a default; an unknown field is an error, and the values are checked before anything starts (`TrainConfig::validate`), each refusal naming its field. `recipes/gpt2-124m/train.toml` is GPT-2 124M on FineWeb with llm.c's settings.

```toml
[batch]
rows = 16          # sequences of one pass
length = 1024
tokens = 524288    # per step: 32 passes accumulated

[optimizer]
kind = "adamw"     # or "muon", "adafactor"
lr = 6e-4
weight_decay = 0.1
[[optimizer.groups]]
matches = "*.bias" # the first group that matches a name applies
lr_scale = 0.5

[schedule]
kind = "cosine"    # or "constant", "wsd"
warmup = 700
steps = 19073
final = 0.0
```

`kapel_train::trainer::run` does each step this way:

1. **Batches.** A thread reads them ahead (section 15.13): rows of `length` tokens from the training sources, each with the token after it as its last target. `data_ms` in the log is the time a step waited for them.
2. **Accumulation.** Each of the step's passes computes its loss and its gradients on a tape of its own; the gradients are added up across passes, those by rows joined.
3. **Clipping.** The global norm of the gradients, `√Σ g²` over all of them, divided by the passes, is read back; above `training.clip`, the gradients are scaled down to it. The scale goes to the optimizer with `1 / passes`: no gradient is rewritten.
4. **The step**, at the learning rate of the schedule (`schedule.rs`): a linear warmup from zero, `(step + 1) / warmup` as llm.c's, then half a cosine down to `final` (llm.c, nanoGPT), a constant, or WSD's plateau then a straight decay over the last part of the run.
5. **The log.** Every step goes to `log.jsonl` in the run's directory: its loss, learning rate, gradient norm, time, tokens per second, model FLOP utilization and, on a card, peak memory. The terminal gets a line every `log.every` steps.
6. **Evaluation.** Every `eval.every` steps and at the end, the mean loss of the first `eval.tokens` validation tokens, computed with the parameters as constants: nothing is recorded for a backward pass.
7. **Checkpoints**, every `checkpoint.every` steps and at the end (section 15.11).

**The model's FLOP utilization.** `flops.rs` counts the operations a step's definition asks for: the products of the blocks and of the output table, and the attention's whole score matrix, as llm.c and Smallem count it, three forward passes for a step, two operations per multiply-add. Over the step's time and the card's peak in the computation's type (`flops::peak_tflops`, from the card's name, or `log.peak_tflops`), that is the MFU of the log. The test `gpt2_small_costs_what_nanogpt_counts` checks the count against nanoGPT's estimate.

`--profile` times each operation (section 15.7) from the second step on, the first compiling the kernels.

## 15.9 Optimizers

`Optimizer` (`optim.rs`) is one of three, with their states for the checkpoints:

- **AdamW**, dense and by rows (section 15.4). Its step is one fused operation in place, `Update::adamw` (`kapel-ops/src/update.rs`): the gradient read in its type, the parameter and both moments updated in `f32` by one kernel, where the step had been fifteen operations and as many temporary tensors per parameter. `AdamWStep::apply` is the arithmetic every backend does, element by element; the conformity suite compares each with the reference, and checks that a shared parameter and a gradient of another shape are refused.
- **Muon**, the variant Smallem measured (DeepSeek-V4's): for each matrix of the blocks, a momentum with Nesterov's look ahead, made orthogonal by ten iterations of Newton-Schulz, then scaled by `0.18 · √max(m, n)` so that its elements have the size of an AdamW step and one learning rate serves both. The matrices of one shape are orthogonalized together, as one batched product. The tables (embeddings, positions, output) and the vectors stay with AdamW.
- **Adafactor** (Shazeer and Stern), without first moment: the second moment of a matrix kept as its means by row and by column, `β₂ = 1 − t^−0.8`, the update divided by `max(1, rms)`.

A group of parameters (`optimizer.groups`) sets, for the names its pattern matches, a factor of the learning rate, whether weight decay applies, whether the parameter trains, and whether its gradient stays rows.

## 15.10 Mixed precision

The parameters and the optimizer's moments are `f32`, the master weights. Each pass converts the parameters to the computation's type (`training.dtype`, bf16 by default) and computes in it; the conversions' derivatives bring the gradients back to `f32` for the optimizer. The losses are `f32`. 8-bit floats for the products, which the plan asks on Ada and Hopper cards, are left for a card that has them: the RTX 3090 is an Ampere.

## 15.11 Checkpoints and resuming

A checkpoint is a directory `step-NNNNNNNN` in the run's (`checkpoint.rs`): `model.safetensors`, the parameters in `f32` under the names of `transformers`, with `config.json` (the directory is a model that `transformers` and `kapel run` load); `optimizer.safetensors`, the moments; `state.json`, the step, the optimizer's steps, the position in the data (the rows read, and for each source its epoch, shard and row) and the tokens seen; and `config.toml`, the run's configuration. The tensors are copied to host memory when the checkpoint is taken, and a thread writes them while the next steps compute, under a temporary name renamed once complete: a run killed while writing leaves the previous checkpoints whole. The oldest are removed beyond `checkpoint.keep`. (On the processor, a download is the tensor itself: the checkpoint takes copies, since the next steps update the tensors in place.)

`kapel train --resume` goes on from the last complete checkpoint: the parameters, the moments and steps of the optimizer, the position in the data. Nothing else in a step depends on the past: on the same machine, the run computes the bits of a run that was not interrupted. The tests `a_resumed_run_computes_the_bits_of_an_uninterrupted_one` (on the processor, with each optimizer, accumulation and clipping) and `a_resumed_run_on_the_card_computes_the_same_bits` (bf16 on the card) kill a run after three steps (`--steps 3` does it), resume it, and compare every parameter with those of a run of seven steps, bit for bit.

## 15.12 As fast as llm.c

The validation of step T1 is GPT-2 124M reaching the target loss of modded-nanogpt's speedrun, 3.28 on FineWeb's validation, in at most 1.2 times llm.c's time with the same settings on the same card. With llm.c's settings, 3.28 takes about 10 billion tokens: some 45 hours on an RTX 3090 for each side, and more disk than the machine had. The validation was therefore cut in two: the time of a step with the same settings, and the losses of a shorter run, both from new weights.

For one pass of 16 × 1024 tokens a step, llm.c takes 0.27 s on the RTX 3090. The first version of the loop took 0.80 s, with its clipping and its optimizer. The gap closed in a dozen changes, each measured by the profile of section 15.7 and by `nsys`, in the order they were made:

| Change | Step (ms) |
| --- | --- |
| The loop as first written (the gradient's norm and AdamW composed of operations) | 795 |
| Long sums on the card in stages, each stage table-free for the last dimensions; AdamW fused (`Update::adamw`) | 455 |
| `CrossEntropy` reading rows by their stride, one block per row instead of one warp | 413 |
| Backward passes of the element-wise functions, of the norms and the sums reading 16-bit inputs in place, without widening them to `f32` first | 379 |
| Element-wise operations eight values a thread, in 128-bit loads and stores (290–435 → 800–840 GB/s) | 351 |
| The loss's products by pieces of 4096 rows instead of 1024 (the products along the vocabulary fill the card) | 329 |
| Sums over the first dimensions (the gradients of biases) table-free too | 328 |
| Gradients kept as views: the tape had made every transposed gradient contiguous | 324 |
| The bias added in the product (`Matmul` with a third input), as cuBLASLt's epilogue does for llm.c | 322 |
| Each attention's output and statistics kept for its backward pass, which no longer computes the forward one again (`Cuda::keep_attention_for_backward`) | 316 |

Before the loop was written, the gradient of a weight, `xᵀ · g`, got a kernel that reads the transposed tiles along their contiguous dimension (13–25 → 54–60 Top/s), and the table was padded to 50,304 columns (section 15.3). Two attempts were measured and undone: a backward pass of the layer norm that kept its weight gradients in registers over several rows, twice as slow for want of registers, and other sizes of tiles for the products, whose current ones stayed the best on these shapes.

With llm.c's settings, 32 passes accumulated for a step of 524,288 tokens:

| | llm.c | Kapel |
| --- | --- | --- |
| Step | 8.46 s | 9.84 s |
| Tokens per second | 62,000 | 53,300 |
| MFU | 70 % | 64 % |
| Peak memory | 12.9 GiB (the card's memory in use, as llm.c reports it) | 7.9 GiB (the allocator's peak, `peak_gib` in `log.jsonl`) |

Kapel takes 1.16 times llm.c's time: within the 1.2 the plan asks. The products are about two thirds of a step and run at 55 to 71 Top/s of the card's 71. The rest is in the layer norm's backward pass, which writes and reads its rows' parts of the weight gradients in `f32`, and in the many small operations around the products.

The losses come from 250 such steps, 131 million tokens, both runs from new weights with the same schedule (25 steps of warmup, then a cosine down to a tenth of 6 × 10⁻⁴), validated every 50 steps on the same 327,680 tokens of FineWeb's validation. llm.c draws its training samples in a shuffled order and Kapel reads them in the files' order, so llm.c was also run with its shuffling turned off, reading the very tokens Kapel reads:

| Step | llm.c, shuffled | llm.c, in order | Kapel, in order |
| --- | --- | --- | --- |
| 50 | 6.925 | 6.997 | 6.908 |
| 100 | 6.454 | 6.531 | 6.476 |
| 150 | 6.217 | 6.305 | 6.246 |
| 200 | 6.057 | 6.160 | 6.096 |
| 250 | 5.971 | 6.084 | 6.014 |

The order of the data moves the loss by 0.11 at step 250, more than anything that separates the two programs: on the same tokens Kapel's loss is the lower one, by 0.07, from initial weights drawn by another generator. The curves leave no sign of a difference in what is computed; the target of 3.28 itself is left to a machine that can give the run its 45 hours.

## 15.13 The data

The data of a run is prepared once, then read by every step without the step waiting for it, and the run knows exactly which tokens it read.

**Preparing.** `kapel prepare INPUT... --tokenizer PATH --output DIR` (`prepare.rs`) reads Parquet files (a row each, the column `text`), JSONL files (a line each, the field `text`) and text files (one document each). One thread reads the documents in order and cuts them into jobs of 256; as many threads as cores tokenize them, each with a tokenizer of its own (the cache of encoded pieces has one lock); the main thread puts the jobs back in order and only then decides: a document whose text is that of an earlier one is left out (exact deduplication, by the first 128 bits of the SHA-256 of its text), an empty one too, and one kept document in `--val-every` goes to the validation split. The decisions depend on the order of the documents alone, so the same inputs give the same files whatever the number of threads; the test `the_same_inputs_give_the_same_files_whatever_the_threads` compares them byte for byte after one thread and five. Documents are encoded as plain text (section 9.2), each preceded by the separator, the tokenizer's `bos_token` or else its `eos_token` (`<|endoftext|>` for GPT-2, as llm.c's FineWeb shards have it).

**The dataset** (`dataset.rs`) is a directory: shards in llm.c's format (`u16` for vocabularies up to 65,536 tokens, `u32` beyond), each document whole in one shard of at most `--shard-tokens` (100 million by default); for each shard an index, the first token of each of its documents; and `dataset.json`, the manifest, with the SHA-256 of the tokenizer, the separator, the counts of documents read, duplicates and empty ones, and for each split its shards with their sizes and SHA-256. llm.c's own files are read as they are, without an index: the separator, given as `data.separator`, marks the documents when they are needed.

**Sources.** `data.train` names one source, `data.sources` several, each with a weight: a dataset's directory or `dataset.json`, local or at an `http(s)://` URL, or a pattern of llm.c files. Each row of a batch comes from one source, drawn with the probability of its weight by a generator that depends only on `data.seed` and the row's rank (SplitMix64). A source goes through its shards in order or, with `data.shuffle`, in an order drawn anew for each epoch, and through each shard's rows in order or in an order drawn for that shard and that epoch. The order of every row thus depends on the seed alone; the position in the data is the rows read and, for each source, its epoch, shard and row, and a checkpoint that saves it reads on exactly as before. `a_resumed_run_over_mixed_shuffled_documents_computes_the_same_bits` kills and resumes a run over two weighted sources, shuffled, packed by documents with their mask, and compares every parameter with those of a run that was not killed; a run with the same seed gives the same bits, one with another seed others.

**Packing.** `data.pack = "stream"` cuts a shard's tokens every `length`, as llm.c does: a row may begin inside a document and end in the next. `"documents"` fills a row with whole documents, as many as fit, and pads the rest (ids of the separator, targets ignored); a document longer than a row is cut into rows of its own. With `data.mask`, each token attends to the earlier tokens of its own document only, and its position counts from the document's first token; the test `documents_of_one_row_see_only_themselves` checks that the logits of three documents in one row are those of the inference pass on each alone, for GPT-2, Llama and Qwen 3. The mask is not a tensor of `rows × length²` values: the attention receives where the document of each token starts (section 4), and the card skips the keys before each document (section 7.6).

**Reading ahead.** `Prefetched` moves the loader to a thread of its own, `data.prefetch` batches ahead (4 by default); each batch comes with the position after it, which is what a checkpoint saves. A remote source downloads its manifest, then each shard with its index when it is reached, into `data.cache` (`~/.cache/kapel/data` by default), checking its SHA-256; while one shard is read the next is downloaded, and the shards read are removed. The test `a_remote_dataset_reads_as_the_local_one_and_keeps_little_on_disk` serves a dataset over HTTP: the batches are those of the local copy, at most two shards stay on disk, and a shard whose bytes were changed is refused. Requests carry no token but to the Hub, and nothing that names the user.

**Measured**, on the Ryzen 7 7700 and the RTX 3090:

| | |
| --- | --- |
| Preparing FineWeb-Edu's first file (726,000 documents, 2.2 GB of Parquet) with GPT-2's tokenizer | 26 s, 29 million tokens a second; 1,489 duplicates left out |
| The same, one thread | 3.5 million tokens a second |
| Tokens compared with `transformers` on 2,000 documents | identical |
| One thread reading batches of 16 × 1024 (stream, in order) | 1.5 billion tokens a second |
| The same, whole documents with their mask, shuffled | 730 million tokens a second; 80 % of the targets real, 20 % padding |
| A step of GPT-2 124M waiting for its batch (16 × 1024, prefetched) | 0.003 ms of 316 |
| The same step with documents and their mask | 317.6 ms, against 315.9 without |

The plan asks that loading take less than 1 % of a step of a 1B model on 8 cards. Eight H100 at 40 % of their bf16 peak (989 Top/s each) train a 1B model, some 6 × 10⁹ operations a token, on about half a million tokens a second; one thread of the loader gives 730 million, and each card's process would have its own. On one card, a step waits 0.003 ms for its batch: a hundred-thousandth of its time.

