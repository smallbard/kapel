# 15. Training: kapel-train

Every operation of the catalogue has had a backward pass since chapter 4: `Run::backward` turns the gradient of an output into the gradients of the inputs, and the conformity suite checks it on every backend. This chapter is about putting those derivatives to work. `kapel-train` records a computation on a tape, runs it backwards to get the gradient of a loss with respect to the parameters, and moves the parameters with an optimizer. The decoders of chapter 10 train on it without a second implementation: the same weights, by the same names, through the same operations.

This is step T0 of the training plan: the differentiation. The training loop, the data, post-training and the tools of experiments come in the steps after it.

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

**The loss of a large vocabulary.** GPT-2 predicts among 50,257 tokens: the logits of a batch of 16 × 1024 positions take 1.6 GB in bf16, and twice that in f32. `linear_cross_entropy(h, w, targets, chunk)` never forms them whole. It takes the final hidden states `h [N, C]` and the output table `w [V, C]`, and for each piece of `chunk` rows computes the logits, their loss by `CrossEntropy`, and at once their gradient, then `gh = g · w` for the piece and `gw += gᵀ · h` (in `f32`). The logits of a piece are let go before the next one is computed. The gradients, formed during the forward pass, are kept in the node, and the backward pass only scales them by the loss's own gradient. The memory of the vocabulary is that of `chunk` rows of logits, 1024 by default, instead of `N`.

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

The distance is the norm of the difference of all gradients over the norm of PyTorch's in f32. In bf16, PyTorch's own gradients are 16.6 % from its f32 ones, and the worst tensor for both is the final norm's weight: that is the precision of bf16 on this model, and Kapel's gradients are as close. Kapel's peak memory is less than half of PyTorch's: the activations kept are those of the derivatives only, and the logits are never whole. With every block checkpointed, it is a fifth.

The example `steps` trains the model from GPT-2's initialization with AdamW at 6 × 10⁻⁴: the loss goes from 11.02 to 7.34 in 30 steps, 0.57 s each.

Two kernels of the card were found on the way, both doing a reduction with one thread for each result, adding its terms in sequence. That was slow and, over millions of terms, imprecise: a sum of 12.8 million values was off by 5 × 10⁻⁵. Sums of more than 128 terms now go in two stages (`reduce_sum_segments`, then `reduce_sum` over the segments): adjacent threads read adjacent values, and the order of the additions is still fixed. The backward pass of a layer norm over 16,384 rows, whose weight gradients are such sums, went from 11 ms to 1 ms. The other, the softmax, no longer runs in training: `CrossEntropy` replaced it.

What is left of the gap to PyTorch is in the attention (half the time of a step, its backward pass computing the forward one again) and in the products, which reach about 20 Top/s of the card's 71. Speed is the target of step T1.

## 15.7 Seeing where the time goes

`kapel_train::profile::Profiled<B>` wraps a backend and times each call by operation and direction (`"matmul"`, `"attention backward"`), waiting for the device before and after each. `report()` lists them, the longest first. The example `gradients` prints it with `--profile`; it is how the reductions above, and a scatter of the first loss that took three quarters of a step, were found.

The card's backend counts the bytes held by its live tensors: `Cuda::memory_in_use`, and `Cuda::peak_memory` since the last `reset_peak_memory`, the equivalents of `torch.cuda.memory_allocated` and `max_memory_allocated`. Buffers kept for reuse are not counted.
