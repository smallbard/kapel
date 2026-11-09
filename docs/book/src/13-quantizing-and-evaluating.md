# 13. Quantizing and evaluating: kapel-quant and kapel eval

Chapters 6, 7 and 10 run quantized models: weights read from GGUF files, or rounded to int8 and int4 as they are loaded. This chapter is about making them. `kapel-quant` quantizes a model with data, by GPTQ and AWQ, and `kapel quantize` writes the result to a GGUF file that llama.cpp reads, or to a model directory. To know what a quantization cost, `kapel eval` measures perplexities and task accuracies exactly as lm-evaluation-harness does.

## 13.1 Rounding is not enough

Rounding to the nearest (RTN) treats each block of weights alone: the scale of a block of 32 is chosen from its largest weight, and each weight goes to the closest of the sixteen codes of Q4\_0. It needs no data, and it is what `Model::quantize` and llama.cpp's `llama-quantize` do. But the model never sees a weight alone. A projection computes `y = W x`, and what matters is the error of `y`, over the inputs `x` the model meets.

For a row `w` of `W` changed by `δ`, the output error over inputs `x₁ … xₙ` is

```text
Σᵢ (δ · xᵢ)² = δ H δᵀ      with  H = Σᵢ xᵢ xᵢᵀ   [K, K]
```

`H` is the *Hessian* of the projection's inputs: its diagonal says how large each input channel is, and its other terms how channels move together. Both methods of this chapter read `H` from calibration texts and spend the rounding error where `H` says it costs least. They differ in what they change.

## 13.2 Calibration, block by block

`kapel_quant::quantize(model, backend, compute, calibration, options, log)` takes the model as loaded (in host memory), a backend to compute on, the type of those computations (`bf16` on a card, `f32` on the processor), and the calibration texts as token ids, one sequence each. `kapel quantize` draws them from a text file: `--samples` windows (128) of `--length` tokens (512) at random places, each after the beginning of sequence if the tokenizer adds one. The draw follows `--seed`, so the same command gives the same file.

The model is computed one block at a time, never whole. Two pieces of `kapel-models` make that possible:

- `Model::embed(ids)` gives the hidden states of a sequence before the first block.
- `Model::block(layer, x, observe)` passes them through one block, without a cache. `observe` is called with the input of each group of projections that read one input, a `Site`: `Attention` (the normed states, read by the query, key and value projections), `Output` (the attention's output), `Mlp` (the normed states read by gate and up), `Down` (the gated unit's output).

For each block, `quantize` runs every calibration sequence through it, accumulates `H = Σ xᵀx` per site on the backend in `f32` (one product per sequence), quantizes the block's projections, then runs the sequences through the *quantized* block to get the inputs of the next one. Each block is thus calibrated on what the quantized model actually feeds it, and the errors of earlier blocks are partly corrected by later ones. The test `blocks_one_at_a_time_give_the_whole_pass` checks that `embed` then `block` for each layer gives the logits of `forward`.

The model is kept twice during the work: in host memory, as it will be written, and on the backend, in the type of the computations, as the next blocks see it. `Model::tensor`, `Model::names` and `Model::replace` read and swap single weights in both.

GPTQ and AWQ compute blocks of Llama's kind (Llama, Mistral, Qwen, Gemma); other models are refused with an error that says so, and can still be rounded.

## 13.3 GPTQ: errors carried forward

GPTQ (Frantar et al., 2022) rounds the columns of `W` one at a time, and after each one changes the columns not yet rounded so that they make up for its error. With `U` the upper triangular factor of `H⁻¹ = Uᵀ U`, rounding column `j` from `w_j` to `q_j` and carrying its error is:

```text
e    = (w_j − q_j) / U[j, j]
w_k −= e · U[j, k]          for every column k > j
```

When two input channels move together, an error on one is cancelled by a change on the other; when a channel is independent of the rest, `U[j, k]` is zero and GPTQ is plain rounding (a test checks that case). `crates/kapel-quant/src/gptq.rs` follows the paper's layout:

- **Batches of 128 columns.** Inside a batch, the error of each column reaches the batch's later columns at once; the errors of the whole batch reach the columns after it in one product.
- **Rows in parallel.** The rows of `W` are independent: pieces of 64 rows run on every core.
- **Scales chosen late.** A block of the format (32 weights for Q4\_0) takes its scale, and Q4\_1 its minimum, from its weights as they are when its first column comes, by the format's own rule. `kapel-core` exposes that rule in pieces: `block_scale` chooses the scale of a block, `block_code` the code of one weight under it, `block_value` the value of a code, and `encode_block` the bytes. Rounding to the nearest is these four in a row, and gives llama.cpp's bytes; GPTQ uses the same pieces, so its blocks are valid blocks of the format.
- **Damping.** `H` gets 1 % of the mean of its diagonal added to it (`Options::damping`), ten times more until it is positive definite. A channel no calibration input ever reached (zero on the diagonal) weighs nothing: its weights become zeros.

The factor `U` costs a few `K³` operations, with `K` up to some 15,000 for the down projection of an 8B model. `linalg.rs` computes it in `f64` on every core, by tiles of 64. With `P` the matrix that reverses the order of indices, the Cholesky factorization `P H P = L Lᵀ` gives `U = P L⁻¹ P`: one factorization and one triangular inverse, where the reference implementation does a Cholesky factorization, an inverse and a second Cholesky factorization.

## 13.4 AWQ: factors and clipping

AWQ (Lin et al., 2023) starts from an observation: a few input channels carry much larger activations than the others, and the weights that multiply them count most. A projection's output does not change if channel `c` is multiplied by `s_c` in the weights and divided by `s_c` in its inputs. Rounding then sees those weights `s_c` times larger, and loses relatively less on them.

`crates/kapel-quant/src/awq.rs` does it in two steps.

1. **Factors.** `s = m^α`, where `m` is the mean magnitude of each input channel over the calibration (accumulated with the Hessian), normalized so that `max(s) · min(s) = 1`. `α` is tried on a grid from 0 (no factor) to almost 1 (`Options::grid`, 20 steps as in the paper). Each try rounds the scaled weights of the site and measures `Σ δ H δᵀ` on the backend, with `H` adjusted for the scaling. The best `α` wins; at 0 nothing changes. The division goes into the operation before the site: the norm's weight for `Attention` and `Mlp` (Gemma's norms store `weight − 1`, and are adjusted for it), the rows of the value projection for `Output` when its width matches, the rows of the up projection for `Down`. The model computes the same function before rounding.
2. **Clipping.** Each block of each row may be brought within `r` times its largest magnitude, `r` from 1 down to one half, the ratio of smallest output error on that block (`δ H_bb δᵀ`, `H_bb` the part of `H` for its columns). Ratio 1 is among those tried, so clipping never makes a block worse. Queries and keys are not clipped: their outputs are seen only through their products, which the error of the projection alone does not measure (as in the paper).

The paper measures the error of the whole attention or MLP for the factors; here it is the projections' own outputs, from the same Hessians GPTQ uses. `Method::AwqGptq` applies AWQ's factors and clipping, then GPTQ.

## 13.5 Writing GGUF

`kapel quantize MODEL --output NAME.gguf` writes a file llama.cpp reads as if its own converter had made it. Three layers do the work.

- `Model::to_gguf(backend)` (`crates/kapel-models/src/gguf.rs`) is the inverse of `Model::from_gguf` (chapter 10): llama.cpp's names (`blk.0.attn_q.weight`), its metadata keys with the types it reads them as (`llama.block_count` as `u32`, `llama.rope.freq_base` as `f32`), Llama's query and key rows reordered so that RoPE pairs neighbours, Llama 3's RoPE scaling written as its per-frequency factors, and norms and biases in `f32`. Quantized weights keep their blocks: the reordering moves whole rows, and a row is whole blocks. Llama, Mistral, Qwen 2 and Qwen 3 are written; another architecture is an error.
- `save_gguf` (`crates/kapel-infer/src/gguf.rs`) adds the tokenizer: the tokens, merges, types and pre-tokenizer of `tokenizer.json`, the special tokens, the chat template, and the sampling defaults of `generation_config.json`; from a GGUF source, its own tokenizer metadata is copied.
- `gguf::write` (`crates/kapel-hub/src/gguf.rs`) writes version 3 of the format: the metadata in order (`Entry` carries each value with its GGUF type), then the tensor descriptions and the data, aligned on 32 bytes, streamed tensor by tensor through a temporary file renamed into place (chapter 8).

Formats: Kapel's int8 and int4 by groups (`int8/G`, `int4/G`) and GGUF's Q8\_0, Q4\_0 and Q4\_1; `f16`, `bf16` and `f32` convert without quantizing. Without `.gguf`, the output is a model directory: safetensors that Kapel reads back, with the tokenizer and template files copied from the source.

The test `quantization_gives_the_bytes_of_llama_cpp` compares Kapel's rounding with llama.cpp's `gguf-py` for every format written. On Llama 3.2 1B, `kapel quantize --format q4_0` gives the same bytes as `llama-quantize`, projection for projection.

## 13.6 Evaluating as lm-evaluation-harness does

A perplexity is only comparable to another computed the same way. Kapel follows lm-evaluation-harness, the reference of most published results, without running its Python on every model: the harness writes the requests of its tasks once, and Kapel scores them.

`tools/eval/export_tasks.py` asks lm-evaluation-harness for the requests of tasks (`wikitext`, `hellaswag`, `arc_easy`, `arc_challenge`, `mmlu`, …) without a model, and writes one JSON line per document: for a task by choice, the `(context, continuation)` pairs, the indices of the right answers (found by the task's own scoring rule) and the length of each answer; for a text, the text with its number of words and bytes.

`kapel eval MODEL --tasks FILE` (`crates/kapel-cli/src/eval.rs`) tokenizes and scores them as the harness's `hf` model does:

| Rule of lm-evaluation-harness | In Kapel |
| --- | --- |
| A context's trailing spaces begin its continuation | `Encoder::pair` |
| Context and continuation encoded together, split where the context alone ends | `Encoder::pair` |
| Special tokens added, unless the text starts with the prefix token | `Encoder::encode` |
| An empty context is the prefix token: `bos_token`, else `eos_token` | `prefix_token`, `--prefix-token` |
| Sequences longer than `max_position_embeddings` cut on the left | `truncate`, `--max-length` |
| A text scored by disjoint windows of that length | `windows` |

A task by choice gives `acc` (the answer of highest log-likelihood is right) and `acc_norm` (the log-likelihood divided by the answer's length in characters); a text gives perplexities by word and by byte, and bits per byte. Groups such as MMLU's 57 subjects are averaged weighted by their documents, as the harness weights them.

The scores come from `Scorer` (`crates/kapel-infer/src/score.rs`): `score(context, continuation)` gives the sum of the log-probabilities of the continuation's tokens, and whether each was the most likely. It keeps one cache from call to call. A sequence that starts like the previous one (the choices of a question share their context; the questions of an MMLU subject share a prompt) reuses the cached positions, minus one: the logits of the context's last token are needed and not kept, so that position is computed again. A model with linear attention, whose states cannot be brought back to an earlier position, starts again from an empty cache instead. The model reads at most `--piece` tokens per pass (512), and only the rows that score a token are computed through the output table (`Model::extend_last`). Each row is a whole vocabulary (151,936 logits for Qwen), so the rows are decoded to `f64` and scored on every core, each row alone and summed in order, which leaves the result independent of the number of threads; on a card this made WikiText 2.4 times faster. The test `scores_match_a_full_pass_whatever_the_cache_holds` checks the scores against a full forward pass, with pieces of three tokens and a cache that holds anything.

**Measures.** With SmolLM2-135M on the processor, 20 documents of each task, against lm-evaluation-harness 0.4.13 (`hf`, `float32`):

| Task | lm-evaluation-harness | Kapel |
| --- | --- | --- |
| WikiText, perplexity by word | 21.42474 | 21.42494 |
| WikiText, bits per byte | 0.829376 | 0.829378 |
| HellaSwag, acc / acc\_norm | 0.35 / 0.30 | 0.35 / 0.30 |
| ARC Easy, acc / acc\_norm | 0.55 / 0.40 | 0.55 / 0.40 |
| ARC Challenge, acc / acc\_norm | 0.15 / 0.25 | 0.15 / 0.25 |
| MMLU, acc | 0.242105 | 0.242105, each of the 57 subjects the same |

The perplexities differ by 0.001 %, the target being 0.5 %.

## 13.7 What a quantization costs

Llama 3.2 1B in Q4\_0, the embedding table kept in 16 bits in both cases, calibrated on wikitext-2's training set (128 windows of 512 tokens for Kapel; `llama-imatrix` with 128 chunks of 512 for llama.cpp). All perplexities by `llama-perplexity` (`-c 512`, the whole test set of wikitext-2), on an RTX 3090:

| Quantization | Perplexity | Loss | Time |
| --- | --- | --- | --- |
| `f16` | 10.1915 | — | — |
| llama.cpp, `llama-quantize` | 11.5043 | +1.313 | 4 s |
| llama.cpp, with an importance matrix | 10.9296 | +0.738 | 25 s |
| Kapel, rounding to the nearest | 11.5031 | +1.312 | 5 s |
| **Kapel, GPTQ** | **10.6066** | **+0.415** | 4 min |
| Kapel, AWQ | 11.0128 | +0.821 | 47 min |
| Kapel, AWQ then GPTQ | 10.6522 | +0.461 | 50 min |

GPTQ loses 44 % less than llama.cpp with its importance matrix, which weighs each weight's rounding by its channel's activations but carries no error forward. AWQ helps less here than in its paper: Q4\_0 already has a scale per block of 32 weights, finer than the groups of 128 AWQ was designed for, and its search is slow, a full rounding and error product per factor tried and a clipping search per block. Speeding it up, and measuring it on coarser formats, is left for later.
