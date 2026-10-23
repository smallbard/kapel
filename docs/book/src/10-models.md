# 10. Models: kapel-models

`kapel-models` describes transformer architectures by configuration rather than by code per model. One `Config` read from `config.json`, and one forward pass written against the `Backend` trait, cover seven model families. The forward pass is a sequence of catalogue operations, so it runs unchanged on the reference, the CPU and the GPU.

## 10.1 From config.json to Config

`Config::from_json` reads the fields of `transformers` configurations, with their aliases (`hidden_size` or `n_embd`, `num_hidden_layers` or `n_layer`, and so on), applies the same defaults, and classifies the model by its `model_type`:

| `model_type` | Block layout | Normalization | Notable features |
| --- | --- | --- | --- |
| `llama`, `mistral` | sequential | RMSNorm | Mistral slides a window on every layer |
| `qwen2` | sequential | RMSNorm | biases on q, k, v |
| `qwen3` | sequential | RMSNorm | an RMSNorm per head on q and k |
| `gemma2` | sandwich | RMSNorm, weights stored as w − 1 | softcapped scores and logits, scaled embeddings, a window on every other layer |
| `phi` | parallel | LayerNorm with bias | partial rotary dimensions |
| `gpt2` | GPT-2 | LayerNorm with bias | learned positions instead of RoPE, fused q-k-v |

Anything else is "unsupported architecture". The configuration also yields the activation function (SiLU, exact or tanh GELU, ReLU), the epsilon of the norms, RoPE's theta, rotary dimensions and scaling (linear, dynamic NTK, Llama 3, YaRN), the sliding windows per layer, attention and logit softcaps, whether the output layer shares the embedding table, and the positions the model was trained for (`max_positions`, from `max_position_embeddings` or `n_positions`, if the file says), which a server takes as the longest sequence it accepts.

A configuration file is untrusted input too. Sizes are bounded (at most 4,096 layers and heads, a hidden size of 2²⁰, a vocabulary of 2²⁴), chosen so that a product of two of them fits in 48 bits, and every derived size uses checked arithmetic. Fuzzing found the reason for these bounds: a configuration announcing 1.2 billion layers used to allocate 19 GB.

## 10.2 Weights

`Model::load(&checkpoint)` keeps every tensor of the checkpoint under its Hugging Face name, still memory-mapped. `Model::upload(&backend)` moves them to the backend's device and `Model::cast(&backend, dtype)` converts float weights, for instance to bf16, leaving quantized and integer weights alone.

Linear layers are stored by `transformers` as `[out, in]`. Kapel computes `x · wᵀ` on a transposed view of the weight, so no copy is made: the backends read the transposition through strides. A quantized weight keeps the same layout, its blocks along `in`, and goes to `QuantizedMatmul` instead of `Matmul`.

**Quantizing at loading.** `Model::quantize(dtype, output)` quantizes the weights of the linear projections (q, k, v, o, gate, up, down, and Phi's dense, fc1 and fc2) by rounding to the nearest (chapter 3), and with `output` the output table too: the embeddings when the model ties them to it, which `GatherRows` then reads quantized. It works on pieces of rows on every core, from the mapped weights, so a large table is never expanded whole. Norms, biases and GPT-2's transposed projections keep their type. On Llama 3.2 1B, int8 costs almost nothing (a wikitext-2 perplexity of 15.61 against 15.59 in f32) and int4 by rounding costs 16 % (18.16); GPTQ, planned in step I5, does better.

## 10.3 Models from GGUF

`Model::from_gguf(&gguf)` reads a model of llama.cpp: Llama (and Mistral, which GGUF files as Llama), Qwen 2 and Qwen 3. The metadata (`llama.block_count`, `llama.attention.head_count_kv`, …) is translated into the `config.json` of `transformers` it came from, and the tensors are renamed (`blk.0.attn_q.weight` to `model.layers.0.self_attn.q_proj.weight`): from there, everything is the model of section 10.2, its weights read in place from the mapped file, quantized or not. Two conversions of llama.cpp's converter are undone in the configuration rather than in the weights:

- For Llama, the query and key projections are stored with each head's components reordered, so that RoPE pairs neighbours (2j with 2j + 1) instead of halves (j with j + D/2). Attention only sees products of queries and keys, so the reordered weights with the interleaved RoPE layout give the original model; Kapel sets `RopeLayout::Interleaved` and copies nothing.
- The RoPE scaling of Llama 3 is stored as its per-frequency factors (`rope_freqs.weight`). Kapel recovers the parameters that give them (factor, low and high frequency factors, original context) and checks that they reproduce every factor.

The tokenizer, chat template and end tokens of the file are read by `kapel-infer` (chapter 11).

## 10.4 The forward pass

`Model::forward(backend, ids)` returns the logits of every position, without a cache: the pass of training, every operation with a gradient. `Model::step(backend, ids, cache)` returns only the last position's logits and adds the new keys and values to a `Cache`; `Model::extend` does the same with the logits of every new position, to score a text by pieces or check proposed tokens. `step_batch` and `extend_batch` do it for several sequences of one `KvCache` at once (section 10.5). All of them run the same code:

1. **Embed.** `GatherRows` picks the rows of the embedding table. Gemma multiplies them by √hidden, rounded to the activation type as `transformers` rounds it. GPT-2 adds learned position embeddings.
2. **For each layer**, depending on the block layout:
   - *Sequential* (Llama, Mistral, Qwen): x + attention(norm(x)), then x + down(act(gate(h)) · up(h)); a fused model computes gate and up in one product and joins them with `Glu`.
   - *Sandwich* (Gemma 2): the same, with an extra norm on the output of the attention and of the MLP.
   - *Parallel* (Phi): x + attention(h) + MLP(h), from a single norm.
   - *GPT-2*: norm, attention, add, norm, MLP, add, with weights stored `[in, out]`.
3. **Attention.** Project q, k and v, with biases when the model has them; GPT-2 and fused models slice one projection. Reshape to `[T, heads, head_dim]`, a view. Qwen 3 normalizes each head of q and k. Apply `Rope` at the absolute positions. Without a cache, call `Attention` with causality, the layer's window, the model's scale and softcap. With one, write the new keys and values into the cache's tables in place (`Write::write_rows`), then call `PagedAttention`, which finds each sequence's positions through its pages (section 10.5). Grouped-query attention needs nothing special: the operation receives fewer key heads than query heads. Project the output.
4. **Final norm** and **output layer**, the embedding table when weights are tied. Gemma 2 caps the logits smoothly: cap · tanh(x / cap).

**Fusing.** `Model::fused(backend)` stacks the query, key and value projections of each layer into one weight (`qkv_proj`), and the gate and up projections into another (`gate_up_proj`). The rows of the products are the same, but there is one product instead of three, and one plus a `Glu` instead of two products and two element-wise operations: a card reads larger weights faster, and launches fewer kernels. Only float weights of one type are fused; quantized projections keep their own.

## 10.5 The cache by pages

`KvCache` holds the keys and values of every sequence in progress. Each attention layer has two tables, `[lines, kv_heads, head_dim]`, on the backend, written in place. A sequence does not own neighbouring lines: lines are handed out by pages of `PAGE` = 16, taken from a common reserve as the sequence grows and given back when it ends, as an operating system hands out memory. Each sequence keeps the list of its pages, and `PagedAttention` finds its keys through it. Writing a position is a write into a free line: nothing is copied as a sequence grows, and ending one only touches lists.

- `add_sequence` starts a sequence and gives its `SequenceId`; `remove_sequence` gives its pages back; `truncate` brings it back to its first positions, as speculative decoding takes back refused tokens.
- `KvCache::new` grows: when a sequence needs more pages than are free, its tables are copied into tables twice as large, and the pages keep their numbers. `KvCache::with_pages(n)` has a fixed number of pages, as a server sizes it for its device, and refuses positions beyond them with `ModelError::CacheFull { needed, free }`; `pages_needed` and `free_pages` say beforehand whether a pass fits.
- `with_dtype` chooses the tables' type, by default that of the keys: `f16`, or `f8e4m3` and `f8e5m2` for half the memory of `bf16`, the keys and values rounded as they are written.

`Cache`, which `Model::step` takes, is a `KvCache` that holds one sequence.

**Shared beginnings.** With `with_prefix_sharing`, sequences share the pages of their common beginning, as the requests of a server share a system prompt. Each full page is indexed by a key: a chained hash of every token before it and of its own (`PageKey`), with the page's sixteen tokens and the hash of its parent kept beside it, so that a match is checked token for token rather than trusted to the hash. `add_sequence_sharing(prompt)` starts a sequence on the pages of the longest beginning of `prompt` the index holds, whole pages short of its last token, which is left to compute for its logits; it returns the positions already held, and the caller computes only the rest. The tokens of each pass are noted as they are written, and each page they fill is offered to the index (the first page of a key wins; an identical page computed twice stays private).

Each page counts the sequences that use it. When the last one ends, an indexed page becomes *idle*: free, and still findable. `free_pages` counts idle pages as free, and when a sequence needs a page, a page never indexed goes first, then the idle page let go least recently, taken out of the index. A sequence gives its pages back last page first, so that the end of a prompt is taken back before its beginning, which more sequences are likely to share. Pages are written only past the end of their sequence, so a shared page, full, is never written again, with one exception: `truncate` inside a shared page. The sequence then gets a page of its own and a copy of the lines it keeps; the copy waits for the next pass that has a backend (`begin`, `reserve`), and meanwhile `fits` says no, so that no recorded pass replays before it. A page cut that no one else uses only leaves the index. After a pass that failed, `clear_prefixes` forgets the idle pages, whose contents may be wrong. `shared_positions` counts the positions taken rather than computed.

**Batches.** `Model::step_batch(backend, cache, batch)` computes, in one pass, the new tokens of several sequences, each entry a sequence and its tokens: the prompts of new sequences, or the last token of sequences being decoded. The tokens of all entries form one list of rows; each knows its sequence and how many positions it sees, which is all `PagedAttention` needs, so sequences of different lengths need no padding. The products have one row per token of every sequence, so the weights are read once for the whole batch: in decoding, where a sequence alone mostly waits for memory, this is what makes a batch cheaper than its sequences one at a time. It returns the logits of each entry's last token; `extend_batch` those of every token.

**Recorded steps.** On a backend that records passes (`Record`, chapter 4; a CUDA graph on the card, chapter 7), `Model::record_step(backend, cache, batch, rows, pages)` computes one token of each sequence as `step_batch` does, then records the same pass with `rows` rows and a table of `pages` pages per sequence. `replay_step(backend, cache, recorded, batch)` runs the recording again for any batch of the same cache with as many sequences or fewer, each within those pages: the tokens, positions, lines, pages and rows are copied into the recording's inputs, and the whole pass is launched in one call instead of one per kernel. The engine of a server (chapter 11) thus records one pass per size of batch, not one per set of sequences, which arrive and leave at every step.

The rows past the batch are padding: each writes one token into a line of its own and sees only its page up to that line (`Plan::pad`). Their lines come from scratch pages, taken from the reserve once and never given back (`scratch_lines`); a recording of `rows` rows takes, when it is made, the lines of its smallest batch, `rows − 1`, so that a replay never needs more. `reserve_padding(rows)` sets them aside beforehand, so that recording a pass never takes pages a pass was counting on; scratch lines are taken all or none. A cache that grows first reserves the pages of its sequences up to the room asked, so that a replay never grows the tables. When the tables grow anyway, a sequence needs more pages than the recording reads, the batch is larger than its rows, or a copy of pages waits, `replay_step` returns `None` and the caller records again.

`record_step` says what it did with `Recorded`: `Nothing` (the backend does not record, or the cache lacks the room; nothing computed), `Made(step, logits)`, or `Unrecorded(logits, error)` when the backend failed to record, for lack of memory say: the step was computed by the eager pass before the capture, so the caller goes on with its logits and only loses the recording. `Cache::recorded_step` records a single sequence with room for `ROOM` = 256 more positions.

## 10.6 Parity with transformers

The models are checked against `transformers`, logit for logit. `tools/parity/reference.py` saves, for each family, a tiny random test model from the Hub plus SmolLM2-135M, together with the logits `transformers` computes for three fixed texts. The test `logits_match_transformers_and_survive_a_save` requires a relative difference below 2e-5, then saves the model with `Model::save`, reloads it, and checks again. Other tests check that a prefill followed by single steps gives the same logits as the whole sequence at once, that sequences computed in one batch give the logits they give alone, and that a fused model gives the logits of the model. The cache's own tests follow pages: whole pages taken and given back, a fixed cache refusing positions, a growing one keeping its page numbers, sequences sharing the pages of a common beginning (and a prompt of exactly two pages keeping its last token to compute), idle pages taken back last page first, and a sequence cut inside a shared page writing into its own copy while the other still finds its pages.

The recorded results are an absolute difference of at most 3e-7 on the tiny models, and 7e-6 relative on SmolLM2, whose logits reach 36. On the GPU in f32, the difference stays within 9e-6 relative. Models read from GGUF are checked against the models they come from (`kapel-infer/tests/gguf.rs`): the same tokens from both tokenizers, and, in Q8\_0, the same most likely token at 95 % of the positions at least. The reference data lives in `data/parity`, outside git; without it, the tests say they were skipped, and they fail if `KAPEL_REQUIRE_PARITY=1` is set.
