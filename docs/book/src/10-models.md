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

Anything else is "unsupported architecture". The configuration also yields the activation function (SiLU, exact or tanh GELU, ReLU), the epsilon of the norms, RoPE's theta, rotary dimensions and scaling (linear, dynamic NTK, Llama 3, YaRN), the sliding windows per layer, attention and logit softcaps, and whether the output layer shares the embedding table.

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

`Model::forward(backend, ids)` returns the logits of every position. `Model::step(backend, ids, cache)` returns only the last position's logits and appends the new keys and values to a `Cache`; `Model::extend` does the same with the logits of every new position, to score a text by pieces or check proposed tokens. All three run the same code:

1. **Embed.** `GatherRows` picks the rows of the embedding table. Gemma multiplies them by √hidden, rounded to the activation type as `transformers` rounds it. GPT-2 adds learned position embeddings.
2. **For each layer**, depending on the block layout:
   - *Sequential* (Llama, Mistral, Qwen): x + attention(norm(x)), then x + down(act(gate(h)) · up(h)).
   - *Sandwich* (Gemma 2): the same, with an extra norm on the output of the attention and of the MLP.
   - *Parallel* (Phi): x + attention(h) + MLP(h), from a single norm.
   - *GPT-2*: norm, attention, add, norm, MLP, add, with weights stored `[in, out]`.
3. **Attention.** Project q, k and v, with biases when the model has them; GPT-2 slices a fused projection. Reshape to `[1, heads, T, head_dim]`. Qwen 3 normalizes each head of q and k. Apply `Rope` at the absolute positions. Concatenate keys and values to the cache. Call `Attention` with causality, the layer's window, the model's scale and softcap. Grouped-query attention needs nothing special: the operation receives fewer key heads than query heads. Project the output.
4. **Final norm** and **output layer**, the embedding table when weights are tied. Gemma 2 caps the logits smoothly: cap · tanh(x / cap).

The cache holds, per layer, keys and values of shape `[1, kv_heads, length, head_dim]`. Each step concatenates the new positions onto them, which copies the cache; the paged cache of the inference plan will replace this (chapter 13).

## 10.5 Parity with transformers

The models are checked against `transformers`, logit for logit. `tools/parity/reference.py` saves, for each family, a tiny random test model from the Hub plus SmolLM2-135M, together with the logits `transformers` computes for three fixed texts. The test `logits_match_transformers_and_survive_a_save` requires a relative difference below 2e-5, then saves the model with `Model::save`, reloads it, and checks again. A second test checks that a prefill followed by single steps gives the same logits as the whole sequence at once.

The recorded results are an absolute difference of at most 3e-7 on the tiny models, and 7e-6 relative on SmolLM2, whose logits reach 36. On the GPU in f32, the difference stays within 9e-6 relative. Models read from GGUF are checked against the models they come from (`kapel-infer/tests/gguf.rs`): the same tokens from both tokenizers, and, in Q8\_0, the same most likely token at 95 % of the positions at least. The reference data lives in `data/parity`, outside git; without it, the tests say they were skipped, and they fail if `KAPEL_REQUIRE_PARITY=1` is set.
