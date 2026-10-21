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

Linear layers are stored by `transformers` as `[out, in]`. Kapel computes `x · wᵀ` on a transposed view of the weight, so no copy is made: the backends read the transposition through strides.

## 10.3 The forward pass

`Model::forward(backend, ids)` returns the logits of every position. `Model::step(backend, ids, cache)` returns only the last position's logits and appends the new keys and values to a `Cache`. Both run the same code:

1. **Embed.** `GatherRows` picks the rows of the embedding table. Gemma multiplies them by √hidden, rounded to the activation type as `transformers` rounds it. GPT-2 adds learned position embeddings.
2. **For each layer**, depending on the block layout:
   - *Sequential* (Llama, Mistral, Qwen): x + attention(norm(x)), then x + down(act(gate(h)) · up(h)).
   - *Sandwich* (Gemma 2): the same, with an extra norm on the output of the attention and of the MLP.
   - *Parallel* (Phi): x + attention(h) + MLP(h), from a single norm.
   - *GPT-2*: norm, attention, add, norm, MLP, add, with weights stored `[in, out]`.
3. **Attention.** Project q, k and v, with biases when the model has them; GPT-2 slices a fused projection. Reshape to `[1, heads, T, head_dim]`. Qwen 3 normalizes each head of q and k. Apply `Rope` at the absolute positions. Concatenate keys and values to the cache. Call `Attention` with causality, the layer's window, the model's scale and softcap. Grouped-query attention needs nothing special: the operation receives fewer key heads than query heads. Project the output.
4. **Final norm** and **output layer**, the embedding table when weights are tied. Gemma 2 caps the logits smoothly: cap · tanh(x / cap).

The cache holds, per layer, keys and values of shape `[1, kv_heads, length, head_dim]`. Each step concatenates the new positions onto them, which copies the cache; the paged cache of the inference plan will replace this (chapter 13).

## 10.4 Parity with transformers

The models are checked against `transformers`, logit for logit. `tools/parity/reference.py` saves, for each family, a tiny random test model from the Hub plus SmolLM2-135M, together with the logits `transformers` computes for three fixed texts. The test `logits_match_transformers_and_survive_a_save` requires a relative difference below 2e-5, then saves the model with `Model::save`, reloads it, and checks again. A second test checks that a prefill followed by single steps gives the same logits as the whole sequence at once.

The recorded results are an absolute difference of at most 3e-7 on the tiny models, and 7e-6 relative on SmolLM2, whose logits reach 36. On the GPU in f32, the difference stays within 9e-6 relative. The reference data lives in `data/parity`, outside git; without it, the tests say they were skipped, and they fail if `KAPEL_REQUIRE_PARITY=1` is set.
