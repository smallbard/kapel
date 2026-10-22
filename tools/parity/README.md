# Parity with the Python ecosystem

These scripts write reference outputs of `transformers`, `tokenizers` and `gguf-py`
into `data/parity/` (not in git). Kapel's tests compare against them when they are
present. When they are not (as in continuous integration), the parity tests pass
without checking anything, and print a line such as
`skipped: data/parity missing (see tools/parity/README.md)` on stderr. To make a
missing reference an error instead, set `KAPEL_REQUIRE_PARITY=1`:

```
KAPEL_REQUIRE_PARITY=1 cargo test --release --workspace
```

The parity tests are `gguf::tests::every_format_dequantizes_as_llama_cpp_does`
(kapel-hub), `tests::encodings_match_the_tokenizers_library` and
`chat::tests::renderings_match_transformers` (kapel-tokenizers),
`logits_match_transformers_and_survive_a_save` (kapel-models), and, on an NVIDIA card,
`models_on_the_card_match_transformers` (kapel-cuda, feature `cuda`: `f32` against
`transformers`, `bf16` against the processor's `bf16`). The generation tests of
kapel-infer (`tests/generation.rs`) compare greedy generations of SmolLM2-135M.

## Generations

`generation_reference.py` saves the greedy generations of `transformers` (256 tokens
after fixed prompts, and after a chat message when the model has a template) in
`data/parity/generation/<name>.json`, in `f32`, and on a card in `bf16`. The models
are read from the cache of `huggingface_hub`, where Kapel reads them too. The
example `generation_parity` of kapel-cli generates the same on the processor in
`f32`, and with the feature `cuda` on the card in `bf16`, and tells where each
generation first differs and by how much the two tokens' logits differ there.

In `f32`, the generations are identical. In `bf16`, rounding decides ties between
tokens early (the logits themselves are rounded to `bf16`): `transformers`' own
`bf16` generations leave its `f32` ones after a few dozen tokens. Kapel's `bf16`
generations are compared to the `f32` ones too, and should follow them about as
long.

Chat templates that write today's date (`strftime_now`, as Llama 3's does) are
rendered at a fixed time on both sides: `FIXED_NOW` in `chat_reference.py` and
`PARITY_TIME` in `crates/kapel-tokenizers/src/chat.rs` must stay equal. Regenerate
`chat_cases.json` after changing them.

```
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install transformers tokenizers safetensors gguf numpy jinja2
.venv/bin/python tools/parity/reference.py             # small models and their logits
.venv/bin/python tools/parity/tokenizers_reference.py  # tokenizer corpus
.venv/bin/python tools/parity/chat_reference.py        # chat templates
.venv/bin/python tools/parity/gguf_reference.py        # GGUF quantization formats
cargo test --release --workspace
cargo run --release -p kapel-models --example parity   # a table of logit differences

.venv/bin/python tools/parity/generation_reference.py qwen3-0.6b  # greedy generations
cargo run --release -p kapel-cli --features cuda --example generation_parity
```
