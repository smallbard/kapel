# Parity with the Python ecosystem

These scripts write reference outputs of `transformers`, `tokenizers` and `gguf-py`
into `data/parity/`. Kapel's tests compare against them when they are present, and
check nothing when they are not (as in continuous integration).

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
```
