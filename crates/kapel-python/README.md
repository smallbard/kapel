# kapel for Python

Kapel is an inference engine for small language models, written in Rust. This
module loads a model, generates text and serves it, from Python.

```python
import kapel

model = kapel.Model("HuggingFaceTB/SmolLM2-135M-Instruct")
print(model.chat([{"role": "user", "content": "What is the capital of France?"}]))

for piece in model.generate("Once upon a time", max_tokens=64, stream=True):
    print(piece, end="", flush=True)
```

A model is a model directory, a GGUF file, or a repository of the Hugging Face
Hub, downloaded once into `~/.cache/kapel/hub` (or `$KAPEL_CACHE`; a gated model
needs `HF_TOKEN`). The options are those of `kapel run`, in the same words:
`device="cuda:0"`, `dtype="bf16"`, `quantize="int4"`, `kv_cache="f8e4m3"`.

`kapel.serve(name, port=8000)` serves the model behind an HTTP API compatible
with OpenAI's, until Ctrl-C.

## Building

```sh
pip install maturin
maturin develop --release -m crates/kapel-python/Cargo.toml        # the processor
maturin develop --release -m crates/kapel-python/Cargo.toml -F cuda # and NVIDIA cards
```

The tests: `pip install pytest && pytest crates/kapel-python/tests`. They need
SmolLM2-135M-Instruct in the cache of the Hub, or `KAPEL_TEST_MODEL`, and are
skipped otherwise.
