# Python

## Install

```sh
pip install ./crates/kapel-python               # from a copy of Kapel's repository
```

The module works with every Python from 3.9 on. For NVIDIA cards, build it with the `cuda` feature (the CUDA toolkit is needed, as for the command):

```sh
pip install maturin
maturin build --release -m crates/kapel-python/Cargo.toml -F cuda -o dist
pip install dist/kapel-*.whl
```

## Generate

```python
import kapel

model = kapel.Model("Qwen/Qwen2.5-0.5B-Instruct")          # device="cuda", quantize="int8", ...

print(model.chat([{"role": "user", "content": "What is the capital of France?"}]))

for piece in model.generate("Once upon a time", max_tokens=64, stream=True):
    print(piece, end="", flush=True)
```

`kapel.Model(name, device="cpu", dtype=None, quantize=None, kv_cache=None, revision="main")` loads a model directory, a GGUF file or a repository of the Hub, as `kapel run` does.

| Method | Gives |
| --- | --- |
| `chat(messages, ...)` | the answer to a conversation, through the model's chat template |
| `generate(prompt, ...)` | what follows a text, or a list of token ids, without the template |
| `chat_prompt(messages)` | the token ids of a conversation, to give `generate` |
| `tokenize(text, add_special_tokens=True)`, `detokenize(ids, skip_special_tokens=False)` | tokens and text |
| `has_chat_template` | whether the model has a chat template |

`generate` and `chat` take `max_tokens`, `temperature` (0 for greedy), `top_k`, `top_p`, `min_p`, `repetition_penalty`, `seed` and `stop`; by default, the model's own settings. With `stream=True` they return an iterator of pieces of text; leaving the loop early stops the generation.

Failures raise `kapel.KapelError`; a wrong option raises `ValueError`.

## Threads

The module releases Python's global lock while a model loads or generates: other threads run meanwhile, and several threads can generate with the same model at once.

```python
from concurrent.futures import ThreadPoolExecutor

questions = ["What is 2 + 2?", "Name a planet.", "Spell 'cat' backwards."]
with ThreadPoolExecutor() as pool:
    answers = list(pool.map(lambda q: model.chat([{"role": "user", "content": q}], max_tokens=32), questions))
```

Each thread computes its own answer. To compute many answers *together*, which is much faster on a card, use the server.

## Serve

```python
kapel.serve("Qwen/Qwen2.5-0.5B-Instruct", port=8000, device="cuda")   # until Ctrl-C
```

It takes the options of `kapel serve`: `host`, `port`, `served_model_name`, `max_model_len`, `cache_memory`, `gpu_memory_utilization`, `max_sequences`, `max_pass_tokens`, `max_requests`, `prefix_sharing`, `timeout`, and those of loading.

The module ships type annotations (`kapel.pyi`): editors complete and check its calls.
