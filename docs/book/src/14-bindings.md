# 14. Kapel from C and Python: kapel-runtime, kapel-capi and kapel-python

Everything so far is a Rust library, generic over its backend. Other programs reach Kapel through two bindings: a C library with a stable header, which any language with a foreign function interface can call, and a Python module. Both are thin. The work is done by `kapel-runtime`, and the bindings only translate types, check pointers and keep errors from crossing the boundary as anything but errors.

## 14.1 One type for every device: kapel-runtime

A `Pipeline<B>` (chapter 11) is generic over its backend `B`, chosen when the program is compiled. A C or Python program chooses the device when it runs. `kapel_runtime::Model` closes that gap: it holds a pipeline on the processor or on a card, behind an enum, and a macro dispatches each call to whichever it holds.

```rust
let model = Model::load("HuggingFaceTB/SmolLM2-135M-Instruct", &LoadOptions::default())?;
let prompt = model.chat_prompt(&[("user", "What is the capital of France?")])?;
let options = GenerateOptions::from(model.default_request());
let outcome = model.generate(&prompt, &options, |piece| {
    print!("{piece}");
    true            // false stops the generation
})?;
```

Its options are those of `kapel run` and `kapel serve`, in the same words, so that documentation written for the command serves the bindings: `LoadOptions::parse` reads `"cuda:1"`, `"bf16"`, `"int4/64"`, `"f8e4m3"` as the command line does. `Model::load` takes a directory, a GGUF file or a Hub repository; `tokenize`, `detokenize` and `chat_prompt` prepare text; `generate` calls back with each piece of final text (chapter 11.6) and returns an `Outcome`: the text, why it ended, and the token counts. `Model::serve(options)` starts the server of chapter 12 and returns at Ctrl-C.

A `Model` can be shared between threads: each call to `generate` makes its own cache. The CUDA backend comes with the `cuda` feature, as for the command.

## 14.2 The C API: kapel-capi

`crates/kapel-capi` builds `libkapel` as a shared library (`libkapel.so`, `libkapel.dylib`, `kapel.dll`) and a static one, and its header `crates/kapel-capi/include/kapel.h` is the reference: each function of the crate implements the one the header declares, and each `#[repr(C)]` structure mirrors its C twin field for field. A short program:

```c
kapel_model *model;
if (kapel_model_load("HuggingFaceTB/SmolLM2-135M-Instruct", NULL, &model) != KAPEL_OK) {
    fprintf(stderr, "%s\n", kapel_last_error());
    return 1;
}
kapel_message messages[] = {{"user", "What is the capital of France?"}};
uint32_t *prompt; size_t len;
kapel_chat_prompt(model, messages, 1, &prompt, &len);
kapel_generation answer;
kapel_generate(model, prompt, len, NULL, print_piece, NULL, &answer);
kapel_string_free(answer.text);
kapel_tokens_free(prompt, len);
kapel_model_free(model);
```

The rules of the API are few, and the header states them.

- **Status codes.** Every call that can fail returns a `kapel_status`: `KAPEL_OK`, `KAPEL_INVALID_ARGUMENT` (a null pointer, text that is not UTF-8, an unknown option), `KAPEL_ERROR` (a model that cannot be read, a pass that failed) or `KAPEL_PANIC`. The message of the last failure is kept per thread, in a `thread_local!`, and `kapel_last_error()` gives it until the next call on that thread.
- **No panic crosses into C.** Unwinding through a C frame is undefined behaviour. Every exported function runs its body inside `std::panic::catch_unwind`; a panic, which would be a bug in Kapel, becomes `KAPEL_PANIC` with its message, and the model involved may be left unusable. Even `kapel_model_free` drops the model inside `catch_unwind`.
- **Who frees what.** Kapel allocates token arrays and strings with Rust's allocator; the caller gives them back with `kapel_tokens_free` and `kapel_string_free`, never `free()`. Null is accepted everywhere a pointer is freed.
- **Structures that can grow.** Each option structure (`kapel_load_options`, `kapel_generate_options`, `kapel_serve_options`) begins with its `size`, which its `*_init` function sets. The library reads `min(size, its own size)` bytes and keeps its defaults for the rest. A program compiled against an older, shorter header keeps working with a newer library; a test builds such a shorter structure and checks it. `KAPEL_API_VERSION` changes only when a change breaks programs built against the previous one.
- **Callbacks.** `kapel_generate` calls a `kapel_piece_callback` with each piece of text (its bytes, followed by a NUL, and the caller's `user_data`); returning nonzero stops the generation, which ends with `KAPEL_FINISH_CANCELLED`.

Two tests keep the header honest. `tests/header.rs` reads `kapel.h` and the crate's source and checks that every function the header declares is exported and every exported function is declared. `tests/c_example.rs` compiles `examples/c/generate.c` with the system's C compiler against the library of the build, runs it on SmolLM2-135M-Instruct, and checks that it prints the greedy answer the Rust side gives. It runs on Linux and macOS, and is skipped, saying so, without a compiler or the model.

## 14.3 The Python module: kapel-python

`crates/kapel-python` is a PyO3 extension built by maturin into a wheel named `kapel`. It targets CPython's stable ABI from 3.9 (`abi3-py39`): one wheel per platform serves every Python version from 3.9 on.

```python
import kapel

model = kapel.Model("HuggingFaceTB/SmolLM2-135M-Instruct")
print(model.chat([{"role": "user", "content": "What is the capital of France?"}]))
for piece in model.generate("Once upon a time", max_tokens=64, stream=True):
    print(piece, end="", flush=True)

kapel.serve("HuggingFaceTB/SmolLM2-135M-Instruct", port=8000)
```

`kapel.Model(name, device=, dtype=, quantize=, kv_cache=, revision=)` loads a model; `tokenize`, `detokenize`, `chat_prompt`, `generate` (from text or token ids) and `chat` (from messages) follow `kapel-runtime`. The sampling arguments default to the model's own settings. Failures raise `kapel.KapelError`; wrong arguments raise Python's usual `TypeError` and `ValueError`.

Two details matter to Python programs.

- **The interpreter's lock is released** while a model loads, generates or serves (`Python::detach`): other Python threads run meanwhile, and several threads can generate with one model at once.
- **Streaming.** With `stream=True`, `generate` and `chat` return a `Stream`, an iterator. The generation runs on a thread of its own and sends its pieces through a bounded channel of 16; `__next__` waits for the next piece with the lock released. When the iterator is dropped, the channel closes, and the generation stops at its next piece.

`kapel.pyi` gives the types and documentation to editors and type checkers; `crates/kapel-python/tests/test_kapel.py` checks the module with pytest on SmolLM2-135M-Instruct: errors of each kind, the chat template, greedy answers the same whole or streamed, seeds, stop strings, a stream left early that stops, and a server that answers until Ctrl-C.

## 14.4 Building and testing the bindings

```
cargo test -p kapel-capi                                   # the header, the C example
cargo build --release -p kapel-capi                        # target/release/libkapel.*
pip install maturin pytest
maturin develop --release -m crates/kapel-python/Cargo.toml
pytest crates/kapel-python/tests
```

The Python module links to libpython outside of maturin, which not every machine has, so `cargo test --workspace` leaves it out in continuous integration (`--exclude kapel-python`). The `bindings` job builds and tests both bindings on Linux, macOS and Windows, with the model cached between runs (chapter 18).
