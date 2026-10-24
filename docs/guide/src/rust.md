# Rust

Kapel is a set of Rust crates; a Rust program uses them directly. Until they are published, depend on them by path (or by Git):

```toml
[dependencies]
kapel-infer = { path = "../kapel/crates/kapel-infer" }
kapel-cpu = { path = "../kapel/crates/kapel-cpu" }
kapel-tokenizers = { path = "../kapel/crates/kapel-tokenizers" }
```

## Generate

```rust
use kapel_infer::{Pipeline, Request};
use kapel_tokenizers::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = kapel_hub::download::snapshot(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "main",
        &kapel_hub::download::default_cache(),
    )?;
    let pipeline = Pipeline::load(&dir, kapel_cpu::Cpu::new(), None)?;
    let prompt = pipeline.chat_prompt(&[Message::new("user", "What is the capital of France?")])?;
    for piece in pipeline.generate(&prompt, &Request::default())? {
        print!("{}", piece?.text);
    }
    Ok(())
}
```

(`kapel-hub` is needed too, for the download.) `Pipeline::load` takes a model directory or a GGUF file, and a backend: `kapel_cpu::Cpu` for the processor, `kapel_cuda::Cuda` for a card (the `cuda` feature of `kapel-cuda`). `Pipeline::load_with` takes `LoadOptions`: the type of the weights, a quantization at loading, the type of the cache. A `Request` holds the sampling settings (`pipeline.default_sampling()` gives the model's), `max_tokens`, `stop` strings and an optional `Constraint` (a JSON schema, a regular expression, a grammar, a list of choices).

The rest of the API, by task:

| Task | Where |
| --- | --- |
| Many requests computed together, coming and going | `Pipeline::engine`, `kapel_infer::Engine` |
| All requests at once | `Pipeline::batch` |
| Speculative decoding | `Pipeline::speculate`, `kapel_infer::Draft` |
| Log-likelihoods of continuations | `kapel_infer::Scorer` |
| An HTTP server in your program | `kapel_serve::serve` |
| Quantizing with GPTQ or AWQ | `kapel_quant::quantize`, then `kapel_infer::save_gguf` |
| The device chosen at run time, one type | `kapel_runtime::Model` |
| Tokenizers and chat templates alone | `kapel-tokenizers`, which depends on no other crate of Kapel |

## API documentation

```sh
cargo doc --workspace --no-deps --open
```

Every public item is documented, with its errors. *The Kapel Book* (`docs/book`) explains how the pieces fit together.
