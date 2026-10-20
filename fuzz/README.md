# Fuzzing

The readers of files that come from strangers (safetensors, GGUF, `tokenizer.json`,
chat templates, `config.json`) are fuzzed with libFuzzer. A target may return an
error; it must never panic, hang or read out of bounds.

```
cargo install cargo-fuzz
cd fuzz
./seed.sh                                   # seed corpora from data/parity
cargo +nightly fuzz run safetensors -- -max_total_time=86400
```

Targets: `safetensors`, `gguf`, `tokenizer_json`, `tokenizer_encode`,
`chat_template`, `model_config`.
