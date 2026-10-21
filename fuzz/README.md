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

`regressions/<target>/` holds inputs that once made a target panic, kept in git;
`seed.sh` copies them into the corpora, and each one can be replayed with
`cargo +nightly fuzz run <target> regressions/<target>/<file>`:

- `gguf/alignment-u64-max`, `gguf/alignment-near-u64-max`: a
  `general.alignment` that overflowed the start of the tensor data.
- `gguf/unknown-type-tensor`: a tensor of a type Kapel does not read, which once
  rejected the whole file.
- `model_config/mlp-width-overflow.json`: a hidden width whose default MLP width
  (4 ×) overflowed; `model_config/huge-head-dim.json`: a head width accepted, that
  overflowed in the forward pass.
