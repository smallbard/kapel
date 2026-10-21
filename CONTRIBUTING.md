# Contributing to Kapel

## Rules

- **Every fast backend is checked against the reference backend.** A new operation
  comes with its reference definition, its gradient, and a conformity test with a
  written tolerance.
- **Results do not depend on the number of threads.** Work is split by tensor
  shapes, never by hardware; tests check it bit for bit.
- **The library does not panic on user input.** Wrong shapes, unsupported types,
  corrupt files: return a typed `Error`. `panic!` is reserved for broken internal
  invariants.
- **Every optimization comes with a measurement**, added to the benchmarks.
- **English** for code, documentation and error messages.

## Before opening a pull request

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

Add a line to `CHANGELOG.md` under *Unreleased*.

## The CUDA backend

`kapel-cuda` is empty unless built with the feature `cuda`, so that the workspace
builds anywhere. With it, it needs the CUDA toolkit 13.2 or later (for `tileiras`) on
the `PATH`, and an NVIDIA device to run:

```
export PATH=/usr/local/cuda/bin:$PATH
cargo clippy -p kapel-cuda --features cuda --all-targets -- -D warnings
cargo test --release -p kapel-cuda --features cuda
```

The tests include the conformity suite in `f32`, `bf16` and `f16`, and, when
`data/parity` is present (see `tools/parity/README.md`), the parity of whole models on
the card. Measure on an idle device: another program sharing it takes a varying share
of its time.

```
cargo run --release -p kapel-cuda --features cuda --example gemm
cargo run --release -p kapel-cuda --features cuda --example attention
cargo run --release -p kapel-cuda --features cuda --example prefill -- MODEL_DIR
```

Two kinds of kernels:

- **Tile kernels** (cutile-rs: products, attention) live in the backend's sources. They
  are compiled by `tileiras` at their first launch and kept in `~/.cache/cutile/kernels`.
  `cutile-compiler` must be built without debug information (see the workspace's
  `Cargo.toml`), or the kernels are compiled unoptimized.
- **Thread kernels** (cuda-oxide: everything else) live in `kernels/`, built apart on the
  nightly of `kernels/rust-toolchain.toml` with `cargo oxide`. After changing them, run
  `kernels/build.sh`, which writes `crates/kapel-cuda/kernels.ptx`, and update the
  parameter table of `crates/kapel-cuda/src/signatures.rs`; a test checks that the two
  agree.

## Unsafe code

Every `unsafe` block carries a `// SAFETY:` comment that says why it is sound.
