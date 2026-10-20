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

## Unsafe code

Every `unsafe` block carries a `// SAFETY:` comment that says why it is sound.
