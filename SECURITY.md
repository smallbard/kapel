# Security policy

Kapel reads files written by others: model weights (safetensors, GGUF), tokenizers
(`tokenizer.json`), configurations (`config.json`) and chat templates. Opening such a
file must never run code, read or write out of bounds, or hang.

## What Kapel guarantees

- **Readers return errors, they do not panic.** A malformed file gives a typed error
  that names what is wrong.
- **Bounded work.** Safetensors headers are limited to 100 MB; GGUF arrays nest at
  most 8 deep; byte-pair encoding is O(n log n) in the length of a word; a chat
  template runs at most 10 million instructions; regexes that exceed their
  backtracking limit reject the text instead of cutting it wrongly.
- **No code execution.** Chat templates run in a sandbox (minijinja) without access
  to files or the network. Kapel never reads pickle files.
- **Audited `unsafe`.** Every `unsafe` block states why it is sound (`// SAFETY:`,
  enforced by clippy). The thread pool, the storage and the CPU backend's
  operations run under Miri.

Memory-mapped files are read in place: a process that truncates or rewrites a model
file while Kapel has it open can make reads fault. Do not modify model files in use.

## How it is checked

- Fuzzing with libFuzzer (`fuzz/`): safetensors, GGUF, `tokenizer.json`, encoding
  with real tokenizers, chat templates, `config.json`. A weekly job runs each target.
- Miri on the crates that hold `unsafe` code.
- `cargo deny`: licenses, advisories, sources.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's "Report a vulnerability"
button (Security tab of the repository), not in a public issue. Include a file or
input that reproduces the problem if you can. You will get an answer within a week.

Supported versions: the latest release, and `main`.
