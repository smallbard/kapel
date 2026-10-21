# Security policy

Kapel reads files written by others: model weights (safetensors, GGUF), tokenizers
(`tokenizer.json`), configurations (`config.json`) and chat templates. Opening such a
file must never run code, read or write out of bounds, or hang.

## What Kapel guarantees

- **Readers return errors, they do not panic.** A malformed file gives a typed error
  that names what is wrong.
- **Bounded work.** Safetensors headers are limited to 100 MB; GGUF arrays nest at
  most 8 deep, and `general.alignment` must be a power of two of at most 64 KiB;
  `config.json` sizes are bounded (layers, heads, widths, vocabulary: see
  `kapel_models::MAX_*`); byte-pair encoding is O(n log n) in the length of a
  word; regexes that exceed their backtracking limit reject the text instead of
  cutting it wrongly.
- **Bounded chat templates, within limits.** A rendering runs at most 10 million
  instructions (minijinja's fuel) and its output is at most 1 MiB
  (`kapel_tokenizers::MAX_OUTPUT`): the text is written to a buffer that refuses
  to grow past it, so a long output fails as it is produced. One string
  repetition (`'x' * n`) is limited by minijinja to 100 MB. What is *not* bounded
  is the memory of intermediate values: a template that doubles a string by
  concatenation (`s ~ s`) in a few steps, or captures a large output with
  `{% set %}…{% endset %}`, can allocate until the process runs out of memory
  within its fuel. Render templates of unknown origin in a process whose memory
  you can afford to lose, or limit it (`ulimit -v`, a cgroup).
- **Paths stay inside their directory.** File names read from files — the shards
  of `model.safetensors.index.json`, the file list of a Hub repository — are
  rejected if they are absolute or contain `..`, `/`, `\` or `:`. Downloads keep
  to the top level of a repository, under a directory named after the commit.
- **Downloads time out.** Connections to the Hub give up after 30 s, and a
  transfer that stalls for 60 s fails.
- **No code execution.** Chat templates run in a sandbox (minijinja) without access
  to files or the network. Kapel never reads pickle files.
- **Audited `unsafe`.** Every `unsafe` block states why it is sound (`// SAFETY:`,
  enforced by clippy). The thread pool, the storage and the CPU backend's
  operations run under Miri.

Memory-mapped files are read in place: a process that truncates or rewrites a model
file while Kapel has it open can make reads fault. Do not modify model files in use.
Kapel itself never does: it writes files (`safetensors::save`, `Model::save`) under
a temporary name and renames them into place, so a model can be saved over the
directory it was loaded from.

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
