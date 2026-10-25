# 18. Safety and security

Kapel reads files written by strangers — weights, tokenizers, configurations, chat templates — and runs low-level code on them. Its security policy states the goal in one sentence: opening such a file must never run code, read or write out of bounds, or hang. This chapter gathers the techniques that keep that promise, which earlier chapters introduced one at a time.

## 18.1 Two kinds of failure

Kapel distinguishes two situations, and handles them differently.

- **Bad input** is expected: a wrong shape, an unsupported type, a corrupt file, a hostile template. It returns a typed error that names what is wrong. The library does not panic on user input.
- **A broken invariant** is a bug in Kapel: two slices that should not overlap, a shape that a check should have rejected. It may panic, because continuing would be worse.

The line between them is drawn by validation at the boundaries. `infer` validates every operation call before a backend sees it (chapter 4); `Matrix::new` validates an operand before the product reads it with raw pointers (chapter 6); the CUDA launcher validates every argument list and size before a kernel is queued (chapter 7). Behind these boundaries, code can assume what was checked.

## 18.2 Unsafe code

`unsafe` appears where Rust cannot express the proof: SIMD intrinsics, memory mapping, raw buffers shared by the thread pool, kernel launches, and the device kernels themselves. The rules are mechanical:

- The workspace lints deny `unsafe_op_in_unsafe_fn` and warn on `clippy::undocumented_unsafe_blocks`, and CI turns warnings into errors. Every `unsafe` block carries a `// SAFETY:` comment saying why it is sound.
- Proofs are carried by types where possible. An `Isa` value proves its instructions exist; a `Matrix` proves its last element is inside its data; a `View` passed to a kernel was built from a checked layout.
- Miri, Rust's interpreter for undefined behaviour, runs weekly on `kapel-core`, the CPU thread pool, and the CPU backend's conformity suite.

## 18.3 Bounded work on untrusted files

| Input | Bound |
| --- | --- |
| safetensors header | 100 MB; every byte range checked against the file |
| GGUF | lengths at most the file size, arrays nested at most 8 deep, alignment a power of two up to 64 KiB, checked offsets |
| `config.json` | layers, heads, widths and vocabulary bounded so products fit in 48 bits |
| BPE | O(n log n) in the length of a word |
| Regular expressions | a backtracking limit; exceeding it rejects the text |
| Chat templates | 10 million steps of fuel, 1 MiB of output, 100 MB for one repeated string |
| File names from files | no absolute path, `..`, `/`, `\` or `:`; downloads stay at the top level of a repository |
| Network | 30 s to connect, 60 s without data |

One bound is honestly incomplete. A template can still grow a string by repeated concatenation, or capture a large output in a variable, within its fuel. `SECURITY.md` says so and advises rendering templates of unknown origin in a process whose memory is limited, with `ulimit -v` or a cgroup.

Kapel never executes code from a model: chat templates run in minijinja's sandbox, with no access to files or the network, and pickle files are never read.

## 18.4 Fuzzing

Six libFuzzer targets in `fuzz/` feed random and mutated inputs to every reader: `safetensors`, `gguf`, `tokenizer_json`, `tokenizer_encode` (encoding with real tokenizers), `chat_template` and `model_config`. A target may return an error; it must never panic, hang or read out of bounds. `fuzz/seed.sh` seeds the corpora from the parity data, and a weekly CI job runs each target for an hour with a 2 GB memory limit.

Every input that once made a target fail is kept in `fuzz/regressions/` and replayed, for instance a GGUF `general.alignment` near `u64::MAX` that overflowed the start of the data, or a configuration whose default MLP width overflowed.

## 18.5 Memory-mapped files

Mapping a file is the one place where safety depends on the outside world: if another process truncates a mapped file, reading it faults. Kapel cannot prevent that, so it documents it (do not modify model files in use) and never does it itself: every write goes to a temporary file that is then renamed into place.

## 18.6 Clients of the server

The server of chapter 12 takes requests from clients it does not know, and treats them as it treats files from strangers.

- **It listens on `127.0.0.1` by default.** Serving on other interfaces takes `--host`; the container image listens on `0.0.0.0`, behind the port mapping its operator chooses. Kapel has no authentication: put it behind a proxy that has one before exposing it.
- **Every request is bounded before it is queued**: a body of 4 MiB at most, a prompt and an answer within `max_model_len` positions, at most 16 answers and 4 stop sequences, token ids inside the vocabulary, numbers within their ranges. Answers in flight are bounded too (429 beyond), and so is the time of a request (504).
- **Constraints are compiled by llguidance under its parser limits.** A schema, regular expression or grammar that does not compile is a 400 before anything is queued; a parser that fails while an answer is generated ends that answer only (`Finish::Error`), not the others of its pass.
- **Nothing a request says runs code.** Chat templates render in minijinja's sandbox, as in section 18.3. Tools and function calls are refused, not passed to the template, and so are log probabilities.
- **A client that goes away gives its pages back** before the next pass, so abandoned requests cannot fill the cache.

## 18.7 Reporting

Vulnerabilities are reported privately through GitHub's "Report a vulnerability" button, with a reproducing input if possible; the policy promises an answer within a week. Supported versions are the latest release and `main`.
