# Preface

This book takes you from the shape of Kapel as a whole down to the loops inside its kernels. It is written for people who will read, change or extend the code: contributors, reviewers, and engineers who want to know exactly what their model runs on.

Kapel is a Rust library for running and, later, training transformer models. Its distinctive trait is discipline rather than size: every fast path is checked against a slow, exact reference; results do not depend on the number of threads; and no input, however malformed, should make it panic. Most chapters show how one of those promises is kept in code.

**How the book is organised.**

| Part | Chapters | What you learn |
| --- | --- | --- |
| I. The big picture | 1–2 | What Kapel is for, its rules, and how its crates fit together |
| II. Foundations | 3–5 | Tensors, the operation catalogue, and the reference that defines correctness |
| III. Backends | 6–7 | How the CPU and CUDA backends make the catalogue fast |
| IV. From files to answers | 8–12 | Model files, tokenizers and chat templates, the models, text generation, and the server |
| V. Practice | 13–14 | Safety and security, testing, benchmarks, CI, and the roadmap |
| Appendix | — | Glossary and an index of files |

**How to read it.** Read Part I first, whatever your goal. Then:

- To add an operation, read chapters 3, 4 and 5, then the backend chapters.
- To tune performance, read chapter 4, then chapter 6 or 7.
- To support a new model or file format, read chapters 8 to 11.
- To serve models, read chapters 11 and 12.
- To review a change, read chapter 13 and the testing part of chapter 14.

**Conventions.** Shapes are written in brackets, outermost first: `[B, H, T, D]` is batch, heads, positions, head width. File references are relative to the repository root, such as `crates/kapel-ops/src/lib.rs`. Code identifiers are in `monospace`. The book describes the repository as of 23 October 2026; where code and book disagree, the code wins, and the book should be fixed.
