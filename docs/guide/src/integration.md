# Integrating Kapel

Three ways to put Kapel inside a program, besides talking to its server over HTTP:

| From | Use | Page |
| --- | --- | --- |
| Python | the `kapel` module: load, generate, chat, serve | [Python](python.md) |
| C, C++, and any language that calls C (Go, C#, Swift, Zig, Java…) | `libkapel` and its header `kapel.h` | [C and other languages](c.md) |
| Rust | the crates themselves | [Rust](rust.md) |

All three take the options of the command line in the same words: `"cuda:1"` for `--device cuda:1`, `"int4"` for `--quantize int4`. A loaded model can be used by several threads at once, each generation with its own cache.

When the program and the model can live in different processes, the [server](serving.md) is often simpler: any HTTP client works, and many programs can share one model.
