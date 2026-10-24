# C and other languages

`libkapel` is a C library: load a model, tokenize, apply the chat template, generate with a callback for each piece of text, or serve. Its header, `crates/kapel-capi/include/kapel.h`, documents every function and structure.

## Build the library

```sh
cargo build --release -p kapel-capi                    # add --features cuda for NVIDIA cards
```

This gives, in `target/release`, a shared library (`libkapel.so` on Linux, `libkapel.dylib` on macOS, `kapel.dll` with its import library `kapel.dll.lib` on Windows) and a static one (`libkapel.a`, `kapel.lib`). Ship the shared library with your program, or link the static one; the static library also needs the system libraries Rust's standard library uses, which `cargo rustc -p kapel-capi --release -- --print native-static-libs` lists.

## A program

```c
#include <stdio.h>
#include "kapel.h"

static int print_piece(const char *piece, size_t len, void *user_data) {
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;                                   /* nonzero stops the generation */
}

int main(void) {
    kapel_model *model;
    if (kapel_model_load("Qwen/Qwen2.5-0.5B-Instruct", NULL, &model) != KAPEL_OK) {
        fprintf(stderr, "%s\n", kapel_last_error());
        return 1;
    }
    kapel_message messages[] = {{"user", "What is the capital of France?"}};
    uint32_t *prompt;
    size_t len;
    kapel_chat_prompt(model, messages, 1, &prompt, &len);

    kapel_generation answer;
    if (kapel_generate(model, prompt, len, NULL, print_piece, NULL, &answer) == KAPEL_OK) {
        printf("\n[%zu tokens]\n", answer.completion_tokens);
        kapel_string_free(answer.text);
    }
    kapel_tokens_free(prompt, len);
    kapel_model_free(model);
    return 0;
}
```

```sh
cc main.c -I crates/kapel-capi/include -L target/release -lkapel -o main
LD_LIBRARY_PATH=target/release ./main                  # DYLD_LIBRARY_PATH on macOS
```

A complete example, with every error checked, is `crates/kapel-capi/examples/c/generate.c`, built by the `Makefile` next to it.

## The rules

- **Errors.** Every call that can fail returns a `kapel_status`. On failure, `kapel_last_error()` gives the message, until the next call on the same thread.
- **Strings** are UTF-8 and end with a NUL byte, in both directions.
- **Memory.** Free what Kapel allocates with `kapel_tokens_free` and `kapel_string_free`, never with `free()`. Free a model with `kapel_model_free`.
- **Options.** Fill an option structure with its `*_init` function, then change the fields you want: `kapel_load_options` (device, type, quantization, cache), `kapel_generate_options` (the model's sampling settings by default), `kapel_serve_options`. A null pointer stands for the defaults.
- **Threads.** A model can be used by several threads at once.
- **Versions.** `KAPEL_API_VERSION` changes only when a change breaks programs built against the previous header; `kapel_api_version()` gives the library's. A program built against an older header keeps working with a newer library.

## From other languages

Any language that calls C functions can use `libkapel` through `kapel.h`: `cgo` in Go, `ctypes` or `cffi` in Python (though the [Python module](python.md) is simpler), P/Invoke in C#, `@cImport` in Zig, the Foreign Function and Memory API in Java. Bind the functions you need, mirror the structures field by field, and keep to the rules above.
