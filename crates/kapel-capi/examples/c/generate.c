/*
 * Answers a question with a model, through the C API of Kapel: the answer is
 * printed as it comes, then a line of counts on the standard error.
 *
 *     generate MODEL "QUESTION" [MAX_TOKENS]
 *
 * MODEL is a model directory, a GGUF file, or a repository of the Hub. The
 * answer is greedy: the same each time. Build it with the Makefile next to
 * it, or by hand:
 *
 *     cargo build --release -p kapel-capi
 *     cc generate.c -I ../../include -L ../../../../target/release -lkapel \
 *        -Wl,-rpath,$(realpath ../../../../target/release) -o generate
 */

#include <stdio.h>
#include <stdlib.h>

#include "kapel.h"

/* Prints each piece as it comes; never stops the generation. */
static int print_piece(const char *piece, size_t len, void *user_data) {
    (void)user_data;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

static int fail(const char *what) {
    fprintf(stderr, "%s: %s\n", what, kapel_last_error());
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL QUESTION [MAX_TOKENS]\n", argv[0]);
        return 2;
    }
    if (kapel_api_version() != KAPEL_API_VERSION) {
        fprintf(stderr, "libkapel has API %u, this program %d\n", kapel_api_version(),
                KAPEL_API_VERSION);
        return 1;
    }

    kapel_load_options load;
    kapel_load_options_init(&load);
    load.device = "cpu";
    kapel_model *model;
    if (kapel_model_load(argv[1], &load, &model) != KAPEL_OK) {
        return fail("load");
    }

    kapel_message messages[] = {{"user", argv[2]}};
    uint32_t *prompt;
    size_t len;
    if (kapel_chat_prompt(model, messages, 1, &prompt, &len) != KAPEL_OK) {
        /* No chat template: the question as a plain prompt. */
        if (kapel_tokenize(model, argv[2], 1, &prompt, &len) != KAPEL_OK) {
            kapel_model_free(model);
            return fail("tokenize");
        }
    }

    kapel_generate_options options;
    if (kapel_generate_options_init(model, &options) != KAPEL_OK) {
        kapel_tokens_free(prompt, len);
        kapel_model_free(model);
        return fail("options");
    }
    options.temperature = 0.0;
    options.max_tokens = argc > 3 ? (size_t)strtoul(argv[3], NULL, 10) : 128;

    kapel_generation result;
    kapel_status status =
        kapel_generate(model, prompt, len, &options, print_piece, NULL, &result);
    kapel_tokens_free(prompt, len);
    if (status != KAPEL_OK) {
        kapel_model_free(model);
        return fail("generate");
    }
    static const char *finishes[] = {"eos", "stop", "length", "cancelled"};
    printf("\n");
    fprintf(stderr, "[%zu prompt tokens, %zu tokens, %s]\n", result.prompt_tokens,
            result.completion_tokens, finishes[result.finish]);
    kapel_string_free(result.text);
    kapel_model_free(model);
    return 0;
}
