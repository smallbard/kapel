/*
 * kapel.h: the C API of Kapel, an inference engine for small language models.
 *
 * Load a model, turn text into tokens and back, apply its chat template,
 * generate text piece by piece, or serve it behind an HTTP API compatible
 * with OpenAI's.
 *
 *     kapel_model *model;
 *     if (kapel_model_load("HuggingFaceTB/SmolLM2-135M-Instruct", NULL, &model) != KAPEL_OK) {
 *         fprintf(stderr, "%s\n", kapel_last_error());
 *         return 1;
 *     }
 *     ...
 *     kapel_model_free(model);
 *
 * Conventions:
 *
 * - Every call that can fail returns a kapel_status. On failure,
 *   kapel_last_error() gives the message, until the next call on the same
 *   thread.
 * - Strings are UTF-8 and end with a NUL byte, in and out.
 * - What Kapel allocates for the caller (token arrays, strings) is freed
 *   with kapel_tokens_free() and kapel_string_free(), never with free().
 * - The option structures start with their size: fill them with their
 *   *_init function first, then change the fields wanted. A program built
 *   against an older header keeps working with a newer library.
 * - A model can be used by several threads at once: each generation has its
 *   own cache.
 *
 * Link with -lkapel (libkapel.so, libkapel.dylib, kapel.dll, or the static
 * libkapel.a with the system libraries it needs).
 */

#ifndef KAPEL_H
#define KAPEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this API. It changes only when a call or a structure
 * changes in a way that breaks programs built against the previous one. */
#define KAPEL_API_VERSION 1

/* What a call returned. */
typedef enum kapel_status {
    KAPEL_OK = 0,
    /* An argument is invalid: a null pointer, text that is not UTF-8, an
     * unknown option. */
    KAPEL_INVALID_ARGUMENT = 1,
    /* The call failed: a model that cannot be read, a pass that failed. */
    KAPEL_ERROR = 2,
    /* A bug in Kapel: it was caught, the model may be left unusable. */
    KAPEL_PANIC = 3
} kapel_status;

/* Why a generation ended. */
typedef enum kapel_finish {
    /* The model gave a token that ends its answers. */
    KAPEL_FINISH_EOS = 0,
    /* A stop string appeared; the answer ends before it. */
    KAPEL_FINISH_STOP = 1,
    /* max_tokens were generated. */
    KAPEL_FINISH_LENGTH = 2,
    /* The callback asked to stop. */
    KAPEL_FINISH_CANCELLED = 3
} kapel_finish;

/* A model loaded on its device, with its tokenizer and chat template. */
typedef struct kapel_model kapel_model;

/* How to load a model: the options of `kapel run`, in the same words. A null
 * string is the default. */
typedef struct kapel_load_options {
    /* sizeof(kapel_load_options), set by kapel_load_options_init. */
    size_t size;
    /* "cpu" (the default), "cuda" or "cuda:N". */
    const char *device;
    /* The type of the float weights: "f32", "bf16" or "f16"; by default
     * f32 on the processor, bf16 on a card. */
    const char *dtype;
    /* The revision of a repository of the Hub; by default "main". */
    const char *revision;
    /* Quantize the linear projections: "int8", "int4" (groups of 128 and
     * 64, or "int8/G", "int4/G"), "q8_0" or "q4_0"; by default not. */
    const char *quantize;
    /* Nonzero to quantize the output table too. */
    int quantize_output;
    /* The type of the cache of keys and values: "f32", "bf16", "f16",
     * "f8e4m3" or "f8e5m2"; by default that of the keys. */
    const char *kv_cache;
    /* Nonzero, on the processor, to quantize the activations of the
     * products by quantized weights to bytes, as llama.cpp does. */
    int byte_activations;
} kapel_load_options;

/* What to generate. kapel_generate_options_init fills it with the model's
 * own sampling settings (its generation_config.json). */
typedef struct kapel_generate_options {
    /* sizeof(kapel_generate_options), set by kapel_generate_options_init. */
    size_t size;
    /* The most tokens to generate (default 512). */
    size_t max_tokens;
    /* The divisor of the logits; 0 for greedy decoding. */
    double temperature;
    /* Only the k most likely tokens; 0 for all. */
    size_t top_k;
    /* Only the most likely tokens whose probabilities add up to top_p. */
    double top_p;
    /* Only the tokens at least min_p times as likely as the most likely. */
    double min_p;
    /* The penalty of the tokens already in the sequence; 1 for none. */
    double repetition_penalty;
    /* The seed of the draws: the same seed gives the same answer. */
    uint64_t seed;
    /* stop_count texts that end the answer; it stops before them. */
    const char *const *stop;
    size_t stop_count;
} kapel_generate_options;

/* A message of a conversation. */
typedef struct kapel_message {
    /* "system", "user", "assistant"... */
    const char *role;
    const char *content;
} kapel_message;

/* What a generation gave. */
typedef struct kapel_generation {
    /* The whole answer; free it with kapel_string_free. */
    char *text;
    kapel_finish finish;
    size_t prompt_tokens;
    size_t completion_tokens;
} kapel_generation;

/* Called with each piece of text as it becomes final (len bytes, followed by
 * a NUL byte), and user_data. Return 0 to go on, anything else to stop. */
typedef int (*kapel_piece_callback)(const char *piece, size_t len, void *user_data);

/* How to serve a model: the options of `kapel serve`. */
typedef struct kapel_serve_options {
    /* sizeof(kapel_serve_options), set by kapel_serve_options_init. */
    size_t size;
    /* The address to listen on (default "127.0.0.1"). */
    const char *host;
    /* The port (default 8000). */
    uint16_t port;
    /* The model's name in the API; by default from the name given. */
    const char *served_model_name;
    /* The most positions of a sequence; 0 for the model's, at most what the
     * cache holds. */
    size_t max_model_len;
    /* The memory of the cache in GiB; 0 for 4 on the processor, and what
     * the weights leave of gpu_memory_utilization on a card. */
    double cache_memory;
    /* On a card, the share of its memory to use (default 0.85). */
    double gpu_memory_utilization;
    /* The most answers computed at once (default 256). */
    size_t max_sequences;
    /* The most tokens of one pass (default 2048). */
    size_t max_pass_tokens;
    /* The most answers in flight; beyond, 429 (default 1024). */
    size_t max_requests;
    /* Nonzero to share the cache of common prompt beginnings (default 1). */
    int prefix_sharing;
    /* The longest a request may take, in seconds (default 600). */
    uint64_t timeout;
} kapel_serve_options;

/* KAPEL_API_VERSION, as the library was built. */
uint32_t kapel_api_version(void);

/* The version of Kapel, such as "0.1.0". */
const char *kapel_version(void);

/* The message of the last failure on this thread, or "" if none. Valid until
 * the next call on this thread. */
const char *kapel_last_error(void);

/* Fills options with the defaults. */
void kapel_load_options_init(kapel_load_options *options);

/* Loads model: a model directory, a GGUF file, or a repository of the Hub
 * ("owner/name", downloaded once into ~/.cache/kapel/hub, or $KAPEL_CACHE; a
 * gated model needs HF_TOKEN). options may be null for the defaults. */
kapel_status kapel_model_load(const char *model, const kapel_load_options *options,
                              kapel_model **out);

/* Frees a model; null is ignored. */
void kapel_model_free(kapel_model *model);

/* The tokens of text, with the special tokens the tokenizer adds (a
 * beginning of sequence, say) if add_special_tokens is nonzero. *ids gets
 * *len tokens, to free with kapel_tokens_free. */
kapel_status kapel_tokenize(const kapel_model *model, const char *text, int add_special_tokens,
                            uint32_t **ids, size_t *len);

/* The text of len tokens, without the special tokens if skip_special_tokens
 * is nonzero. *text is to free with kapel_string_free. */
kapel_status kapel_detokenize(const kapel_model *model, const uint32_t *ids, size_t len,
                              int skip_special_tokens, char **text);

/* The tokens of a conversation of count messages in the model's chat
 * template, ending with the start of the assistant's answer: the prompt to
 * give kapel_generate. Fails if the model has no chat template. */
kapel_status kapel_chat_prompt(const kapel_model *model, const kapel_message *messages,
                               size_t count, uint32_t **ids, size_t *len);

/* Fills options with the model's own sampling settings. */
kapel_status kapel_generate_options_init(const kapel_model *model,
                                         kapel_generate_options *options);

/* Generates what follows the len tokens of prompt. options may be null for
 * the model's settings, callback null for none. On success, *result holds
 * the answer; free its text with kapel_string_free. result may be null. */
kapel_status kapel_generate(const kapel_model *model, const uint32_t *prompt, size_t len,
                            const kapel_generate_options *options,
                            kapel_piece_callback callback, void *user_data,
                            kapel_generation *result);

/* Fills options with the defaults of `kapel serve`. */
void kapel_serve_options_init(kapel_serve_options *options);

/* Loads model as kapel_model_load does and serves it behind the HTTP API
 * compatible with OpenAI's (/v1/chat/completions, /v1/completions,
 * /v1/models, /health, /metrics), until the process gets SIGINT (Ctrl-C).
 * Either options may be null for the defaults. */
kapel_status kapel_serve(const char *model, const kapel_load_options *load,
                         const kapel_serve_options *options);

/* Frees tokens from kapel_tokenize or kapel_chat_prompt; null is ignored. */
void kapel_tokens_free(uint32_t *ids, size_t len);

/* Frees a string from Kapel; null is ignored. */
void kapel_string_free(char *text);

#ifdef __cplusplus
}
#endif

#endif /* KAPEL_H */
