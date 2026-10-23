# 12. Serving: kapel-serve

`kapel-serve` puts a `Pipeline` behind HTTP, with an API that OpenAI's clients use unchanged. The requests of all clients are served together by one `Engine` (chapter 11): each pass of the model computes a token of every answer in progress and pieces of new prompts. This chapter follows a request from the socket to the card and back, then describes what the server measures and how it was validated.

## 12.1 Two worlds and the channels between them

The server has two kinds of threads, which never share a lock on the model.

- **The engine's thread** (`worker.rs`, named `kapel-engine`) owns the `Engine` and is the only thread that touches the backend. A recorded pass captures all the work of the card's stream, so no other thread may queue any (chapter 7).
- **The HTTP threads**, a tokio runtime running axum, read requests, tokenize them and stream the answers.

They meet through channels. A handler sends `Submission`s (a prepared request, the index of its answer, and the sending half of an unbounded tokio channel) over a standard `mpsc` channel. Between two passes, the engine's thread takes every submission waiting, admits it with `Engine::add_prepared`, and after each pass sends every token to its answer's channel as `Output::Piece`. The answer ends with `Output::Done` (how it ended, and the prompt positions taken from shared pages) or `Output::Failed`. When nothing is waiting or running, the thread blocks on the channel instead of spinning.

The work that may take a while before a request is queued runs on tokio's blocking pool (`spawn_blocking`), away from both the engine and the HTTP reactor: parsing the JSON, rendering the chat template, tokenizing, and compiling a constraint with `Pipeline::prepare`. A JSON schema can take milliseconds to compile; the engine never waits for it.

The pipeline is shared as an `Arc<Pipeline<B>>`: the handlers read its tokenizer, template and default sampling, and the engine's thread borrows it for its `Engine`. A test checks that pipelines are `Send + Sync` and prepared requests `Send`.

## 12.2 Routes

| Route | What it does |
| --- | --- |
| `POST /v1/chat/completions` | a conversation through the model's chat template |
| `POST /v1/completions` | plain prompts: a text, several, or token ids |
| `GET /v1/models` | the one model served, with its `max_model_len` |
| `GET /health` | 200 while the engine runs on a working device, 503 otherwise |
| `GET /metrics` | Prometheus' text format (section 12.6) |

Any other route answers 404 with OpenAI's error body. A request names the model it wants; a name other than the served one is a 404 `not_found_error`, as vLLM gives.

## 12.3 Reading OpenAI's requests

`api.rs` reads requests leniently: clients send many fields, and an unknown one is ignored. A known field that Kapel cannot honour is refused with a 400 that names it, rather than silently ignored, because an answer that quietly lacks what was asked for is worse than an error.

| Honoured | Refused (400) |
| --- | --- |
| `messages` (text content, or parts of type `text`), `prompt` | images and other content parts; `tool_calls` in messages |
| `max_tokens`, `max_completion_tokens` | `tools` (unless empty) |
| `temperature` (0 to 2), `top_p`, `seed` | `logprobs`, `top_logprobs` |
| `n` (1 to 16), `stop` (up to 4) | `echo`, `suffix`, `best_of` other than `n` |
| `frequency_penalty`, `presence_penalty` (−2 to 2) | more than one constraint |
| `stream`, `stream_options.include_usage` | |
| `response_format`: `text`, `json_object`, `json_schema` | other formats |

vLLM's extensions are read too: `top_k`, `min_p`, `repetition_penalty`, `ignore_eos`, and the constraints `guided_json` (a schema, or its text), `guided_regex`, `guided_choice` and `guided_grammar` (llguidance's Lark syntax). What a request leaves out comes from the model's `generation_config.json`. A request without a seed gets a fresh one; the `n` answers of a request draw from seeds that follow each other.

The length of an answer is what the request asks, or else the rest of the context; `/v1/completions` defaults to 16 tokens, as OpenAI's does. A prompt and its answer must fit in `max_model_len` positions, or the request is refused before it is queued. So is a constraint that does not compile, or a token id outside the vocabulary.

## 12.4 Answers, whole and streamed

A whole answer is OpenAI's `chat.completion` or `text_completion` object: an `id` (`chatcmpl-…` or `cmpl-…`, also sent as the `x-request-id` header), the choices with their text and `finish_reason` (`stop` for an end token or a stop sequence, `length`), and `usage` with `prompt_tokens_details.cached_tokens`, the prompt positions that prefix sharing (chapter 10) took from the cache instead of computing.

A streamed answer is a stream of server-sent events, each `data:` line a JSON chunk:

```
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Paris"},"finish_reason":null}],...}
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],...}
data: {"object":"chat.completion.chunk","choices":[],"usage":{...}}
data: [DONE]
```

The first chunk of each answer carries the role; a piece with no new text (the bytes of an unfinished character, or the start of a possible stop sequence) is not sent; with `include_usage`, every chunk has `"usage": null` and a last chunk, without choices, carries the usage. A failure mid-stream sends an `error` object, then `[DONE]`. axum sends a keep-alive comment while a request waits for its first token.

The stream is built from the receiving end of the answer's channel (`futures_util::stream::unfold`), with a `Tracker` that collects the texts, the endings and the usage.

## 12.5 Cancellation and limits

A client that closes its connection drops the response's stream, which drops the receiving end of its channel. Before each pass, the engine's thread looks at the channels of every answer it serves; a closed one cancels its request (`Engine::cancel`), and its pages go back to the cache at once. A whole answer's handler is dropped the same way when its connection closes.

| Limit | Default | Beyond it |
| --- | --- | --- |
| Body of a request | 4 MiB | 413 |
| Prompt and answer | `max_model_len` positions | 400 |
| Answers waiting or in progress | 1024 | 429 `rate_limit_error` |
| Time from arrival to last token | 600 s | 504, or an error event in a stream |

`/health` turns to 503 when the engine's thread has stopped, or when a pass failed with an error of the device and a small upload and download no longer come back: a CUDA context that failed for good stays failed, and an orchestrator should restart the server. Other failures of a pass end the requests in it, and the server goes on.

## 12.6 What the server says

`/metrics` gives, in Prometheus' text format:

| Metric | Kind |
| --- | --- |
| `kapel_requests_total{outcome="ok|refused|cancelled|failed"}` | counter |
| `kapel_prompt_tokens_total`, `kapel_generation_tokens_total` | counters |
| `kapel_preemptions_total` | counter: requests suspended for lack of pages |
| `kapel_prefix_cache_hit_tokens_total` | counter: prompt positions taken from shared pages |
| `kapel_passes_total`, `kapel_recorded_passes_total` | counters |
| `kapel_requests_waiting`, `kapel_requests_running` | gauges |
| `kapel_cache_pages`, `kapel_cache_usage_ratio` | gauges |
| `kapel_time_to_first_token_seconds` | histogram, from arrival to the first token |
| `kapel_inter_token_latency_seconds` | histogram, between two tokens of an answer |
| `kapel_request_duration_seconds` | histogram, from arrival to the last token |

Counters and histograms are atomics, added to by the handlers and the engine's thread without locks; the gauges are copied from `Engine::stats` after each pass.

Every request leaves one line of JSON on the standard error when it ends, however it ends:

```
{"ts":1792745959.687,"event":"request","id":"chatcmpl-…","endpoint":"chat","outcome":"ok","prompt_tokens":44,"completion_tokens":33,"cached_tokens":32,"first_token_ms":105.0,"duration_ms":745.1,"finish":["stop"],"error":null}
```

Refused requests (`refused`, with their status), failed passes (`pass_failed`), the start (`listening`) and the end of the engine (`engine_stopped`) have lines of their own.

## 12.7 `kapel serve`

```
kapel serve MODEL [--device cuda] [--host 127.0.0.1] [--port 8000]
```

`kapel serve` loads a model as `kapel run` does (the same `--device`, `--dtype`, `--quantize`, `--kv-cache` and other options of chapter 11) and serves it. Its own options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--served-model-name NAME` | the Hub repository, from the name or a path of the Hub's cache | the model's name in the API |
| `--max-model-len N` | the model's `max_position_embeddings`, at most what the cache holds | positions of one sequence |
| `--gpu-memory-utilization F` | 0.85 | on the card, the share of its memory to use |
| `--cache-memory GIB` | 4 on the processor | the memory of the cache, given |
| `--max-sequences N` | 256 | answers computed at once |
| `--max-pass-tokens N` | 2048 | tokens of one pass |
| `--max-requests N` | 1024 | answers in flight |
| `--no-prefix-sharing` | sharing on | |
| `--timeout SECONDS` | 600 | |

On the card, the cache takes what the weights leave. Once the model is loaded, the backend frees its pool and asks the driver for the free memory (`Cuda::memory`); the budget is the share of the total memory given, minus what is already used and what the passes need besides the cache: the activations of the widest pass, the logits of the most sequences, and the buffers of the recorded passes, one set per size of batch. What is left, divided by the bytes of a page (layers × 2 × key-value heads × head width × element size × 16), is the number of pages. With Llama 3.2 3B in bf16 on an RTX 3090, that is 5197 pages, 83 152 positions, 8.9 GiB.

## 12.8 The container image

The `Dockerfile` builds in two stages. The first, from NVIDIA's CUDA 13.3 `devel` image, installs Rust and clang (the CUDA bindings are generated by bindgen) and builds `kapel` with the `cuda` feature. The second, from the `runtime` image, holds the binary and `tileiras`, the compiler that cutile-rs runs at the first launch of a tile kernel; it runs as an unprivileged user, keeps models from the Hub in `/cache`, checks `/health` every 30 seconds, and listens on `0.0.0.0:8000`:

```
docker build -t kapel .
docker run --gpus all -p 8000:8000 -v ~/.cache/kapel:/cache kapel \
    meta-llama/Llama-3.2-3B-Instruct --device cuda
```

`--build-arg FEATURES=` builds for the processor only.

## 12.9 Validation

`tests/server.rs` serves SmolLM2-135M on the processor over a real socket and checks, against the pipeline alone: a whole answer and the same answer streamed, six requests at once each giving its greedy text, a chat held to a list of choices with `n = 2`, a streamed chat with its role first, the errors (unknown model, log probabilities, too many tokens, a regular expression that does not compile, a prompt of the wrong type), and a client that reads one chunk and goes away, whose request the metrics count as cancelled while the running requests fall back to zero.

`tools/serve/openai_client.py` drives a server with OpenAI's official Python client, unchanged but for its `base_url`: the list of models, whole and streamed chats with their usage, repeated greedy answers whose prompt is then cached, completions, `n = 3`, structured output with `client.chat.completions.parse` and a pydantic model, `json_object`, `guided_choice` through `extra_body`, the 404 and 400 errors as the client's exception types, sixteen concurrent streams of `AsyncOpenAI`, and a stream closed early. With version 3.37 of the client, every check passes.

The load is measured by the example `load`, which draws requests as Smallem's simulator of serving did: arrivals as a Poisson process, prompts of 16 to 128 random tokens, answers of 32 to 384 tokens with the end tokens ignored, all streamed. `--rates max` sends them all at once, which gives the most the server sustains; `--fractions` then measures that, and each fraction of it:

```
cargo run --release -p kapel-serve --example load -- http://127.0.0.1:8000 \
    --requests 600 --fractions 0.2,0.4,0.6,0.8,0.9
```

Llama 3.2 3B Instruct in bf16 on an RTX 3090, 600 requests per point:

| Load | Requests/s | Tokens/s | First token p50 | First token p99 | Per token p50 | Per token p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all at once | 20.5 | 4400 | 7.0 s | 21 s | 50 ms | 62 ms |
| 20 % | 4.1 | 853 | 22 ms | 35 ms | 9.8 ms | 10.4 ms |
| 40 % | 8.0 | 1668 | 25 ms | 43 ms | 10.6 ms | 11.6 ms |
| 60 % | 11.7 | 2435 | 29 ms | 51 ms | 14.0 ms | 15.5 ms |
| 80 % | 14.9 | 3100 | 37 ms | 63 ms | 17.9 ms | 25.4 ms |
| 90 % | 16.3 | 3392 | 44 ms | 83 ms | 25.7 ms | 29.0 ms |

The target was a first token within 500 ms for 99 % of the requests up to 80 % of the maximal throughput; it holds up to 90 %, at 83 ms. Prompts are computed by pieces within each pass's budget of tokens, so a new request waits at most a pass or two, whatever the others are doing.
