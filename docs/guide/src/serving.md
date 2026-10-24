# Serving a model

`kapel serve` puts a model behind an HTTP API compatible with OpenAI's. The requests of all clients are computed together, so a server answers many users with one copy of the model.

## Start a server

```sh
kapel serve Qwen/Qwen2.5-0.5B-Instruct
```

```
kapel: a cache of 10922 pages (174752 positions, 4.0 GiB); sequences of up to 32768 positions
{"ts":1792777537.943,"event":"listening","address":"127.0.0.1:8000",...}
```

In another terminal:

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [{"role": "user", "content": "Name three French cities."}],
  "max_tokens": 60
}'
```

The answer is OpenAI's `chat.completion` object. Add `"stream": true` to receive it token by token as server-sent events.

On an NVIDIA card:

```sh
kapel serve meta-llama/Llama-3.2-3B-Instruct --device cuda
```

## Use OpenAI's clients

Any client of OpenAI's API works; point it at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
answer = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(answer.choices[0].message.content)
```

The same goes for LangChain, LlamaIndex, and anything with a setting for an OpenAI-compatible base URL. The server needs no key; the clients that require one accept any.

## What the API offers

| Route | |
| --- | --- |
| `POST /v1/chat/completions` | a conversation, through the model's chat template |
| `POST /v1/completions` | plain text continued, or token ids |
| `GET /v1/models` | the model served |
| `GET /health` | 200 while the server works |
| `GET /metrics` | metrics in Prometheus' format: requests, tokens, latencies, cache |

Requests accept `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `n`, `frequency_penalty`, `presence_penalty` and `stream`, as OpenAI defines them.

**Structured outputs.** An answer can be held to a form, and is then guaranteed to have it:

- `"response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {...}}}` for JSON that follows a schema (OpenAI's client's `client.chat.completions.parse` with a pydantic model works);
- `"response_format": {"type": "json_object"}` for any JSON object;
- vLLM's extensions `guided_json` (a schema), `guided_regex`, `guided_grammar` (a Lark grammar) and `guided_choice` (one of a list of texts).

**Not supported**, and refused with a 400 error that names the field rather than silently ignored: images, tools and function calls, and log probabilities.

## Options

| Option | Default | |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | the address to listen on; `0.0.0.0` for every interface |
| `--port PORT` | `8000` | |
| `--served-model-name NAME` | the name given | the model's name in the API |
| `--max-model-len N` | the model's | the most positions of a request, prompt and answer |
| `--gpu-memory-utilization F` | `0.85` | on a card, the share of its memory to use; the cache takes what the weights leave |
| `--cache-memory GIB` | 4 on the processor | the memory of the cache instead |
| `--max-sequences N` | 256 | the most answers computed at once |
| `--max-requests N` | 1024 | the most answers in flight; beyond, 429 |
| `--timeout SECONDS` | 600 | the longest a request may take |
| `--no-prefix-sharing` | | do not share the cache of prompts that begin alike |

`--device`, `--dtype`, `--quantize`, `--kv-cache` and the other loading options of `kapel run` apply too.

**Security.** The server has no authentication and listens on `127.0.0.1` by default. To expose it, put it behind a reverse proxy that authenticates (nginx, Caddy, a cloud gateway).

## In a container

The repository's `Dockerfile` builds an image for NVIDIA cards:

```sh
docker build -t kapel .
docker run --gpus all -p 8000:8000 -v ~/.cache/kapel:/cache kapel \
    meta-llama/Llama-3.2-3B-Instruct --device cuda
```

The image keeps downloaded models in `/cache` (mounted above, so they survive the container), listens on `0.0.0.0:8000`, and checks `/health` every 30 seconds. Pass `-e HF_TOKEN` for gated models. `docker build --build-arg FEATURES= -t kapel .` builds an image for the processor only.

## From Python, without a separate process

```python
import kapel
kapel.serve("Qwen/Qwen2.5-0.5B-Instruct", port=8000)   # until Ctrl-C
```
