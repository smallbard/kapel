# 11. Generating text: kapel-infer and kapel-cli

`kapel-infer` is the first piece of the inference plan: it loads a model directory or a GGUF file onto a backend and generates text token by token, making the same choices as `transformers`. With greedy decoding, its tokens match the reference generations of `transformers` saved in `data/parity/generation`. `kapel-cli` wraps it in the `kapel run` command.

## 11.1 The pipeline

```rust
let pipeline = Pipeline::load("SmolLM2-135M-Instruct", kapel_cpu::Cpu::new(), None)?;
let prompt = pipeline.chat_prompt(&[Message::new("user", "Hello!")])?;
for piece in pipeline.generate(&prompt, &Request::default())? {
    print!("{}", piece?.text);
}
```

`Pipeline::load(dir, backend, dtype)` gathers everything a generation needs: the model, uploaded to the backend and optionally cast; the tokenizer; the chat template, if the model has one; the default sampling settings of `generation_config.json`; and the tokens that end an answer, taken from `generation_config.json`, else `config.json`, else the tokenizer's `eos_token`.

`Pipeline::load_with(dir, backend, &options)` does the same with `LoadOptions`: the type of the float weights; a format to quantize the linear projections to (and the output table, optionally) before anything else, while they are still the mapped weights of the checkpoint; `fuse`, to stack the projections that read the same input (`Model::fused`, chapter 10), what a card computes faster; and `cache_dtype`, the type of the caches of keys and values, `f16` or `f8e4m3` for instance. `Pipeline::cache()` makes an empty cache of that type, and `Pipeline::backend()` gives the backend.

A file is read as GGUF (`Pipeline::load_gguf`). The model comes from `Model::from_gguf` (chapter 10), and everything else from the metadata. The tokenizer is written back as the `tokenizer.json` it was converted from: the tokens and merges of a byte-level BPE, the splitting pattern that `tokenizer.ggml.pre` names (Llama 3's, or Qwen's with its NFC normalization), the control and user-defined tokens as added tokens, and a beginning of sequence when the file asks for one, or by llama.cpp's default for its pre-tokenizer. `kapel-tokenizers` then reads it as any other. The end tokens are those the file names (end of sequence, of turn, of message) and the control tokens that llama.cpp recognizes as ends (`<|eot_id|>`, `<|im_end|>`, …): the same set as `generation_config.json` gives for the models measured. A GGUF file has no sampling settings, so the default is greedy.

`prompt(text)` encodes raw text; `chat_prompt(messages)` renders a conversation through the template first. A `Request` says how to choose tokens, how many to generate at most (256 by default), and which stop sequences end the answer; `ignore_eos` takes the end tokens as any other, so that answers have known lengths, to measure speed.

## 11.2 Prefill, then one token at a time

`generate` returns a `Generation`, an iterator. Its first step passes the whole prompt to `Model::step` with an empty cache: this is the *prefill*, where the products have many rows and the GPU or the CPU's blocked product is efficient. Each later step passes only the last token, which reads the keys and values of every earlier position from the cache: this is *decoding*, where products have one row and speed is bound by memory bandwidth, which is why the CPU has a separate few-rows path, and reads quantized weights in place (chapter 6).

On a backend that records passes, each generated token goes through `Cache::recorded_step` (chapter 10): on the card, the decoding pass is a CUDA graph recorded once and replayed, one launch per token instead of a few hundred. With greedy decoding, the most likely token is found on the device by `ArgMax`, and only its index comes back; otherwise the logits are downloaded to the host as `f64`. A token is chosen, and the loop continues until one of three endings, reported by `finish()`:

| Ending | Cause |
| --- | --- |
| `Finish::Eos` | the model produced a token that ends its answers |
| `Finish::Stop(text)` | a stop sequence appeared; the answer stops before it |
| `Finish::Length` | the maximum number of tokens was generated |

After an error, the iterator ends: the cache may hold part of a failed step, so nothing can follow.

## 11.3 Batches

A server answers many requests at once, and a pass that reads every weight for one token mostly waits for memory. `Pipeline::batch(&requests)` makes a `Batch` of several prompts, each with its `Request`, in one `KvCache` that grows (`batch_in` takes a cache of fixed size, or of another type). `Batch::step` makes one pass of the model:

- while some prompts are not computed, it computes them, several per pass when they are short (up to 8192 tokens a pass, a longer prompt alone);
- then it computes one token of every sequence in progress, in one pass (`Model::step_batch`, or a recorded step replayed while the same sequences continue).

It returns the pieces it gave, each with the index of its request. A sequence that ends gives its pages back to the cache, and the next steps no longer include it. `tokens(i)`, `finish(i)`, `in_progress()` and `is_finished()` follow the requests. When every request is greedy, the tokens are found on the device; otherwise the logits come back and each row is sampled on the machine's threads, each request with its own generator (the seed plus its index), so that requests with the same seed do not draw the same tokens.

The test `a_batch_gives_the_greedy_tokens_of_each_prompt` checks that a batch gives each prompt the tokens it gets alone.

## 11.4 Choosing a token

`Sampling` applies the filters of `transformers`, in the same order:

1. **Repetition penalty**: the logit of every token already in the sequence, prompt included, is divided by the factor if positive, multiplied if negative.
2. **Greedy**: with a temperature of 0, take the most likely token, the first on a tie.
3. **Temperature**: divide the logits.
4. **top-k**: keep the k most likely tokens, and those tied with the last.
5. **top-p**: keep the most likely tokens until their probabilities add up to p, at least one.
6. **min-p**: keep the tokens at least min-p times as likely as the most likely one.
7. **Draw** from what is left.

`Sampling::from_generation_config` follows `transformers`' defaults: greedy unless `do_sample` is set, and then a temperature of 1 and a top-k of 50 unless the file says otherwise. The draws use a small xoshiro256\*\* generator seeded through SplitMix64, written in the crate, so the same seed gives the same tokens on every machine.

## 11.5 Text that is never taken back

A token does not always map to finished text. A character may be split over two tokens, and the start of a stop sequence may appear before it is complete. A `Piece` therefore carries a token and the text it makes *final*: the concatenation of all pieces' texts is the answer, and no later token changes what was already given out.

The text stream decodes the whole generated sequence again at each token, because spaces and multi-byte characters depend on their neighbours, and gives out only the new suffix. It holds back a trailing U+FFFD, the mark of an unfinished character, and the longest ending that could be the start of a stop sequence. When a stop sequence appears, possibly straddling text already held back, the answer is cut just before it.

## 11.6 The kapel command

```
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B --prompt "Why is the sky blue?"
```

`kapel run MODEL` accepts a local directory, a GGUF file, or a Hub repository, which it downloads once into the cache. Without `--prompt`, it holds a conversation: one message per line on standard input, an empty line to end, each answer added to the history. The main options are:

| Option | Effect |
| --- | --- |
| `--device cpu\|cuda[:N]` | where to compute; CUDA needs the `cuda` feature and defaults to bf16 weights |
| `--dtype f32\|bf16\|f16` | the type of the float weights: f32 by default on the CPU, bf16 on the GPU |
| `--quantize FORMAT`, `--quantize-output` | quantize the linear projections at loading (`int8`, `int4`, `int8/G`, `int4/G`, `q8_0`, `q4_0`), and the output table too |
| `--kv-cache f32\|bf16\|f16\|f8e4m3\|f8e5m2` | the type of the cache of keys and values, by default that of the keys; f8 halves its memory |
| `--byte-activations` | on the CPU, quantize the activations of the quantized products to bytes, as llama.cpp does: faster, and the results change a little |
| `--system TEXT`, `--raw` | a system message; or skip the chat template |
| `--max-tokens N`, `--stop TEXT` | limits of each answer |
| `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`, `--seed` | override the model's sampling settings |
| `--revision REV` | a branch, tag or commit of a Hub repository |
| `--stats` | prompt and generation speeds on standard error |

The example `generation_parity` compares Kapel's greedy generations with those of `transformers` and, at the first difference, reports the gap between the two candidates' logits: a gap near zero is a tie that rounding decides, not a bug.

Two more examples measure speed and quality. `profile` times each operation of the backend while a model generates, and lists them by type and, for quantized products, by weight shape with the bandwidth reached; it is how the CPU work of step I1 found where the time of a token went. `perplexity` scores a text as llama.cpp's `llama-perplexity` does (chunks of 512 tokens, the first replaced by the beginning of sequence, each scored on its second half), by pieces of 16 tokens after the cache, so with the kernels of generation; the options of `kapel run` choose them. On wikitext-2, Llama 3.2 1B in Q4\_K\_M scores 16.21 with Kapel, in f32 activations as in bytes, and 16.25 with llama.cpp. `--kv-cache` measures what a smaller cache costs: over 20 chunks, 16.206 in f16 (as in f32), 16.264 in f8e4m3 and 16.586 in f8e5m2, whose two bits of mantissa round the keys too coarsely.

`throughput` measures generation by batches as vLLM's `vllm bench throughput` does with random prompts: `--sequences` prompts of `--input` random tokens, each continued by `--output` tokens, greedy and past the end tokens, all in one `Batch`. It prints the time of the prefill and of the decoding, and the tokens generated per second. On an RTX 3090, Llama 3.2 3B in bf16 generates 3328 tokens per second for 64 sequences of 128 + 256 tokens, against 3710 for vLLM 0.32 (90 %), and 105 for one sequence, against 108.5; `--kv-cache f8e4m3` gives the same throughput. `tools/parity/README.md` gives both commands.
