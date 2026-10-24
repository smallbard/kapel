# 11. Generating text: kapel-infer and kapel-cli

`kapel-infer` is the first piece of the inference plan: it loads a model directory or a GGUF file onto a backend and generates text token by token, making the same choices as `transformers`. With greedy decoding, its tokens match the reference generations of `transformers` saved in `data/parity/generation`. `kapel-cli` wraps it in the `kapel run` command; the engine that serves many requests at once is the base of the server of chapter 12. `Scorer`, which gives the log-likelihood of continuations rather than generating, serves evaluation (chapter 13).

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

`prompt(text)` encodes raw text; `chat_prompt(messages)` renders a conversation through the template first. A `Request` says how to choose tokens, how many to generate at most (256 by default), which stop sequences end the answer, and the form the answer must take, if any (`constraint`, section 11.4); `ignore_eos` takes the end tokens as any other, so that answers have known lengths, to measure speed.

## 11.2 Prefill, then one token at a time

`generate` returns a `Generation`, an iterator. Its first step passes the whole prompt to `Model::step` with an empty cache: this is the *prefill*, where the products have many rows and the GPU or the CPU's blocked product is efficient. Each later step passes only the last token, which reads the keys and values of every earlier position from the cache: this is *decoding*, where products have one row and speed is bound by memory bandwidth, which is why the CPU has a separate few-rows path, and reads quantized weights in place (chapter 6).

On a backend that records passes, each generated token goes through `Cache::recorded_step` (chapter 10): on the card, the decoding pass is a CUDA graph recorded once and replayed, one launch per token instead of a few hundred. With greedy decoding, the most likely token is found on the device by `ArgMax`, and only its index comes back; otherwise the logits are downloaded to the host as `f64`. A token is chosen, and the loop continues until one of three endings, reported by `finish()`:

| Ending | Cause |
| --- | --- |
| `Finish::Eos` | the model produced a token that ends its answers |
| `Finish::Stop(text)` | a stop sequence appeared; the answer stops before it |
| `Finish::Length` | the maximum number of tokens was generated |
| `Finish::Error(message)` | the request failed on its own (its constraint's parser) |

After an error, the iterator ends: the cache may hold part of a failed step, so nothing can follow.

## 11.3 Serving many requests: the engine

A server answers many requests at once, and a pass that reads every weight for one token mostly waits for memory. An `Engine` (`crates/kapel-infer/src/engine.rs`) serves requests that come and go, all computed together: *continuous batching*. `Pipeline::engine(EngineOptions)` makes one; `EngineOptions` gives the pages of its cache (`None` for a cache that grows), whether sequences share the pages of their common beginnings (`prefix_sharing`, chapter 10), the most tokens of one pass (`max_pass_tokens`, 2048 by default), the most sequences at once (`max_sequences`, 256) and a share of pages to keep free at admission (`reserve`).

`add(prompt, &request)` queues a request and gives its `RequestId`; `cancel(id)` ends it wherever it is, its pages given back. The work of checking a request and compiling its constraint, which can take a while, may be done on another thread: `Pipeline::prepare` gives a `Prepared` request, and `add_prepared` queues it. That is how the server keeps the engine's thread for the passes (chapter 12).

Each `step()` makes one pass:

1. **Admit.** Waiting requests enter, in order, while the free pages hold their prompt (less what prefix sharing gives them) and one more token, beyond the pages the prompts already admitted will still take and the reserve. Alone, a request takes what there is.
2. **Plan.** One token of every request that decodes, first; then pieces of the prompts not yet computed, in the order of admission, until the pass has `max_pass_tokens` tokens. A long prompt is thus computed over several passes and never holds up the others for long.
3. **Suspend if needed.** If the pages free do not hold the pass, the request admitted last is *suspended*: its pages go back, and it waits again at the front of the queue, its prompt and the tokens it had generated to be computed anew. Its full pages often stay in the cache, idle but indexed, and are taken back rather than computed. A request alone that the cache cannot hold ends with `CacheFull`.
4. **Compute.** A pass of decoding only goes through a recorded pass (chapter 10), one per size of batch: the sizes are 1, 2, 4, 8, 16, 24, 32, 48, then multiples of 32, and a smaller batch is completed with padding rows. The engine sets aside, when it starts, the scratch pages of the largest recording, so that recording never takes pages a pass was counting on. A size the backend fails to record (the device short of memory, say) is computed as it comes, and tried again only after 256 passes. Any other pass goes through `Model::step_batch`.
5. **Choose.** The requests whose last position was computed get a token: found on the device by `ArgMax` when all of them are greedy and unconstrained, otherwise sampled on the machine's threads from the downloaded logits.

`step` returns `Event`s: a request, its `Piece`, and its `Finish` if it ended (the request is then gone from the engine), with the prompt positions it took from shared pages. A pass that fails gives a `StepError`, with the error and the requests that were in progress, which all end; the shared pages no sequence uses are forgotten, since what the failed pass wrote may be wrong. `stats()` gives an `EngineStats`: requests waiting and running, pages and free pages, suspensions, shared positions, passes and recorded passes.

`Pipeline::batch(&requests)` makes a `Batch`: an engine whose requests all come at the start, in a cache that grows (`batch_in` takes a cache of fixed size, or of another type), with up to 8192 tokens a pass so that short prompts are computed together. `Batch::step` returns the pieces it gave, each with the index of its request; `tokens(i)`, `finish(i)`, `in_progress()` and `is_finished()` follow the requests. Each request draws from its own generator (the seed plus its index), so that requests with the same seed do not draw the same tokens.

The test `a_batch_gives_the_greedy_tokens_of_each_prompt` checks that a batch gives each prompt the tokens it gets alone. `tests/engine.rs` checks the engine on the processor: requests that arrive while others are computed, in a cache so small that they are suspended, prompts in pieces of 7 tokens and shared beginnings, each get the greedy tokens of `transformers`; a cancelled request gives its pages back; answers held to a list, a regular expression and a JSON schema take that form.

## 11.4 Choosing a token

`Sampling` applies the filters of `transformers`, in the same order, after the two penalties of OpenAI's API:

0. **Frequency and presence penalties**: the logit of each token generated so far (the prompt's are left alone) loses `frequency_penalty` times the number of times it was generated, plus `presence_penalty` once.
1. **Repetition penalty**: the logit of every token already in the sequence, prompt included, is divided by the factor if positive, multiplied if negative.
2. **Greedy**: with a temperature of 0, take the most likely token, the first on a tie.
3. **Temperature**: divide the logits.
4. **top-k**: keep the k most likely tokens, and those tied with the last.
5. **top-p**: keep the most likely tokens until their probabilities add up to p, at least one.
6. **min-p**: keep the tokens at least min-p times as likely as the most likely one.
7. **Draw** from what is left.

`Sampling::from_generation_config` follows `transformers`' defaults: greedy unless `do_sample` is set, and then a temperature of 1 and a top-k of 50 unless the file says otherwise. The draws use a small xoshiro256\*\* generator seeded through SplitMix64, written in the crate, so the same seed gives the same tokens on every machine.

### Answers held to a form

A `Request` may carry a `Constraint` (`crates/kapel-infer/src/constraint.rs`): a JSON schema, any JSON object, a regular expression matched whole, a grammar in llguidance's Lark syntax, or one text of a list. The constraint is compiled by llguidance into a parser that reads the answer as it is generated; before each token, the parser gives the set of tokens that can continue the answer in its form, and the others get a logit of −∞. Sampling then chooses among the allowed tokens only, with the same filters.

llguidance works on the bytes of the tokens, kept in a tree. A `Grammars` is made once per pipeline, at the first constraint: the bytes of every id (`Tokenizer::token_bytes`, chapter 9), special tokens marked so that no text spells them, ids beyond the tokenizer empty. A list of texts becomes a Lark grammar of alternatives, each a JSON string. Each answer then gets its own `Guide`. The model may have several end tokens; they are all allowed when llguidance allows its own (the answer may end), and alone allowed once nothing can follow, so a complete answer ends with `Finish::Eos`. An end token is not fed to the parser.

A constraint that does not compile is refused when the request is made (`Pipeline::check_constraint` checks one). A parser that fails during an answer, out of its limits say, ends that request only, with `Finish::Error`; the others go on. Constrained requests choose on the host, since the mask is computed there.

## 11.5 Speculative decoding

A pass of decoding reads every weight for one token, and costs about as much for a few: the device waits for memory. `Pipeline::speculate(prompt, request, draft)` (`crates/kapel-infer/src/speculative.rs`) lets a cheap proposer guess the next tokens, and checks them all in one pass of the model, as training does, where each position predicts the next. It gives the same items as `generate`, and `stats()` counts the passes, the tokens proposed and those accepted.

Three proposers, the variants of `Draft`:

- `Draft::Model { pipeline, tokens }`: a smaller model of the same tokenizer (checked token by token, through `token_bytes`), on the same backend, proposing `tokens` tokens one at a time with its own cache. Its single-token steps go through recorded passes, so each costs one launch.
- `Draft::Ngram { tokens, longest, shortest }`: prompt lookup, no second model. `lookup` finds the latest earlier place where the last `n` tokens of the sequence appeared, `n` from `longest` down to `shortest`, and proposes what followed. It guesses well when the answer copies its prompt (code edits, summaries, quotes), little on open questions.
- `Draft::MultiToken`: the model's own multi-token prediction layer, for MiMo (`crates/kapel-infer/src/multi_token.rs`). At position `i`, the layer reads the model's final hidden state there (before the final norm) and token `i + 1`, and predicts token `i + 2` (`Model::multi_token`, chapter 10). Each pass gives the hidden states of the positions it read; after the check, the layer reads those now final and proposes one token for the next pass.

**The rule.** Keeping the guesses the model would have drawn would be another draw. To keep the model's distribution exactly (Leviathan, Kalman and Matias, 2023; Chen et al., 2023), `verify` accepts each proposal `x`, drawn from the proposer's distribution `q`, with probability `min(1, p(x)/q(x))`, `p` being the model's. At the first refusal it draws from `max(0, p − q)`, renormalized, and stops; when every proposal is accepted, the pass gives one token more, drawn from `p` at the last position. For this, the sampler is split into `distribution` (penalties, temperature and filters, as section 11.4) and `draw`. A distribution is a `Choice`: `Token(t)` when certain (greedy, or a proposal without probabilities, such as a prompt lookup's) or `Probabilities`. A certain choice needs no draw: in greedy decoding, the rule keeps the proposals as long as they are the model's choice, then gives the model's choice. Greedy answers are thus the same as without a proposer, and sampled ones follow the same distribution, though not the same draws for a given seed.

**A round.** The proposer gives up to `k` proposals, fewer near `max_tokens`. The model reads, in one pass, the tokens it has not seen (the last token, or the prompt at first) and the proposals, with `Model::extend_last`, which gives the logits of the last `k + 1` positions only. In greedy decoding, the choices are made on the device by `ArgMax`, one number per position coming back; otherwise the rows are read and each position's distribution computed with the proposals before it. After the check, the model's cache and the proposer's are cut back to the accepted tokens (`Cache::truncate`); the token drawn by the check is in no cache yet, and starts the next round. The finishing rules of `generate` apply token by token, and end the round early.

For the greedy answers to be the same bits, the positions of one sequence must give the same logits computed together as one at a time. The products do (each row is computed alone), and paged attention cuts the positions into chunks sized by the sequences and heads of a pass, not by its queries (chapter 7); `proposals_checked_together_give_the_logits_of_steps` in `crates/kapel-cuda/tests/recorded.rs` checks it bit for bit.

**Tests.** `verify` is checked by a chi-square test: over 200,000 checks of two positions of five tokens, where the model's second distribution depends on the first token, the pairs produced follow the model's distribution and not the proposer's, for a proposer that draws and one that proposes fixed tokens. `crates/kapel-infer/tests/speculative.rs` checks that greedy speculation gives the greedy tokens of SmolLM2-135M saved from `transformers`, with three proposers: the model quantized to int4 (sometimes wrong), the model itself (always right), and prompt lookup; and that the multi-token layer of the small MiMo of `data/parity/mimo` proposes without changing the tokens.

**Measures.** The example `speculative` of `kapel-cli` answers six chat questions, 256 greedy tokens each, alone and with each proposer, and checks the answers are the same. On an RTX 3090 in bf16:

| Model | Proposer | Tokens/s | Speed-up | Same answers | Accepted | Tokens per pass |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3 8B | alone | 49.9 | — | — | — | 1 |
| Qwen3 8B | Qwen3 0.6B, k = 3 | 85.2 | 1.71× | 6/6 | 53 % | 2.56 |
| Qwen3 8B | Qwen3 0.6B, k = 4 | 85.6 | 1.72× | 6/6 | 47 % | 2.84 |
| Qwen3 8B | Qwen3 0.6B, k = 6 | 80.2 | 1.61× | 6/6 | 36 % | 3.13 |
| Qwen3 8B | Qwen3 0.6B, k = 8 | 72.6 | 1.46× | 6/6 | 29 % | 3.28 |
| Qwen3 8B | prompt lookup, k = 4 | 53.4 | 1.07× | 6/6 | 17 % | 1.10 |
| MiMo-7B-RL | alone | 54.7 | — | — | — | 1 |
| MiMo-7B-RL | its multi-token layer | 83.8 | 1.53× | 6/6 | 76 % | 1.75 |

Beyond four proposals, the proposer's passes cost more than the extra tokens accepted bring.

**Limits.** A request held to a constraint is refused (the parser would have to be taken back with the refused proposals). Speculation runs one sequence at a time: the engine and the server do not use it.

## 11.6 Text that is never taken back

A token does not always map to finished text. A character may be split over two tokens, and the start of a stop sequence may appear before it is complete. A `Piece` therefore carries a token and the text it makes *final*: the concatenation of all pieces' texts is the answer, and no later token changes what was already given out.

Spaces and multi-byte characters depend on their neighbours, so the text of the tokens is not the sum of their texts. The text stream decodes the last few tokens twice, as Hugging Face's `DecodeStream` does: the tokens from `prefix` to `read`, and the same with the new ones. What the second adds to the first is new text, after the text of the tokens before `read`, already final. When the new text ends with a whole character, the window moves: `prefix` to `read`, `read` to the end. The cost of a token thus stays the same however long the answer, and the text is the whole sequence's, decoded at once: the test `streamed_text_is_the_whole_text` checks it for every tokenizer of `data/parity`. Of that text, the stream gives out only the new suffix. It holds back a trailing U+FFFD, the mark of an unfinished character, and the longest ending that could be the start of a stop sequence. When a stop sequence appears, possibly straddling text already held back, the answer is cut just before it.

## 11.7 The kapel command

`kapel serve`, the server, is the subject of chapter 12, and `kapel quantize` and `kapel eval`, which make quantized models and measure them, of chapter 13; this section is about `kapel run`.

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
| `--stats` | prompt and generation speeds on standard error, and the proposals accepted |
| `--draft MODEL`, `--prompt-lookup`, `--multi-token` | speculative decoding (section 11.5): a smaller model of the same tokenizer proposes, or the tokens of the conversation, or the model's multi-token layer; one of them |
| `--speculate K` | the proposals per pass (default 4) |

The example `generation_parity` compares Kapel's greedy generations with those of `transformers` and, at the first difference, reports the gap between the two candidates' logits: a gap near zero is a tie that rounding decides, not a bug.

Two more examples measure speed and quality. `profile` times each operation of the backend while a model generates, and lists them by type and, for quantized products, by weight shape with the bandwidth reached; it is how the CPU work of step I1 found where the time of a token went. `perplexity` scores a text as llama.cpp's `llama-perplexity` does (chunks of 512 tokens, the first replaced by the beginning of sequence, each scored on its second half), by pieces of 16 tokens after the cache, so with the kernels of generation; the options of `kapel run` choose them. On wikitext-2, Llama 3.2 1B in Q4\_K\_M scores 16.21 with Kapel, in f32 activations as in bytes, and 16.25 with llama.cpp. `--kv-cache` measures what a smaller cache costs: over 20 chunks, 16.206 in f16 (as in f32), 16.264 in f8e4m3 and 16.586 in f8e5m2, whose two bits of mantissa round the keys too coarsely.

`throughput` measures generation by batches as vLLM's `vllm bench throughput` does with random prompts: `--sequences` prompts of `--input` random tokens, each continued by `--output` tokens, greedy and past the end tokens, all in one `Batch`. It prints the time of the prefill and of the decoding, and the tokens generated per second. On an RTX 3090, Llama 3.2 3B in bf16 generates 3328 tokens per second for 64 sequences of 128 + 256 tokens, against 3710 for vLLM 0.32 (90 %), and 105 for one sequence, against 108.5; `--kv-cache f8e4m3` gives the same throughput. `tools/parity/README.md` gives both commands.
