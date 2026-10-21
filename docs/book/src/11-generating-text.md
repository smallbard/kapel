# 11. Generating text: kapel-infer and kapel-cli

`kapel-infer` is the first piece of the inference plan: it loads a model directory onto a backend and generates text token by token, making the same choices as `transformers`. With greedy decoding, its tokens match the reference generations of `transformers` saved in `data/parity/generation`. `kapel-cli` wraps it in the `kapel run` command.

## 11.1 The pipeline

```rust
let pipeline = Pipeline::load("SmolLM2-135M-Instruct", kapel_cpu::Cpu::new(), None)?;
let prompt = pipeline.chat_prompt(&[Message::new("user", "Hello!")])?;
for piece in pipeline.generate(&prompt, &Request::default())? {
    print!("{}", piece?.text);
}
```

`Pipeline::load(dir, backend, dtype)` gathers everything a generation needs: the model, uploaded to the backend and optionally cast; the tokenizer; the chat template, if the model has one; the default sampling settings of `generation_config.json`; and the tokens that end an answer, taken from `generation_config.json`, else `config.json`, else the tokenizer's `eos_token`.

`prompt(text)` encodes raw text; `chat_prompt(messages)` renders a conversation through the template first. A `Request` says how to choose tokens, how many to generate at most (256 by default), and which stop sequences end the answer.

## 11.2 Prefill, then one token at a time

`generate` returns a `Generation`, an iterator. Its first step passes the whole prompt to `Model::step` with an empty cache: this is the *prefill*, where the products have many rows and the GPU or the CPU's blocked product is efficient. Each later step passes only the last token, which reads the keys and values of every earlier position from the cache: this is *decoding*, where products have one row and speed is bound by memory bandwidth, which is why the CPU has a separate few-rows path (chapter 6).

The logits are downloaded to the host as `f64`, a token is chosen, and the loop continues until one of three endings, reported by `finish()`:

| Ending | Cause |
| --- | --- |
| `Finish::Eos` | the model produced a token that ends its answers |
| `Finish::Stop(text)` | a stop sequence appeared; the answer stops before it |
| `Finish::Length` | the maximum number of tokens was generated |

After an error, the iterator ends: the cache may hold part of a failed step, so nothing can follow.

## 11.3 Choosing a token

`Sampling` applies the filters of `transformers`, in the same order:

1. **Repetition penalty**: the logit of every token already in the sequence, prompt included, is divided by the factor if positive, multiplied if negative.
2. **Greedy**: with a temperature of 0, take the most likely token, the first on a tie.
3. **Temperature**: divide the logits.
4. **top-k**: keep the k most likely tokens, and those tied with the last.
5. **top-p**: keep the most likely tokens until their probabilities add up to p, at least one.
6. **min-p**: keep the tokens at least min-p times as likely as the most likely one.
7. **Draw** from what is left.

`Sampling::from_generation_config` follows `transformers`' defaults: greedy unless `do_sample` is set, and then a temperature of 1 and a top-k of 50 unless the file says otherwise. The draws use a small xoshiro256\*\* generator seeded through SplitMix64, written in the crate, so the same seed gives the same tokens on every machine.

## 11.4 Text that is never taken back

A token does not always map to finished text. A character may be split over two tokens, and the start of a stop sequence may appear before it is complete. A `Piece` therefore carries a token and the text it makes *final*: the concatenation of all pieces' texts is the answer, and no later token changes what was already given out.

The text stream decodes the whole generated sequence again at each token, because spaces and multi-byte characters depend on their neighbours, and gives out only the new suffix. It holds back a trailing U+FFFD, the mark of an unfinished character, and the longest ending that could be the start of a stop sequence. When a stop sequence appears, possibly straddling text already held back, the answer is cut just before it.

## 11.5 The kapel command

```
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B --prompt "Why is the sky blue?"
```

`kapel run MODEL` accepts a local directory or a Hub repository, which it downloads once into the cache. Without `--prompt`, it holds a conversation: one message per line on standard input, an empty line to end, each answer added to the history. The main options are:

| Option | Effect |
| --- | --- |
| `--device cpu\|cuda[:N]` | where to compute; CUDA needs the `cuda` feature and defaults to bf16 weights |
| `--dtype f32\|bf16\|f16` | the type of the weights |
| `--system TEXT`, `--raw` | a system message; or skip the chat template |
| `--max-tokens N`, `--stop TEXT` | limits of each answer |
| `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`, `--seed` | override the model's sampling settings |
| `--revision REV` | a branch, tag or commit of a Hub repository |
| `--stats` | prompt and generation speeds on standard error |

On the CPU, weights keep their stored type unless `--dtype` is given, although the help text says the default is f32. The example `generation_parity` compares Kapel's greedy generations with those of `transformers` and, at the first difference, reports the gap between the two candidates' logits: a gap near zero is a tie that rounding decides, not a bug.
