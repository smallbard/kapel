# Evaluating a model

`kapel eval` measures a model on tasks of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), the tool behind most published scores: the perplexity of WikiText, and the accuracies of HellaSwag, ARC and MMLU. Its numbers are those lm-evaluation-harness gives for the same model (the same accuracies, perplexities within 0.001 %), so they can be compared with published ones, and a quantized model can be compared with its original.

## 1. Export the tasks, once

lm-evaluation-harness turns each task into requests to a model. A script asks it for these requests, without a model, and writes them to a file. It needs Python and lm-evaluation-harness once; Kapel then needs neither.

```sh
pip install "lm-eval[hf]"
python tools/eval/export_tasks.py \
    --tasks wikitext,hellaswag,arc_easy,arc_challenge,mmlu \
    --output tasks.jsonl
```

Any task of lm-evaluation-harness whose requests are log-likelihoods works: tasks by choice (the most likely answer wins) and perplexities. Tasks that generate text are not replayed.

## 2. Score a model

```sh
kapel eval Qwen/Qwen2.5-0.5B-Instruct --tasks tasks.jsonl --limit 20 --device cuda
```

```
arc_challenge                                20  acc 0.3000  acc_norm 0.2000
arc_easy                                     20  acc 0.6500  acc_norm 0.5500
hellaswag                                    20  acc 0.3500  acc_norm 0.2500
...
wikitext                                     20  word ppl    18.6349  byte ppl 1.7310  bits/byte 0.7916
mmlu (57 tasks)                            1140  acc 0.4807  acc_norm 0.4807
```

| Metric | Meaning |
| --- | --- |
| `acc` | the share of questions where the most likely answer is the right one |
| `acc_norm` | the same, the likelihood of each answer divided by its length in characters |
| `word ppl`, `byte ppl` | the perplexity of the text by word and by byte: lower is better |
| `bits/byte` | bits per byte of text: lower is better |

`--limit N` takes the first `N` documents of each task, for a quick estimate: the run above takes two and a half minutes on an RTX 3090. Without it, everything is scored, which takes hours on the processor for MMLU's 14,000 questions. `--only mmlu` keeps some tasks or groups. `--output scores.json` writes the scores as JSON. `--device cuda` and the other loading options of `kapel run` apply, including `--quantize`, to measure a quantization at loading.

## Compare a quantization with its original

```sh
kapel eval meta-llama/Llama-3.2-1B --tasks tasks.jsonl --only wikitext,hellaswag --output f16.json
kapel eval llama-3.2-1b-q4_0.gguf    --tasks tasks.jsonl --only wikitext,hellaswag --output q4.json
```

In `bf16` on a card, a perplexity moves in its fourth digit with `--piece`, the tokens computed per pass (the sums are rounded in another order); compare models with the same options. On a few hundred questions, differences of one or two points of accuracy are noise; perplexity is more sensitive, and shows a quantization's loss first.

For a perplexity comparable with llama.cpp's `llama-perplexity` (chunks of 512 tokens, as llama.cpp reports), use the example `perplexity` of the repository:

```sh
cargo run --release -p kapel-cli --example perplexity -- MODEL wiki.test.raw
```
