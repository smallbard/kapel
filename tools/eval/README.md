# Evaluation

`kapel eval` replays tasks of
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with
Kapel. lm-evaluation-harness writes the requests of its tasks without a model, once;
Kapel scores them, as many times and with as many models as wanted, without Python.

```
.venv/bin/pip install "lm-eval[hf]"
.venv/bin/python tools/eval/export_tasks.py \
    --tasks wikitext,hellaswag,arc_easy,arc_challenge,mmlu --output data/eval/tasks.jsonl
kapel eval MODEL --tasks data/eval/tasks.jsonl [--limit N] [--output scores.json]
```

Each line of the file is a document: the requests of a task by choice (HellaSwag,
ARC, MMLU), with the index of the right answers and the length of each, or a text
whose perplexity is measured (WikiText). `kapel eval` gives `acc` and `acc_norm` (the
log-likelihood divided by the answer's length in characters) for the first, the
perplexity by word and by byte and the bits per byte for the second, by task and by
group (MMLU's 57 subjects, weighted by their number of questions, as
lm-evaluation-harness weights them).

The requests are tokenized and scored as lm-evaluation-harness's `hf` model does:

- the trailing spaces of a context move to the start of its continuation; context and
  continuation are encoded together, then split where the context encoded alone ends;
- the tokenizer adds its special tokens, unless the text starts with the prefix token
  (the tokenizer's `bos_token`, else its `eos_token`; `--prefix-token` otherwise);
- an empty context is the prefix token;
- a sequence longer than the model's `max_position_embeddings` (`--max-length`) is cut
  on the left;
- a text is scored by disjoint windows of that length, the first after the prefix
  token.

Choices of one question share their context, and questions of one task often share a
prompt: the cache keeps the positions a sequence shares with the one before
(`kapel_infer::Scorer`).

## Against lm-evaluation-harness

```
lm_eval --model hf --model_args pretrained=MODEL_DIR,dtype=float32 --device cpu \
    --tasks wikitext,hellaswag,arc_easy,arc_challenge,mmlu --limit 20 --batch_size 8
kapel eval MODEL_DIR --tasks data/eval/tasks.jsonl --limit 20
```

`--limit` takes the first documents of each task in both.
