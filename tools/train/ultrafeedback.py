"""UltraFeedback's pairs as the recipe of DPO reads them.

    python tools/train/ultrafeedback.py OUT [--tokenizer MODEL] [--max-tokens 1024]
                                            [--train 4000] [--test 500]

Downloads trl-lib/ultrafeedback_binarized (UltraFeedback, each prompt's best
answer against a worse one, by GPT-4's scores) and writes into OUT
`train.jsonl` and `test.jsonl`, `{"chosen": [user, assistant], "rejected":
[user, assistant]}`, the prompt the messages both begin with, as TRL reads
them. Only the pairs whose two conversations, in MODEL's chat template, fit
in `--max-tokens` with 16 to spare are kept, so that neither Kapel nor TRL
cuts them: the
first `--train` of the training split, the first `--test` of its test split.

Needs: huggingface_hub, pyarrow, transformers.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--test", type=int, default=500)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, wanted in [("train", a.train), ("test", a.test)]:
        path = hf_hub_download("trl-lib/ultrafeedback_binarized",
                               f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        kept, seen = [], 0
        for row in pq.read_table(path, columns=["chosen", "rejected"]).to_pylist():
            if len(kept) == wanted:
                break
            seen += 1
            # The tokens of the conversation, with room for what a template
            # adds to a prompt and its completion encoded apart (Qwen 3's
            # empty block of thoughts).
            fits = all(
                len(tok(tok.apply_chat_template(row[side], tokenize=False))["input_ids"])
                <= a.max_tokens - 16
                for side in ("chosen", "rejected")
            )
            if fits and row["chosen"] != row["rejected"]:
                kept.append({"chosen": row["chosen"], "rejected": row["rejected"]})
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(kept)} pairs of {seen} read fit in {a.max_tokens} tokens")


if __name__ == "__main__":
    main()
