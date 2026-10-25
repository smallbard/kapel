"""GSM8K as the recipes of post-training read it.

    python tools/train/gsm8k.py OUT

Downloads GSM8K (openai/gsm8k, `main`) and writes into OUT:

- `train.jsonl`: the 7,473 problems of the training split as examples of
  supervised fine-tuning, `{"prompt": [user], "completion": [assistant]}`,
  the answer as GSM8K writes it (its reasoning, then `#### 72`);
- `prompts.jsonl`: the same problems as prompts for GRPO, `{"prompt":
  [user], "answer": …}`, the reward reading the number after `####`;
- `test.jsonl`: the 1,319 problems of the test split, the same way, for the
  accuracy.

Needs: huggingface_hub, pyarrow.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def rows(split):
    path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    return pq.read_table(path).to_pylist()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    train, test = rows("train"), rows("test")

    def user(r):
        return [{"role": "user", "content": r["question"]}]

    with open(out / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps({"prompt": user(r),
                                "completion": [{"role": "assistant", "content": r["answer"]}]})
                    + "\n")
    for name, split in [("prompts.jsonl", train), ("test.jsonl", test)]:
        with open(out / name, "w") as f:
            for r in split:
                f.write(json.dumps({"prompt": user(r), "answer": r["answer"]}) + "\n")
    print(f"{len(train)} problems to train on, {len(test)} to test, in {out}")


if __name__ == "__main__":
    main()
