"""FineWeb in GPT-2's tokens, as llm.c writes them, for the recipes of
pre-training.

    python tools/train/fineweb_gpt2.py OUT [--train N]

Downloads into OUT the shards of chrisdryden/FineWebTokenizedGPT2, made by
llm.c's dev/data/fineweb.py from FineWeb's sample of 10 billion tokens:
`fineweb_val_000000.bin` (100 million tokens, the validation) and the first N
of `fineweb_train_000001.bin`… (100 million tokens each, 200 MB). The run of
GPT-2 124M to the end reads 10 billion tokens, all 100 shards (20 GB); the
first 250 steps read 131 million, two shards.

Needs: huggingface_hub.
"""

import argparse

from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--train", type=int, default=2)
    a = p.parse_args()
    names = ["fineweb_val_000000.bin"]
    names += [f"fineweb_train_{i:06d}.bin" for i in range(1, a.train + 1)]
    for name in names:
        hf_hub_download("chrisdryden/FineWebTokenizedGPT2", name, repo_type="dataset",
                        local_dir=a.out)
        print(name)


if __name__ == "__main__":
    main()
