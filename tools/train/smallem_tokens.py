"""Smallem's tokens as files of llm.c, which `kapel train` reads.

    python tools/train/smallem_tokens.py SMALLEM_DATA OUT

SMALLEM_DATA is a directory of Smallem's `prepare` (`data/bpe`): its
`*train.bin` and `*val.bin` hold the tokens, little-endian `u16`, without a
header. Each is written into OUT with llm.c's header of 256 `i32` (magic
20240520, version 1, the number of tokens) as `smallem_train_NNN.bin` and
`smallem_val_NNN.bin`, so that `data.train = "OUT/smallem_train_*.bin"`
reads them.

Needs nothing but Python.
"""

import struct
import sys
from pathlib import Path


def convert(source, target):
    tokens = source.stat().st_size // 2
    header = struct.pack("<256i", 20240520, 1, tokens, *([0] * 253))
    with open(source, "rb") as src, open(target, "wb") as dst:
        dst.write(header)
        while chunk := src.read(1 << 24):
            dst.write(chunk)
    return tokens


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for i, source in enumerate(sorted(data.glob(f"*{split}.bin"))):
            target = out / f"smallem_{split}_{i:03}.bin"
            print(f"{source} → {target}: {convert(source, target):,} tokens")


if __name__ == "__main__":
    main()
