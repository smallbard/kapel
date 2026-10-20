"""Reference encodings of the `tokenizers` library, for Kapel's tokenizer checks.

For each `data/parity/<name>/tokenizer.json`, writes `tokenizer_cases.json`: a
corpus of texts with their ids, with and without special tokens, and the decoded
text of those ids.

Usage: python tools/parity/tokenizers_reference.py
Also downloads the Llama 3 tokenizer into data/parity/llama3-tokenizer/.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

CORPUS = [
    "",
    " ",
    "Hello world",
    " Hello  world  ",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's fine; we've seen it, you'd agree. DON'T SHOUT!",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1,000,000 and 2026-10-20",
    "émigré, naïve café — « guillemets » à l'œuvre",
    "東京は日本の首都です。北京欢迎你",
    "Emoji: 🙂👍🏽 🇫🇷 and symbols ∑∫√∞ ≠ ≤",
    "tabs\tand\nnew\n\nlines\r\nwindows",
    "    indented code:\n    def f(x):\n        return x ** 2  # square\n",
    "fn main() { println!(\"{}\", vec![1, 2, 3].iter().sum::<i32>()); }",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "<s>[INST] What is RoPE? [/INST]</s>",
    "<bos><start_of_turn>user\nhello<end_of_turn>",
    "URL: https://huggingface.co/docs?q=tokenizers&page=2#bpe",
    "Mixed non-breaking thin space and zero​width",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Ünïcödé ÇÀPS and ligatures ﬁ ﬂ; combining é vs é",
]


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "parity"
    llama3 = root / "llama3-tokenizer"
    llama3.mkdir(parents=True, exist_ok=True)
    if not (llama3 / "tokenizer.json").exists():
        path = hf_hub_download("unsloth/Llama-3.2-1B", "tokenizer.json")
        (llama3 / "tokenizer.json").write_bytes(Path(path).read_bytes())
    for path in sorted(root.glob("*/tokenizer.json")):
        tokenizer = Tokenizer.from_file(str(path))
        cases = []
        for text in CORPUS:
            with_special = tokenizer.encode(text, add_special_tokens=True).ids
            without = tokenizer.encode(text, add_special_tokens=False).ids
            cases.append({
                "text": text,
                "ids": with_special,
                "ids_no_special": without,
                "decoded": tokenizer.decode(without, skip_special_tokens=False),
            })
        out = path.parent / "tokenizer_cases.json"
        out.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
        print(f"{path.parent.name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
