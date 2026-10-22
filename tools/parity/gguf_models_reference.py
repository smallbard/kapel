"""GGUF models and the models of `transformers` they were converted from, for
Kapel's parity of models read from GGUF.

For each model, downloads a GGUF file in Q8_0 and the directory of the model
of `transformers` it comes from, and writes
`data/parity/gguf-models/<name>.json` with their paths. Kapel's tests then
check that the GGUF file gives the same tokens as the original tokenizer, and
nearly the same predictions (Q8_0 rounds the weights) as the original model.

Usage: python tools/parity/gguf_models_reference.py [names...]
Needs: huggingface_hub.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# name: (GGUF repository, file, original repository)
MODELS = {
    "qwen3-0.6b": ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "Qwen/Qwen3-0.6B"),
    "llama3.2-1b-instruct": (
        "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "unsloth/Llama-3.2-1B-Instruct",
    ),
}


def main(names):
    root = Path(__file__).resolve().parents[2] / "data" / "parity" / "gguf-models"
    root.mkdir(parents=True, exist_ok=True)
    for name in names or list(MODELS):
        repo, file, original = MODELS[name]
        gguf = hf_hub_download(repo, file)
        patterns = ["*.json", "*.safetensors", "*.jinja"]
        directory = snapshot_download(original, allow_patterns=patterns)
        out = root / f"{name}.json"
        out.write_text(json.dumps({"name": name, "gguf": gguf, "directory": directory}, indent=1))
        print(f"{name}: saved to {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
