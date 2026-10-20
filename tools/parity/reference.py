"""Reference outputs of `transformers`, for Kapel's parity checks.

For each model, saves into `data/parity/<name>/`:
- the model itself (`config.json`, `model.safetensors`, tokenizer files), in float32;
- `reference.safetensors`: the token ids of fixed texts (`ids_<i>`, int32) and the
  logits of the model on them (`logits_<i>`, float32);
- `texts.json`: the texts and their token ids, for the tokenizer check.

Usage: python tools/parity/reference.py [model ids...]
Needs: torch, transformers, safetensors.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "qwen2": "yujiepan/qwen2.5-tiny-random",
    "qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gemma": "hf-internal-testing/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-tiny-v2/tiny-random-Gemma2ForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
}

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Kapel loads models written by others, and gives the same logits.",
    "Numbers: 12345 + 67890 = 80235; émigré, naïve, 東京, 🙂 and\ttabs\n\nnew lines.",
]


def main(names):
    root = Path(__file__).resolve().parents[2] / "data" / "parity"
    for name in names or MODELS:
        try:
            run(root, name)
        except Exception as error:  # one failing model must not stop the others
            print(f"{name}: FAILED: {error}", flush=True)


def run(root, name):
    if True:
        model_id = MODELS.get(name, name)
        out = root / name.replace("/", "--")
        out.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        tensors, texts = {}, []
        for i, text in enumerate(TEXTS):
            ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                logits = model(ids).logits[0].float()
            tensors[f"ids_{i}"] = ids[0].to(torch.int32).contiguous()
            tensors[f"logits_{i}"] = logits.contiguous()
            texts.append({"text": text, "ids": ids[0].tolist()})
        save_file(tensors, out / "reference.safetensors")
        (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1))
        print(f"  saved to {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
