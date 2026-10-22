"""Greedy generations of `transformers`, for Kapel's generation parity.

For each model, writes `data/parity/generation/<name>.json`: the directory of the
model's files (the snapshot of the Hugging Face cache, which Kapel reads as is),
and for each prompt its token ids and the tokens `transformers` generates after
it, greedily, in float32 (on the card if there is one, without TF32) and, on a
card, in bfloat16.

Usage: python tools/parity/generation_reference.py [names or model ids...]
Needs: torch, transformers, huggingface_hub. Gated models need HF_TOKEN.
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

PROMPTS = [
    "The history of the Roman Empire begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]
CHAT = "Explain in a few paragraphs why the sky is blue."
NEW_TOKENS = 256


def main(names):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[2] / "data" / "parity" / "generation"
    root.mkdir(parents=True, exist_ok=True)
    for name in names or ["smollm2-135m", "smollm2-135m-instruct", "qwen3-0.6b"]:
        try:
            run(root, name)
        except Exception as error:  # one failing model must not stop the others
            print(f"{name}: FAILED: {error}", flush=True)


def generate(model, ids):
    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids], device=model.device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=model.device),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
        )
    return out[0, len(ids):].tolist()


def run(root, name):
    model_id = MODELS.get(name, name)
    print(f"{name}: {model_id}", flush=True)
    directory = snapshot_download(
        model_id, allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer.model"]
    )
    tokenizer = AutoTokenizer.from_pretrained(directory)
    cases = [{"text": p, "chat": False, "ids": tokenizer(p).input_ids} for p in PROMPTS]
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": CHAT}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        cases.append({"text": CHAT, "chat": True, "ids": list(ids)})
    card = torch.cuda.is_available()
    runs = [("f32", torch.float32)] + ([("bf16", torch.bfloat16)] if card else [])
    weights = sum(f.stat().st_size for f in Path(directory).glob("*.safetensors"))
    for label, dtype in runs:
        # In float32, a model stored in 16 bits takes twice its files: on the
        # processor if the card cannot hold it.
        size = weights * (2 if dtype == torch.float32 else 1)
        device = "cuda" if card and size < 0.8 * torch.cuda.mem_get_info()[0] else "cpu"
        print(f"  {label} on {device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(directory, dtype=dtype).to(device)
        model.eval()
        for case in cases:
            case[label] = generate(model, case["ids"])
            print(f"  {label}: {len(case[label])} tokens", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = {"model": model_id, "directory": directory, "cases": cases}
    (root / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"  saved to {root / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
