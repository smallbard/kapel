"""A small Qwen3.6-35B-A3B with random weights, for Kapel's parity checks.

Qwen3.6-35B-A3B (`qwen3_5_moe`) alternates three layers of Gated DeltaNet and
one of gated attention, each followed by a mixture of experts with a shared
expert. This script builds the same architecture at a small size with
`transformers`, and saves it as the published checkpoint is laid out: a
`config.json` of type `qwen3_5_moe` whose sizes are under `text_config`, and
weights under `model.language_model.` with the experts stacked
(`mlp.experts.gate_up_proj [E, 2I, H]`, `mlp.experts.down_proj [E, H, I]`).

Saves into `data/parity/qwen3_5_moe/`, as `reference.py` does for the other
models: the model, the tokenizer of Qwen3.6-35B-A3B, and `reference.safetensors`
with `ids_<i>` and `logits_<i>`. The last text is longer than DeltaNet's chunks
of 64 positions, so that its state is carried from chunk to chunk.

Usage: python tools/parity/moe_reference.py
Needs: torch, transformers (with `qwen3_5_moe`), safetensors, huggingface_hub
(the tokenizer).
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch import nn
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

SOURCE = "Qwen/Qwen3.6-35B-A3B"

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Kapel loads models written by others, and gives the same logits.",
    "Numbers: 12345 + 67890 = 80235; émigré, naïve, 東京, 🙂 and\ttabs\n\nnew lines.",
    " ".join(
        f"Line {i}: the state of a linear attention is carried from one chunk to the next."
        for i in range(12)
    ),
]


def main():
    torch.manual_seed(0)
    out = Path(__file__).resolve().parents[2] / "data" / "parity" / "qwen3_5_moe"
    out.mkdir(parents=True, exist_ok=True)
    source = snapshot_download(SOURCE, allow_patterns=["tokenizer*", "*.json", "*.txt", "*.jinja"])
    tokenizer = AutoTokenizer.from_pretrained(source)
    published = json.loads((Path(source) / "config.json").read_text())
    text = dict(published["text_config"])
    text.update(
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        mtp_num_hidden_layers=0,
        initializer_range=0.1,
    )
    config = Qwen3_5MoeTextConfig(**text)
    config._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                # Zero-centred norms of zeros, or gated norms of ones, would
                # hide a misplaced offset.
                nn.init.uniform_(p, -0.5, 0.5) if "linear_attn" not in name else nn.init.uniform_(p, 0.5, 1.5)
            elif name.endswith("A_log"):
                nn.init.uniform_(p, -2.0, 1.0)
            elif name.endswith("dt_bias"):
                nn.init.uniform_(p, -1.0, 1.0)
            elif name.endswith("conv1d.weight"):
                nn.init.uniform_(p, -0.5, 0.5)
            elif "experts" in name:
                nn.init.normal_(p, std=0.1)

    weights = {}
    for name, value in model.state_dict().items():
        if name.startswith("model."):
            name = "model.language_model." + name[len("model.") :]
        weights[name] = value.contiguous()
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    raw = dict(published)
    raw.pop("vision_config", None)
    raw["text_config"] = config.to_dict()
    (out / "config.json").write_text(json.dumps(raw, indent=1))
    tokenizer.save_pretrained(out)

    tensors, texts = {}, []
    for i, line in enumerate(TEXTS):
        ids = tokenizer(line, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        tensors[f"ids_{i}"] = ids[0].to(torch.int32).contiguous()
        tensors[f"logits_{i}"] = logits.contiguous()
        texts.append({"text": line, "ids": ids[0].tolist()})
        print(f"text {i}: {ids.shape[1]} tokens", flush=True)
    save_file(tensors, out / "reference.safetensors")
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
