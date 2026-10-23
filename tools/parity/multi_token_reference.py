"""A small MiMo model with random weights, and the outputs of its multi-token
prediction layer, for Kapel's parity checks.

MiMo is Qwen 2 with a multi-token prediction (MTP) layer, which `transformers`
loads but never runs: vLLM runs it, to propose tokens. This script builds the
layer as MiMo's `modeling_mimo.py` declares it and runs it as vLLM's
`mimo_mtp.py` does. At position `i`, the layer reads the model's final hidden
state there, before the final norm, and the embedding of token `i + 1` (zero
at position 0), and predicts token `i + 2`.

Saves into `data/parity/mimo/`, as `reference.py` does for the other models:
the model (`config.json` of type `mimo`, `model.safetensors` with
`model.mtp_layers.0.*`, the tokenizer of MiMo-7B-RL), and `reference.safetensors`
with `ids_<i>`, `logits_<i>` and `mtp_logits_<i>` (`[T - 1, vocab]`).

Usage: python tools/parity/multi_token_reference.py
Needs: torch, transformers, safetensors, huggingface_hub (MiMo-7B-RL's tokenizer).
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch import nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Kapel loads models written by others, and gives the same logits.",
    "Numbers: 12345 + 67890 = 80235; émigré, naïve, 東京, 🙂 and\ttabs\n\nnew lines.",
]


class MiMoMTPLayers(nn.Module):
    """As `modeling_mimo.py` declares it."""

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.input_layernorm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(h * 2, h, bias=False)
        self.final_layernorm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)


def main():
    torch.manual_seed(0)
    out = Path(__file__).resolve().parents[2] / "data" / "parity" / "mimo"
    out.mkdir(parents=True, exist_ok=True)
    source = snapshot_download("XiaomiMiMo/MiMo-7B-RL", allow_patterns=["tokenizer*", "*.json", "*.txt"])
    tokenizer = AutoTokenizer.from_pretrained(source)
    vocab = json.loads((Path(source) / "config.json").read_text())["vocab_size"]
    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=640000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        initializer_range=0.1,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).eval()
    mtp = MiMoMTPLayers(config).eval()
    for module in [model, mtp]:
        for name, p in module.named_parameters():
            if name.endswith("norm.weight"):
                # Norms of ones would hide a missing one.
                nn.init.uniform_(p, 0.5, 1.5)
    weights = {k: v.contiguous() for k, v in model.state_dict().items()}
    weights.update({f"model.mtp_layers.0.{k}": v.contiguous() for k, v in mtp.state_dict().items()})
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    raw = config.to_dict()
    raw.update(model_type="mimo", architectures=["MiMoForCausalLM"], num_nextn_predict_layers=1)
    (out / "config.json").write_text(json.dumps(raw, indent=1))
    tokenizer.save_pretrained(out)

    captured = {}
    model.model.norm.register_forward_hook(lambda m, args, out: captured.update(hidden=args[0]))
    tensors, texts = {}, []
    for i, text in enumerate(TEXTS):
        ids = tokenizer(text, return_tensors="pt").input_ids
        t = ids.shape[1]
        with torch.no_grad():
            logits = model(ids).logits[0].float()
            hidden = captured["hidden"][:, : t - 1]
            embeds = model.model.embed_tokens(ids[:, 1:]).clone()
            embeds[:, 0] = 0  # position 0, as vLLM masks it
            x = mtp.input_proj(
                torch.cat([mtp.hidden_layernorm(hidden), mtp.token_layernorm(embeds)], dim=-1)
            )
            positions = torch.arange(t - 1).unsqueeze(0)
            cos_sin = model.model.rotary_emb(x, positions)
            mask = torch.full((t - 1, t - 1), float("-inf")).triu(1)[None, None]
            residual = x
            a, _ = mtp.self_attn(mtp.input_layernorm(x), position_embeddings=cos_sin, attention_mask=mask)
            x = residual + a
            x = x + mtp.mlp(mtp.post_attention_layernorm(x))
            mtp_logits = model.lm_head(mtp.final_layernorm(x))[0].float()
        tensors[f"ids_{i}"] = ids[0].to(torch.int32).contiguous()
        tensors[f"logits_{i}"] = logits.contiguous()
        tensors[f"mtp_logits_{i}"] = mtp_logits.contiguous()
        texts.append({"text": text, "ids": ids[0].tolist()})
    save_file(tensors, out / "reference.safetensors")
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
