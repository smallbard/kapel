"""The small Qwen 3.5 mixture of experts of `moe_reference.py`, written as a GGUF
file the way llama.cpp's converter writes `qwen35moe` (`conversion/qwen.py`,
`Qwen3_5MoeTextModel` and the classes it derives from), for Kapel's parity
checks of `Model::from_gguf`.

The converter's changes to the weights, reproduced here:
- the norms centred on zero get their 1 added (`w + 1`), but the gated norm of
  linear attention (`linear_attn.norm`);
- `A_log` becomes `ssm_a = −exp(A_log)`, `dt_bias` becomes `ssm_dt.bias`, the
  convolution's weight `[C, 1, K]` is squeezed to `[C, K]`;
- the value heads of linear attention go from grouped by key head to tiled, in
  the value rows of `in_proj_qkv` and of the convolution, the rows of
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, and the columns of
  `out_proj`;
- the stacked `gate_up_proj` of the experts is cut into `ffn_gate_exps` and
  `ffn_up_exps`.

Writes `data/parity/qwen3_5_moe/model-f32.gguf`, every tensor in `f32`.

Usage: python tools/parity/moe_gguf.py
Needs: numpy, safetensors, gguf (gguf-py).
"""

import json
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file

NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
}


def tiled(x, dim, k_heads, per_k, head_dim):
    """Value heads from grouped by key head to tiled, along `dim`."""
    shape = list(x.shape)
    x = x.reshape(shape[:dim] + [k_heads, per_k, head_dim] + shape[dim + 1 :])
    x = np.swapaxes(x, dim, dim + 1)
    return np.ascontiguousarray(x).reshape(shape)


def main():
    out = Path(__file__).resolve().parents[2] / "data" / "parity" / "qwen3_5_moe"
    raw = json.loads((out / "config.json").read_text())
    c = raw["text_config"]
    weights = load_file(out / "model.safetensors")
    hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    per_k = hv // hk

    writer = gguf.GGUFWriter(str(out / "model-f32.gguf"), "qwen35moe")
    writer.add_block_count(c["num_hidden_layers"])
    writer.add_context_length(c["max_position_embeddings"])
    writer.add_embedding_length(c["hidden_size"])
    writer.add_head_count(c["num_attention_heads"])
    writer.add_head_count_kv(c["num_key_value_heads"])
    writer.add_rope_freq_base(c["rope_parameters"]["rope_theta"])
    writer.add_layer_norm_rms_eps(c["rms_norm_eps"])
    writer.add_expert_count(c["num_experts"])
    writer.add_expert_used_count(c["num_experts_per_tok"])
    writer.add_key_length(c["head_dim"])
    writer.add_value_length(c["head_dim"])
    writer.add_expert_feed_forward_length(c["moe_intermediate_size"])
    writer.add_expert_shared_feed_forward_length(c["shared_expert_intermediate_size"])
    writer.add_ssm_conv_kernel(c["linear_conv_kernel_dim"])
    writer.add_ssm_state_size(dk)
    writer.add_ssm_group_count(hk)
    writer.add_ssm_time_step_rank(hv)
    writer.add_ssm_inner_size(dv * hv)
    writer.add_full_attention_interval(c["full_attention_interval"])
    rotary = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
    writer.add_rope_dimension_count(rotary)
    writer.add_rope_dimension_sections([11, 11, 10, 0])

    def put(name, value):
        writer.add_tensor(name, np.ascontiguousarray(value.astype(np.float32)))

    put("token_embd.weight", weights["model.language_model.embed_tokens.weight"])
    put("output_norm.weight", weights["model.language_model.norm.weight"] + 1)
    put("output.weight", weights["lm_head.weight"])
    for layer in range(c["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}."
        for name in sorted(k[len(prefix) :] for k in weights if k.startswith(prefix)):
            w = weights[prefix + name]
            if name == "mlp.experts.gate_up_proj":
                half = w.shape[1] // 2
                put(f"blk.{layer}.ffn_gate_exps.weight", w[:, :half])
                put(f"blk.{layer}.ffn_up_exps.weight", w[:, half:])
                continue
            if name.endswith("norm.weight") and name != "linear_attn.norm.weight":
                w = w + 1
            if name == "linear_attn.A_log":
                w = -np.exp(w.astype(np.float32))
            if name.startswith("linear_attn.") and hk != hv:
                if name == "linear_attn.in_proj_qkv.weight":
                    q = w[: 2 * hk * dk]
                    w = np.concatenate([q, tiled(w[2 * hk * dk :], 0, hk, per_k, dv)])
                elif name == "linear_attn.conv1d.weight":
                    w = w.squeeze(1)
                    w = np.concatenate([w[: 2 * hk * dk], tiled(w[2 * hk * dk :], 0, hk, per_k, dv)])
                elif name == "linear_attn.in_proj_z.weight":
                    w = tiled(w, 0, hk, per_k, dv)
                elif name in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
                    w = tiled(w, 0, hk, per_k, 1)
                elif name in ("linear_attn.A_log", "linear_attn.dt_bias"):
                    w = tiled(w[:, None], 0, hk, per_k, 1)[:, 0]
                elif name == "linear_attn.out_proj.weight":
                    w = tiled(w, 1, hk, per_k, dv)
            if name == "mlp.shared_expert_gate.weight":
                w = w[0]
            put(f"blk.{layer}.{NAMES[name]}", w)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"saved {out / 'model-f32.gguf'}")


if __name__ == "__main__":
    main()
