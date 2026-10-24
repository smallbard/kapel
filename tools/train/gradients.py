"""T0's validation: Kapel's gradients and peak memory against PyTorch's.

    python tools/train/gradients.py tokens MODEL TEXT OUT.bin
        tokenizes TEXT with MODEL's tokenizer into little-endian u32 ids;
    python tools/train/gradients.py torch MODEL TOKENS [--rows 16] [--length 1024]
        [--dtype bf16|f32] [--output GRADS.safetensors]
        the loss, gradients, time and peak memory of PyTorch on the card, with
        nanoGPT's model (`model.py`, its attention by `scaled_dot_product_attention`)
        and GPT-2's weights loaded as its `from_pretrained` does: the model in
        f32, the pass under autocast for bf16, as nanoGPT trains;
    python tools/train/gradients.py compare REFERENCE A [B ...]
        the relative error of each file's gradients to REFERENCE's, by tensor
        and overall.

The names and layouts are those of Kapel (`transformers`' `GPT2LMHeadModel`).
nanoGPT's model has no dropout here, so that both sides compute the same
function.
"""

import argparse
import json
import sys
import time

import numpy as np


def tokens(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = open(args.text, encoding="utf-8").read()
    ids = tok(text)["input_ids"]
    np.asarray(ids, dtype="<u4").tofile(args.out)
    print(f"{len(ids)} tokens written to {args.out}", file=sys.stderr)


# The projections GPT-2's checkpoints store as `Conv1D`, `[in, out]`:
# nanoGPT's `Linear` holds them transposed.
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def nanogpt(model_dir):
    """nanoGPT's GPT, with the weights of the checkpoint in `model_dir`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file

    config = json.load(open(f"{model_dir}/config.json"))
    C, H, L = config["n_embd"], config["n_head"], config["n_layer"]
    V, P = config["vocab_size"], config["n_positions"]
    eps = config.get("layer_norm_epsilon", 1e-5)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(C, 3 * C)
            self.c_proj = nn.Linear(C, C)

        def forward(self, x):
            B, T, _ = x.size()
            q, k, v = self.c_attn(x).split(C, dim=2)
            q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(C, 4 * C)
            self.c_proj = nn.Linear(4 * C, C)

        def forward(self, x):
            return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(C, eps=eps)
            self.attn = Attention()
            self.ln_2 = nn.LayerNorm(C, eps=eps)
            self.mlp = Mlp()

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))

    class Gpt(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(V, C), wpe=nn.Embedding(P, C),
                h=nn.ModuleList([Block() for _ in range(L)]), ln_f=nn.LayerNorm(C, eps=eps)))
            self.lm_head = nn.Linear(C, V, bias=False)
            self.lm_head.weight = self.transformer.wte.weight

        def forward(self, idx, targets):
            t = self.transformer
            pos = torch.arange(idx.size(1), device=idx.device)
            x = t.wte(idx) + t.wpe(pos)
            for block in t.h:
                x = block(x)
            logits = self.lm_head(t.ln_f(x))
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    model = Gpt()
    weights = load_file(f"{model_dir}/model.safetensors")
    state = {}
    for name, w in weights.items():
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue
        if not name.startswith("transformer."):
            name = "transformer." + name
        state[name] = w.t() if name.endswith(TRANSPOSED) else w
    state["lm_head.weight"] = state["transformer.wte.weight"]
    model.load_state_dict(state)
    return model


def run_torch(args):
    import torch
    from safetensors.torch import save_file

    ids = np.fromfile(args.tokens, dtype="<u4").astype(np.int64)
    n = args.rows * args.length
    x = torch.from_numpy(ids[:n]).view(args.rows, args.length).cuda()
    y = torch.from_numpy(ids[1 : n + 1]).view(args.rows, args.length).cuda()
    model = nanogpt(args.model).cuda()
    best = float("inf")
    for i in range(args.repeat):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            loss = model(x, y)
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        best = min(best, seconds)
        print(
            f"torch: run {i + 1}: loss {loss.item():.6f}, {seconds:.3f} s, "
            f"peak {peak / 2**30:.3f} GiB",
            file=sys.stderr,
        )
    print(json.dumps({"loss": loss.item(), "seconds": best,
                      "tokens_per_s": n / best, "peak_bytes": peak}))
    if args.output:
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().float().cpu()
                grads[name] = (g.t() if name.endswith(TRANSPOSED) else g).contiguous()
        # The output table is the embedding table: one gradient.
        grads.pop("lm_head.weight", None)
        save_file(grads, args.output)
        print(f"torch: gradients written to {args.output}", file=sys.stderr)


def compare(args):
    from safetensors.numpy import load_file

    ref = {k: v.astype(np.float64) for k, v in load_file(args.reference).items()}
    for path in args.others:
        other = {k: v.astype(np.float64) for k, v in load_file(path).items()}
        missing = sorted(set(ref) ^ set(other))
        if missing:
            print(f"{path}: names differ: {missing[:8]}")
        num = den = 0.0
        worst = (0.0, "")
        for name in sorted(set(ref) & set(other)):
            a, b = ref[name], other[name]
            if a.shape != b.shape:
                print(f"{path}: {name}: shapes {a.shape} and {b.shape}")
                continue
            d = np.linalg.norm(a - b)
            r = np.linalg.norm(a)
            num += d * d
            den += r * r
            rel = d / max(r, 1e-30)
            worst = max(worst, (rel, name))
        print(f"{path}: overall relative error {np.sqrt(num / den):.3e}, "
              f"worst tensor {worst[1]} {worst[0]:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("model")
    t.add_argument("text")
    t.add_argument("out")
    r = sub.add_parser("torch")
    r.add_argument("model")
    r.add_argument("tokens")
    r.add_argument("--rows", type=int, default=16)
    r.add_argument("--length", type=int, default=1024)
    r.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--output")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("others", nargs="+")
    args = parser.parse_args()
    {"tokens": tokens, "torch": run_torch, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
