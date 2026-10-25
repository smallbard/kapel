"""Kapel's LoRA adapters against PEFT's, both ways.

    python tools/train/peft_parity.py MODEL WORKDIR

MODEL is a model directory (a snapshot of the Hugging Face cache). The script

1. runs Kapel's `lora write` example: adapters of rank 4 on every projection and
   on the embeddings, all factors drawn, written as PEFT writes them; loads them
   with `PeftModel.from_pretrained` and compares the logits with Kapel's;
2. draws an adapter with PEFT (`init_lora_weights=False`: no factor at zero),
   saves it, and compares the logits Kapel computes with it (`lora read`) with
   PEFT's.

Both in float32 on the processor, on the ids of the example. It prints the
largest difference of each, and fails above 1e-4.
Needs: torch, transformers, peft.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

IDS = [3, 17, 4, 4, 22, 0, 9, 13]


def kapel(*args):
    cmd = ["cargo", "run", "--release", "-q", "-p", "kapel-train", "--example", "lora", "--", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def peft_logits(model):
    with torch.no_grad():
        return model(torch.tensor([IDS])).logits[0].double().flatten()


def main():
    model_dir, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # Kapel's adapter, read by PEFT.
    out = work / "kapel"
    kapel("write", model_dir, str(out))
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    peft = PeftModel.from_pretrained(base, str(out))
    expected = torch.tensor(json.load(open(out / "kapel_logits.json")), dtype=torch.float64)
    a = (peft_logits(peft) - expected).abs().max().item()
    print(f"Kapel's adapter in PEFT: largest difference {a:.2e}")

    # PEFT's adapter, read by Kapel.
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    gpt2 = base.config.model_type == "gpt2"
    embedding = "wte" if gpt2 else "embed_tokens"
    linear = ["c_attn", "c_proj", "c_fc"] if gpt2 else [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(r=4, lora_alpha=8, target_modules=linear + [embedding],
                        init_lora_weights=False, fan_in_fan_out=gpt2)
    model = get_peft_model(base, config)
    expected = peft_logits(model)
    theirs = work / "peft"
    model.save_pretrained(str(theirs))
    got = torch.tensor(json.loads(kapel("read", model_dir, str(theirs))), dtype=torch.float64)
    b = (got - expected).abs().max().item()
    print(f"PEFT's adapter in Kapel: largest difference {b:.2e}")
    sys.exit(0 if max(a, b) < 1e-4 else 1)


if __name__ == "__main__":
    main()
