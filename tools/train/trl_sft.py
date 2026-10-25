"""Supervised fine-tuning with LoRA by TRL, the reference of step T4.

    python tools/train/trl_sft.py MODEL TRAIN.jsonl OUT [--steps N] [--seed S]

TRAIN holds prompt-completion examples (`{"prompt": [...], "completion":
[...]}`), as `kapel prepare --chat` reads them. The settings are those of
Kapel's recipe (recipes/sft-lora-gsm8k.toml): LoRA of rank 16, alpha 32, no
dropout, on every projection; AdamW at 2e-4 with betas (0.9, 0.999), no
weight decay, a linear decay without warmup, gradients clipped to 1; 16
examples a step, cut to 512 tokens, one epoch without packing, the loss on
the completions only; bf16. Writes the adapter into OUT, and the loss of
each step into OUT/log.jsonl.

Needs: torch, transformers, peft, trl, datasets.
"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class Log(TrainerCallback):
    def __init__(self, path):
        self.file = open(path, "w")
        self.start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            line = {"step": state.global_step, "loss": logs["loss"],
                    "seconds": time.time() - self.start}
            self.file.write(json.dumps(line) + "\n")
            self.file.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("train")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    data = load_dataset("json", data_files=a.train, split="train")
    config = SFTConfig(
        output_dir=a.out,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=a.steps,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        max_length=512,
        packing=False,
        completion_only_loss=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=a.seed,
        dataloader_drop_last=True,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tok,
                         peft_config=lora, callbacks=[Log(f"{a.out}/log.jsonl")])
    start = time.time()
    trainer.train()
    seconds = time.time() - start
    trainer.model.save_pretrained(a.out)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps({"seconds": seconds, "steps": trainer.state.global_step,
                      "peak_gib": peak}))


if __name__ == "__main__":
    main()
