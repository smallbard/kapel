"""DPO with LoRA by TRL, the reference of the recipe dpo-ultrafeedback.

    python tools/train/trl_dpo.py MODEL TRAIN.jsonl TEST.jsonl OUT [--steps N] [--seed S]

TRAIN and TEST hold pairs (`{"chosen": [...], "rejected": [...]}`), as
tools/train/ultrafeedback.py writes them. The settings are those of Kapel's
recipe (recipes/dpo-ultrafeedback/train.toml): LoRA of rank 16, alpha 32, no
dropout, on every projection, the reference the model without its adapters;
β = 0.1, the sigmoid loss; AdamW at 5e-5 with betas (0.9, 0.999), no weight
decay, a linear decay without warmup, gradients clipped to 1; 8 pairs a
step, at most 1,024 tokens each, one epoch; bf16. The pairs of TEST are
measured every 100 steps. Writes the adapter into OUT, and each step's and
evaluation's numbers into OUT/log.jsonl, under Kapel's names.

Needs: torch, transformers, peft, trl, datasets.
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

NAMES = {
    "loss": "loss",
    "rewards/accuracies": "accuracy",
    "rewards/chosen": "chosen_reward",
    "rewards/rejected": "rejected_reward",
    "rewards/margins": "margin",
    "eval_loss": "val_loss",
    "eval_rewards/accuracies": "val_accuracy",
    "eval_rewards/chosen": "val_chosen_reward",
    "eval_rewards/rejected": "val_rejected_reward",
    "eval_rewards/margins": "val_margin",
}


class Log(TrainerCallback):
    def __init__(self, path):
        self.file = open(path, "w")
        self.start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        line = {NAMES[k]: v for k, v in (logs or {}).items() if k in NAMES}
        if line:
            line = {"step": state.global_step, **line, "seconds": time.time() - self.start}
            self.file.write(json.dumps(line) + "\n")
            self.file.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("train")
    p.add_argument("test")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    train = load_dataset("json", data_files=a.train, split="train")
    test = load_dataset("json", data_files=a.test, split="train")
    config = DPOConfig(
        output_dir=a.out,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=a.steps,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        beta=0.1,
        max_length=1024,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to=[],
        seed=a.seed,
        dataloader_drop_last=True,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=config, train_dataset=train, eval_dataset=test,
                         processing_class=tok, peft_config=lora,
                         callbacks=[Log(f"{a.out}/log.jsonl")])
    trainer.evaluate()
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
