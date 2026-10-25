# Training against PyTorch

`gradients.py` checks step T0 of the training plan: Kapel's gradients and peak
memory against PyTorch's, on GPT-2 124M. It needs PyTorch with CUDA,
`transformers` (for the tokenizer) and `safetensors`.

```sh
G=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openai-community/gpt2'))")
python tools/train/gradients.py tokens $G wiki.test.raw tokens.bin
python tools/train/gradients.py torch $G tokens.bin --dtype f32 --output torch-f32.safetensors
python tools/train/gradients.py torch $G tokens.bin --dtype bf16 --output torch-bf16.safetensors
cargo run --release -p kapel-train --features cuda --example gradients -- \
    $G tokens.bin --device cuda --dtype bf16 --output kapel-bf16.safetensors
python tools/train/gradients.py compare torch-f32.safetensors torch-bf16.safetensors kapel-bf16.safetensors
```

The PyTorch side is nanoGPT's model, with GPT-2's weights, the model in f32 and
the pass under autocast to bf16. The comparison gives, for each file, the norm
of the difference of all gradients over the norm of the reference's, and the
tensor that differs most. The results are in chapter 15 of *The Kapel Book*.

# Post-training against PEFT and TRL

Step T4 of the training plan (*The Kapel Book*, chapter 16). They need
`peft` and `trl` besides PyTorch and `transformers`.

- `peft_parity.py MODEL WORKDIR`: Kapel's LoRA adapters read by PEFT, and
  PEFT's read by Kapel, the logits compared (section 16.2).
- `gsm8k.py OUT`: GSM8K as the recipes read it, examples of fine-tuning,
  prompts for GRPO and the test's prompts.
- `trl_sft.py MODEL TRAIN.jsonl OUT`: the fine-tuning of
  `recipes/sft-lora-gsm8k` by TRL, with the same settings.

```sh
python tools/train/gsm8k.py gsm8k
kapel prepare gsm8k/train.jsonl --chat --tokenizer Qwen/Qwen3-0.6B --output sft
python tools/train/trl_sft.py $QWEN gsm8k/train.jsonl trl
cargo run --release -p kapel-train --features cuda --example tasks -- \
    $QWEN gsm8k/test.jsonl --adapter trl
```

`$QWEN` is a directory of Qwen 3 0.6B, a snapshot of the Hugging Face cache.
The example `tasks` answers the test's prompts greedily, with the adapter
merged, and prints the accuracy: it measures Kapel's adapters and TRL's alike.
