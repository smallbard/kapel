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
