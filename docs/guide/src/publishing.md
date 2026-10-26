# Publishing a model

A checkpoint is already a model `transformers` loads, but it holds what a run needs to go on and others do not: the weights in `f32`, the optimizer's moments (twice the weights again), the run's state. `kapel export` writes what a repository of the Hugging Face Hub holds:

```sh
kapel export recipes/sft-gsm8k/run --output qwen3-0.6b-gsm8k
hf upload my-name/qwen3-0.6b-gsm8k qwen3-0.6b-gsm8k
```

The first argument is a checkpoint (`run/step-00000467`), or a run's directory for its last one. The directory written holds:

- `model.safetensors` in bf16 (`--dtype f16` or `f32` otherwise) and `config.json`, which `transformers`, `kapel run` and the tools that read `transformers`' models load;
- the tokenizer's files and `generation_config.json` of the model the run started from (a model of the Hub, or a directory); for a model trained from a `config.json` alone, `--tokenizer MODEL` gives them;
- `README.md`, a model card: the method, the base model, the steps and tokens, the last losses and accuracies of the run, its configuration, and how to load it. The card shows file names only, not the paths of your machine.

`hf upload` comes with `huggingface_hub` (`pip install huggingface_hub`, then `hf auth login`). Check the card, and add what only you know: the data's license, what the model is for, its limits.

## Adapters

A run with LoRA writes adapters in PEFT's format, and `kapel export` writes an adapter repository: `adapter_model.safetensors` and `adapter_config.json`, which names the base model by its name on the Hub, and a card for PEFT.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(model, "my-name/qwen3-0.6b-grpo")
```

`--merge` writes the model with its adapter merged instead, a whole model anyone can load without PEFT:

```sh
kapel export recipes/grpo-gsm8k/run --merge --output qwen3-0.6b-grpo
```

A run that trained other parameters beside its adapters (`trainable = true` in `[[optimizer.groups]]`) is refused: PEFT has no place for them.

## Checking it

Before uploading, load the directory as others will:

```sh
kapel run qwen3-0.6b-gsm8k --prompt "What is 6 × 7?"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("qwen3-0.6b-gsm8k")
```

Kapel's export of a checkpoint of the recipe `scaling-smallem` gives, in `transformers`, the validation loss Kapel measured (3.4819 against 3.4820, the difference that of bf16).
