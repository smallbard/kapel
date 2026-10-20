"""Reference renderings of chat templates by `transformers`.

For each model of data/parity with a chat template, writes chat_cases.json:
conversations and the text `apply_chat_template` renders for them.

Usage: python tools/parity/chat_reference.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And times 3?"},
    ],
    [{"role": "user", "content": "Multi\nline  text with émojis 🙂 and \"quotes\"."}],
]


# Chat models whose tokenizer alone is fetched, for their templates.
TOKENIZERS = {
    "smollm2-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "llama3-instruct": "unsloth/Llama-3.2-1B-Instruct",
    "gemma2-it": "unsloth/gemma-2-2b-it",
    "mistral-instruct": "unsloth/mistral-7b-instruct-v0.3",
}


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "parity"
    for name, repo in TOKENIZERS.items():
        target = root / name
        if not (target / "tokenizer_config.json").exists():
            try:
                AutoTokenizer.from_pretrained(repo).save_pretrained(target)
            except Exception as error:
                print(f"{name}: {error}")
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            tokenizer = AutoTokenizer.from_pretrained(directory)
        except Exception:
            continue
        if not tokenizer.chat_template:
            continue
        cases = []
        for messages in CONVERSATIONS:
            for generation in (False, True):
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
                except Exception as error:
                    text = None
                    print(f"  {directory.name}: {error}")
                cases.append({"messages": messages, "add_generation_prompt": generation, "text": text})
        (directory / "chat_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
        print(f"{directory.name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
