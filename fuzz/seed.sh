#!/bin/sh
# Seed corpora from the reference files of data/parity (see tools/parity).
set -e
cd "$(dirname "$0")"
data=../data/parity
mkdir -p corpus/safetensors corpus/gguf corpus/tokenizer_json corpus/tokenizer_encode corpus/chat_template corpus/model_config
for d in "$data"/*/; do
  n=$(basename "$d")
  [ -f "$d/config.json" ] && cp "$d/config.json" "corpus/model_config/$n.json"
  [ -f "$d/chat_template.jinja" ] && cp "$d/chat_template.jinja" "corpus/chat_template/$n.jinja"
  [ -f "$d/reference.safetensors" ] && cp "$d/reference.safetensors" "corpus/safetensors/$n.safetensors"
done
cp "$data/gpt2/tokenizer.json" corpus/tokenizer_json/gpt2.json 2>/dev/null || true
cp "$data/gguf/quants.gguf" corpus/gguf/ 2>/dev/null || true
printf 'Hello world' > corpus/tokenizer_encode/hello
# Inputs that once crashed a target (see README.md).
for t in regressions/*/; do
  n=$(basename "$t")
  mkdir -p "corpus/$n"
  cp "$t"* "corpus/$n/"
done
