#!/usr/bin/env bash
# The measures Kapel tracks: runs the example `bench` of kapel-cli on a fixed
# list of setups, and the recipes of training for a few steps each
# (tools/bench/recipes.py), and writes one JSON report with all of them.
#
#     tools/bench/run.sh cpu|cuda|train|all OUTPUT.json
#
# The models come from the Hub (into $KAPEL_CACHE); the text of the
# perplexities is wikitext-2's test set, downloaded once into
# $KAPEL_BENCH_DATA (default ~/.cache/kapel/bench). Compare the report with a
# baseline with tools/bench/compare.py.
set -euo pipefail

suite=${1:?usage: run.sh cpu|cuda|train|all OUTPUT.json}
output=${2:?usage: run.sh cpu|cuda|train|all OUTPUT.json}
root=$(cd "$(dirname "$0")/../.." && pwd)
data=${KAPEL_BENCH_DATA:-$HOME/.cache/kapel/bench}
text=$data/wikitext-2-raw/wiki.test.raw

if [ ! -f "$text" ]; then
    mkdir -p "$data"
    curl -sSL -o "$data/wikitext-2-raw-v1.zip" \
        https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    (cd "$data" && unzip -q -o wikitext-2-raw-v1.zip)
fi

features=()
if [ "$suite" != cpu ]; then features=(--features cuda); fi
cargo build --release -p kapel-cli "${features[@]}" --bin kapel --example bench --manifest-path "$root/Cargo.toml"
bench=$root/target/release/examples/bench

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
run() {
    echo "== $1" >&2
    "$bench" "$2" --name "$1" --text "$text" --output "$work/$1.json" "${@:3}"
}

if [ "$suite" = cpu ] || [ "$suite" = all ]; then
    run qwen2.5-0.5b-cpu-f32 Qwen/Qwen2.5-0.5B-Instruct --device cpu
    run qwen2.5-0.5b-cpu-q4_0 Qwen/Qwen2.5-0.5B-Instruct --device cpu --quantize q4_0 --quantize-output
    run llama3.2-1b-cpu-q4_0 unsloth/Llama-3.2-1B-Instruct --device cpu --quantize q4_0 --quantize-output
fi
if [ "$suite" = cuda ] || [ "$suite" = all ]; then
    run qwen2.5-0.5b-cuda-bf16 Qwen/Qwen2.5-0.5B-Instruct --device cuda
    run llama3.2-3b-cuda-bf16 unsloth/Llama-3.2-3B-Instruct --device cuda --batch 64
    run llama3.2-3b-cuda-q4_0 unsloth/Llama-3.2-3B-Instruct --device cuda --quantize q4_0 --quantize-output
fi

if [ "$suite" = train ] || [ "$suite" = all ]; then
    python3 "$root/tools/bench/recipes.py" "$work" --kapel "$root/target/release/kapel"
fi

python3 - "$work" "$output" <<'PY'
import json, pathlib, sys
reports = [json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))]
pathlib.Path(sys.argv[2]).write_text(json.dumps(reports, indent=2) + "\n")
print(f"{len(reports)} setups written to {sys.argv[2]}", file=sys.stderr)
PY
