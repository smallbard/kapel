"""The recipes' speeds, memories and losses, measured on a few steps each.

    python3 tools/bench/recipes.py OUTPUT_DIR [--kapel PATH] [--work DIR] [--only NAME]

Each recipe of `recipes/` runs for a few steps (`kapel train RECIPE --steps N
--dir …`), in a copy of the repository's layout under WORK (default
$KAPEL_BENCH_DATA/recipes, or ~/.cache/kapel/bench/recipes) where its data
are prepared once, by the commands of its header, and kept. Each writes
OUTPUT_DIR/train-NAME.json, a setup of tools/bench/run.sh's report: the mean
speed of the steps after the first (which compiles the kernels), the peak
memory, and the loss of the last step, which the arithmetic alone fixes.
tools/bench/compare.py compares them with a baseline, as the other setups.

Needs: python3 with huggingface_hub, pyarrow and transformers (for the data).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Each recipe: the steps measured, and the commands that prepare its data
# (run from WORK, `{kapel}` the binary), with a file each makes.
RECIPES = [
    {
        "name": "gpt2-124m",
        "steps": 20,
        "prepare": [
            ("data/fineweb/fineweb_train_000001.bin",
             ["{python}", "{root}/tools/train/fineweb_gpt2.py", "data/fineweb", "--train", "1"]),
        ],
    },
    {
        "name": "llama-1b",
        "steps": 4,
        "prepare": [
            ("data/fineweb-parquet/sample/10BT/000_00000.parquet",
             ["hf", "download", "HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet",
              "--repo-type", "dataset", "--local-dir", "data/fineweb-parquet"]),
            ("data/fineweb-llama3/dataset.json",
             ["{kapel}", "prepare", "data/fineweb-parquet/sample/10BT/000_00000.parquet",
              "--tokenizer", "unsloth/Llama-3.2-1B", "--output", "data/fineweb-llama3",
              "--val-every", "1000", "--limit", "100000", "--quiet"]),
        ],
    },
    {
        "name": "sft-gsm8k",
        "steps": 30,
        "prepare": [
            ("recipes/sft-gsm8k/gsm8k/train.jsonl",
             ["{python}", "{root}/tools/train/gsm8k.py", "recipes/sft-gsm8k/gsm8k"]),
            ("recipes/sft-gsm8k/data/dataset.json",
             ["{kapel}", "prepare", "recipes/sft-gsm8k/gsm8k/train.jsonl", "--chat",
              "--tokenizer", "Qwen/Qwen3-0.6B", "--output", "recipes/sft-gsm8k/data", "--quiet"]),
        ],
    },
    {
        "name": "sft-lora-gsm8k",
        "steps": 30,
        "prepare": [
            ("recipes/sft-lora-gsm8k/gsm8k/train.jsonl",
             ["{python}", "{root}/tools/train/gsm8k.py", "recipes/sft-lora-gsm8k/gsm8k"]),
            ("recipes/sft-lora-gsm8k/data/dataset.json",
             ["{kapel}", "prepare", "recipes/sft-lora-gsm8k/gsm8k/train.jsonl", "--chat",
              "--tokenizer", "Qwen/Qwen3-0.6B", "--output", "recipes/sft-lora-gsm8k/data",
              "--quiet"]),
        ],
    },
    {
        "name": "dpo-ultrafeedback",
        "steps": 20,
        "prepare": [
            ("recipes/dpo-ultrafeedback/ultrafeedback/train.jsonl",
             ["{python}", "{root}/tools/train/ultrafeedback.py",
              "recipes/dpo-ultrafeedback/ultrafeedback"]),
        ],
    },
    {
        "name": "grpo-gsm8k",
        "steps": 3,
        "prepare": [
            ("recipes/grpo-gsm8k/gsm8k/prompts.jsonl",
             ["{python}", "{root}/tools/train/gsm8k.py", "recipes/grpo-gsm8k/gsm8k"]),
        ],
    },
]


def measure(recipe, kapel, work, out):
    name = recipe["name"]
    values = {"python": sys.executable, "root": str(ROOT), "kapel": kapel}
    for made, command in recipe["prepare"]:
        if not (work / made).exists():
            command = [c.format(**values) for c in command]
            print(f"== {name}: {' '.join(command)}", file=sys.stderr)
            subprocess.run(command, cwd=work, check=True)
    # The recipe as the repository has it now, beside its data.
    place = work / "recipes" / name
    place.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "recipes" / name).iterdir():
        if f.is_file() and f.suffix in (".toml", ".json"):
            shutil.copy(f, place / f.name)
    run = work / "runs" / name
    shutil.rmtree(run, ignore_errors=True)
    print(f"== {name}: {recipe['steps']} steps", file=sys.stderr)
    subprocess.run([kapel, "train", str(place / "train.toml"), "--steps", str(recipe["steps"]),
                    "--dir", str(run), "--quiet"], check=True)
    steps = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    steps = [s for s in steps if "loss" in s]
    later = [s for s in steps if s["step"] > 1]
    report = {
        "name": f"train-{name}",
        "model": f"recipes/{name}/train.toml",
        "device": "cuda",
        "steps": recipe["steps"],
        "results": {
            "train_tokens_per_s": sum(s["tokens_per_s"] for s in later) / len(later),
            "peak_gib": max(s["peak_gib"] for s in steps),
            "loss": steps[-1]["loss"],
        },
    }
    (out / f"train-{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.rmtree(run, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("output_dir")
    p.add_argument("--kapel", default=str(ROOT / "target/release/kapel"))
    p.add_argument("--work", default=os.path.join(
        os.environ.get("KAPEL_BENCH_DATA", os.path.expanduser("~/.cache/kapel/bench")), "recipes"))
    p.add_argument("--only", action="append")
    a = p.parse_args()
    work, out = Path(a.work), Path(a.output_dir)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for recipe in RECIPES:
        if not a.only or recipe["name"] in a.only:
            measure(recipe, a.kapel, work, out)


if __name__ == "__main__":
    main()
