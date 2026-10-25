"""A run of Kapel in Weights & Biases.

    python tools/train/wandb_log.py RUN_DIR [--project NAME] [--name NAME] [--online]

Reads the run's `log.jsonl` (one line a step, and the evaluations) and
logs each line at its step; the configuration of the run's last checkpoint
(`train.toml`, or that of `step-*/config.toml`) becomes the run's config.
Offline by default: the run is written under `wandb/` and `wandb sync`
sends it; with `--online`, it is sent as it is read, to the account that
`wandb login` set up.

TensorBoard's files (`log.tensorboard = true`) are another way in:
`wandb sync --sync-tensorboard RUN_DIR/tensorboard`.

Needs: wandb.
"""

import argparse
import json
import os
import tomllib
from pathlib import Path


def config_of(run):
    candidates = [run / "train.toml"] + sorted(run.glob("step-*/config.toml"))[-1:]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--project", default="kapel")
    parser.add_argument("--name")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if not args.online:
        os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    run = wandb.init(project=args.project, name=args.name or args.run.resolve().name,
                     config=config_of(args.run))
    lines = 0
    with open(args.run / "log.jsonl") as f:
        for line in f:
            record = json.loads(line)
            step = record.pop("step")
            values = {k: v for k, v in record.items() if isinstance(v, (int, float))}
            run.log(values, step=step)
            lines += 1
    run.finish()
    print(f"{lines} lines of {args.run / 'log.jsonl'} logged")


if __name__ == "__main__":
    main()
