"""Compares measures of tools/bench/run.sh with a baseline, setup by setup.

    python3 tools/bench/compare.py BASELINE.json CURRENT.json [--tolerance 0.10]

A speed more than `--tolerance` below the baseline's is a regression; a
perplexity that moved by more than 0.01 % means the numbers changed, which a
change must explain (and then update the baseline). A speed more than the
tolerance above the baseline's is reported, to update the baseline with.
Prints a Markdown table (also appended to $GITHUB_STEP_SUMMARY when set), and
exits with 1 on a regression or a changed perplexity.
"""

import argparse
import json
import os
import sys

SPEEDS = {
    "prefill_tokens_per_s": "prefill tok/s",
    "decode_tokens_per_s": "decode tok/s",
    "batch_tokens_per_s": "batch tok/s",
}
PERPLEXITY_TOLERANCE = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    def load(path):
        with open(path, encoding="utf-8") as f:
            return {r["name"]: r for r in json.load(f)}

    baseline, current = load(args.baseline), load(args.current)
    rows, failures, faster = [], [], []
    for name, report in current.items():
        base = baseline.get(name)
        if base is None:
            rows.append(f"| {name} | (no baseline) | | | |")
            continue
        for key, label in [*SPEEDS.items(), ("perplexity", "perplexity")]:
            now, then = report["results"].get(key), base["results"].get(key)
            if now is None or then is None:
                continue
            change = now / then - 1
            if key == "perplexity":
                verdict = "changed" if abs(change) > PERPLEXITY_TOLERANCE else "same"
                if verdict == "changed":
                    failures.append(f"{name}: perplexity {then:.4f} -> {now:.4f}")
            elif change < -args.tolerance:
                verdict = "slower"
                failures.append(f"{name}: {label} {then:.1f} -> {now:.1f} ({change:+.1%})")
            elif change > args.tolerance:
                verdict = "faster"
                faster.append(f"{name}: {label} {then:.1f} -> {now:.1f} ({change:+.1%})")
            else:
                verdict = "ok"
            rows.append(f"| {name} | {label} | {then:.4g} | {now:.4g} | {change:+.1%} {verdict} |")
    for name in baseline.keys() - current.keys():
        rows.append(f"| {name} | (not measured) | | | |")

    table = "\n".join(
        ["| Setup | Measure | Baseline | Now | Change |", "| --- | --- | --- | --- | --- |", *rows]
    )
    notes = []
    if failures:
        notes.append("**Regressions or changed numbers:**\n" + "\n".join(f"- {f}" for f in failures))
    if faster:
        notes.append("**Faster than the baseline** (update it):\n" + "\n".join(f"- {f}" for f in faster))
    report = "\n\n".join([table, *notes]) + "\n"
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Measures\n\n" + report)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
