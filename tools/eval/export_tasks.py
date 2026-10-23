"""The requests of tasks of lm-evaluation-harness, for `kapel eval` to replay.

lm-evaluation-harness turns each document of a task into requests to a model:
the log-likelihood of continuations after a context (a question and its
answers), or of a whole text (perplexity). This script asks lm-evaluation-harness
for these requests, without a model, and writes them one document per line of
JSON, with what scores the answers: the right choice, found by the task's own
rule, and the lengths that normalize them; the words and bytes of a text.

    {"task": "hellaswag", "group": "hellaswag", "doc": 0, "type": "choice",
     "requests": [[context, continuation], ...], "gold": [2], "lengths": [...]}
    {"task": "wikitext", "group": "wikitext", "doc": 0, "type": "rolling",
     "text": "...", "words": 3247, "bytes": 18190}

Usage:
    pip install "lm-eval[hf]"
    python tools/eval/export_tasks.py --tasks wikitext,hellaswag,arc_easy,arc_challenge,mmlu \\
        --output data/eval/tasks.jsonl [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

from lm_eval.tasks import TaskManager, get_task_dict


def flatten(tasks, group=None):
    """(group, name, task) for every task of a nested dictionary of groups."""
    for key, value in tasks.items():
        name = getattr(key, "group_name", None) or getattr(key, "group", None) or str(key)
        if isinstance(value, dict):
            yield from flatten(value, group or name)
        else:
            yield group or name, name, value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", required=True, help="tasks or groups, separated by commas")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="the first N documents of each task")
    args = parser.parse_args()

    tasks = get_task_dict(args.tasks.split(","), TaskManager())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for group, name, task in flatten(tasks):
            task.build_all_requests(limit=args.limit, rank=0, world_size=1)
            by_doc = {}
            for instance in task.instances:
                by_doc.setdefault(instance.doc_id, []).append(instance)
            for doc_id, instances in sorted(by_doc.items()):
                instances.sort(key=lambda i: i.idx)
                doc = instances[0].doc
                kind = instances[0].request_type
                line = {"task": name, "group": group, "doc": doc_id}
                if kind == "loglikelihood":
                    n = len(instances)
                    # The right choices, by the task's own rule: those whose
                    # log-likelihood alone highest gives an accuracy of 1.
                    gold = []
                    for i in range(n):
                        onehot = [(1.0 if j == i else 0.0, False) for j in range(n)]
                        if task.process_results(doc, onehot).get("acc") == 1.0:
                            gold.append(i)
                    choices = task.doc_to_choice(doc)
                    line.update(
                        type="choice",
                        requests=[list(i.arguments) for i in instances],
                        gold=gold,
                        lengths=[len(c) for c in choices],
                    )
                elif kind == "loglikelihood_rolling":
                    scores = task.process_results(doc, [0.0])
                    words = scores.get("word_perplexity", (0, 0))[1]
                    size = scores.get("byte_perplexity", (0, 0))[1]
                    line.update(type="rolling", text=instances[0].arguments[0], words=words, bytes=size)
                else:
                    sys.exit(f"{name}: requests of type {kind} are not replayed by kapel eval")
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
                counts[name] = counts.get(name, 0) + 1
    print(f"wrote {sum(counts.values())} documents of {len(counts)} tasks to {args.output}")


if __name__ == "__main__":
    main()
