from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluator.splits import (
    get_heldout_tasks,
    get_test_tasks,
    get_val_tasks,
    test_estimated_full_score,
    val_estimated_full_score,
)


def aggregate_records(records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(record)
    per_task = []
    task_names = _expected_tasks(split, by_task)
    for task in task_names:
        task_records = by_task[task]
        rewards = [float(record.get("reward", 0)) for record in task_records]
        runtimes = [
            float(record["runtime_sec"])
            for record in task_records
            if record.get("runtime_sec") is not None
        ]
        per_task.append(
            {
                "task": task,
                "mean": mean(rewards) if rewards else 0.0,
                "num_trials": len(task_records),
                "num_successes": sum(1 for value in rewards if value >= 1),
                "num_crashes": sum(1 for record in task_records if record.get("status") == "crash"),
                "mean_runtime": mean(runtimes) if runtimes else None,
            }
        )
    split_mean = mean(item["mean"] for item in per_task) if per_task else 0.0
    runtime_values = [item["mean_runtime"] for item in per_task if item["mean_runtime"] is not None]
    estimate = (
        val_estimated_full_score(split_mean)
        if split == "val"
        else test_estimated_full_score(split_mean)
    )
    return {
        "split": split,
        "split_mean": split_mean,
        "estimated_full_score": estimate,
        "num_trials": sum(item["num_trials"] for item in per_task),
        "num_successes": sum(item["num_successes"] for item in per_task),
        "num_crashes": sum(item["num_crashes"] for item in per_task),
        "mean_runtime": mean(runtime_values) if runtime_values else None,
        "per_task": per_task,
    }


def _expected_tasks(split: str, by_task: dict[str, list[dict[str, Any]]]) -> list[str]:
    if split == "val":
        return get_val_tasks()
    if split == "test":
        return get_test_tasks()
    if split == "heldout":
        return get_heldout_tasks()
    return sorted(by_task)


def write_summary(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "mean",
                "num_trials",
                "num_successes",
                "num_crashes",
                "mean_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["per_task"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_json", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records_json.read_text(encoding="utf-8"))
    summary = aggregate_records(records, args.split)
    write_summary(summary, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
