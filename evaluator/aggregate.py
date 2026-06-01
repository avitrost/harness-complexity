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
    get_tb2_core_tasks,
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
        input_tokens = _numeric_values(task_records, "input_tokens")
        output_tokens = _numeric_values(task_records, "output_tokens")
        cached_tokens = _numeric_values(task_records, "cached_tokens")
        total_tokens = _numeric_values(task_records, "total_tokens")
        costs = _numeric_values(task_records, "cost_usd")
        model_calls = _numeric_values(task_records, "model_calls")
        per_task.append(
            {
                "task": task,
                "mean": mean(rewards) if rewards else 0.0,
                "num_trials": len(task_records),
                "num_successes": sum(1 for value in rewards if value >= 1),
                "num_crashes": sum(1 for record in task_records if record.get("status") == "crash"),
                "mean_runtime": mean(runtimes) if runtimes else None,
                "total_input_tokens": _sum_or_none(input_tokens),
                "total_output_tokens": _sum_or_none(output_tokens),
                "total_cached_tokens": _sum_or_none(cached_tokens),
                "total_tokens": _sum_or_none(total_tokens),
                "mean_total_tokens": mean(total_tokens) if total_tokens else None,
                "total_cost_usd": _sum_or_none(costs),
                "mean_cost_usd": mean(costs) if costs else None,
                "model_calls": int(sum(model_calls)) if model_calls else None,
            }
        )
    split_mean = mean(item["mean"] for item in per_task) if per_task else 0.0
    runtime_values = [item["mean_runtime"] for item in per_task if item["mean_runtime"] is not None]
    all_input_tokens = _numeric_values(records, "input_tokens")
    all_output_tokens = _numeric_values(records, "output_tokens")
    all_cached_tokens = _numeric_values(records, "cached_tokens")
    all_total_tokens = _numeric_values(records, "total_tokens")
    all_costs = _numeric_values(records, "cost_usd")
    all_model_calls = _numeric_values(records, "model_calls")
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
        "total_input_tokens": _sum_or_none(all_input_tokens),
        "total_output_tokens": _sum_or_none(all_output_tokens),
        "total_cached_tokens": _sum_or_none(all_cached_tokens),
        "total_tokens": _sum_or_none(all_total_tokens),
        "mean_total_tokens": mean(all_total_tokens) if all_total_tokens else None,
        "total_cost_usd": _sum_or_none(all_costs),
        "mean_cost_usd": mean(all_costs) if all_costs else None,
        "model_calls": int(sum(all_model_calls)) if all_model_calls else None,
        "per_task": per_task,
    }


def _expected_tasks(split: str, by_task: dict[str, list[dict[str, Any]]]) -> list[str]:
    if split == "val":
        return get_val_tasks()
    if split == "test":
        return get_test_tasks()
    if split == "heldout":
        return get_heldout_tasks()
    if split == "tb2-core":
        return get_tb2_core_tasks()
    return sorted(by_task)


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _sum_or_none(values: Any) -> float | None:
    items = list(values)
    return sum(items) if items else None


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
                "total_input_tokens",
                "total_output_tokens",
                "total_cached_tokens",
                "total_tokens",
                "mean_total_tokens",
                "total_cost_usd",
                "mean_cost_usd",
                "model_calls",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["per_task"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_json", type=Path)
    parser.add_argument(
        "--split", choices=("val", "test", "heldout", "tblite", "tb2-core"), required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records_json.read_text(encoding="utf-8"))
    summary = aggregate_records(records, args.split)
    write_summary(summary, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
