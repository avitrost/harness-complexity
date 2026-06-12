from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bootstrap_ci import bootstrap_ci  # noqa: E402

DEFAULT_AGGREGATE_DIR = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/aggregate_openai_deepseek_20260609"
)
DEFAULT_LOW10_ROOT = Path(
    "/home/trost/harness-complexity/final_test/tb2_core_model_sweep_low10_global60_slurm_20260603_224026"
)
DEFAULT_MEDIUM10_ROOT = Path(
    "/home/trost/harness-complexity/final_test/tb2_core_model_sweep_medium10_global45_20260601_223918"
)

MODELS = {
    "gpt_5_4_mini": "gpt-5.4-mini",
    "gpt_5_4": "gpt-5.4",
    "gpt_5_5": "gpt-5.5",
}
TASKS = (
    "bn-fit-modify",
    "circuit-fibsqrt",
    "polyglot-c-py",
    "sparql-university",
    "mteb-retrieve",
    "cobol-modernization",
    "password-recovery",
    "model-extraction-relu-logits",
    "large-scale-text-editing",
)
PRIOR_CANDIDATES = (
    "seed_codex_1000",
    "seed_codex_1300",
    "seed_codex_400",
    "seed_codex_700",
    "seed_codex_compressed",
    "seed_codex_full",
    "seed_minimal_agent",
    "seed_terminus_2_compressed",
)
HARNESS_ALIASES = {
    "seed_codex_1000": "c1000",
    "seed_codex_1300": "c1300",
    "seed_codex_400": "c400",
    "seed_codex_700": "c700",
    "seed_codex_compressed": "c_comp",
    "seed_codex_full": "c_full",
    "seed_mini_swe_agent_barebones": "mini_bare",
    "seed_mini_swe_agent_barebones_v2_persistent": "bbv2_bash_persistent",
    "seed_mini_swe_agent_barebones_v2_rich_terminal": "bbv2_rich_terminal",
    "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples": "bbv2_rich_no_examples",
    "seed_mini_swe_agent_v2": "mini_v2",
    "seed_minimal_agent": "minimal",
    "seed_terminus_2_compressed": "term2_comp",
}
HARNESS_ALIAS_ORDER = (
    "c1000",
    "c1300",
    "c400",
    "c700",
    "c_comp",
    "c_full",
    "mini_bare",
    "bbv2_bash_persistent",
    "bbv2_rich_terminal",
    "bbv2_rich_no_examples",
    "mini_v2",
    "minimal",
    "term2_comp",
)
CORRUPTED_FAILURE_CLASSES = {
    "rate_limit",
    "provider_transport",
    "infra",
    "harbor_cache",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--low10-root", type=Path, default=DEFAULT_LOW10_ROOT)
    parser.add_argument("--medium10-root", type=Path, default=DEFAULT_MEDIUM10_ROOT)
    parser.add_argument("--ci-samples", type=int, default=10000)
    parser.add_argument("--suffix", default="with_prior_openai10")
    parser.add_argument("--allow-outside-slurm", action="store_true")
    args = parser.parse_args()

    if not args.allow_outside_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to run merge outside Slurm. Submit with sbatch or srun.")

    current_candidate = read_csv(args.aggregate_dir / "aggregate_by_candidate.csv")
    current_task = read_csv(args.aggregate_dir / "aggregate_by_task.csv")
    print(
        f"Loaded current aggregate: {len(current_candidate)} candidate rows, "
        f"{len(current_task)} task rows",
        flush=True,
    )
    prior_attempts, prior_meta = prior_attempt_rows(args)
    print(f"Loaded prior OpenAI10 attempts: {len(prior_attempts)} rows", flush=True)
    prior_candidate = aggregate(
        prior_attempts,
        ("provider", "config_id", "model", "effort", "candidate"),
        args.ci_samples,
    )
    print(f"Aggregated prior candidate rows: {len(prior_candidate)}", flush=True)
    prior_task = aggregate(
        prior_attempts,
        ("provider", "config_id", "model", "effort", "candidate", "task"),
        args.ci_samples,
    )
    print(f"Aggregated prior task rows: {len(prior_task)}", flush=True)

    merged_candidate, skipped_candidate = merge_rows(
        current_candidate,
        prior_candidate,
        ("provider", "config_id", "model", "effort", "candidate"),
    )
    merged_task, skipped_task = merge_rows(
        current_task,
        prior_task,
        ("provider", "config_id", "model", "effort", "candidate", "task"),
    )

    candidate_path = args.aggregate_dir / f"aggregate_by_candidate_{args.suffix}.csv"
    task_path = args.aggregate_dir / f"aggregate_by_task_{args.suffix}.csv"
    scores_path = args.aggregate_dir / f"scores_by_harness_{args.suffix}.csv"
    config_path = args.aggregate_dir / f"config_summary_{args.suffix}.csv"
    summary_path = args.aggregate_dir / f"summary_{args.suffix}.json"

    write_csv(candidate_path, merged_candidate)
    write_csv(task_path, merged_task)
    write_csv(
        scores_path,
        scores_by_harness(merged_candidate),
        ["provider", "model", "effort", *HARNESS_ALIAS_ORDER],
    )
    write_csv(
        config_path,
        config_summary(merged_candidate),
        [
            "provider",
            "model",
            "effort",
            "N",
            "mean_score",
            "successes",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ],
    )

    summary = {
        "created_files": {
            "aggregate_by_candidate": str(candidate_path),
            "aggregate_by_task": str(task_path),
            "scores_by_harness": str(scores_path),
            "config_summary": str(config_path),
        },
        "current_candidate_rows": len(current_candidate),
        "current_task_rows": len(current_task),
        "prior_openai10_attempt_rows": len(prior_attempts),
        "prior_openai10_candidate_rows": len(prior_candidate),
        "prior_openai10_task_rows": len(prior_task),
        "merged_candidate_rows": len(merged_candidate),
        "merged_task_rows": len(merged_task),
        "skipped_existing_candidate_rows": len(skipped_candidate),
        "skipped_existing_task_rows": len(skipped_task),
        "prior_meta": prior_meta,
        "notes": [
            "Existing aggregate files were left untouched.",
            "OpenAI low/medium non-mini harness cells are filled from prior 10-attempt roots.",
            "Current mini_bare/mini_v2 low/medium cells are retained from the provider matrix aggregate.",
            "Prior token columns are sourced from older harness model_accounting fields.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    if preferred:
        fields = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    else:
        fields = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prior_attempt_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leaf_histogram: dict[str, int] = defaultdict(int)
    missing: list[str] = []
    for effort, root in (("low", args.low10_root), ("medium", args.medium10_root)):
        for model_dir, model in MODELS.items():
            config_id = f"openai_{slug(model)}_{effort}"
            for candidate in PRIOR_CANDIDATES:
                for task in TASKS:
                    for attempt in range(1, 11):
                        attempt_dir = root / model_dir / candidate / task / f"attempt_{attempt:02d}"
                        run_dir, record, valid_count = newest_valid_record(attempt_dir, task)
                        leaf_histogram[str(valid_count)] += 1
                        if record is None:
                            missing.append(str(attempt_dir))
                            rows.append(
                                attempt_row(
                                    effort=effort,
                                    model=model,
                                    config_id=config_id,
                                    candidate=candidate,
                                    task=task,
                                    attempt=attempt,
                                    source_run_id=root.name,
                                    selected_run_dir=None,
                                )
                            )
                            continue
                        rows.append(
                            attempt_row(
                                effort=effort,
                                model=model,
                                config_id=config_id,
                                candidate=candidate,
                                task=task,
                                attempt=attempt,
                                source_run_id=root.name,
                                selected_run_dir=run_dir,
                                record=record,
                            )
                        )
    meta = {
        "prior_sources": {
            "low10": str(args.low10_root),
            "medium10": str(args.medium10_root),
        },
        "selection_rule": (
            "For each attempt_XX, selected the newest timestamp leaf containing a parseable "
            "task record; attempts 01-10 only."
        ),
        "valid_leaf_count_histogram": dict(
            sorted(leaf_histogram.items(), key=lambda item: int(item[0]))
        ),
        "missing_attempt_count": len(missing),
        "missing_examples": missing[:20],
    }
    return rows, meta


def newest_valid_record(
    attempt_dir: Path, task: str
) -> tuple[Path | None, dict[str, Any] | None, int]:
    try:
        run_dirs = sorted((path for path in attempt_dir.iterdir() if path.is_dir()), reverse=True)
    except OSError:
        return None, None, 0
    valid_count = 0
    selected_run_dir: Path | None = None
    selected_record: dict[str, Any] | None = None
    for run_dir in run_dirs:
        record = record_from_run_dir(run_dir, task)
        if not record:
            continue
        valid_count += 1
        if selected_record is None:
            selected_run_dir = run_dir
            selected_record = record
    return selected_run_dir, selected_record, valid_count


def record_from_run_dir(run_dir: Path, task: str) -> dict[str, Any] | None:
    try:
        trial_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
    except OSError:
        return None
    for trial_dir in sorted(trial_dirs):
        result = read_json(trial_dir / "result.json")
        if not isinstance(result, dict):
            continue
        task_name = result.get("task") or result.get("task_name") or result.get("task_id")
        if str(task_name) != task:
            continue
        reward = extract_reward(result)
        status = result.get("status")
        if not status:
            if result.get("exception_info"):
                status = "crash"
            elif reward == 1:
                status = "success"
            elif reward == 0:
                status = "failure"
            else:
                status = "unknown"
        accounting = token_accounting(result, trial_dir)
        record = {
            "task": task,
            "reward": 1 if float(reward or 0) >= 1 else 0,
            "status": str(status),
        }
        record.update(accounting)
        return record
    return None


def extract_reward(result: dict[str, Any]) -> float | int | None:
    reward = result.get("reward", result.get("score"))
    if reward is None and "success" in result:
        reward = 1 if result["success"] else 0
    verifier = result.get("verifier_result")
    if reward is None and isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if isinstance(rewards, dict):
            for key in ("reward", "score", "success"):
                if key in rewards:
                    return rewards[key]
            numeric = [
                value
                for value in rewards.values()
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if numeric:
                return numeric[0]
    return reward


def token_accounting(result: dict[str, Any], trial_dir: Path) -> dict[str, int]:
    accounting = result.get("agent_result")
    if not isinstance(accounting, dict):
        harness_result = read_json(trial_dir / "agent" / "harness-result.json")
        if isinstance(harness_result, dict):
            accounting = harness_result.get("model_accounting")
    if not isinstance(accounting, dict):
        return {}
    mapped = {
        "input_tokens": accounting.get("input_tokens", accounting.get("n_input_tokens")),
        "output_tokens": accounting.get("output_tokens", accounting.get("n_output_tokens")),
        "cached_tokens": accounting.get("cached_tokens", accounting.get("n_cache_tokens")),
        "total_tokens": accounting.get("total_tokens", accounting.get("n_total_tokens")),
    }
    input_tokens = optional_int(mapped["input_tokens"])
    output_tokens = optional_int(mapped["output_tokens"])
    total_tokens = optional_int(mapped["total_tokens"])
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    mapped["input_tokens"] = input_tokens
    mapped["output_tokens"] = output_tokens
    mapped["cached_tokens"] = optional_int(mapped["cached_tokens"])
    mapped["total_tokens"] = total_tokens
    return {key: value for key, value in mapped.items() if value is not None}


def attempt_row(
    *,
    effort: str,
    model: str,
    config_id: str,
    candidate: str,
    task: str,
    attempt: int,
    source_run_id: str,
    selected_run_dir: Path | None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(record.get("status") or "unknown") if record else "unknown"
    reward: Any = record.get("reward") if record else "N/A"
    if reward is None:
        reward = "N/A"
    failure_class = failure_class_for(status, reward)
    return {
        "provider": "openai",
        "config_id": config_id,
        "model": model,
        "effort": effort,
        "candidate": candidate,
        "task": task,
        "attempt": attempt,
        "status": status,
        "reward": reward,
        "failure_class": failure_class,
        "corrupted": "1" if failure_class in CORRUPTED_FAILURE_CLASSES else "0",
        "api_input_tokens": token(record, "input_tokens"),
        "api_output_tokens": token(record, "output_tokens"),
        "api_cached_tokens": token(record, "cached_tokens"),
        "api_total_tokens": token(record, "total_tokens"),
        "source_run_id": source_run_id,
        "selected_run_dir": str(selected_run_dir) if selected_run_dir else "N/A",
    }


def failure_class_for(status: str, reward: Any) -> str:
    if status == "success":
        return "success"
    if reward != "N/A":
        return "task_failure"
    return "unknown"


def aggregate(
    attempt_rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    ci_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)

    rows: list[dict[str, Any]] = []
    for key_values, records in sorted(grouped.items()):
        rewards = [
            value
            for value in (as_float(record.get("reward")) for record in records)
            if value is not None
        ]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        ci = bootstrap_ci(
            [
                {"task": record["task"], "reward": as_float(record.get("reward")) or 0.0}
                for record in records
                if record.get("reward") != "N/A"
            ],
            samples=ci_samples,
            seed=7,
        )
        row = dict(zip(keys, key_values, strict=True))
        row.update(
            {
                "score": score,
                "ci95_low": csv_value(ci["q025"]),
                "ci95_high": csv_value(ci["q975"]),
                "num_attempts": len(records),
                "num_successes": sum(1 for value in rewards if value >= 1),
                "num_crashes": sum(1 for record in records if record.get("status") == "crash"),
                "num_corrupted": sum(1 for record in records if record.get("corrupted") == "1"),
                "api_input_tokens": sum_csv(records, "api_input_tokens"),
                "api_output_tokens": sum_csv(records, "api_output_tokens"),
                "api_cached_tokens": sum_csv(records, "api_cached_tokens"),
                "api_total_tokens": sum_csv(records, "api_total_tokens"),
            }
        )
        for failure_class in sorted({str(record.get("failure_class")) for record in records}):
            row[f"failure_{failure_class}"] = sum(
                1 for record in records if record.get("failure_class") == failure_class
            )
        source_ids = sorted({str(record.get("source_run_id")) for record in records})
        row["source_run_id"] = source_ids[0] if len(source_ids) == 1 else ";".join(source_ids)
        rows.append({key: csv_value(value) for key, value in row.items()})
    return rows


def merge_rows(
    current: list[dict[str, str]],
    additions: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[tuple[str, ...]]]:
    seen = {tuple(str(row.get(key, "")) for key in keys) for row in current}
    merged: list[dict[str, Any]] = list(current)
    skipped: list[tuple[str, ...]] = []
    for row in additions:
        key = tuple(str(row.get(part, "")) for part in keys)
        if key in seen:
            skipped.append(key)
            continue
        merged.append(row)
        seen.add(key)
    return merged, skipped


def scores_by_harness(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        alias = HARNESS_ALIASES.get(str(row.get("candidate")))
        if not alias:
            continue
        key = (str(row.get("provider")), str(row.get("model")), str(row.get("effort")))
        out = grouped.setdefault(key, {"provider": key[0], "model": key[1], "effort": key[2]})
        score = as_float(row.get("score"))
        out[alias] = "" if score is None else f"{score:.3f}"
    rows = list(grouped.values())
    rows.sort(key=sort_key)
    for row in rows:
        for alias in HARNESS_ALIAS_ORDER:
            row.setdefault(alias, "")
    return rows


def config_summary(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(str(row.get("provider")), str(row.get("model")), str(row.get("effort")))].append(
            row
        )
    rows: list[dict[str, Any]] = []
    for key, records in sorted(grouped.items(), key=lambda item: sort_key_dict(item[0])):
        n = int(sum(as_float(record.get("num_attempts")) or 0 for record in records))
        successes = int(sum(as_float(record.get("num_successes")) or 0 for record in records))
        rows.append(
            {
                "provider": key[0],
                "model": key[1],
                "effort": key[2],
                "N": n,
                "mean_score": f"{(successes / n):.3f}" if n else "N/A",
                "successes": successes,
                "input_tokens": sum_csv(records, "api_input_tokens"),
                "output_tokens": sum_csv(records, "api_output_tokens"),
                "total_tokens": sum_csv(records, "api_total_tokens"),
            }
        )
    return rows


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return sort_key_dict((str(row["provider"]), str(row["model"]), str(row["effort"])))


def sort_key_dict(key: tuple[str, str, str]) -> tuple[int, int, int]:
    provider_order = {"openai": 0, "deepseek": 1, "anthropic": 2}
    model_order = {
        "gpt-5.4": 0,
        "gpt-5.4-mini": 1,
        "gpt-5.5": 2,
        "deepseek-v4-flash": 3,
        "deepseek-v4-pro": 4,
    }
    effort_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
    return (
        provider_order.get(key[0], 99),
        model_order.get(key[1], 99),
        effort_order.get(key[2], 99),
    )


def token(record: dict[str, Any] | None, key: str) -> int | str:
    if not record:
        return "N/A"
    value = record.get(key)
    if value in (None, "") or isinstance(value, bool):
        return "N/A"
    try:
        return int(value)
    except (TypeError, ValueError):
        return "N/A"


def optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value in (None, "", "N/A") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sum_csv(rows: list[dict[str, Any]], key: str) -> str:
    values = [value for value in (as_float(row.get(key)) for row in rows) if value is not None]
    return csv_value(sum(values) if values else None)


def csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6g}"
    return value


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace(".", "_").replace("-", "_")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
