from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.parse_results import parse_records  # noqa: E402
from evaluator.run_val import run_split  # noqa: E402
from evaluator.splits import get_tb2_core_tasks  # noqa: E402
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET  # noqa: E402
from scripts.bootstrap_ci import bootstrap_ci  # noqa: E402
from scripts.run_tb2_core import TB2_CORE_SPLIT  # noqa: E402

DEFAULT_OUT_ROOT = Path("/wbl-fast/usrs/trost/harness-complexity/final_test")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_PARTITION = "m7i-cpu2"


@dataclass(frozen=True)
class RetryVariant:
    name: str
    base_harness: str
    candidate_dir: Path
    retry_limit: int
    loc: int


@dataclass(frozen=True)
class AttemptSpec:
    variant: RetryVariant
    task: str
    attempt: int


VARIANTS = (
    RetryVariant("mini_v2_r0", "seed_mini_swe_agent_v2", Path("seeds/mini_swe_agent_v2"), 0, 478),
    RetryVariant("mini_v2_r1", "seed_mini_swe_agent_v2", Path("seeds/mini_swe_agent_v2"), 1, 478),
    RetryVariant("mini_v2_r2", "seed_mini_swe_agent_v2", Path("seeds/mini_swe_agent_v2"), 2, 478),
    RetryVariant(
        "barebones_v2_r0",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        0,
        478,
    ),
    RetryVariant(
        "barebones_v2_r1",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        1,
        478,
    ),
    RetryVariant(
        "barebones_v2_r2",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        2,
        478,
    ),
    RetryVariant(
        "barebones_v2_r3",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        3,
        478,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=45)
    parser.add_argument("--slurm-partition", default=DEFAULT_PARTITION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--harbor-bin")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--ci-samples", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.backend == "slurm-pyxis" and not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to run Harbor/evals outside Slurm. Submit with sbatch/salloc/srun.")

    tasks = args.tasks or get_tb2_core_tasks()
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    specs = _attempt_specs(tasks, args.trials)
    manifest = _manifest(args, tasks, specs)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _configure_environment(args)
    print(
        f"[mini-swe-retry-matrix] starting {len(specs)} attempts at concurrency {args.concurrency}",
        flush=True,
    )
    attempt_summaries = _run_attempt_pool(root, args, specs)
    attempts = _write_attempts_csv(root, specs, attempt_summaries)
    _write_aggregate_csvs(root, attempts, args.ci_samples)
    summary = {
        "run_id": args.run_id,
        "out_root": str(root),
        "attempts": len(attempts),
        "expected_attempts": len(specs),
        "returncode": 0 if all(item.get("returncode", 0) == 0 for item in attempt_summaries) else 1,
        "dry_run": args.dry_run,
        "outputs": {
            "attempts": str(root / "attempts.csv"),
            "scores_by_harness": str(root / "scores_by_harness.csv"),
            "aggregate_by_task": str(root / "aggregate_by_task.csv"),
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return int(summary["returncode"])


def _attempt_specs(tasks: list[str], trials: int) -> list[AttemptSpec]:
    return [
        AttemptSpec(variant=variant, task=task, attempt=attempt)
        for variant in VARIANTS
        for task in tasks
        for attempt in range(1, trials + 1)
    ]


def _manifest(args: argparse.Namespace, tasks: list[str], specs: list[AttemptSpec]) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "split": TB2_CORE_SPLIT,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "trials": args.trials,
        "tasks": tasks,
        "variants": [variant.__dict__ | {"candidate_dir": str(variant.candidate_dir)} for variant in VARIANTS],
        "attempts": len(specs),
        "concurrency": args.concurrency,
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
    }


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    os.environ["TERMINAL_MODEL_PROVIDER"] = "openai"
    if args.backend == "slurm-pyxis" and args.slurm_partition:
        os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition


def _run_attempt_pool(
    root: Path,
    args: argparse.Namespace,
    specs: list[AttemptSpec],
) -> list[dict[str, Any]]:
    worker_count = min(args.concurrency, len(specs))
    summaries: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_attempt, root, args, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), start=1):
            summaries.append(future.result())
            if index == len(futures) or index % max(1, args.concurrency) == 0:
                elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 1.0)
                rate = index / elapsed
                print(
                    f"[mini-swe-retry-matrix] completed {index}/{len(futures)} "
                    f"({rate:.2f} attempts/sec)",
                    flush=True,
                )
    return summaries


def _run_attempt(root: Path, args: argparse.Namespace, spec: AttemptSpec) -> dict[str, Any]:
    out_dir = root / args.model.replace(".", "_").replace("-", "_") / spec.variant.name / spec.task / f"attempt_{spec.attempt:02d}"
    try:
        summary = run_split(
            split=TB2_CORE_SPLIT,
            candidate_dir=ROOT / spec.variant.candidate_dir,
            budget=spec.variant.loc,
            out_dir=out_dir,
            tasks=[spec.task],
            trials=1,
            concurrency=1,
            dry_run=args.dry_run,
            harbor_bin=args.harbor_bin,
            backend=args.backend,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            max_retries=args.max_retries,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            agent_env=(
                "TERMINAL_MODEL_PROVIDER=openai",
                f"OPENAI_TERMINAL_MODEL={args.model}",
                f"OPENAI_TERMINAL_REASONING_EFFORT={args.reasoning_effort}",
                "OPENAI_AUTH_MODE=codex",
                f"MINI_SWE_FORMAT_RETRY_LIMIT={spec.variant.retry_limit}",
            ),
        )
        return {
            "harness": spec.variant.name,
            "base_harness": spec.variant.base_harness,
            "retry_limit": spec.variant.retry_limit,
            "model": args.model,
            "effort": args.reasoning_effort,
            "task": spec.task,
            "attempt": spec.attempt,
            "out_dir": str(out_dir),
            "returncode": int(summary.get("returncode", 0)),
            "ran": bool(summary.get("ran", True)),
            "dry_run": bool(summary.get("dry_run", False)),
        }
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "harness": spec.variant.name,
            "base_harness": spec.variant.base_harness,
            "retry_limit": spec.variant.retry_limit,
            "model": args.model,
            "effort": args.reasoning_effort,
            "task": spec.task,
            "attempt": spec.attempt,
            "out_dir": str(out_dir),
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _write_attempts_csv(
    root: Path,
    specs: list[AttemptSpec],
    attempt_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = {
        (
            str(item["harness"]),
            str(item["task"]),
            int(item["attempt"]),
        ): item
        for item in attempt_summaries
    }
    rows = []
    for spec in specs:
        key = (spec.variant.name, spec.task, spec.attempt)
        summary = summaries.get(key, {})
        out_dir = Path(str(summary.get("out_dir") or ""))
        record = _first_record(out_dir)
        harness_result = _latest_harness_result(out_dir)
        row = {
            "harness": spec.variant.name,
            "base_harness": spec.variant.base_harness,
            "retry_limit": spec.variant.retry_limit,
            "model": str(summary.get("model") or DEFAULT_MODEL),
            "effort": str(summary.get("effort") or DEFAULT_REASONING_EFFORT),
            "task": spec.task,
            "attempt": spec.attempt,
            "reward": _csv_value(record.get("reward") if record else "N/A"),
            "status": str(record.get("status") if record else "unknown"),
            "returncode": summary.get("returncode", "N/A"),
            "out_dir": str(out_dir),
            "commands": _csv_value(_len_or_none(harness_result.get("commands"))),
            "model_calls": _csv_value(_nested_value(harness_result, ("model_accounting", "model_calls"))),
            "input_tokens": _csv_value(record.get("input_tokens") if record else None),
            "output_tokens": _csv_value(record.get("output_tokens") if record else None),
            "total_tokens": _csv_value(record.get("total_tokens") if record else None),
            "termination_reason": _csv_value(harness_result.get("termination_reason")),
        }
        rows.append(row)
    _write_csv(root / "attempts.csv", rows)
    (root / "attempts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _write_aggregate_csvs(root: Path, attempts: list[dict[str, Any]], ci_samples: int) -> None:
    candidate_rows = _aggregate_rows(
        attempts,
        ("harness", "base_harness", "retry_limit", "model", "effort"),
        ci_samples,
    )
    task_rows = _aggregate_rows(
        attempts,
        ("harness", "base_harness", "retry_limit", "model", "effort", "task"),
        ci_samples,
    )
    _write_csv(root / "scores_by_harness.csv", candidate_rows)
    _write_csv(root / "aggregate_by_task.csv", task_rows)


def _aggregate_rows(
    attempts: list[dict[str, Any]],
    keys: tuple[str, ...],
    ci_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    rows = []
    for values, records in sorted(grouped.items()):
        rewards = [value for value in (_float(record["reward"]) for record in records) if value is not None]
        ci = bootstrap_ci(
            [
                {"task": record["task"], "reward": _float(record["reward"]) or 0.0}
                for record in records
                if record["reward"] != "N/A"
            ],
            samples=ci_samples,
            seed=7,
        )
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "N": len(records),
                "score": _csv_value(sum(rewards) / len(rewards) if rewards else 0.0),
                "ci95_low": _csv_value(ci["q025"]),
                "ci95_high": _csv_value(ci["q975"]),
                "successes": sum(1 for value in rewards if value >= 1.0),
                "input_tokens": _sum_csv(records, "input_tokens"),
                "output_tokens": _sum_csv(records, "output_tokens"),
                "total_tokens": _sum_csv(records, "total_tokens"),
                "avg_commands": _csv_value(_mean_numeric(records, "commands")),
                "avg_model_calls": _csv_value(_mean_numeric(records, "model_calls")),
            }
        )
        rows.append(row)
    return rows


def _first_record(out_dir: Path) -> dict[str, Any] | None:
    records = parse_records(out_dir) if out_dir.exists() else []
    return records[0] if records else None


def _latest_harness_result(out_dir: Path) -> dict[str, Any]:
    candidates = sorted(out_dir.glob("*/**/agent/harness-result.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _len_or_none(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float(value: Any) -> float | None:
    if value in (None, "", "N/A") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_csv(rows: list[dict[str, Any]], key: str) -> str:
    values = [value for value in (_float(row.get(key)) for row in rows) if value is not None]
    return _csv_value(sum(values) if values else None)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in (_float(row.get(key)) for row in rows) if value is not None]
    return sum(values) / len(values) if values else None


def _csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"mini_swe_retry_matrix_gpt55_high_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
