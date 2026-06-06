from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.aggregate import aggregate_records, write_summary  # noqa: E402
from evaluator.parse_results import parse_records  # noqa: E402
from evaluator.run_val import run_split  # noqa: E402
from evaluator.splits import get_tb2_core_tasks  # noqa: E402
from plumbing.harbor_adapter import (  # noqa: E402
    TERMINAL_BENCH_DATASET,
    detect_harbor_executable,
    harbor_help,
)
from scripts.bootstrap_ci import bootstrap_ci  # noqa: E402
from scripts.run_tb2_core import SEED_CANDIDATES, TB2_CORE_SPLIT, EvalCandidate  # noqa: E402

DEFAULT_OUT_ROOT = Path("/wbl-fast/usrs/trost/harness-complexity/final_test")
OPENAI_MODELS = ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5")
CLAUDE_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8")
CLAUDE_EFFORTS = {
    "claude-haiku-4-5": ("none",),
    "claude-sonnet-4-6": ("low", "medium", "high"),
    "claude-opus-4-8": ("low", "medium", "high"),
}
DEEPSEEK_CONFIGS = (
    ("deepseek_flash_none", "deepseek-v4-flash", "none"),
    ("deepseek_flash_high", "deepseek-v4-flash", "high"),
    ("deepseek_flash_max", "deepseek-v4-flash", "max"),
    ("deepseek_pro_none", "deepseek-v4-pro", "none"),
    ("deepseek_pro_high", "deepseek-v4-pro", "high"),
    ("deepseek_pro_max", "deepseek-v4-pro", "max"),
)
MINI_BACKFILL_CANDIDATES = {
    "seed_mini_swe_agent_v2",
    "seed_mini_swe_agent_barebones",
}
CORRUPTED_CLASSES = {"rate_limit", "provider_transport", "infra", "harbor_cache"}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    config_id: str
    model: str
    effort: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class MatrixAttempt:
    provider: str
    config_id: str
    model: str
    effort: str
    candidate: EvalCandidate
    task: str
    attempt: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--candidate", action="append", dest="candidate_names")
    parser.add_argument("--provider-config", action="append", dest="config_ids")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2400)
    parser.add_argument("--slurm-partition", default="m7i-cpu2")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--ci-samples", type=int, default=10000)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")

    tasks = args.tasks or get_tb2_core_tasks()
    candidates = _select_candidates(args.candidate_names)
    configs = _select_configs(_provider_configs(tuple(candidate.name for candidate in candidates)), args.config_ids)
    all_attempts = _attempts(configs, candidates, tasks, args.trials)
    attempts = _shard_attempts(all_attempts, args.shard_count, args.shard_index)
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args, configs, candidates, tasks, all_attempts, attempts)
    _write_manifest(root, args, manifest)

    if args.aggregate_only:
        aggregate = _write_outputs(root, all_attempts, args)
        (root / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(json.dumps({"out_root": str(root), **aggregate}, indent=2))
        return 0 if not aggregate["corrupted_attempts"] else 1

    os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition
    harbor_help_text = _harbor_help_text(args.harbor_bin)
    print(
        f"[provider-matrix] starting {len(attempts)}/{len(all_attempts)} attempts "
        f"at concurrency {args.concurrency}",
        flush=True,
    )
    summaries = _run_attempt_pool(root, args, attempts, harbor_help_text)
    summary_dir = _shard_dir(root, args)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "attempt_summaries.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    shard_summary = {
        "run_id": args.run_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "attempts_total": len(all_attempts),
        "attempts_selected": len(attempts),
        "attempts_completed": len(summaries),
    }
    (summary_dir / "summary.json").write_text(json.dumps(shard_summary, indent=2), encoding="utf-8")
    if args.skip_aggregate or args.shard_count > 1:
        print(json.dumps({"out_root": str(root), **shard_summary}, indent=2))
        return 0
    aggregate = _write_outputs(root, attempts, args)
    (root / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), **aggregate}, indent=2))
    return 0 if not aggregate["corrupted_attempts"] else 1


def _provider_configs(candidate_names: tuple[str, ...]) -> list[ProviderConfig]:
    all_candidates = candidate_names
    mini_candidates = tuple(name for name in candidate_names if name in MINI_BACKFILL_CANDIDATES)
    configs: list[ProviderConfig] = []
    for model in OPENAI_MODELS:
        configs.append(
            ProviderConfig("openai", f"openai_{_slug(model)}_high", model, "high", all_candidates)
        )
        for effort in ("low", "medium"):
            configs.append(
                ProviderConfig(
                    "openai",
                    f"openai_{_slug(model)}_{effort}",
                    model,
                    effort,
                    mini_candidates,
                )
            )
    for model in CLAUDE_MODELS:
        for effort in CLAUDE_EFFORTS[model]:
            configs.append(
                ProviderConfig(
                    "anthropic",
                    f"anthropic_{_slug(model)}_{effort}",
                    model,
                    effort,
                    all_candidates,
                )
            )
    for config_id, model, effort in DEEPSEEK_CONFIGS:
        configs.append(ProviderConfig("deepseek", config_id, model, effort, all_candidates))
    return configs


def _select_configs(
    configs: list[ProviderConfig],
    requested_ids: list[str] | None,
) -> list[ProviderConfig]:
    if not requested_ids:
        return configs
    by_id = {config.config_id: config for config in configs}
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"unknown provider config(s): {', '.join(missing)}")
    return [by_id[item] for item in requested_ids]


def _select_candidates(requested_names: list[str] | None) -> list[EvalCandidate]:
    by_name = {candidate.name: candidate for candidate in SEED_CANDIDATES}
    if not requested_names:
        return list(SEED_CANDIDATES)
    missing = sorted(set(requested_names) - set(by_name))
    if missing:
        raise ValueError(f"unknown candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in requested_names]


def _attempts(
    configs: list[ProviderConfig],
    candidates: list[EvalCandidate],
    tasks: list[str],
    trials: int,
) -> list[MatrixAttempt]:
    by_candidate = {candidate.name: candidate for candidate in candidates}
    attempts = []
    for config in configs:
        for candidate_name in config.candidates:
            candidate = by_candidate.get(candidate_name)
            if candidate is None:
                continue
            for task in tasks:
                for attempt in range(1, trials + 1):
                    attempts.append(
                        MatrixAttempt(
                            config.provider,
                            config.config_id,
                            config.model,
                            config.effort,
                            candidate,
                            task,
                            attempt,
                        )
                    )
    return attempts


def _shard_attempts(
    attempts: list[MatrixAttempt],
    shard_count: int,
    shard_index: int,
) -> list[MatrixAttempt]:
    return [
        attempt
        for index, attempt in enumerate(attempts)
        if index % shard_count == shard_index
    ]


def _manifest(
    args: argparse.Namespace,
    configs: list[ProviderConfig],
    candidates: list[EvalCandidate],
    tasks: list[str],
    all_attempts: list[MatrixAttempt],
    selected_attempts: list[MatrixAttempt],
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "split": TB2_CORE_SPLIT,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "tasks": tasks,
        "trials": args.trials,
        "scheduler": "global_provider_attempt_pool",
        "global_concurrency": args.concurrency,
        "attempt_concurrency": 1,
        "attempt_cells": len(all_attempts),
        "selected_attempt_cells": len(selected_attempts),
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "skip_aggregate": args.skip_aggregate,
        "aggregate_only": args.aggregate_only,
        "configs": [config.__dict__ for config in configs],
        "candidates": [
            {
                "name": candidate.name,
                "kind": candidate.kind,
                "candidate_dir": str(candidate.candidate_dir) if candidate.candidate_dir else None,
                "loc": candidate.loc,
            }
            for candidate in candidates
        ],
    }


def _write_manifest(root: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if args.shard_count == 1 or args.shard_index == 0 or args.aggregate_only:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shard_dir = _shard_dir(root, args)
    if shard_dir != root:
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def _shard_dir(root: Path, args: argparse.Namespace) -> Path:
    if args.shard_count == 1 and not args.aggregate_only:
        return root
    return root / "_shards" / f"shard_{args.shard_index:04d}"


def _harbor_help_text(harbor_bin: str | None) -> str | None:
    executable = harbor_bin or detect_harbor_executable() or "harbor"
    return harbor_help(executable, "run")


def _run_attempt_pool(
    root: Path,
    args: argparse.Namespace,
    attempts: list[MatrixAttempt],
    harbor_help_text: str | None,
) -> list[dict[str, Any]]:
    worker_count = min(args.concurrency, len(attempts))
    if worker_count < 1:
        return []
    summaries = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_attempt, root, args, attempt, harbor_help_text)
            for attempt in attempts
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            summaries.append(future.result())
            if index == len(futures) or index % max(1, min(args.concurrency, 100)) == 0:
                elapsed = max(time.monotonic() - started, 1)
                rate = index / elapsed
                print(
                    f"[provider-matrix] completed {index}/{len(futures)} "
                    f"({rate:.2f} attempts/sec)",
                    flush=True,
                )
    return summaries


def _run_attempt(
    root: Path,
    args: argparse.Namespace,
    attempt: MatrixAttempt,
    harbor_help_text: str | None,
) -> dict[str, Any]:
    out_dir = _attempt_dir(root, attempt)
    if attempt.candidate.candidate_dir is None:
        raise ValueError(f"candidate {attempt.candidate.name} has no candidate_dir")
    try:
        summary = run_split(
            split=TB2_CORE_SPLIT,
            candidate_dir=ROOT / attempt.candidate.candidate_dir,
            budget=attempt.candidate.loc or 0,
            out_dir=out_dir,
            tasks=[attempt.task],
            trials=1,
            concurrency=1,
            dry_run=args.dry_run,
            harbor_bin=args.harbor_bin,
            harbor_help_text=harbor_help_text,
            backend=args.backend,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            max_retries=args.max_retries,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            agent_env=_agent_env(attempt),
        )
        return {
            **_attempt_identity(attempt, out_dir),
            "returncode": int(summary.get("returncode", 0)),
            "ran": bool(summary.get("ran", True)),
            "dry_run": bool(summary.get("dry_run", False)),
        }
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            **_attempt_identity(attempt, out_dir),
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _agent_env(attempt: MatrixAttempt) -> tuple[str, ...]:
    env = [
        f"TERMINAL_MODEL_PROVIDER={attempt.provider}",
        f"OPENAI_TERMINAL_MODEL={attempt.model}",
        f"OPENAI_TERMINAL_REASONING_EFFORT={attempt.effort}",
    ]
    if attempt.provider == "openai":
        env.append("OPENAI_AUTH_MODE=codex")
    return tuple(env)


def _attempt_dir(root: Path, attempt: MatrixAttempt) -> Path:
    return (
        root
        / attempt.provider
        / attempt.config_id
        / _slug(attempt.model)
        / attempt.effort
        / attempt.candidate.name
        / attempt.task
        / f"attempt_{attempt.attempt:02d}"
    )


def _attempt_identity(attempt: MatrixAttempt, out_dir: Path) -> dict[str, Any]:
    return {
        "provider": attempt.provider,
        "config_id": attempt.config_id,
        "model": attempt.model,
        "effort": attempt.effort,
        "candidate": attempt.candidate.name,
        "task": attempt.task,
        "attempt": attempt.attempt,
        "out_dir": str(out_dir),
    }


def _write_outputs(
    root: Path,
    attempts: list[MatrixAttempt],
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempt_rows = [_attempt_row(root, attempt) for attempt in attempts]
    _write_csv(root / "attempts.csv", attempt_rows)
    (root / "attempts.json").write_text(json.dumps(attempt_rows, indent=2), encoding="utf-8")

    candidate_rows = _aggregate_rows(attempt_rows, ("provider", "config_id", "model", "effort", "candidate"), args)
    task_rows = _aggregate_rows(
        attempt_rows,
        ("provider", "config_id", "model", "effort", "candidate", "task"),
        args,
    )
    _write_csv(root / "aggregate_by_candidate.csv", candidate_rows)
    _write_csv(root / "aggregate_by_task.csv", task_rows)
    _write_nested_summaries(root, attempts)

    corrupted = [row for row in attempt_rows if row["corrupted"] == "1"]
    (root / "corrupted_attempts.json").write_text(
        json.dumps(corrupted, indent=2),
        encoding="utf-8",
    )
    return {
        "attempts": len(attempt_rows),
        "corrupted_attempts": len(corrupted),
        "aggregate_by_candidate_rows": len(candidate_rows),
        "aggregate_by_task_rows": len(task_rows),
    }


def _attempt_row(root: Path, attempt: MatrixAttempt) -> dict[str, Any]:
    out_dir = _attempt_dir(root, attempt)
    records = parse_records(out_dir)
    record = records[0] if records else {}
    usage = _api_usage(out_dir)
    status = str(record.get("status") or _status_from_summary(out_dir))
    reward = record.get("reward")
    failure_class = _failure_class(out_dir, status, attempt.candidate.name)
    return {
        **_attempt_identity(attempt, out_dir),
        "status": status,
        "reward": reward if reward is not None else "N/A",
        "failure_class": failure_class,
        "corrupted": "1" if failure_class in CORRUPTED_CLASSES else "0",
        "api_input_tokens": _csv_value(usage.get("input_tokens")),
        "api_output_tokens": _csv_value(usage.get("output_tokens")),
        "api_cached_tokens": _csv_value(usage.get("cached_tokens")),
        "api_total_tokens": _csv_value(usage.get("total_tokens")),
    }


def _status_from_summary(out_dir: Path) -> str:
    summary = _read_json(out_dir / "summary.json")
    if isinstance(summary, dict) and summary.get("returncode"):
        return "crash"
    return "unknown"


def _failure_class(out_dir: Path, status: str, candidate: str) -> str:
    if status == "success":
        return "success"
    text = _failure_text(out_dir).lower()
    if any(item in text for item in ("rate limit", "too many requests", " 429 ", "status_code\": 429")):
        return "rate_limit"
    if any(item in text for item in ("cybersecurity risk", "trusted access for cyber", "policy")):
        return "provider_policy"
    if any(item in text for item in ("directory not empty", "file exists", ".cache/harbor/tasks")):
        return "harbor_cache"
    if any(item in text for item in ("context", "maximum context", "token limit", "too many tokens")) and "mini_swe_agent" in candidate:
        return "context_overflow_no_compaction"
    if any(item in text for item in ("slurm", "pyxis", "srun", "connection reset", "transport")):
        return "infra"
    if "api call failed" in text or "http" in text and "failed" in text:
        return "provider_transport"
    return "task_failure" if status in {"failure", "crash"} else "unknown"


def _failure_text(out_dir: Path) -> str:
    chunks = []
    for name in ("stderr.log", "stdout.log", "summary.json"):
        path = out_dir / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[-4000:])
    for path in sorted(out_dir.glob("**/exception.txt")):
        chunks.append(path.read_text(encoding="utf-8", errors="replace")[-4000:])
    for path in sorted(out_dir.glob("**/harness-result.json")):
        chunks.append(path.read_text(encoding="utf-8", errors="replace")[-4000:])
    return "\n".join(chunks)


def _api_usage(out_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for path in sorted(out_dir.glob("**/agent/model-call-*.json")):
        payload = _read_json(path)
        metadata = payload.get("request_metadata") if isinstance(payload, dict) else None
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        if not isinstance(usage, dict) or not usage:
            continue
        for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or value is None:
                continue
            try:
                totals[key] = totals.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue
    return totals


def _aggregate_rows(
    attempt_rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in attempt_rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    rows = []
    for key_values, records in sorted(grouped.items()):
        rewards = [_float(row["reward"]) for row in records if _float(row["reward"]) is not None]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        ci = bootstrap_ci(
            [
                {"task": row["task"], "reward": _float(row["reward"]) or 0.0}
                for row in records
                if row["reward"] != "N/A"
            ],
            samples=args.ci_samples,
            seed=7,
        )
        base = dict(zip(keys, key_values, strict=True))
        base.update(
            {
                "score": score,
                "ci95_low": _csv_value(ci["q025"]),
                "ci95_high": _csv_value(ci["q975"]),
                "num_attempts": len(records),
                "num_successes": sum(1 for value in rewards if value >= 1),
                "num_crashes": sum(1 for row in records if row["status"] == "crash"),
                "num_corrupted": sum(1 for row in records if row["corrupted"] == "1"),
                "api_input_tokens": _sum_csv(records, "api_input_tokens"),
                "api_output_tokens": _sum_csv(records, "api_output_tokens"),
                "api_cached_tokens": _sum_csv(records, "api_cached_tokens"),
                "api_total_tokens": _sum_csv(records, "api_total_tokens"),
            }
        )
        for failure_class in sorted({str(row["failure_class"]) for row in records}):
            base[f"failure_{failure_class}"] = sum(
                1 for row in records if row["failure_class"] == failure_class
            )
        rows.append(base)
    return rows


def _write_nested_summaries(root: Path, attempts: list[MatrixAttempt]) -> None:
    candidate_dirs = sorted(
        {
            _attempt_dir(root, attempt).parents[1]
            for attempt in attempts
        }
    )
    for candidate_dir in candidate_dirs:
        records = parse_records(candidate_dir)
        (candidate_dir / "records.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        summary = aggregate_records(records, TB2_CORE_SPLIT)
        write_summary(summary, candidate_dir)


def _sum_csv(records: list[dict[str, Any]], key: str) -> str:
    values = []
    for row in records:
        value = _float(row[key])
        if value is not None:
            values.append(value)
    return _csv_value(sum(values) if values else None)


def _float(value: Any) -> float | None:
    if value == "N/A" or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace(".", "_").replace("-", "_")


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tb2_provider_matrix_gapfill_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
