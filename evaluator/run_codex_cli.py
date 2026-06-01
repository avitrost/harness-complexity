from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from evaluator.aggregate import aggregate_records, write_summary
from evaluator.parse_results import parse_records
from evaluator.run_val import BACKENDS, _backend_error
from evaluator.splits import (
    TB2_CORE_CONCURRENCY,
    TB2_CORE_TRIALS,
    VAL_CONCURRENCY,
    get_tb2_core_tasks,
    get_test_tasks,
    get_val_tasks,
)
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET, HarborRunSpec, build_harbor_command
from plumbing.openai_client import terminal_model, terminal_reasoning_effort

OFFICIAL_CODEX_AGENT_NAME = "codex"
LEGACY_TIMEOUT_SEC = 7200


def run_codex_cli_split(
    split: str,
    out_dir: Path,
    tasks: list[str],
    trials: int,
    concurrency: int,
    backend: str,
    codex_model: str,
    codex_reasoning_effort: str,
    timeout_sec: int,
    dry_run: bool,
    harbor_bin: str | None = None,
    harbor_help_text: str | None = None,
    dataset: str = TERMINAL_BENCH_DATASET,
    dataset_path: Path | None = None,
    max_retries: int = 0,
    verifier_timeout_multiplier: float | None = None,
    retry_include: tuple[str, ...] = (),
    retry_exclude: tuple[str, ...] = (),
    codex_auth_json_path: Path | None = None,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not tasks:
        raise ValueError(f"no tasks configured for {split} split")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = HarborRunSpec(
        candidate_dir=Path("."),
        out_dir=out_dir,
        tasks=tasks,
        trials=trials,
        concurrency=concurrency,
        split=split,
        backend=backend,
        dataset=dataset,
        dataset_path=dataset_path,
        max_retries=max_retries,
        verifier_timeout_multiplier=verifier_timeout_multiplier,
        retry_include=retry_include,
        retry_exclude=retry_exclude,
        agent_name=OFFICIAL_CODEX_AGENT_NAME,
        agent_model_name=codex_model,
        agent_kwargs=(f"reasoning_effort={codex_reasoning_effort}",),
        agent_env=_codex_agent_env(codex_auth_json_path),
        include_candidate_dir_kwarg=False,
    )
    plan = build_harbor_command(spec, executable=harbor_bin, help_text=harbor_help_text)
    command_json = {
        "split": split,
        "backend": backend,
        "dataset": dataset,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "codex_model": codex_model,
        "codex_reasoning_effort": codex_reasoning_effort,
        "codex_agent": OFFICIAL_CODEX_AGENT_NAME,
        "codex_auth_json_path": str(_resolve_codex_auth_path(codex_auth_json_path) or ""),
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
        "verifier_timeout_multiplier": verifier_timeout_multiplier,
        "retry_include": list(retry_include),
        "retry_exclude": list(retry_exclude),
        "command": plan.command,
        "runnable": plan.runnable,
        "task_flag": plan.task_flag,
        "note": plan.note,
    }
    (out_dir / "command.json").write_text(json.dumps(command_json, indent=2), encoding="utf-8")
    if dry_run or not plan.runnable:
        summary = {"split": split, "dry_run": dry_run, "ran": False, **command_json}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    backend_error = _backend_error(backend)
    if backend_error:
        summary = {"split": split, "ran": False, "returncode": 1, "error": backend_error}
        summary.update(command_json)
        (out_dir / "records.json").write_text("[]\n", encoding="utf-8")
        write_summary(summary, out_dir)
        return summary
    result = subprocess.run(plan.command, check=False, capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    records = parse_records(out_dir)
    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = aggregate_records(records, split)
    summary.update({"ran": True, "returncode": result.returncode, **command_json})
    write_summary(summary, out_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test", "tb2-core"), default="tb2-core")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="slurm-pyxis")
    parser.add_argument("--codex-model", default=terminal_model())
    parser.add_argument("--codex-reasoning-effort", default=terminal_reasoning_effort())
    parser.add_argument("--timeout-sec", type=int, default=LEGACY_TIMEOUT_SEC)
    parser.add_argument("--codex-auth-json-path", type=Path)
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--retry-include", action="append", default=[])
    parser.add_argument("--retry-exclude", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = args.tasks or _default_tasks(args.split)
    summary = run_codex_cli_split(
        split=args.split,
        out_dir=args.out_dir,
        tasks=tasks,
        trials=args.trials or (TB2_CORE_TRIALS if args.split == "tb2-core" else 1),
        concurrency=args.concurrency
        or (TB2_CORE_CONCURRENCY if args.split == "tb2-core" else VAL_CONCURRENCY),
        backend=args.backend,
        codex_model=args.codex_model,
        codex_reasoning_effort=args.codex_reasoning_effort,
        timeout_sec=args.timeout_sec,
        dry_run=args.dry_run,
        harbor_bin=args.harbor_bin,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        max_retries=args.max_retries,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        retry_include=tuple(args.retry_include),
        retry_exclude=tuple(args.retry_exclude),
        codex_auth_json_path=args.codex_auth_json_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ran", True) or args.dry_run else 1


def _default_tasks(split: str) -> list[str]:
    if split == "tb2-core":
        return get_tb2_core_tasks()
    if split == "val":
        return get_val_tasks()
    return get_test_tasks()


def _codex_agent_env(codex_auth_json_path: Path | None = None) -> tuple[str, ...]:
    auth_path = _resolve_codex_auth_path(codex_auth_json_path)
    if auth_path is None:
        return ()
    return (f"CODEX_AUTH_JSON_PATH={auth_path}",)


def _resolve_codex_auth_path(codex_auth_json_path: Path | None = None) -> Path | None:
    if codex_auth_json_path is not None:
        return codex_auth_json_path.expanduser().resolve()
    env_path = os.getenv("CODEX_AUTH_JSON_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    default = Path.home() / ".codex" / "auth.json"
    if default.is_file():
        return default.resolve()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
