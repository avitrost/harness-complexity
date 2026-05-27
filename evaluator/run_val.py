from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from evaluator.aggregate import aggregate_records, write_summary
from evaluator.parse_results import parse_records
from evaluator.splits import VAL_CONCURRENCY, VAL_TRIALS, get_val_tasks
from plumbing.harbor_adapter import HarborRunSpec, build_harbor_command
from plumbing.openai_client import check_terminal_model_available, using_codex_auth

BACKENDS = {"docker", "slurm-pyxis"}


def run_split(
    split: str,
    candidate_dir: Path,
    budget: int,
    out_dir: Path,
    tasks: list[str],
    trials: int,
    concurrency: int,
    dry_run: bool,
    harbor_bin: str | None = None,
    harbor_help_text: str | None = None,
    backend: str = "docker",
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not tasks:
        raise ValueError(f"no tasks configured for {split} split")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = HarborRunSpec(candidate_dir, out_dir, tasks, trials, concurrency, split, backend)
    plan = build_harbor_command(spec, executable=harbor_bin, help_text=harbor_help_text)
    command_json = {
        "split": split,
        "budget": budget,
        "backend": backend,
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
        (out_dir / "stderr.log").write_text(backend_error + "\n", encoding="utf-8")
        (out_dir / "stdout.log").write_text("", encoding="utf-8")
        (out_dir / "records.json").write_text("[]\n", encoding="utf-8")
        summary = aggregate_records([], split)
        summary.update(
            {
                "ran": False,
                "returncode": 1,
                "error": backend_error,
                **command_json,
            }
        )
        write_summary(summary, out_dir)
        return summary
    api_error = _terminal_model_error()
    if api_error:
        (out_dir / "stderr.log").write_text(api_error + "\n", encoding="utf-8")
        (out_dir / "stdout.log").write_text("", encoding="utf-8")
        (out_dir / "records.json").write_text("[]\n", encoding="utf-8")
        summary = aggregate_records([], split)
        summary.update(
            {
                "ran": False,
                "returncode": 1,
                "error": api_error,
                **command_json,
            }
        )
        write_summary(summary, out_dir)
        return summary
    result = subprocess.run(plan.command, check=False, capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    records = parse_records(out_dir)
    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = aggregate_records(records, split)
    summary.update({"ran": True, "returncode": result.returncode, "command": plan.command})
    write_summary(summary, out_dir)
    return summary


def _backend_error(backend: str) -> str | None:
    if backend == "docker" and shutil.which("docker") is None:
        return "Docker is not installed or not on PATH. Please install Docker and try again."
    if backend == "slurm-pyxis":
        missing = [name for name in ("srun", "enroot") if shutil.which(name) is None]
        if missing:
            return f"Slurm/Pyxis backend missing required command(s): {', '.join(missing)}."
    return None


def _terminal_model_error() -> str | None:
    if not using_codex_auth() and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is required for terminal model validation."
    if os.getenv("HARBOR_TERMINAL_MODEL_PREFLIGHT", "0") != "1":
        return None
    try:
        check_terminal_model_available()
    except Exception as exc:
        return (
            "Terminal model preflight failed. Check OPENAI_API_KEY or Codex auth, "
            f"quota, and model access before running validation. ({type(exc).__name__}: {exc})"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="docker")
    parser.add_argument("--concurrency", type=int, default=VAL_CONCURRENCY)
    args = parser.parse_args()
    summary = run_split(
        "val",
        args.candidate_dir,
        args.budget,
        args.out_dir,
        get_val_tasks(),
        VAL_TRIALS,
        args.concurrency,
        args.dry_run,
        args.harbor_bin,
        backend=args.backend,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ran", True) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
