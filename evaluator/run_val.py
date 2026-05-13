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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = HarborRunSpec(candidate_dir, out_dir, tasks, trials, concurrency, split)
    plan = build_harbor_command(spec, executable=harbor_bin, help_text=harbor_help_text)
    command_json = {
        "split": split,
        "budget": budget,
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
    if shutil.which("docker") is None:
        message = "Docker is not installed or not on PATH. Please install Docker and try again."
        (out_dir / "stderr.log").write_text(message + "\n", encoding="utf-8")
        (out_dir / "stdout.log").write_text("", encoding="utf-8")
        (out_dir / "records.json").write_text("[]\n", encoding="utf-8")
        summary = aggregate_records([], split)
        summary.update(
            {
                "ran": False,
                "returncode": 1,
                "error": message,
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


def _terminal_model_error() -> str | None:
    if not using_codex_auth() and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is required for terminal model validation."
    try:
        check_terminal_model_available()
    except Exception:
        return (
            "Terminal model preflight failed. Check OPENAI_API_KEY or Codex auth, "
            "quota, and model access before running validation."
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--harbor-bin")
    args = parser.parse_args()
    summary = run_split(
        "val",
        args.candidate_dir,
        args.budget,
        args.out_dir,
        get_val_tasks(),
        VAL_TRIALS,
        VAL_CONCURRENCY,
        args.dry_run,
        args.harbor_bin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ran", True) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
