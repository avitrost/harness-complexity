from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from evaluator.aggregate import aggregate_records, write_summary
from evaluator.parse_results import parse_records
from evaluator.run_val import BACKENDS, _backend_error
from evaluator.splits import VAL_CONCURRENCY, get_test_tasks, get_val_tasks
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET, HarborRunSpec, build_harbor_command
from plumbing.openai_client import terminal_model, terminal_reasoning_effort
from plumbing.terminus_2_agent import (
    DEFAULT_TERMINUS_2_PARSER_NAME,
    DEFAULT_TERMINUS_2_REASONING_EFFORT,
    terminus_2_agent_import_path,
)


def run_terminus_2_split(
    split: str,
    out_dir: Path,
    tasks: list[str],
    trials: int,
    concurrency: int,
    backend: str,
    terminus_model: str,
    parser_name: str,
    reasoning_effort: str | None,
    dry_run: bool,
    harbor_bin: str | None = None,
    harbor_help_text: str | None = None,
    dataset: str = TERMINAL_BENCH_DATASET,
    dataset_path: Path | None = None,
    max_retries: int = 0,
    verifier_timeout_multiplier: float | None = None,
    retry_include: tuple[str, ...] = (),
    retry_exclude: tuple[str, ...] = (),
    record_terminal_session: bool = False,
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
    agent_kwargs = [
        f"parser_name={parser_name}",
        f"record_terminal_session={str(record_terminal_session).lower()}",
    ]
    if reasoning_effort:
        agent_kwargs.append(f"reasoning_effort={reasoning_effort}")
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
        agent_import_path=terminus_2_agent_import_path(),
        agent_model_name=terminus_model,
        agent_kwargs=tuple(agent_kwargs),
    )
    plan = build_harbor_command(spec, executable=harbor_bin, help_text=harbor_help_text)
    command_json = {
        "split": split,
        "backend": backend,
        "dataset": dataset,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "terminus_model": terminus_model,
        "parser_name": parser_name,
        "reasoning_effort": reasoning_effort,
        "record_terminal_session": record_terminal_session,
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
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=VAL_CONCURRENCY)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="slurm-pyxis")
    parser.add_argument("--terminus-model", default=terminal_model())
    parser.add_argument("--parser-name", default=DEFAULT_TERMINUS_2_PARSER_NAME)
    parser.add_argument(
        "--reasoning-effort",
        default=terminal_reasoning_effort() or DEFAULT_TERMINUS_2_REASONING_EFFORT,
    )
    parser.add_argument(
        "--record-terminal-session",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--retry-include", action="append", default=[])
    parser.add_argument("--retry-exclude", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = args.tasks or (get_val_tasks() if args.split == "val" else get_test_tasks())
    summary = run_terminus_2_split(
        split=args.split,
        out_dir=args.out_dir,
        tasks=tasks,
        trials=args.trials,
        concurrency=args.concurrency,
        backend=args.backend,
        terminus_model=args.terminus_model,
        parser_name=args.parser_name,
        reasoning_effort=args.reasoning_effort,
        dry_run=args.dry_run,
        harbor_bin=args.harbor_bin,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        max_retries=args.max_retries,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        retry_include=tuple(args.retry_include),
        retry_exclude=tuple(args.retry_exclude),
        record_terminal_session=args.record_terminal_session,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ran", True) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
