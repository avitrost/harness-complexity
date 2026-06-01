from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.aggregate import aggregate_records, write_summary
from evaluator.parse_results import parse_records
from evaluator.run_val import run_split
from evaluator.splits import get_tb2_core_tasks
from scripts.run_tb2_core import SEED_CANDIDATES, TB2_CORE_SPLIT, EvalCandidate

SUPPORTED_CODEX_BACKEND_MODELS = ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5")


@dataclass(frozen=True)
class AttemptSpec:
    model: str
    candidate: EvalCandidate
    task: str
    attempt: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=45)
    parser.add_argument("--max-candidate-workers", type=int)
    parser.add_argument("--slurm-partition", default="m7i-cpu2")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    models = tuple(args.models or SUPPORTED_CODEX_BACKEND_MODELS)
    tasks = args.tasks or get_tb2_core_tasks()
    candidates = _select_candidates(args.candidates)
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    specs = _attempt_specs(models, candidates, tasks, args.trials)
    manifest = _manifest(args, models, candidates, tasks, len(specs))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _configure_environment(args)
    if args.max_candidate_workers is not None:
        print(
            "[tb2-model-sweep] --max-candidate-workers is ignored by the global pool",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[tb2-model-sweep] starting {len(specs)} attempt cells with "
        f"global concurrency {args.concurrency}",
        flush=True,
    )
    attempt_summaries = _run_attempt_pool(root, args, specs)
    summaries = _write_summaries(root, models, candidates, attempt_summaries, args)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), "summaries": summaries}, indent=2))
    return 0 if all(item["returncode"] == 0 for item in summaries) else 1


def _manifest(
    args: argparse.Namespace,
    models: tuple[str, ...],
    candidates: list[EvalCandidate],
    tasks: list[str],
    attempt_cells: int,
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "models": list(models),
        "reasoning_effort": args.reasoning_effort,
        "trials": args.trials,
        "tasks": tasks,
        "scheduler": "global_attempt_pool",
        "global_concurrency": args.concurrency,
        "attempt_concurrency": 1,
        "effective_max_in_flight": args.concurrency,
        "legacy_max_candidate_workers": args.max_candidate_workers,
        "attempt_cells": attempt_cells,
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
        "include_codex_cli": False,
        "include_terminus_2": False,
        "candidates": [_candidate_manifest(candidate) for candidate in candidates],
    }


def _candidate_manifest(candidate: EvalCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "kind": candidate.kind,
        "candidate_dir": str(candidate.candidate_dir) if candidate.candidate_dir else None,
        "loc": candidate.loc,
    }


def _select_candidates(requested_names: list[str] | None) -> list[EvalCandidate]:
    if not requested_names:
        return list(SEED_CANDIDATES)
    by_name = {candidate.name: candidate for candidate in SEED_CANDIDATES}
    missing = sorted(set(requested_names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown seed candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in requested_names]


def _attempt_specs(
    models: tuple[str, ...],
    candidates: list[EvalCandidate],
    tasks: list[str],
    trials: int,
) -> list[AttemptSpec]:
    return [
        AttemptSpec(model=model, candidate=candidate, task=task, attempt=attempt)
        for model in models
        for candidate in candidates
        for task in tasks
        for attempt in range(1, trials + 1)
    ]


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    if args.backend == "slurm-pyxis" and args.slurm_partition:
        os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition


def _run_attempt_pool(
    root: Path,
    args: argparse.Namespace,
    specs: list[AttemptSpec],
) -> list[dict[str, Any]]:
    worker_count = min(args.concurrency, len(specs))
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_attempt, root, spec, args) for spec in specs]
        for index, future in enumerate(as_completed(futures), start=1):
            summary = future.result()
            summaries.append(summary)
            if index == len(futures) or index % max(1, args.concurrency) == 0:
                print(
                    f"[tb2-model-sweep] completed {index}/{len(futures)} attempt cells",
                    flush=True,
                )
    return summaries


def _run_attempt(root: Path, spec: AttemptSpec, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = (
        root
        / _model_run_id(spec.model)
        / spec.candidate.name
        / spec.task
        / f"attempt_{spec.attempt:02d}"
    )
    if spec.candidate.candidate_dir is None:
        raise ValueError(f"Candidate {spec.candidate.name} has no candidate_dir")
    try:
        summary = run_split(
            split=TB2_CORE_SPLIT,
            candidate_dir=ROOT / spec.candidate.candidate_dir,
            budget=spec.candidate.loc or 0,
            out_dir=out_dir,
            tasks=[spec.task],
            trials=1,
            concurrency=1,
            dry_run=args.dry_run,
            harbor_bin=args.harbor_bin,
            backend=args.backend,
            agent_env=(
                f"OPENAI_TERMINAL_MODEL={spec.model}",
                f"OPENAI_TERMINAL_REASONING_EFFORT={args.reasoning_effort}",
            ),
        )
        return {
            "model": spec.model,
            "candidate": spec.candidate.name,
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
            "model": spec.model,
            "candidate": spec.candidate.name,
            "task": spec.task,
            "attempt": spec.attempt,
            "out_dir": str(out_dir),
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _write_summaries(
    root: Path,
    models: tuple[str, ...],
    candidates: list[EvalCandidate],
    attempt_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summaries_by_cell = {
        (
            str(summary["model"]),
            str(summary["candidate"]),
            str(summary["task"]),
            int(summary["attempt"]),
        ): summary
        for summary in attempt_summaries
    }
    model_summaries = []
    for model in models:
        model_dir = root / _model_run_id(model)
        candidate_summaries = []
        for candidate in candidates:
            candidate_dir = model_dir / candidate.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            records = parse_records(candidate_dir)
            (candidate_dir / "records.json").write_text(
                json.dumps(records, indent=2),
                encoding="utf-8",
            )
            summary = aggregate_records(records, TB2_CORE_SPLIT)
            expected_attempts = len(args.tasks or get_tb2_core_tasks()) * args.trials
            attempt_items = [
                item
                for key, item in summaries_by_cell.items()
                if key[0] == model and key[1] == candidate.name
            ]
            returncode = 0 if all(item.get("returncode", 0) == 0 for item in attempt_items) else 1
            summary.update(
                {
                    "model": model,
                    "candidate": candidate.name,
                    "expected_attempts": expected_attempts,
                    "attempt_cells": len(attempt_items),
                    "global_concurrency": args.concurrency,
                    "returncode": returncode,
                    "ran": any(item.get("ran", True) for item in attempt_items),
                }
            )
            write_summary(summary, candidate_dir)
            candidate_summaries.append(summary)
        model_returncode = (
            0 if all(item.get("returncode", 0) == 0 for item in candidate_summaries) else 1
        )
        model_summary = {
            "model": model,
            "run_id": _model_run_id(model),
            "returncode": model_returncode,
            "candidates": candidate_summaries,
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "summary.json").write_text(
            json.dumps(model_summary, indent=2),
            encoding="utf-8",
        )
        model_summaries.append(model_summary)
    return model_summaries


def _model_run_id(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tb2_core_model_sweep_medium10_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
