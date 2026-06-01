from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.run_codex_cli import run_codex_cli_split  # noqa: E402
from evaluator.run_terminus_2 import run_terminus_2_split  # noqa: E402
from evaluator.run_val import run_split  # noqa: E402
from evaluator.tblite import TBLITE_DATASET_ID, TBLITE_SPLIT  # noqa: E402
from plumbing.codex_cli_agent import DEFAULT_TIMEOUT_SEC  # noqa: E402
from plumbing.openai_client import terminal_model, terminal_reasoning_effort  # noqa: E402
from plumbing.terminus_2_agent import DEFAULT_TERMINUS_2_PARSER_NAME  # noqa: E402

DEFAULT_SOURCE_ROOT = Path("final_test/tblite5x_fixed_20260528_201823")
DEFAULT_MINIMAL_CONCURRENCY = 112
DEFAULT_DEFAULT_CONCURRENCY = 24


@dataclass(frozen=True)
class Candidate:
    name: str
    category: str
    kind: str
    budget: int | None
    candidate_dir: Path | None


@dataclass(frozen=True)
class CatchupJob:
    candidate: Candidate
    tier: int
    tasks: tuple[str, ...]
    concurrency: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--extra-source-root", type=Path, action="append", default=[])
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--target-attempts", type=int, default=4)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--minimal-concurrency", type=int, default=DEFAULT_MINIMAL_CONCURRENCY)
    parser.add_argument("--default-concurrency", type=int, default=DEFAULT_DEFAULT_CONCURRENCY)
    parser.add_argument("--candidate-concurrency", action="append", default=[])
    parser.add_argument("--candidate", action="append", dest="candidate_names")
    parser.add_argument("--codex-model", default=terminal_model())
    parser.add_argument("--codex-reasoning-effort", default=terminal_reasoning_effort())
    parser.add_argument("--terminus-model", default=terminal_model())
    parser.add_argument("--terminus-parser-name", default=DEFAULT_TERMINUS_2_PARSER_NAME)
    parser.add_argument("--terminus-reasoning-effort", default=terminal_reasoning_effort())
    parser.add_argument(
        "--terminus-record-terminal-session",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_attempts < 1:
        raise ValueError("--target-attempts must be >= 1")

    manifest = json.loads((args.source_root / "manifest.json").read_text(encoding="utf-8"))
    tasks = list(manifest["tasks"])
    candidates = _select_candidates(_load_candidates(manifest), args.candidate_names)
    source_roots = [args.source_root, *args.extra_source_root]
    overrides = _parse_concurrency_overrides(args.candidate_concurrency)
    out_root = args.out_root / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    plans: list[CatchupJob] = []
    coverage: list[dict[str, Any]] = []
    for candidate in candidates:
        counts = _count_attempts(source_roots, candidate.name)
        missing = {task: max(0, args.target_attempts - counts.get(task, 0)) for task in tasks}
        tier_tasks = [
            tuple(task for task in tasks if missing[task] >= tier)
            for tier in range(1, max(missing.values(), default=0) + 1)
        ]
        tier_tasks = [items for items in tier_tasks if items]
        concurrency_budget = _candidate_concurrency(
            candidate.name,
            overrides,
            args.minimal_concurrency,
            args.default_concurrency,
        )
        allocations = _allocate_concurrency(concurrency_budget, tier_tasks)
        candidate_jobs = [
            CatchupJob(candidate, index + 1, tasks_for_tier, concurrency)
            for index, (tasks_for_tier, concurrency) in enumerate(
                zip(tier_tasks, allocations, strict=True)
            )
        ]
        plans.extend(candidate_jobs)
        coverage.append(
            {
                "candidate": candidate.name,
                "completed_attempts": sum(counts.get(task, 0) for task in tasks),
                "missing_attempts": sum(missing.values()),
                "max_missing_for_task": max(missing.values(), default=0),
                "jobs": [
                    {
                        "tier": job.tier,
                        "tasks": len(job.tasks),
                        "concurrency": job.concurrency,
                    }
                    for job in candidate_jobs
                ],
            }
        )

    catchup_manifest = {
        "run_id": args.run_id,
        "source_roots": [str(root) for root in source_roots],
        "target_attempts": args.target_attempts,
        "dataset": manifest.get("dataset", TBLITE_DATASET_ID),
        "dataset_path": manifest["dataset_path"],
        "tasks": tasks,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "coverage": coverage,
        "planned_jobs": [
            {
                "candidate": job.candidate.name,
                "tier": job.tier,
                "tasks": len(job.tasks),
                "concurrency": job.concurrency,
            }
            for job in plans
        ],
    }
    (out_root / "manifest.json").write_text(
        json.dumps(catchup_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(catchup_manifest, indent=2), flush=True)
    if args.dry_run or not plans:
        return 0

    max_workers = len(plans)
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_job, job, out_root, manifest, args): job for job in plans}
        for future in as_completed(futures):
            job = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "candidate": job.candidate.name,
                    "tier": job.tier,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            summaries.append(summary)
            print(json.dumps({"completed_catchup_job": summary}), flush=True)

    summaries.sort(key=lambda item: (item.get("candidate", ""), item.get("tier", 0)))
    (out_root / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    return 0 if all("error" not in item for item in summaries) else 1


def _load_candidates(manifest: dict[str, Any]) -> list[Candidate]:
    result = []
    for item in manifest["candidates"]:
        candidate_dir = item.get("candidate_dir")
        result.append(
            Candidate(
                name=item["name"],
                category=item["category"],
                kind=item["kind"],
                budget=item.get("budget"),
                candidate_dir=Path(candidate_dir) if candidate_dir else None,
            )
        )
    return result


def _select_candidates(
    candidates: list[Candidate],
    requested_names: list[str] | None,
) -> list[Candidate]:
    if not requested_names:
        return candidates
    by_name = {candidate.name: candidate for candidate in candidates}
    missing = sorted(set(requested_names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in requested_names]


def _count_attempts(source_roots: list[Path], candidate: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_root in source_roots:
        candidate_root = source_root / candidate
        if not candidate_root.exists():
            continue
        for result_path in candidate_root.glob("**/result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task = result.get("task_name")
            if isinstance(task, str) and task:
                counts[task] = counts.get(task, 0) + 1
    return counts


def _parse_concurrency_overrides(items: list[str]) -> dict[str, int]:
    result = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected NAME=INT for --candidate-concurrency, got {item!r}")
        parsed = int(value)
        if parsed < 1:
            raise ValueError("candidate concurrency overrides must be >= 1")
        result[name] = parsed
    return result


def _candidate_concurrency(
    candidate: str,
    overrides: dict[str, int],
    minimal_concurrency: int,
    default_concurrency: int,
) -> int:
    if candidate in overrides:
        return overrides[candidate]
    if candidate == "seed_minimal_agent":
        return minimal_concurrency
    return default_concurrency


def _allocate_concurrency(budget: int, tiers: list[tuple[str, ...]]) -> list[int]:
    if not tiers:
        return []
    budget = max(1, budget)
    weights = [len(items) for items in tiers]
    total_weight = sum(weights)
    raw = [budget * weight / total_weight for weight in weights]
    allocations = [
        max(1, min(len(tiers[index]), math.floor(value))) for index, value in enumerate(raw)
    ]
    while sum(allocations) < budget:
        candidates = [index for index, items in enumerate(tiers) if allocations[index] < len(items)]
        if not candidates:
            break
        index = max(candidates, key=lambda item: raw[item] - math.floor(raw[item]))
        allocations[index] += 1
    while sum(allocations) > budget:
        candidates = [index for index, value in enumerate(allocations) if value > 1]
        if not candidates:
            break
        index = min(candidates, key=lambda item: raw[item] - math.floor(raw[item]))
        allocations[index] -= 1
    return allocations


def _run_job(
    job: CatchupJob,
    out_root: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = out_root / job.candidate.name / f"tier_{job.tier:02d}"
    common = {
        "split": TBLITE_SPLIT,
        "out_dir": out_dir,
        "tasks": list(job.tasks),
        "trials": 1,
        "concurrency": job.concurrency,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "harbor_bin": args.harbor_bin,
        "dataset": manifest.get("dataset", TBLITE_DATASET_ID),
        "dataset_path": Path(manifest["dataset_path"]),
        "max_retries": int(manifest.get("max_retries", 0)),
        "verifier_timeout_multiplier": manifest.get("verifier_timeout_multiplier"),
        "retry_include": tuple(manifest.get("retry_include", ())),
        "retry_exclude": tuple(manifest.get("retry_exclude", ())),
    }
    if job.candidate.kind == "codex_cli":
        summary = run_codex_cli_split(
            **common,
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            timeout_sec=args.timeout_sec,
        )
    elif job.candidate.kind == "terminus_2":
        summary = run_terminus_2_split(
            **common,
            terminus_model=args.terminus_model or manifest.get("terminus_model", terminal_model()),
            parser_name=args.terminus_parser_name
            or manifest.get("terminus_parser_name", DEFAULT_TERMINUS_2_PARSER_NAME),
            reasoning_effort=args.terminus_reasoning_effort
            or manifest.get("terminus_reasoning_effort"),
            record_terminal_session=bool(
                args.terminus_record_terminal_session
                or manifest.get("terminus_record_terminal_session", False)
            ),
        )
    else:
        if job.candidate.candidate_dir is None:
            raise ValueError(f"Candidate {job.candidate.name} has no candidate_dir")
        summary = run_split(
            **common,
            candidate_dir=ROOT / job.candidate.candidate_dir,
            budget=job.candidate.budget or 0,
        )
    summary.update(
        {
            "candidate": job.candidate.name,
            "tier": job.tier,
            "planned_tasks": len(job.tasks),
            "planned_concurrency": job.concurrency,
        }
    )
    return summary


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tblite4x_catchup_{timestamp}"


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    raise SystemExit(main())
