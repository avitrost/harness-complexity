from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.run_codex_cli import run_codex_cli_split  # noqa: E402
from evaluator.run_terminus_2 import run_terminus_2_split  # noqa: E402
from evaluator.run_val import run_split  # noqa: E402
from evaluator.tblite import (  # noqa: E402
    TBLITE_CONCURRENCY,
    TBLITE_DATASET_ID,
    TBLITE_REVISION,
    TBLITE_SPLIT,
    TBLITE_TRIALS,
    discover_tblite_tasks,
    materialize_tblite,
    select_tblite_tasks,
)
from plumbing.codex_cli_agent import DEFAULT_TIMEOUT_SEC  # noqa: E402
from plumbing.openai_client import terminal_model, terminal_reasoning_effort  # noqa: E402
from plumbing.slurm_pyxis_environment import prepare_dockerfile_sqsh  # noqa: E402
from plumbing.terminus_2_agent import DEFAULT_TERMINUS_2_PARSER_NAME  # noqa: E402

DEFAULT_TBLITE_MAX_RETRIES = 2
DEFAULT_TBLITE_RETRY_EXCLUDE = ("VerifierTimeoutError",)
DEFAULT_TBLITE_VERIFIER_TIMEOUT_MULTIPLIER = 3.0


@dataclass(frozen=True)
class EvalCandidate:
    name: str
    category: str
    kind: str
    budget: int | None
    candidate_dir: Path | None = None


SEED_CANDIDATES = (
    EvalCandidate("seed_minimal_agent", "seed", "harness", None, Path("seeds/minimal_agent")),
    EvalCandidate("seed_codex_400", "seed", "harness", 400, Path("seeds/codex_400")),
    EvalCandidate("seed_codex_700", "seed", "harness", 700, Path("seeds/codex_700")),
    EvalCandidate("seed_codex_1000", "seed", "harness", 1000, Path("seeds/codex_1000")),
    EvalCandidate("seed_codex_1300", "seed", "harness", 1300, Path("seeds/codex_1300")),
    EvalCandidate(
        "seed_codex_compressed",
        "seed",
        "harness",
        None,
        Path("seeds/codex_compressed"),
    ),
    EvalCandidate("seed_codex_full", "seed", "harness", None, Path("seeds/codex_full")),
    EvalCandidate(
        "seed_mini_swe_agent_barebones",
        "seed",
        "harness",
        None,
        Path("seeds/mini_swe_agent_barebones"),
    ),
    EvalCandidate(
        "seed_mini_swe_agent_v2",
        "seed",
        "harness",
        None,
        Path("seeds/mini_swe_agent_v2"),
    ),
    EvalCandidate(
        "seed_terminus_2_compressed",
        "seed",
        "harness",
        None,
        Path("seeds/terminus_2_compressed"),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--trials", type=int, default=TBLITE_TRIALS)
    parser.add_argument("--combined-concurrency", type=int, default=TBLITE_CONCURRENCY)
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--cache-root", type=Path, default=Path("external_datasets"))
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--revision", default=TBLITE_REVISION)
    parser.add_argument("--download-method", choices=("git", "snapshot"), default="git")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--prebuild-slurm-images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prebuild-workers", type=int, default=4)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--candidate", action="append", dest="candidate_names")
    parser.add_argument("--include-codex-cli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-terminus-2", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--max-retries", type=int, default=DEFAULT_TBLITE_MAX_RETRIES)
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=DEFAULT_TBLITE_VERIFIER_TIMEOUT_MULTIPLIER,
    )
    parser.add_argument("--retry-include", action="append")
    parser.add_argument("--retry-exclude", action="append")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path or materialize_tblite(
        cache_root=args.cache_root,
        revision=args.revision,
        local_files_only=args.local_files_only,
        download_workers=args.download_workers,
        download_method=args.download_method,
    )
    available_tasks = discover_tblite_tasks(dataset_path)
    tasks = select_tblite_tasks(dataset_path, args.tasks)
    prebuilt_images = (
        _prebuild_slurm_images(dataset_path, tasks, args.prebuild_workers)
        if args.backend == "slurm-pyxis" and args.prebuild_slurm_images and not args.dry_run
        else []
    )
    candidates = _select_candidates(
        _all_candidates(
            args.include_codex_cli,
            args.include_terminus_2,
        ),
        args.candidate_names,
    )
    retry_include = tuple(args.retry_include or ())
    retry_exclude = tuple(
        args.retry_exclude if args.retry_exclude is not None else DEFAULT_TBLITE_RETRY_EXCLUDE
    )
    per_candidate_concurrency = max(1, args.combined_concurrency // len(candidates))
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id,
        "split": TBLITE_SPLIT,
        "dataset": TBLITE_DATASET_ID,
        "revision": args.revision,
        "download_method": args.download_method,
        "dataset_path": str(dataset_path),
        "available_task_count": len(available_tasks),
        "task_count": len(tasks),
        "tasks": tasks,
        "trials": args.trials,
        "combined_concurrency": args.combined_concurrency,
        "per_candidate_concurrency": per_candidate_concurrency,
        "actual_combined_concurrency": per_candidate_concurrency * len(candidates),
        "backend": args.backend,
        "max_retries": args.max_retries,
        "verifier_timeout_multiplier": args.verifier_timeout_multiplier,
        "retry_include": list(retry_include),
        "retry_exclude": list(retry_exclude),
        "codex_model": args.codex_model,
        "codex_reasoning_effort": args.codex_reasoning_effort,
        "terminus_model": args.terminus_model,
        "terminus_parser_name": args.terminus_parser_name,
        "terminus_reasoning_effort": args.terminus_reasoning_effort,
        "terminus_record_terminal_session": args.terminus_record_terminal_session,
        "prebuild_slurm_images": args.prebuild_slurm_images,
        "prebuild_workers": args.prebuild_workers,
        "prebuilt_image_count": len(prebuilt_images),
        "prebuilt_images": prebuilt_images,
        "candidates": [_candidate_manifest(candidate) for candidate in candidates],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        summaries = list(
            pool.map(
                lambda candidate: _run_candidate(
                    candidate,
                    root,
                    dataset_path,
                    tasks,
                    args.trials,
                    per_candidate_concurrency,
                    args.backend,
                    args.dry_run,
                    args.harbor_bin,
                    args.codex_model,
                    args.codex_reasoning_effort,
                    args.terminus_model,
                    args.terminus_parser_name,
                    args.terminus_reasoning_effort,
                    args.terminus_record_terminal_session,
                    args.timeout_sec,
                    args.max_retries,
                    args.verifier_timeout_multiplier,
                    retry_include,
                    retry_exclude,
                ),
                candidates,
            )
        )
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), "summaries": summaries}, indent=2))
    return 0 if all(item.get("ran", True) or args.dry_run for item in summaries) else 1


def _all_candidates(
    include_codex_cli: bool,
    include_terminus_2: bool,
) -> list[EvalCandidate]:
    candidates = list(SEED_CANDIDATES)
    if include_codex_cli:
        candidates.append(EvalCandidate("codex_cli", "baseline", "codex_cli", None))
    if include_terminus_2:
        candidates.append(EvalCandidate("terminus_2", "baseline", "terminus_2", None))
    return candidates


def _select_candidates(
    candidates: list[EvalCandidate],
    requested_names: list[str] | None,
) -> list[EvalCandidate]:
    if not requested_names:
        return candidates
    by_name = {candidate.name: candidate for candidate in candidates}
    missing = sorted(set(requested_names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in requested_names]


def _candidate_manifest(candidate: EvalCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "category": candidate.category,
        "kind": candidate.kind,
        "budget": candidate.budget,
        "candidate_dir": str(candidate.candidate_dir) if candidate.candidate_dir else None,
    }


def _prebuild_slurm_images(
    dataset_path: Path,
    tasks: list[str],
    workers: int,
) -> list[dict[str, str]]:
    workers = max(1, workers)

    def build(task: str) -> dict[str, str]:
        environment_dir = dataset_path / task / "environment"
        sqsh = prepare_dockerfile_sqsh(environment_dir)
        item = {"task": task, "sqsh": str(sqsh)}
        print(json.dumps({"prebuilt": item}), flush=True)
        return item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(build, tasks))


def _run_candidate(
    candidate: EvalCandidate,
    root: Path,
    dataset_path: Path,
    tasks: list[str],
    trials: int,
    concurrency: int,
    backend: str,
    dry_run: bool,
    harbor_bin: str | None,
    codex_model: str,
    codex_reasoning_effort: str,
    terminus_model: str,
    terminus_parser_name: str,
    terminus_reasoning_effort: str | None,
    terminus_record_terminal_session: bool,
    timeout_sec: int,
    max_retries: int,
    verifier_timeout_multiplier: float | None,
    retry_include: tuple[str, ...],
    retry_exclude: tuple[str, ...],
) -> dict[str, Any]:
    out_dir = root / candidate.name
    if candidate.kind == "codex_cli":
        return run_codex_cli_split(
            split=TBLITE_SPLIT,
            out_dir=out_dir,
            tasks=tasks,
            trials=trials,
            concurrency=concurrency,
            backend=backend,
            codex_model=codex_model,
            codex_reasoning_effort=codex_reasoning_effort,
            timeout_sec=timeout_sec,
            dry_run=dry_run,
            harbor_bin=harbor_bin,
            dataset=TBLITE_DATASET_ID,
            dataset_path=dataset_path,
            max_retries=max_retries,
            verifier_timeout_multiplier=verifier_timeout_multiplier,
            retry_include=retry_include,
            retry_exclude=retry_exclude,
        )
    if candidate.kind == "terminus_2":
        return run_terminus_2_split(
            split=TBLITE_SPLIT,
            out_dir=out_dir,
            tasks=tasks,
            trials=trials,
            concurrency=concurrency,
            backend=backend,
            terminus_model=terminus_model,
            parser_name=terminus_parser_name,
            reasoning_effort=terminus_reasoning_effort,
            record_terminal_session=terminus_record_terminal_session,
            dry_run=dry_run,
            harbor_bin=harbor_bin,
            dataset=TBLITE_DATASET_ID,
            dataset_path=dataset_path,
            max_retries=max_retries,
            verifier_timeout_multiplier=verifier_timeout_multiplier,
            retry_include=retry_include,
            retry_exclude=retry_exclude,
        )
    if candidate.candidate_dir is None:
        raise ValueError(f"Candidate {candidate.name} does not have a harness path")
    return run_split(
        split=TBLITE_SPLIT,
        candidate_dir=ROOT / candidate.candidate_dir,
        budget=candidate.budget or 0,
        out_dir=out_dir,
        tasks=tasks,
        trials=trials,
        concurrency=concurrency,
        dry_run=dry_run,
        harbor_bin=harbor_bin,
        backend=backend,
        dataset=TBLITE_DATASET_ID,
        dataset_path=dataset_path,
        max_retries=max_retries,
        verifier_timeout_multiplier=verifier_timeout_multiplier,
        retry_include=retry_include,
        retry_exclude=retry_exclude,
    )


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tblite5x_{timestamp}"


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    raise SystemExit(main())
