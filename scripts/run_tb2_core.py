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
from evaluator.splits import (  # noqa: E402
    TB2_CORE_CONCURRENCY,
    TB2_CORE_TRIALS,
    get_tb2_core_tasks,
)
from plumbing.codex_cli_agent import DEFAULT_TIMEOUT_SEC  # noqa: E402
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET  # noqa: E402
from plumbing.openai_client import terminal_model, terminal_reasoning_effort  # noqa: E402
from plumbing.terminus_2_agent import DEFAULT_TERMINUS_2_PARSER_NAME  # noqa: E402

TB2_CORE_SPLIT = "tb2-core"


@dataclass(frozen=True)
class EvalCandidate:
    name: str
    kind: str
    candidate_dir: Path | None = None
    loc: int | None = None


SEED_CANDIDATES = (
    EvalCandidate("seed_minimal_agent", "harness", Path("seeds/minimal_agent"), 100),
    EvalCandidate("seed_codex_400", "harness", Path("seeds/codex_400"), 400),
    EvalCandidate("seed_codex_700", "harness", Path("seeds/codex_700"), 700),
    EvalCandidate("seed_codex_1000", "harness", Path("seeds/codex_1000"), 1000),
    EvalCandidate("seed_codex_1300", "harness", Path("seeds/codex_1300"), 1300),
    EvalCandidate("seed_codex_compressed", "harness", Path("seeds/codex_compressed"), 1660),
    EvalCandidate("seed_codex_full", "harness", Path("seeds/codex_full"), 2210),
    EvalCandidate(
        "seed_codex_full_minimal_prompt",
        "harness",
        Path("seeds/codex_full_minimal_prompt"),
        2210,
    ),
    EvalCandidate(
        "seed_codex_full_minimal_surfaces",
        "harness",
        Path("seeds/codex_full_minimal_surfaces"),
        2210,
    ),
    EvalCandidate(
        "seed_mini_swe_agent_barebones",
        "harness",
        Path("seeds/mini_swe_agent_barebones"),
        149,
    ),
    EvalCandidate("seed_mini_swe_agent_v2", "harness", Path("seeds/mini_swe_agent_v2"), 478),
    EvalCandidate(
        "seed_mini_swe_agent_barebones_v2",
        "harness",
        Path("seeds/mini_swe_agent_barebones_v2"),
        478,
    ),
    EvalCandidate(
        "seed_mini_swe_agent_barebones_v2_codex_prompt",
        "harness",
        Path("seeds/mini_swe_agent_barebones_v2_codex_prompt"),
        478,
    ),
    EvalCandidate(
        "seed_mini_swe_agent_barebones_v2_persistent",
        "harness",
        Path("seeds/mini_swe_agent_barebones_v2_persistent"),
        478,
    ),
    EvalCandidate(
        "seed_mini_swe_agent_barebones_v2_rich_terminal",
        "harness",
        Path("seeds/mini_swe_agent_barebones_v2_rich_terminal"),
        478,
    ),
    EvalCandidate(
        "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples",
        "harness",
        Path("seeds/mini_swe_agent_barebones_v2_rich_terminal_no_examples"),
        478,
    ),
    EvalCandidate(
        "seed_terminus_2_compressed", "harness", Path("seeds/terminus_2_compressed"), 634
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--trials", type=int, default=TB2_CORE_TRIALS)
    parser.add_argument("--concurrency", type=int, default=TB2_CORE_CONCURRENCY)
    parser.add_argument("--max-candidate-workers", type=int, default=1)
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
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--retry-include", action="append", default=[])
    parser.add_argument("--retry-exclude", action="append", default=[])
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_candidate_workers < 1:
        raise ValueError("--max-candidate-workers must be >= 1")

    tasks = args.tasks or get_tb2_core_tasks()
    candidates = _select_candidates(
        _all_candidates(args.include_codex_cli, args.include_terminus_2),
        args.candidate_names,
    )
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id,
        "split": TB2_CORE_SPLIT,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "tasks": tasks,
        "trials": args.trials,
        "concurrency_per_candidate": args.concurrency,
        "max_candidate_workers": args.max_candidate_workers,
        "backend": args.backend,
        "codex_model": args.codex_model,
        "codex_reasoning_effort": args.codex_reasoning_effort,
        "terminus_model": args.terminus_model,
        "terminus_reasoning_effort": args.terminus_reasoning_effort,
        "candidates": [_candidate_manifest(candidate) for candidate in candidates],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    worker_count = min(args.max_candidate_workers, len(candidates))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        summaries = list(
            pool.map(
                lambda candidate: _run_candidate(candidate, root, tasks, args),
                candidates,
            )
        )
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), "summaries": summaries}, indent=2))
    return 0 if all(item.get("ran", True) or args.dry_run for item in summaries) else 1


def _all_candidates(include_codex_cli: bool, include_terminus_2: bool) -> list[EvalCandidate]:
    candidates = list(SEED_CANDIDATES)
    if include_codex_cli:
        candidates.append(EvalCandidate("codex_cli", "codex_cli"))
    if include_terminus_2:
        candidates.append(EvalCandidate("terminus_2", "terminus_2"))
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
        "kind": candidate.kind,
        "candidate_dir": str(candidate.candidate_dir) if candidate.candidate_dir else None,
        "loc": candidate.loc,
    }


def _run_candidate(
    candidate: EvalCandidate,
    root: Path,
    tasks: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = root / candidate.name
    common = {
        "split": TB2_CORE_SPLIT,
        "out_dir": out_dir,
        "tasks": tasks,
        "trials": args.trials,
        "concurrency": args.concurrency,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "harbor_bin": args.harbor_bin,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "max_retries": args.max_retries,
        "verifier_timeout_multiplier": args.verifier_timeout_multiplier,
        "retry_include": tuple(args.retry_include),
        "retry_exclude": tuple(args.retry_exclude),
    }
    if candidate.kind == "codex_cli":
        return run_codex_cli_split(
            **common,
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            timeout_sec=args.timeout_sec,
        )
    if candidate.kind == "terminus_2":
        return run_terminus_2_split(
            **common,
            terminus_model=args.terminus_model,
            parser_name=args.terminus_parser_name,
            reasoning_effort=args.terminus_reasoning_effort,
            record_terminal_session=args.terminus_record_terminal_session,
        )
    if candidate.candidate_dir is None:
        raise ValueError(f"Candidate {candidate.name} has no candidate_dir")
    return run_split(
        **common,
        candidate_dir=ROOT / candidate.candidate_dir,
        budget=candidate.loc or 0,
        agent_env=(
            f"OPENAI_TERMINAL_MODEL={args.codex_model}",
            f"OPENAI_TERMINAL_REASONING_EFFORT={args.codex_reasoning_effort}",
        ),
    )


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tb2_core_{timestamp}"


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    raise SystemExit(main())
