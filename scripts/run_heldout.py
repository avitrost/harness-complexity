from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.run_val import run_split  # noqa: E402
from evaluator.splits import HELDOUT_CONCURRENCY, HELDOUT_TRIALS, get_heldout_tasks  # noqa: E402

BEST_CANDIDATES = [
    (
        128,
        "iter_008_cand_01",
        Path("experience/B0128/run_combined20_20260516_010740/iter_008_cand_01/workspace"),
    ),
    (
        256,
        "iter_004_cand_01",
        Path("experience/B0256/run_combined20_20260516_010740/iter_004_cand_01/workspace"),
    ),
    (
        512,
        "iter_003_cand_02",
        Path("experience/B0512/run_combined20_20260516_010740/iter_003_cand_02/workspace"),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--trials", type=int, default=HELDOUT_TRIALS)
    parser.add_argument("--combined-concurrency", type=int, default=HELDOUT_CONCURRENCY)
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = get_heldout_tasks()
    per_candidate_concurrency = max(1, args.combined_concurrency // len(BEST_CANDIDATES))
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id,
        "split": "heldout",
        "tasks": tasks,
        "trials": args.trials,
        "combined_concurrency": args.combined_concurrency,
        "per_candidate_concurrency": per_candidate_concurrency,
        "candidates": [
            {"budget": budget, "candidate": name, "workspace": str(path)}
            for budget, name, path in BEST_CANDIDATES
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=len(BEST_CANDIDATES)) as pool:
        summaries = list(
            pool.map(
                lambda candidate: _run_candidate(
                    candidate,
                    root,
                    tasks,
                    args.trials,
                    per_candidate_concurrency,
                    args.backend,
                    args.dry_run,
                ),
                BEST_CANDIDATES,
            )
        )
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), "summaries": summaries}, indent=2))
    return 0 if all(item.get("ran", True) or args.dry_run for item in summaries) else 1


def _run_candidate(
    candidate: tuple[int, str, Path],
    root: Path,
    tasks: list[str],
    trials: int,
    concurrency: int,
    backend: str,
    dry_run: bool,
) -> dict[str, Any]:
    budget, name, workspace = candidate
    out_dir = root / f"B{budget:04d}_{name}"
    return run_split(
        "heldout",
        workspace,
        budget,
        out_dir,
        tasks,
        trials,
        concurrency,
        dry_run,
        backend=backend,
    )


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"heldout_combined20_{timestamp}"


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    raise SystemExit(main())
