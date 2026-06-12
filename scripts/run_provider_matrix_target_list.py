from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_provider_matrix as matrix  # noqa: E402
from scripts.run_tb2_core import SEED_CANDIDATES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset", default=matrix.TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--slurm-partition", default="m7i-cpu2")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--verifier-timeout-multiplier", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if not args.dry_run and args.backend == "slurm-pyxis" and not os.environ.get("SLURM_JOB_ID"):
        print("Refusing to run heavy work outside Slurm. Use sbatch, salloc, or srun first.", file=sys.stderr)
        return 2

    targets = _read_targets(args.targets_json)
    selected = matrix._shard_attempts(targets, args.shard_count, args.shard_index)
    os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition
    harbor_help_text = matrix._harbor_help_text(args.harbor_bin)
    print(
        f"[provider-matrix-targets] starting {len(selected)}/{len(targets)} attempts "
        f"from {args.targets_json} at concurrency {args.concurrency}",
        flush=True,
    )
    run_args = argparse.Namespace(
        backend=args.backend,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        dry_run=args.dry_run,
        harbor_bin=args.harbor_bin,
        max_retries=args.max_retries,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        concurrency=args.concurrency,
        slurm_partition=args.slurm_partition,
    )
    summaries = matrix._run_attempt_pool(args.run_root, run_args, selected, harbor_help_text)
    summary = {
        "run_root": str(args.run_root),
        "targets_json": str(args.targets_json),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "targets_total": len(targets),
        "targets_selected": len(selected),
        "attempts_completed": len(summaries),
    }
    out_dir = args.run_root / "_target_runs" / _target_run_id(args.targets_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"shard_{args.shard_index:04d}.json").write_text(
        json.dumps({"summary": summary, "attempts": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _read_targets(path: Path) -> list[matrix.MatrixAttempt]:
    candidates = {candidate.name: candidate for candidate in SEED_CANDIDATES}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["targets"] if isinstance(payload, dict) and "targets" in payload else payload
    attempts = []
    for row in rows:
        candidate_name = str(row["candidate"])
        candidate = candidates.get(candidate_name)
        if candidate is None:
            raise ValueError(f"unknown candidate in targets: {candidate_name}")
        attempts.append(
            matrix.MatrixAttempt(
                provider=str(row["provider"]),
                config_id=str(row["config_id"]),
                model=str(row["model"]),
                effort=str(row["effort"]),
                candidate=candidate,
                task=str(row["task"]),
                attempt=int(row["attempt"]),
            )
        )
    return attempts


def _target_run_id(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
