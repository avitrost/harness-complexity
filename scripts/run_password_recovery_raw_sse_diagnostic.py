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

from evaluator.run_val import run_split  # noqa: E402
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET  # noqa: E402
from scripts.run_tb2_core import TB2_CORE_SPLIT  # noqa: E402

DEFAULT_OUT_ROOT = Path("/wbl-fast/usrs/trost/harness-complexity/final_test")
DEFAULT_PARTITION = "m7i-cpu2"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--slurm-partition", default=DEFAULT_PARTITION)
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task", default="password-recovery")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--retry-limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.retry_limit < 0:
        raise ValueError("--retry-limit must be >= 0")
    if args.backend == "slurm-pyxis" and not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to run Harbor/evals outside Slurm. Use sbatch, salloc, or srun.")

    if args.backend == "slurm-pyxis" and args.slurm_partition:
        os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition

    root = args.out_root / args.run_id
    out_dir = (
        root
        / args.model.replace(".", "_").replace("-", "_")
        / f"mini_v2_r{args.retry_limit}"
        / args.task
    )
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "task": args.task,
        "trials": args.trials,
        "candidate": "seed_mini_swe_agent_v2",
        "retry_limit": args.retry_limit,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
        "raw_sse_trace_dir": "__TRACE_DIR__/raw_sse",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = run_split(
        split=TB2_CORE_SPLIT,
        candidate_dir=ROOT / "seeds/mini_swe_agent_v2",
        budget=478,
        out_dir=out_dir,
        tasks=[args.task],
        trials=args.trials,
        concurrency=1,
        dry_run=args.dry_run,
        harbor_bin=args.harbor_bin,
        backend=args.backend,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        max_retries=0,
        agent_env=(
            "TERMINAL_MODEL_PROVIDER=openai",
            f"OPENAI_TERMINAL_MODEL={args.model}",
            f"OPENAI_TERMINAL_REASONING_EFFORT={args.reasoning_effort}",
            "OPENAI_AUTH_MODE=codex",
            f"MINI_SWE_FORMAT_RETRY_LIMIT={args.retry_limit}",
            "CODEX_RAW_SSE_TRACE_DIR=__TRACE_DIR__/raw_sse",
        ),
    )
    raw_files = sorted(str(path) for path in out_dir.glob("*/**/agent/raw_sse/*"))
    diagnostic_summary = {
        **manifest,
        "root": str(root),
        "out_dir": str(out_dir),
        "summary": summary,
        "raw_sse_files": raw_files,
    }
    (root / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostic_summary, indent=2))
    return int(summary.get("returncode", 0))


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"password_recovery_raw_sse_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
