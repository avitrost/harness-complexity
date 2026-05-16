from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.bootstrap_ci import bootstrap_ci

DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run budgets sequentially. To parallelize manually, run optimize_budget for each "
            "budget in a separate shell, then rerun this command with --skip-optimization."
        )
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--codex-model", default="gpt-5.5")
    parser.add_argument("--terminal-model", default="gpt-5.4-mini")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--codex-bin")
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="docker")
    parser.add_argument("--k", type=int, default=2, dest="candidates_per_iteration")
    parser.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--run-final-test", action="store_true")
    args = parser.parse_args()
    budgets = _parse_budgets(args.budgets)
    if args.concurrency is not None and args.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if (
        args.resume
        and not args.dry_run
        and not any(_run_dir(budget, args.dry_run, run_id).exists() for budget in budgets)
    ):
        raise RuntimeError(f"no matching run directories found for --run-id {run_id}")
    os.environ["OPENAI_TERMINAL_MODEL"] = args.terminal_model
    if not args.skip_optimization:
        for budget in budgets:
            _run(
                [
                    sys.executable,
                    "-m",
                    "evaluator.optimize_budget",
                    "--budget",
                    str(budget),
                    "--cycles",
                    str(args.cycles),
                    "--codex-model",
                    args.codex_model,
                    "--codex-reasoning-effort",
                    args.codex_reasoning_effort,
                    "--backend",
                    args.backend,
                    "--k",
                    str(args.candidates_per_iteration),
                    "--run-id",
                    run_id,
                    *(
                        ("--resume",)
                        if args.resume and _run_dir(budget, args.dry_run, run_id).exists()
                        else ()
                    ),
                    *(
                        ("--concurrency", str(args.concurrency))
                        if args.concurrency is not None
                        else ()
                    ),
                    *(("--codex-bin", args.codex_bin) if args.codex_bin else ()),
                    *(("--dry-run",) if args.dry_run else ()),
                ]
            )
    if args.dry_run:
        return 0
    _clear_selection_outputs(Path("results"))
    for budget in budgets:
        _run(
            [
                sys.executable,
                "scripts/select_best.py",
                "--budget-dir",
                f"experience/B{budget:04d}",
                "--out-dir",
                "results",
                "--run-id",
                run_id,
            ]
        )
    if args.run_final_test:
        for row in _selected_rows(Path("results/selected_candidates.json")):
            budget = int(row["budget"])
            final_dir = Path(f"final_test/B{budget:04d}")
            _run(
                [
                    sys.executable,
                    "-m",
                    "evaluator.run_test",
                    "--candidate-dir",
                    str(row["candidate_dir"]),
                    "--budget",
                    str(budget),
                    "--out-dir",
                    str(final_dir),
                    "--backend",
                    args.backend,
                    *(
                        ("--concurrency", str(args.concurrency))
                        if args.concurrency is not None
                        else ()
                    ),
                    *(("--dry-run",) if args.dry_run else ()),
                ]
            )
            if not args.dry_run:
                _write_bootstrap(final_dir)
    _run([sys.executable, "scripts/plot_complexity_curve.py", "--run-id", run_id])
    return 0


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one budget is required")
    return budgets


def _run_dir(budget: int, dry_run: bool, run_id: str) -> Path:
    prefix = "dry_run" if dry_run else "run"
    return Path(f"experience/B{budget:04d}/{prefix}_{run_id}")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _clear_selection_outputs(out_dir: Path) -> None:
    for name in (
        "selected_candidates.json",
        "selected_candidates.csv",
        "pareto_frontier.json",
        "pareto_frontier.csv",
    ):
        (out_dir / name).unlink(missing_ok=True)


def _selected_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if isinstance(payload, dict):
        return [payload] if payload else []
    return [item for item in payload if isinstance(item, dict)]


def _write_bootstrap(final_dir: Path) -> None:
    records_path = final_dir / "records.json"
    if not records_path.exists():
        return
    records = json.loads(records_path.read_text(encoding="utf-8"))
    ci = bootstrap_ci(records)
    (final_dir / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
