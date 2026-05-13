from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.bootstrap_ci import bootstrap_ci

BUDGETS = (64, 128, 256, 512)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run budgets sequentially. To parallelize manually, run optimize_budget for each "
            "budget in a separate shell, then rerun this command with --skip-optimization."
        )
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--codex-model", default="gpt-5.5-medium")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-optimization", action="store_true")
    args = parser.parse_args()
    if not args.skip_optimization:
        for budget in BUDGETS:
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
                    *(("--dry-run",) if args.dry_run else ()),
                ]
            )
    for budget in BUDGETS:
        _run(
            [
                sys.executable,
                "scripts/select_best.py",
                "--budget-dir",
                f"experience/B{budget:04d}",
                "--out-dir",
                "results",
            ]
        )
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
                *(("--dry-run",) if args.dry_run else ()),
            ]
        )
        if not args.dry_run:
            _write_bootstrap(final_dir)
    _run([sys.executable, "scripts/plot_complexity_curve.py"])
    return 0


def _run(command: list[str]) -> None:
    subprocess.run(command, check=False)


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
