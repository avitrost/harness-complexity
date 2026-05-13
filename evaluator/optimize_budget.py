from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluator.validate_candidate import validate_candidate
from scripts.make_workspace import make_workspace

BUDGETS = {64, 128, 256, 512}


def optimize_budget(
    budget: int,
    cycles: int = 10,
    codex_model: str = "gpt-5.5-medium",
    dry_run: bool = False,
    codex_bin: str | None = None,
) -> list[dict[str, Any]]:
    if budget not in BUDGETS:
        raise ValueError(f"unsupported budget: {budget}")
    budget_dir = Path(f"experience/B{budget:04d}")
    budget_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    source = Path("seeds/seed_minimal.py")
    for cycle in range(1, cycles + 1):
        iter_dir = budget_dir / f"iter_{cycle:03d}"
        workspace = iter_dir / "workspace"
        make_workspace(workspace, source)
        command = build_codex_command(
            workspace, budget, codex_model, repair=False, codex_bin=codex_bin
        )
        (iter_dir / "codex_command.json").write_text(
            json.dumps(command, indent=2), encoding="utf-8"
        )
        if not dry_run:
            _run_codex(command, workspace)
        validation = validate_candidate(workspace / "candidate" / "harness.py", budget)
        for repair in range(1, 3):
            if validation["ok"] or dry_run:
                break
            repair_command = build_codex_command(
                workspace,
                budget,
                codex_model,
                repair=True,
                codex_bin=codex_bin,
            )
            _run_codex(repair_command, workspace)
            validation = validate_candidate(workspace / "candidate" / "harness.py", budget)
        (iter_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if validation["ok"] and not dry_run:
            _run_val(workspace, budget, iter_dir)
        else:
            summary = {"split": "val", "split_mean": 0.0, "invalid": not validation["ok"]}
            (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        source = workspace / "candidate" / "harness.py"
        reports.append({"iteration": cycle, "workspace": str(workspace), "valid": validation["ok"]})
    return reports


def build_codex_command(
    workspace: Path,
    budget: int,
    codex_model: str,
    repair: bool,
    codex_bin: str | None = None,
) -> list[str]:
    prompt = (
        f"Edit only candidate/harness.py and proposal.md. Keep candidate/harness.py at most "
        f"{budget} Black-formatted physical lines. Improve general TerminalBench behavior "
        "without task-specific hacks."
    )
    if repair:
        prompt = f"Repair validation failures. {prompt}"
    return [
        resolve_codex_executable(codex_bin),
        "exec",
        "--model",
        codex_model,
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        prompt,
    ]


def resolve_codex_executable(codex_bin: str | None = None) -> str:
    if codex_bin:
        resolved = shutil.which(codex_bin) or codex_bin
        if Path(resolved).exists() or shutil.which(resolved):
            return resolved
        raise RuntimeError(f"Codex CLI not found: {codex_bin}")
    for name in ("codex.cmd", "codex.exe", "codex"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise RuntimeError(
        "Codex CLI was not found. Install/authenticate Codex CLI or pass "
        "--codex-bin C:\\path\\to\\codex.cmd."
    )


def _run_codex(command: list[str], workspace: Path) -> None:
    result = subprocess.run(command, cwd=workspace, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI exited with status {result.returncode}")


def _run_val(workspace: Path, budget: int, iter_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluator.run_val",
            "--candidate-dir",
            str(workspace),
            "--budget",
            str(budget),
            "--out-dir",
            str(iter_dir),
        ],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, choices=sorted(BUDGETS), required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--codex-model", default="gpt-5.5-medium")
    parser.add_argument("--codex-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reports = optimize_budget(
        args.budget,
        args.cycles,
        args.codex_model,
        args.dry_run,
        args.codex_bin,
    )
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
