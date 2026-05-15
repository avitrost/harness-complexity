from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import check_terminal_model_available, using_codex_auth
from scripts.make_workspace import make_workspace

BUDGETS = {64, 128, 256, 512}
CODEX_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def optimize_budget(
    budget: int,
    cycles: int = 10,
    codex_model: str = "gpt-5.5",
    codex_reasoning_effort: str = "medium",
    dry_run: bool = False,
    codex_bin: str | None = None,
) -> list[dict[str, Any]]:
    if budget not in BUDGETS:
        raise ValueError(f"unsupported budget: {budget}")
    if not dry_run:
        _require_docker()
        _require_terminal_model()
    budget_dir = Path(f"experience/B{budget:04d}")
    budget_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for cycle in range(1, cycles + 1):
        iter_dir = budget_dir / f"iter_{cycle:03d}"
        if iter_dir.exists():
            shutil.rmtree(iter_dir)
        iter_dir.mkdir(parents=True, exist_ok=True)
        workspace = iter_dir / "workspace"
        with tempfile.TemporaryDirectory(
            prefix=f"harness_complexity_B{budget:04d}_iter_{cycle:03d}_"
        ) as temp_dir:
            codex_workspace = Path(temp_dir) / "workspace"
            make_workspace(codex_workspace, Path("candidate"))
            command = build_codex_command(
                budget,
                codex_model,
                codex_reasoning_effort,
                repair=False,
                codex_bin=codex_bin,
            )
            (iter_dir / "codex_command.json").write_text(
                json.dumps(command, indent=2), encoding="utf-8"
            )
            if not dry_run:
                _run_codex(command, codex_workspace)
            _copy_workspace(codex_workspace, workspace)
            validation = validate_candidate(workspace / "candidate" / "harness.py", budget)
            for repair in range(1, 3):
                if validation["ok"] or dry_run:
                    break
                repair_command = build_codex_command(
                    budget,
                    codex_model,
                    codex_reasoning_effort,
                    repair=True,
                    codex_bin=codex_bin,
                )
                _run_codex(repair_command, codex_workspace)
                _copy_workspace(codex_workspace, workspace)
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
        reports.append({"iteration": cycle, "workspace": str(workspace), "valid": validation["ok"]})
    return reports


def build_codex_command(
    budget: int,
    codex_model: str,
    codex_reasoning_effort: str,
    repair: bool,
    codex_bin: str | None = None,
) -> list[str]:
    prompt = (
        "You are in an isolated workspace. Edit only candidate/harness.py and proposal.md. "
        "Do not inspect parent directories, absolute repository paths, or prior experiment "
        f"artifacts. Keep candidate/harness.py at most {budget} Black-formatted physical "
        "lines. Improve general TerminalBench behavior without task-specific hacks."
    )
    if repair:
        prompt = f"Repair validation failures. {prompt}"
    return [
        resolve_codex_executable(codex_bin),
        "exec",
        "--model",
        codex_model,
        "-c",
        f'model_reasoning_effort="{codex_reasoning_effort}"',
        "--sandbox",
        "danger-full-access",
        "--ephemeral",
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


def _copy_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required for validation but was not found on PATH. "
            "Install/start Docker Desktop, or rerun with --dry-run."
        )


def _require_openai_api_key() -> None:
    if not using_codex_auth() and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for validation because candidate harnesses call "
            "the fixed terminal model through plumbing.openai_client."
        )


def _require_terminal_model() -> None:
    _require_openai_api_key()
    try:
        check_terminal_model_available()
    except Exception as exc:
        raise RuntimeError(
            "Terminal model preflight failed. Check OPENAI_API_KEY or Codex auth, "
            "quota, and model access before running optimization."
        ) from exc


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
    parser.add_argument("--codex-model", default="gpt-5.5")
    parser.add_argument(
        "--codex-reasoning-effort", choices=CODEX_REASONING_EFFORTS, default="medium"
    )
    parser.add_argument("--codex-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        reports = optimize_budget(
            args.budget,
            args.cycles,
            args.codex_model,
            args.codex_reasoning_effort,
            args.dry_run,
            args.codex_bin,
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
