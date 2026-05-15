from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluator.run_val import BACKENDS
from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import check_terminal_model_available, using_codex_auth
from scripts.count_loc import count_loc
from scripts.make_workspace import make_workspace

BUDGETS = (128, 256, 512, 1024, 2048)
CODEX_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
DEFAULT_K = 2


def optimize_budget(
    budget: int,
    cycles: int = 10,
    codex_model: str = "gpt-5.5",
    codex_reasoning_effort: str = "medium",
    dry_run: bool = False,
    codex_bin: str | None = None,
    backend: str = "docker",
    candidates_per_iteration: int = DEFAULT_K,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    if budget not in BUDGETS:
        raise ValueError(f"unsupported budget: {budget}")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if candidates_per_iteration < 1:
        raise ValueError("candidates_per_iteration must be >= 1")
    if not dry_run:
        _require_backend(backend)
        _require_terminal_model()
    budget_dir = Path(f"experience/B{budget:04d}")
    budget_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _new_run_dir(budget_dir, dry_run, run_id)
    reports = []
    seed_report = _ensure_seed_candidate(run_dir, budget, dry_run, backend)
    if seed_report:
        reports.append(seed_report)
    for iteration in range(1, cycles + 1):
        history_dirs = _history_dirs(run_dir)
        for candidate_index in range(1, candidates_per_iteration + 1):
            iter_dir = _proposal_dir(run_dir, iteration, candidate_index)
            if iter_dir.exists():
                shutil.rmtree(iter_dir)
            iter_dir.mkdir(parents=True, exist_ok=True)
            workspace = iter_dir / "workspace"
            with tempfile.TemporaryDirectory(
                prefix=(
                    f"harness_complexity_B{budget:04d}_iter_{iteration:03d}_"
                    f"cand_{candidate_index:02d}_"
                )
            ) as temp_dir:
                codex_workspace = Path(temp_dir) / "workspace"
                make_workspace(codex_workspace, Path("candidate"), history_dirs)
                _pad_seed_to_bucket(codex_workspace / "candidate" / "harness.py", budget)
                _write_history_index(codex_workspace / "history", history_dirs)
                command = build_codex_command(
                    budget,
                    codex_model,
                    codex_reasoning_effort,
                    repair=False,
                    codex_bin=codex_bin,
                    iteration=iteration,
                    candidate_index=candidate_index,
                    candidates_per_iteration=candidates_per_iteration,
                )
                _write_json(iter_dir / "codex_command.json", command)
                _write_meta(iter_dir, budget, iteration, candidate_index, "proposal")
                if not dry_run:
                    _run_codex(command, codex_workspace, iter_dir)
                _copy_workspace(codex_workspace, workspace)
                _strip_workspace_history(workspace)
                validation = validate_candidate(
                    workspace / "candidate" / "harness.py",
                    max_lines=budget,
                    min_lines=_budget_min_lines(budget),
                )
            _write_json(iter_dir / "validation.json", validation, sort_keys=True)
            if validation["ok"] and not dry_run:
                _run_val(workspace, budget, iter_dir, backend)
            else:
                summary = {
                    "split": "val",
                    "split_mean": 0.0,
                    "invalid": not validation["ok"],
                    "dry_run": dry_run,
                }
                _write_json(iter_dir / "summary.json", summary)
            reports.append(
                {
                    "iteration": iteration,
                    "candidate": candidate_index,
                    "workspace": str(workspace),
                    "valid": validation["ok"],
                }
            )
    return reports


def build_codex_command(
    budget: int,
    codex_model: str,
    codex_reasoning_effort: str,
    repair: bool,
    codex_bin: str | None = None,
    iteration: int | None = None,
    candidate_index: int | None = None,
    candidates_per_iteration: int = DEFAULT_K,
) -> list[str]:
    slot = (
        f" This is iteration {iteration}, candidate {candidate_index} of "
        f"{candidates_per_iteration}."
        if iteration is not None and candidate_index is not None
        else ""
    )
    min_lines = _budget_min_lines(budget)
    line_rule = (
        f"Keep candidate/harness.py between {min_lines} and {budget}"
        if min_lines > 1
        else f"Keep candidate/harness.py at most {budget}"
    )
    prompt = (
        "You are in an isolated Meta-Harness workspace."
        f"{slot} Read history/ first: it contains prior candidate source, proposals, "
        "validation, scores, and terminal traces for this budget. Edit only "
        "candidate/harness.py and proposal.md. Do not inspect parent directories, "
        "absolute repository paths, experience/ directly, final_test/, or results/. "
        f"{line_rule} Black-formatted physical lines. "
        "Improve general TerminalBench behavior without task-specific hacks. Propose "
        "one new harness; you may base it on any prior harness in history/."
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


def _ensure_seed_candidate(
    run_dir: Path,
    budget: int,
    dry_run: bool,
    backend: str,
) -> dict[str, Any] | None:
    iter_dir = run_dir / "iter_000_seed"
    summary = _read_json(iter_dir / "summary.json")
    if (
        (iter_dir / "validation.json").exists()
        and summary
        and (iter_dir / "workspace" / "candidate" / "harness.py").exists()
        and not (summary.get("dry_run") and not dry_run)
    ):
        return None
    workspace = iter_dir / "workspace"
    if iter_dir.exists():
        shutil.rmtree(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)
    make_workspace(workspace, Path("candidate"))
    _pad_seed_to_bucket(workspace / "candidate" / "harness.py", budget)
    _strip_workspace_history(workspace)
    _write_meta(iter_dir, budget, 0, 0, "seed")
    validation = validate_candidate(
        workspace / "candidate" / "harness.py",
        max_lines=budget,
        min_lines=_budget_min_lines(budget),
    )
    _write_json(iter_dir / "validation.json", validation, sort_keys=True)
    if validation["ok"] and not dry_run:
        _run_val(workspace, budget, iter_dir, backend)
    else:
        summary = {
            "split": "val",
            "split_mean": 0.0,
            "invalid": not validation["ok"],
            "dry_run": dry_run,
        }
        _write_json(iter_dir / "summary.json", summary)
    return {"iteration": 0, "candidate": 0, "workspace": str(workspace), "valid": validation["ok"]}


def _proposal_dir(budget_dir: Path, iteration: int, candidate_index: int) -> Path:
    return budget_dir / f"iter_{iteration:03d}_cand_{candidate_index:02d}"


def _budget_min_lines(budget: int) -> int:
    ordered = sorted(BUDGETS)
    index = ordered.index(budget)
    return 1 if index == 0 else ordered[index - 1] + 1


def _pad_seed_to_bucket(path: Path, budget: int) -> None:
    min_lines = _budget_min_lines(budget)
    while (needed := min_lines - int(count_loc(path)["physical_loc"])) > 0:
        text = path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n".join("# bucket padding" for _ in range(needed)) + "\n"
        path.write_text(text, encoding="utf-8")


def _new_run_dir(budget_dir: Path, dry_run: bool, run_id: str | None) -> Path:
    prefix = "dry_run" if dry_run else "run"
    label = _clean_run_id(run_id) if run_id else datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = budget_dir / f"{prefix}_{label}"
    if run_id and run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = budget_dir / f"{prefix}_{label}_{suffix:02d}"
    run_dir.mkdir(parents=True)
    (budget_dir / "latest_run.txt").write_text(f"{run_dir.name}\n", encoding="utf-8")
    return run_dir


def _clean_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("run_id must contain at least one safe path character")
    return cleaned


def _run_codex(command: list[str], workspace: Path, iter_dir: Path) -> None:
    result = subprocess.run(command, cwd=workspace, check=False, capture_output=True, text=True)
    (iter_dir / "codex_stdout.log").write_text(result.stdout, encoding="utf-8")
    (iter_dir / "codex_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI exited with status {result.returncode}")


def _copy_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))


def _strip_workspace_history(workspace: Path) -> None:
    shutil.rmtree(workspace / "history", ignore_errors=True)


def _history_dirs(budget_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(budget_dir.iterdir())
        if path.is_dir()
        and _read_json(path / "validation.json").get("ok") is True
        and (path / "summary.json").exists()
        and _read_json(path / "summary.json").get("dry_run") is not True
        and (path / "workspace" / "candidate" / "harness.py").exists()
    ]


def _write_history_index(history_dir: Path, history_dirs: list[Path]) -> None:
    rows = []
    for path in history_dirs:
        summary = _read_json(path / "summary.json")
        validation = _read_json(path / "validation.json")
        count = _extract_count(validation)
        rows.append(
            {
                "dir": path.name,
                "split_mean": summary.get("split_mean"),
                "estimated_full_score": summary.get("estimated_full_score"),
                "num_trials": summary.get("num_trials"),
                "num_crashes": summary.get("num_crashes"),
                "physical_loc": count.get("physical_loc"),
                "valid": validation.get("ok", False),
            }
        )
    _write_json(history_dir / "index.json", rows, sort_keys=True)


def _extract_count(validation: dict[str, Any]) -> dict[str, Any]:
    for check in validation.get("checks", []):
        data = check.get("json")
        if isinstance(data, dict) and "physical_loc" in data:
            return data
    return {}


def _write_meta(
    iter_dir: Path,
    budget: int,
    iteration: int,
    candidate_index: int,
    role: str,
) -> None:
    _write_json(
        iter_dir / "candidate_meta.json",
        {
            "budget": budget,
            "iteration": iteration,
            "candidate": candidate_index,
            "role": role,
            "default_k": DEFAULT_K,
        },
        sort_keys=True,
    )


def _write_json(path: Path, payload: Any, sort_keys: bool = False) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _require_backend(backend: str) -> None:
    if backend == "docker" and shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required for validation but was not found on PATH. "
            "Install/start Docker Desktop, or rerun with --dry-run."
        )
    if backend == "slurm-pyxis":
        missing = [name for name in ("srun", "enroot") if shutil.which(name) is None]
        if missing:
            raise RuntimeError(f"Slurm/Pyxis backend missing: {', '.join(missing)}")


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


def _run_val(workspace: Path, budget: int, iter_dir: Path, backend: str) -> None:
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
            "--backend",
            backend,
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
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="docker")
    parser.add_argument("--k", type=int, default=DEFAULT_K, dest="candidates_per_iteration")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        reports = optimize_budget(
            budget=args.budget,
            cycles=args.cycles,
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            dry_run=args.dry_run,
            codex_bin=args.codex_bin,
            backend=args.backend,
            candidates_per_iteration=args.candidates_per_iteration,
            run_id=args.run_id,
        )
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
