from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

from evaluator.history_artifacts import write_history_artifacts
from evaluator.run_val import BACKENDS
from evaluator.splits import VAL_CONCURRENCY, VAL_TRIALS, get_val_tasks
from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import (
    check_terminal_model_available,
    terminal_model,
    using_codex_auth,
)
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
    resume: bool = False,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    if budget not in BUDGETS:
        raise ValueError(f"unsupported budget: {budget}")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if candidates_per_iteration < 1:
        raise ValueError("candidates_per_iteration must be >= 1")
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not dry_run:
        _require_backend(backend)
        _require_terminal_model()
    budget_dir = Path(f"experience/B{budget:04d}")
    budget_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _new_run_dir(budget_dir, dry_run, run_id, resume)
    reports = []
    seed_report = _ensure_seed_candidate(run_dir, budget, dry_run, backend, concurrency, resume)
    if seed_report:
        reports.append(seed_report)
    for iteration in range(1, cycles + 1):
        history_dirs = _history_dirs(run_dir, before_iteration=iteration)
        val_runs = []
        iteration_reports = []
        for candidate_index in range(1, candidates_per_iteration + 1):
            iter_dir = _proposal_dir(run_dir, iteration, candidate_index)
            if resume and _candidate_complete(iter_dir, dry_run):
                iteration_reports.append(
                    {
                        "iteration": iteration,
                        "candidate": candidate_index,
                        "workspace": str(iter_dir / "workspace"),
                        "valid": _read_json(iter_dir / "validation.json").get("ok", False),
                        "skipped": True,
                    }
                )
                continue
            if resume and iter_dir.exists():
                raise RuntimeError(
                    f"incomplete candidate exists: {iter_dir}. Wait for it to finish "
                    "or remove that directory after confirming no jobs are active."
                )
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
                _sync_agent_alias_from_candidate(codex_workspace)
                write_history_artifacts(codex_workspace / "history", history_dirs)
                command = build_codex_command(
                    budget,
                    codex_model,
                    codex_reasoning_effort,
                    repair=False,
                    codex_bin=codex_bin,
                    workspace=codex_workspace,
                    iteration=iteration,
                    cycles=cycles,
                    candidate_index=candidate_index,
                    candidates_per_iteration=candidates_per_iteration,
                )
                _write_json(iter_dir / "codex_command.json", command)
                _write_meta(iter_dir, budget, iteration, candidate_index, "proposal")
                original_candidate = _read_workspace_text(codex_workspace, "candidate/harness.py")
                original_agent = _read_workspace_text(codex_workspace, "agents/baseline_kira.py")
                if not dry_run:
                    _run_codex(command, codex_workspace, iter_dir)
                _sync_candidate_from_agent_alias(
                    codex_workspace,
                    original_candidate,
                    original_agent,
                )
                _copy_workspace(codex_workspace, workspace)
                _strip_workspace_history(workspace)
                validation = validate_candidate(
                    workspace / "candidate" / "harness.py",
                    max_lines=budget,
                    min_lines=_budget_min_lines(budget),
                )
            _write_json(iter_dir / "validation.json", validation, sort_keys=True)
            if validation["ok"] and not dry_run:
                val_runs.append((workspace, budget, iter_dir, backend))
            else:
                summary = {
                    "split": "val",
                    "split_mean": 0.0,
                    "invalid": not validation["ok"],
                    "dry_run": dry_run,
                }
                _write_json(iter_dir / "summary.json", summary)
            iteration_reports.append(
                {
                    "iteration": iteration,
                    "candidate": candidate_index,
                    "workspace": str(workspace),
                    "valid": validation["ok"],
                }
            )
        _run_val_batch(val_runs, concurrency or VAL_CONCURRENCY)
        reports.extend(iteration_reports)
    return reports


def build_codex_command(
    budget: int,
    codex_model: str,
    codex_reasoning_effort: str,
    repair: bool,
    codex_bin: str | None = None,
    workspace: Path | None = None,
    iteration: int | None = None,
    cycles: int | None = None,
    candidate_index: int | None = None,
    candidates_per_iteration: int = DEFAULT_K,
) -> list[str]:
    min_lines = _budget_min_lines(budget)
    line_rule = (
        f"Keep candidate/harness.py between {min_lines} and {budget}"
        if min_lines > 1
        else f"Keep candidate/harness.py at most {budget}"
    )
    iteration_label = str(iteration) if iteration is not None else "the next"
    horizon_line = (
        f"\n\nThis is iteration {iteration_label} of {cycles}. Late rounds should "
        "prefer consolidating the strongest "
        "frontier-preserving changes over speculative probes."
        if cycles is not None
        else ""
    )
    prompt = (
        f"Run iteration {iteration_label} of the scaffold evolution loop (harness track)."
        f" Model: {terminal_model()}. Start from agents/baseline_kira.py "
        "as the parent. Before editing, inspect references/terminus_kira.py "
        "and references/open_source_harnesses.md as strong harness references. "
        "Codex is the most important GPT reference; also consider "
        "Terminus-KIRA, opencode, gemini-cli, and qwen-code. Prefer concrete "
        "patterns when they fit, but do not force them.\n\n"
        f"## Eval split: {len(get_val_tasks())} selected TB2 optimization tasks x "
        f"{VAL_TRIALS} trials\n\n"
        "This reference example uses the selected TB2 optimization tasks. Focus on "
        "scaffold changes that help the agent solve complex, long-horizon tasks.\n\n"
        "## Line budget\n"
        f"{line_rule} after Black formatting.\n\n"
        "Consider whether a useful reference-harness pattern can be compressed or adapted "
        "into the counted harness.\n\n"
        "## Run directories\n"
        "All logs and results for this run are under `logs/`.\n"
        "- `logs/evolution_summary.jsonl` — past results\n"
        "- `logs/frontier_val.json` — frontier\n"
        "- `logs/reports/` — post-eval reports\n"
        "- Write proposal.md to: `proposal.md`"
        f"{horizon_line}"
    )
    if repair:
        prompt = f"Repair validation failures. {prompt}"
    command = [
        resolve_codex_executable(codex_bin),
        "exec",
        "--model",
        codex_model,
        "-c",
        f'model_reasoning_effort="{codex_reasoning_effort}"',
        "-c",
        "sandbox_workspace_write.exclude_tmpdir_env_var=true",
        "-c",
        "sandbox_workspace_write.exclude_slash_tmp=true",
        "-c",
        "sandbox_workspace_write.writable_roots=[]",
        "--sandbox",
        "workspace-write",
    ]
    if workspace is not None:
        command.extend(["--cd", str(workspace)])
    command.extend(
        [
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            prompt,
        ]
    )
    return command


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
    concurrency: int | None,
    resume: bool,
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
    if resume and iter_dir.exists():
        raise RuntimeError(
            f"incomplete seed candidate exists: {iter_dir}. Wait for it to finish "
            "or remove that directory after confirming no jobs are active."
        )
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
        _run_val(workspace, budget, iter_dir, backend, concurrency)
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


def _new_run_dir(
    budget_dir: Path,
    dry_run: bool,
    run_id: str | None,
    resume: bool = False,
) -> Path:
    prefix = "dry_run" if dry_run else "run"
    label = _clean_run_id(run_id) if run_id else datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = budget_dir / f"{prefix}_{label}"
    if resume:
        if not run_id:
            raise ValueError("--resume requires --run-id")
        if not run_dir.exists():
            raise RuntimeError(f"run directory does not exist: {run_dir}")
        (budget_dir / "latest_run.txt").write_text(f"{run_dir.name}\n", encoding="utf-8")
        return run_dir
    if run_id and run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = budget_dir / f"{prefix}_{label}_{suffix:02d}"
    run_dir.mkdir(parents=True)
    (budget_dir / "latest_run.txt").write_text(f"{run_dir.name}\n", encoding="utf-8")
    return run_dir


def _candidate_complete(iter_dir: Path, dry_run: bool) -> bool:
    summary = _read_json(iter_dir / "summary.json")
    return (
        (iter_dir / "validation.json").exists()
        and summary
        and (iter_dir / "workspace" / "candidate" / "harness.py").exists()
        and not (summary.get("dry_run") and not dry_run)
    )


def _clean_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("run_id must contain at least one safe path character")
    return cleaned


def _run_codex(command: list[str], workspace: Path, iter_dir: Path) -> None:
    wrapped = _bwrap_codex_command(command, workspace)
    _write_json(iter_dir / "codex_sandbox_command.json", wrapped)
    result = subprocess.run(
        wrapped,
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    (iter_dir / "codex_stdout.log").write_text(result.stdout, encoding="utf-8")
    (iter_dir / "codex_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI exited with status {result.returncode}")


def _bwrap_codex_command(command: list[str], workspace: Path) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap (bwrap) is required for isolated Codex proposer runs")
    workspace = workspace.resolve(strict=True)
    tmp_dir = workspace / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    auth = _codex_auth_file()
    if not auth.exists():
        raise RuntimeError(f"Codex auth file not found: {auth}")
    codex_home = auth.parent.resolve(strict=True)
    args = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--clearenv",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
    ]
    seen_dirs: set[str] = set()
    for path in (workspace, codex_home, Path.home(), Path(sys.prefix).resolve()):
        _add_bwrap_parent_dirs(args, path, seen_dirs)
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(path).exists():
            args.extend(["--ro-bind", path, path])
    resolve_dir = Path("/run/systemd/resolve")
    if resolve_dir.exists():
        _add_bwrap_parent_dirs(args, resolve_dir, seen_dirs)
        args.extend(["--ro-bind", str(resolve_dir), str(resolve_dir)])
    prefix = Path(sys.prefix).resolve()
    if prefix.exists() and not _covered_by_system_bind(prefix):
        args.extend(["--ro-bind", str(prefix), str(prefix)])
    executable = Path(command[0]).resolve()
    if executable.exists() and not _covered_by_system_bind(executable):
        _add_bwrap_parent_dirs(args, executable, seen_dirs)
        args.extend(["--ro-bind", str(executable.parent), str(executable.parent)])
    args.extend(
        [
            "--dir",
            str(codex_home),
            "--bind",
            str(auth),
            str(codex_home / "auth.json"),
            "--bind",
            str(workspace),
            str(workspace),
            "--setenv",
            "HOME",
            str(Path.home()),
            "--setenv",
            "CODEX_HOME",
            str(codex_home),
            "--setenv",
            "PATH",
            _sandbox_path(),
            "--setenv",
            "TMPDIR",
            str(tmp_dir),
            "--chdir",
            str(workspace),
        ]
    )
    return args + command


def _codex_auth_file() -> Path:
    return Path(os.getenv("CODEX_HOME", Path.home() / ".codex")).expanduser() / "auth.json"


def _add_bwrap_parent_dirs(args: list[str], path: Path, seen: set[str]) -> None:
    current = Path("/")
    for part in path.resolve(strict=False).parent.parts[1:]:
        current /= part
        key = str(current)
        if key not in seen:
            args.extend(["--dir", key])
            seen.add(key)


def _covered_by_system_bind(path: Path) -> bool:
    return any(path.is_relative_to(Path(root)) for root in ("/usr", "/bin", "/lib", "/lib64"))


def _sandbox_path() -> str:
    entries = [
        str(Path(sys.prefix).resolve() / "bin"),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    return ":".join(dict.fromkeys(entries))


def _copy_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    _write_workspace_file(
        destination,
        "candidate/harness.py",
        _read_workspace_text(source, "candidate/harness.py"),
    )
    for relative in ("proposal.md", "AGENTS.md"):
        if _workspace_file_exists(source, relative):
            _write_workspace_file(destination, relative, _read_workspace_text(source, relative))


def _strip_workspace_history(workspace: Path) -> None:
    for name in ("agents", "history", "jobs", "logs", "references"):
        _remove_path(workspace / name)


def _sync_agent_alias_from_candidate(workspace: Path) -> None:
    if _workspace_file_exists(workspace, "candidate/harness.py"):
        _write_workspace_file(
            workspace,
            "agents/baseline_kira.py",
            _read_workspace_text(workspace, "candidate/harness.py"),
        )


def _sync_candidate_from_agent_alias(
    workspace: Path,
    original_candidate: str,
    original_agent: str,
) -> None:
    if not _workspace_file_exists(workspace, "agents/baseline_kira.py"):
        return
    agent_text = _read_workspace_text(workspace, "agents/baseline_kira.py")
    candidate_text = (
        _read_workspace_text(workspace, "candidate/harness.py")
        if _workspace_file_exists(workspace, "candidate/harness.py")
        else ""
    )
    if agent_text != original_agent:
        _write_workspace_file(workspace, "candidate/harness.py", agent_text)
    elif candidate_text != original_candidate:
        _write_workspace_file(workspace, "agents/baseline_kira.py", candidate_text)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_workspace_text(workspace: Path, relative: str) -> str:
    return _workspace_file(workspace, relative).read_text(encoding="utf-8")


def _write_workspace_file(workspace: Path, relative: str, text: str) -> None:
    path = _workspace_output_path(workspace, relative)
    path.write_text(text, encoding="utf-8")


def _workspace_file_exists(workspace: Path, relative: str) -> bool:
    path = _workspace_relative_path(workspace, relative)
    _assert_safe_existing_parents(workspace, path)
    if path.is_symlink():
        _raise_unsafe_workspace_path(path, "is a symlink")
    if path.exists() and not path.is_file():
        _raise_unsafe_workspace_path(path, "is not a regular file")
    if path.exists():
        _assert_under_workspace(workspace, path.resolve(strict=True))
    return path.exists()


def _workspace_file(workspace: Path, relative: str) -> Path:
    path = _workspace_relative_path(workspace, relative)
    _assert_safe_existing_parents(workspace, path)
    if path.is_symlink():
        _raise_unsafe_workspace_path(path, "is a symlink")
    if not path.is_file():
        _raise_unsafe_workspace_path(path, "is not a regular file")
    _assert_under_workspace(workspace, path.resolve(strict=True))
    return path


def _workspace_output_path(workspace: Path, relative: str) -> Path:
    path = _workspace_relative_path(workspace, relative)
    current = workspace
    for part in Path(relative).parts[:-1]:
        current /= part
        if current.is_symlink():
            _raise_unsafe_workspace_path(current, "has a symlink parent")
        if current.exists() and not current.is_dir():
            _raise_unsafe_workspace_path(current, "is not a regular directory")
        if not current.exists():
            current.mkdir()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        _raise_unsafe_workspace_path(path, "is not a regular file")
    _assert_under_workspace(workspace, path.resolve(strict=False))
    return path


def _workspace_relative_path(workspace: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        _raise_unsafe_workspace_path(path, "is not workspace-relative")
    target = workspace / path
    _assert_under_workspace(workspace, target.resolve(strict=False))
    return target


def _assert_safe_existing_parents(workspace: Path, path: Path) -> None:
    current = workspace
    relative = path.relative_to(workspace)
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            _raise_unsafe_workspace_path(current, "has a symlink parent")
        if current.exists() and not current.is_dir():
            _raise_unsafe_workspace_path(current, "is not a regular directory")
        if not current.exists():
            return


def _assert_under_workspace(workspace: Path, path: Path) -> None:
    root = workspace.resolve(strict=True)
    if not path.is_relative_to(root):
        _raise_unsafe_workspace_path(path, "escapes workspace")


def _raise_unsafe_workspace_path(path: Path, reason: str) -> None:
    raise RuntimeError(f"unsafe workspace path: {path} ({reason})")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _history_dirs(budget_dir: Path, before_iteration: int | None = None) -> list[Path]:
    rows = []
    for path in sorted(budget_dir.iterdir()):
        if before_iteration is not None:
            iteration = _candidate_iteration(path.name)
            if iteration is not None and iteration >= before_iteration:
                continue
        if (
            path.is_dir()
            and _read_json(path / "validation.json").get("ok") is True
            and (path / "summary.json").exists()
            and _read_json(path / "summary.json").get("dry_run") is not True
            and (path / "workspace" / "candidate" / "harness.py").exists()
        ):
            rows.append(path)
    return rows


def _candidate_iteration(name: str) -> int | None:
    if name == "iter_000_seed":
        return 0
    match = re.fullmatch(r"iter_(\d{3})_cand_\d{2}", name)
    return int(match.group(1)) if match else None


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


def _run_val(
    workspace: Path,
    budget: int,
    iter_dir: Path,
    backend: str,
    concurrency: int | None,
) -> None:
    concurrency_args = ["--concurrency", str(concurrency)] if concurrency is not None else []
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
            *concurrency_args,
        ],
        check=False,
    )


def _run_val_batch(
    runs: list[tuple[Path, int, Path, str]],
    total_concurrency: int,
) -> None:
    if not runs:
        return
    concurrencies = _split_concurrency(total_concurrency, len(runs))
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        futures = [
            pool.submit(_run_val, workspace, budget, iter_dir, backend, concurrency)
            for (workspace, budget, iter_dir, backend), concurrency in zip(runs, concurrencies)
        ]
        for future in futures:
            future.result()


def _split_concurrency(total: int, count: int) -> list[int]:
    if count < 1:
        return []
    if total < count:
        return [1] * count
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int)
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
            resume=args.resume,
            concurrency=args.concurrency,
        )
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
