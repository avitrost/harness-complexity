from pathlib import Path

import pytest

from evaluator.optimize_budget import (
    _bwrap_codex_command,
    _budget_min_lines,
    _candidate_complete,
    _copy_workspace,
    _history_dirs,
    _new_run_dir,
    _run_val,
    _run_val_batch,
    _split_concurrency,
    _sync_agent_alias_from_candidate,
    _sync_candidate_from_agent_alias,
    build_codex_command,
)


def test_build_codex_command_uses_resolved_exec_binary(tmp_path: Path) -> None:
    codex = tmp_path / "codex.cmd"
    codex.write_text("", encoding="utf-8")
    command = build_codex_command(
        budget=128,
        codex_model="gpt-5.5",
        codex_reasoning_effort="medium",
        repair=False,
        codex_bin=str(codex),
        workspace=tmp_path,
    )
    assert command[:4] == [str(codex), "exec", "--model", "gpt-5.5"]
    assert command[4:6] == ["-c", 'model_reasoning_effort="medium"']
    assert "sandbox_workspace_write.exclude_tmpdir_env_var=true" in command
    assert "sandbox_workspace_write.exclude_slash_tmp=true" in command
    assert "sandbox_workspace_write.writable_roots=[]" in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--cd") + 1] == str(tmp_path)
    assert "danger-full-access" not in command
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert "scaffold evolution loop (harness track)" in command[-1]
    assert "Start from agents/baseline_kira.py" in command[-1]
    assert "open_source_harnesses.md" in command[-1]
    assert "Codex is the most important GPT reference" in command[-1]
    assert "implementation depth scaled to the available line budget" in command[-1]
    assert "logs/frontier_val.json" in command[-1]
    assert "Keep candidate/harness.py at most 128 nonblank, non-comment source lines" in command[-1]
    assert "Blank lines and comments do not count" in command[-1]
    assert "near-duplicate numbered functions" in command[-1]
    assert "many tiny helpers" in command[-1]
    assert "rule tables" in command[-1]
    assert "top-level rule catalog" in command[-1]
    assert "single function or method" in command[-1]
    assert "large fixture strings" in command[-1]
    assert "pass Ruff" in command[-1]

    large_command = build_codex_command(
        budget=8192,
        codex_model="gpt-5.5",
        codex_reasoning_effort="medium",
        repair=False,
        codex_bin=str(codex),
        workspace=tmp_path,
    )
    assert "distinct reachable subsystems" in large_command[-1]
    assert "Prefer compact loops and data structures" in large_command[-1]
    assert "signal farms" in large_command[-1]
    assert "prefix-family method grids" in large_command[-1]
    assert "PolicyRule/list catalogs" in large_command[-1]
    assert "SignalSpec/metric fields" in large_command[-1]


def test_resume_run_dir_requires_existing_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()

    assert _new_run_dir(tmp_path, False, "existing", resume=True) == run_dir
    with pytest.raises(RuntimeError):
        _new_run_dir(tmp_path, False, "missing", resume=True)


def test_candidate_complete_requires_summary_validation_and_workspace(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter_001_cand_01"
    candidate_dir = iter_dir / "workspace" / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "harness.py").write_text("x = 1\n", encoding="utf-8")
    (iter_dir / "validation.json").write_text('{"ok": true}\n', encoding="utf-8")
    (iter_dir / "summary.json").write_text('{"dry_run": false}\n', encoding="utf-8")

    assert _candidate_complete(iter_dir, dry_run=False)
    (iter_dir / "summary.json").write_text('{"dry_run": true}\n', encoding="utf-8")
    assert not _candidate_complete(iter_dir, dry_run=False)
    assert _candidate_complete(iter_dir, dry_run=True)


def test_run_val_passes_concurrency(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "evaluator.optimize_budget.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    _run_val(tmp_path / "workspace", 128, tmp_path / "iter", "slurm-pyxis", concurrency=20)

    command = calls[0][0]
    assert command[command.index("--concurrency") + 1] == "20"


def test_run_val_batch_splits_total_concurrency(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "evaluator.optimize_budget._run_val",
        lambda workspace, budget, iter_dir, backend, concurrency: calls.append(concurrency),
    )

    _run_val_batch(
        [
            (tmp_path / "w1", 128, tmp_path / "i1", "slurm-pyxis"),
            (tmp_path / "w2", 128, tmp_path / "i2", "slurm-pyxis"),
        ],
        total_concurrency=20,
    )

    assert sorted(calls) == [10, 10]
    assert _split_concurrency(21, 2) == [11, 10]


def test_8192_budget_bucket_starts_after_4096() -> None:
    assert _budget_min_lines(8192) == 4097


def test_history_dirs_exclude_current_iteration(tmp_path: Path) -> None:
    _complete_candidate(tmp_path / "iter_000_seed")
    _complete_candidate(tmp_path / "iter_001_cand_01")
    _complete_candidate(tmp_path / "iter_002_cand_01")

    history = [path.name for path in _history_dirs(tmp_path, before_iteration=2)]

    assert history == ["iter_000_seed", "iter_001_cand_01"]


def test_copy_workspace_excludes_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "candidate").mkdir(parents=True)
    (source / "agents").mkdir(parents=True)
    (source / "history" / "old").mkdir(parents=True)
    (source / "jobs").mkdir(parents=True)
    (source / "logs").mkdir(parents=True)
    (source / "references").mkdir(parents=True)
    (source / "candidate" / "harness.py").write_text("x = 1\n", encoding="utf-8")
    (source / "agents" / "baseline_kira.py").write_text("x = 1\n", encoding="utf-8")
    (source / "history" / "old" / "summary.json").write_text("{}\n", encoding="utf-8")
    (source / "jobs" / "README.md").write_text("jobs\n", encoding="utf-8")
    (source / "logs" / "frontier_val.json").write_text("{}\n", encoding="utf-8")
    (source / "references" / "terminus_kira.py").write_text("ref\n", encoding="utf-8")

    destination = tmp_path / "destination"
    _copy_workspace(source, destination)

    assert (destination / "candidate" / "harness.py").exists()
    assert not (destination / "agents").exists()
    assert not (destination / "history").exists()
    assert not (destination / "jobs").exists()
    assert not (destination / "logs").exists()
    assert not (destination / "references").exists()


def test_bwrap_codex_command_mounts_only_workspace_and_minimal_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "home" / ".codex"
    workspace.mkdir()
    codex_home.mkdir(parents=True)
    (codex_home / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr("evaluator.optimize_budget.shutil.which", lambda name: "/usr/bin/bwrap")

    command = _bwrap_codex_command(["/usr/bin/codex", "exec", "prompt"], workspace)

    assert command[:6] == [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--clearenv",
    ]
    triples = list(zip(command, command[1:], command[2:]))
    assert ("--bind", str(workspace.resolve()), str(workspace.resolve())) in triples
    assert str(codex_home / "auth.json") in command
    assert "--chdir" in command
    assert command[-3:] == ["/usr/bin/codex", "exec", "prompt"]


def test_copy_workspace_rejects_symlinked_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    outside = tmp_path / "outside.py"
    outside.write_text("x = 2\n", encoding="utf-8")
    (source / "candidate").mkdir(parents=True)
    (source / "candidate" / "harness.py").symlink_to(outside)

    with pytest.raises(RuntimeError, match="unsafe workspace path"):
        _copy_workspace(source, tmp_path / "destination")


def test_agent_alias_syncs_back_to_counted_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate" / "harness.py"
    agent = workspace / "agents" / "baseline_kira.py"
    candidate.parent.mkdir(parents=True)
    agent.parent.mkdir(parents=True)
    candidate.write_text("x = 1\n", encoding="utf-8")

    _sync_agent_alias_from_candidate(workspace)
    original_candidate = candidate.read_text(encoding="utf-8")
    original_agent = agent.read_text(encoding="utf-8")
    agent.write_text("x = 2\n", encoding="utf-8")

    _sync_candidate_from_agent_alias(workspace, original_candidate, original_agent)

    assert candidate.read_text(encoding="utf-8") == "x = 2\n"


def test_agent_alias_sync_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.py"
    candidate = workspace / "candidate" / "harness.py"
    agent = workspace / "agents" / "baseline_kira.py"
    candidate.parent.mkdir(parents=True)
    agent.parent.mkdir(parents=True)
    outside.write_text("x = 3\n", encoding="utf-8")
    candidate.write_text("x = 1\n", encoding="utf-8")
    agent.symlink_to(outside)

    with pytest.raises(RuntimeError, match="unsafe workspace path"):
        _sync_candidate_from_agent_alias(workspace, "x = 1\n", "x = 1\n")


def _complete_candidate(iter_dir: Path) -> None:
    candidate_dir = iter_dir / "workspace" / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "harness.py").write_text("x = 1\n", encoding="utf-8")
    (iter_dir / "validation.json").write_text('{"ok": true}\n', encoding="utf-8")
    (iter_dir / "summary.json").write_text('{"dry_run": false}\n', encoding="utf-8")
