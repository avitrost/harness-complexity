from pathlib import Path

import pytest

from evaluator.optimize_budget import (
    _candidate_complete,
    _history_dirs,
    _new_run_dir,
    _run_val,
    _run_val_batch,
    _split_concurrency,
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
    )
    assert command[:4] == [str(codex), "exec", "--model", "gpt-5.5"]
    assert command[4:6] == ["-c", 'model_reasoning_effort="medium"']
    assert command[6:8] == ["--sandbox", "danger-full-access"]
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert "Do not inspect parent directories" in command[-1]
    assert "Read history/ first" in command[-1]


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


def test_history_dirs_exclude_current_iteration(tmp_path: Path) -> None:
    _complete_candidate(tmp_path / "iter_000_seed")
    _complete_candidate(tmp_path / "iter_001_cand_01")
    _complete_candidate(tmp_path / "iter_002_cand_01")

    history = [path.name for path in _history_dirs(tmp_path, before_iteration=2)]

    assert history == ["iter_000_seed", "iter_001_cand_01"]


def _complete_candidate(iter_dir: Path) -> None:
    candidate_dir = iter_dir / "workspace" / "candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "harness.py").write_text("x = 1\n", encoding="utf-8")
    (iter_dir / "validation.json").write_text('{"ok": true}\n', encoding="utf-8")
    (iter_dir / "summary.json").write_text('{"dry_run": false}\n', encoding="utf-8")
