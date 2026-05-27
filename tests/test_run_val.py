from pathlib import Path

from evaluator.run_val import run_split


class _Completed:
    stdout = ""
    stderr = ""
    returncode = 0


def test_run_split_fails_before_harbor_without_docker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("evaluator.run_val.shutil.which", lambda name: None)
    summary = run_split(
        split="val",
        candidate_dir=tmp_path,
        budget=128,
        out_dir=tmp_path / "out",
        tasks=["task"],
        trials=1,
        concurrency=1,
        dry_run=False,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )
    assert summary["ran"] is False
    assert "Docker" in summary["error"]


def test_run_split_fails_before_harbor_without_openai_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("evaluator.run_val.shutil.which", lambda name: "docker")
    summary = run_split(
        split="val",
        candidate_dir=tmp_path,
        budget=128,
        out_dir=tmp_path / "out",
        tasks=["task"],
        trials=1,
        concurrency=1,
        dry_run=False,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )
    assert summary["ran"] is False
    assert "OPENAI_API_KEY" in summary["error"]


def test_run_split_slurm_backend_skips_docker_preflight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "evaluator.run_val.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in {"srun", "enroot"} else None,
    )
    summary = run_split(
        split="val",
        candidate_dir=tmp_path,
        budget=128,
        out_dir=tmp_path / "out",
        tasks=["task"],
        trials=1,
        concurrency=1,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
        backend="slurm-pyxis",
    )
    assert summary["backend"] == "slurm-pyxis"
    assert "--environment-import-path" in summary["command"]


def test_run_split_skips_terminal_model_preflight_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("HARBOR_TERMINAL_MODEL_PREFLIGHT", raising=False)
    monkeypatch.setattr("evaluator.run_val.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        "evaluator.run_val.check_terminal_model_available",
        lambda: (_ for _ in ()).throw(RuntimeError("preflight should not run")),
    )
    monkeypatch.setattr("evaluator.run_val.subprocess.run", lambda *args, **kwargs: _Completed())

    summary = run_split(
        split="val",
        candidate_dir=tmp_path,
        budget=128,
        out_dir=tmp_path / "out",
        tasks=["task"],
        trials=1,
        concurrency=1,
        dry_run=False,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    assert summary["ran"] is True
    assert summary["returncode"] == 0
