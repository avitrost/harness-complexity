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
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
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


def test_run_split_accepts_local_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "OpenThoughts-TBLite"
    summary = run_split(
        split="tblite",
        candidate_dir=tmp_path,
        budget=400,
        out_dir=tmp_path / "out",
        tasks=["acl-permissions-inheritance"],
        trials=5,
        concurrency=7,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--path --include-task-name --n-attempts --n-concurrent",
        dataset="open-thoughts/OpenThoughts-TBLite",
        dataset_path=dataset_path,
        max_retries=2,
        verifier_timeout_multiplier=3.0,
        retry_exclude=("VerifierTimeoutError",),
    )

    command = summary["command"]
    assert "--path" in command
    assert str(dataset_path) in command
    assert "--dataset" not in command
    assert summary["dataset_path"] == str(dataset_path)
    assert command[command.index("--max-retries") + 1] == "2"
    assert command[command.index("--verifier-timeout-multiplier") + 1] == "3.0"
    assert summary["retry_exclude"] == ["VerifierTimeoutError"]


def test_run_split_passes_agent_env_to_harbor(tmp_path: Path) -> None:
    summary = run_split(
        split="tb2-core",
        candidate_dir=tmp_path,
        budget=400,
        out_dir=tmp_path / "out",
        tasks=["bn-fit-modify"],
        trials=1,
        concurrency=1,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
        agent_env=(
            "OPENAI_TERMINAL_MODEL=gpt-5.5",
            "OPENAI_TERMINAL_REASONING_EFFORT=medium",
        ),
    )

    command = summary["command"]
    assert command.count("--agent-env") == 2
    assert "OPENAI_TERMINAL_MODEL=gpt-5.5" in command
    assert "OPENAI_TERMINAL_REASONING_EFFORT=medium" in command
    assert summary["agent_env"] == [
        "OPENAI_TERMINAL_MODEL=gpt-5.5",
        "OPENAI_TERMINAL_REASONING_EFFORT=medium",
    ]


def test_run_split_sets_agent_env_on_harbor_process(monkeypatch, tmp_path: Path) -> None:
    captured_env = {}

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return _Completed()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("evaluator.run_val.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("evaluator.run_val.subprocess.run", fake_run)

    summary = run_split(
        split="tb2-core",
        candidate_dir=tmp_path,
        budget=400,
        out_dir=tmp_path / "out",
        tasks=["bn-fit-modify"],
        trials=1,
        concurrency=1,
        dry_run=False,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
        agent_env=(
            "OPENAI_TERMINAL_MODEL=gpt-5.5",
            "OPENAI_TERMINAL_REASONING_EFFORT=medium",
        ),
    )

    assert summary["ran"] is True
    assert captured_env["OPENAI_TERMINAL_MODEL"] == "gpt-5.5"
    assert captured_env["OPENAI_TERMINAL_REASONING_EFFORT"] == "medium"


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
