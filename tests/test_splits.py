from pathlib import Path

from evaluator import splits
from plumbing.harbor_adapter import HarborRunSpec, build_harbor_command


def test_split_definitions() -> None:
    assert splits.get_val_tasks()[:5] == [
        "fix-git",
        "qemu-alpine-ssh",
        "sparql-university",
        "sqlite-db-truncate",
        "write-compressor",
    ]
    assert len(splits.get_val_tasks()) == 20
    assert splits.get_test_tasks() == []
    assert len(splits.get_heldout_tasks()) == 69
    assert set(splits.get_val_tasks()).isdisjoint(splits.get_heldout_tasks())
    assert splits.VAL_TRIALS == 4
    assert splits.VAL_CONCURRENCY == 160
    assert splits.TEST_TRIALS == 4
    assert splits.TEST_CONCURRENCY == 160
    assert splits.HELDOUT_TRIALS == 2
    assert splits.HELDOUT_CONCURRENCY == 300


def test_score_formulas() -> None:
    assert splits.val_estimated_full_score(0.4) == 0.4
    assert splits.test_estimated_full_score(0.4) == 0.4


def test_harbor_command_dry_run_constructs_task_flags() -> None:
    spec = HarborRunSpec(Path("candidate"), Path("out"), ["a", "b"], 4, 8, "val")
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )
    assert plan.runnable is True
    assert plan.task_flag == "--include-task-name"
    assert plan.command.count("--include-task-name") == 2
    assert "terminal-bench@2.0" in plan.command
    assert "--model" not in plan.command


def test_harbor_command_can_use_local_dataset_path() -> None:
    spec = HarborRunSpec(
        Path("candidate"),
        Path("out"),
        ["a", "b"],
        4,
        8,
        "tblite",
        dataset_path=Path("external_datasets/OpenThoughts-TBLite/revision"),
    )
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--path --include-task-name --n-attempts --n-concurrent",
    )

    assert plan.runnable is True
    assert "--path" in plan.command
    assert "--dataset" not in plan.command
    assert "external_datasets/OpenThoughts-TBLite/revision" in plan.command
    assert plan.command.count("--include-task-name") == 2


def test_harbor_command_can_use_named_agent_model_kwargs_and_env() -> None:
    spec = HarborRunSpec(
        Path("."),
        Path("out"),
        ["a"],
        1,
        2,
        "codex-cli",
        agent_name="codex",
        agent_model_name="gpt-test",
        agent_kwargs=("reasoning_effort=none",),
        agent_env=("CODEX_AUTH_JSON_PATH=/tmp/auth.json",),
        include_candidate_dir_kwarg=False,
    )
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    assert plan.command[plan.command.index("--agent") + 1] == "codex"
    assert "--agent-import-path" not in plan.command
    assert "candidate_dir=." not in plan.command
    assert plan.command[plan.command.index("--model") + 1] == "gpt-test"
    assert "reasoning_effort=none" in plan.command
    assert "CODEX_AUTH_JSON_PATH=/tmp/auth.json" in plan.command


def test_harbor_command_can_add_retry_and_verifier_timeout_flags() -> None:
    spec = HarborRunSpec(
        Path("candidate"),
        Path("out"),
        ["a"],
        4,
        8,
        "tblite",
        max_retries=2,
        verifier_timeout_multiplier=3.0,
        retry_exclude=("VerifierTimeoutError",),
    )
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    assert plan.command[plan.command.index("--max-retries") + 1] == "2"
    assert plan.command[plan.command.index("--verifier-timeout-multiplier") + 1] == "3.0"
    assert plan.command[plan.command.index("--retry-exclude") + 1] == "VerifierTimeoutError"


def test_harbor_command_can_use_slurm_pyxis_environment() -> None:
    spec = HarborRunSpec(Path("candidate"), Path("out"), ["a"], 1, 1, "val", "slurm-pyxis")
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )
    assert "--environment-import-path" in plan.command
    assert "--environment-build-timeout-multiplier" in plan.command
    assert "plumbing.slurm_pyxis_environment:SlurmPyxisEnvironment" in plan.command
    assert plan.command.count("--environment-kwarg") == 3


def test_harbor_command_can_override_slurm_pyxis_partition(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_SLURM_PYXIS_PARTITION", "m7i-cpu2")
    spec = HarborRunSpec(Path("candidate"), Path("out"), ["a"], 1, 1, "val", "slurm-pyxis")
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    assert plan.command.count("--environment-kwarg") == 4
    assert "slurm_partition=m7i-cpu2" in plan.command


def test_harbor_command_uses_known_flags_when_help_probe_times_out(monkeypatch) -> None:
    monkeypatch.setattr("plumbing.harbor_adapter.harbor_help", lambda *args: None)
    spec = HarborRunSpec(Path("candidate"), Path("out"), ["a"], 1, 1, "val")
    plan = build_harbor_command(spec, executable="harbor")

    assert plan.runnable is True
    assert "timed out" in plan.note
