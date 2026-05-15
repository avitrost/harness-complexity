from pathlib import Path

from evaluator import splits
from plumbing.harbor_adapter import HarborRunSpec, build_harbor_command


def test_split_definitions() -> None:
    assert splits.get_val_tasks() == [
        "fix-git",
        "qemu-alpine-ssh",
        "sparql-university",
        "sqlite-db-truncate",
        "write-compressor",
    ]
    assert len(splits.get_test_tasks()) == 15
    assert splits.VAL_TRIALS == 4
    assert splits.VAL_CONCURRENCY == 8
    assert splits.TEST_TRIALS == 5
    assert splits.TEST_CONCURRENCY == 8


def test_score_formulas() -> None:
    assert splits.val_estimated_full_score(1.0) == 0.361193 + 0.295842
    assert splits.test_estimated_full_score(1.0) == 0.510101 + 0.108900


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


def test_harbor_command_can_use_slurm_pyxis_environment() -> None:
    spec = HarborRunSpec(Path("candidate"), Path("out"), ["a"], 1, 1, "val", "slurm-pyxis")
    plan = build_harbor_command(
        spec,
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )
    assert "--environment-import-path" in plan.command
    assert "plumbing.slurm_pyxis_environment:SlurmPyxisEnvironment" in plan.command
    assert plan.command.count("--environment-kwarg") == 3
