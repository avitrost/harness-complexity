import sys

from evaluator import run_experiment


def test_parallel_budgets_keep_per_budget_concurrency(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment",
            "--budgets",
            "128,1024,8192",
            "--concurrency",
            "160",
            "--parallel-budgets",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(run_experiment, "_run", lambda command: calls.append(command))

    assert run_experiment.main() == 0

    by_budget = {int(call[call.index("--budget") + 1]): call for call in calls}
    assert sorted(by_budget) == [128, 1024, 8192]
    assert [int(call[call.index("--concurrency") + 1]) for call in calls] == [160] * 3
