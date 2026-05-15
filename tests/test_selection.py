from scripts.select_best import pareto_frontier, select_best


def test_select_best_uses_registered_tie_breakers() -> None:
    rows = [
        {
            "iteration": 2,
            "val_split_mean": 0.5,
            "actual_loc": 70,
            "crash_rate": 0.0,
            "mean_runtime": 10.0,
        },
        {
            "iteration": 1,
            "val_split_mean": 0.5,
            "actual_loc": 64,
            "crash_rate": 0.5,
            "mean_runtime": 1.0,
        },
    ]
    assert select_best(rows)["iteration"] == 1


def test_select_best_prefers_score_before_loc() -> None:
    rows = [
        {"iteration": 1, "val_split_mean": 0.4, "actual_loc": 1},
        {"iteration": 2, "val_split_mean": 0.6, "actual_loc": 100},
    ]
    assert select_best(rows)["iteration"] == 2


def test_pareto_frontier_keeps_non_dominated_rows() -> None:
    rows = [
        {"iteration": 1, "val_split_mean": 0.5, "actual_loc": 100, "crash_rate": 0.0},
        {"iteration": 2, "val_split_mean": 0.6, "actual_loc": 90, "crash_rate": 0.0},
        {"iteration": 3, "val_split_mean": 0.7, "actual_loc": 150, "crash_rate": 0.0},
    ]
    assert {row["iteration"] for row in pareto_frontier(rows)} == {2, 3}
