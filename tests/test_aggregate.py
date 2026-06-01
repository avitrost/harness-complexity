from evaluator.aggregate import aggregate_records


def test_aggregate_records_reports_token_metrics_by_task() -> None:
    summary = aggregate_records(
        [
            {
                "task": "a",
                "trial": 1,
                "reward": 1,
                "status": "success",
                "runtime_sec": 10,
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": 80,
                "total_tokens": 120,
                "cost_usd": 0.01,
                "model_calls": 2,
            },
            {
                "task": "a",
                "trial": 2,
                "reward": 0,
                "status": "failure",
                "runtime_sec": 20,
                "input_tokens": 50,
                "output_tokens": 10,
                "cached_tokens": 40,
                "total_tokens": 60,
                "cost_usd": 0.02,
                "model_calls": 1,
            },
        ],
        "custom",
    )

    assert summary["total_tokens"] == 180
    assert summary["mean_total_tokens"] == 90
    assert summary["total_cost_usd"] == 0.03
    assert summary["model_calls"] == 3
    [task] = summary["per_task"]
    assert task["task"] == "a"
    assert task["total_input_tokens"] == 150
    assert task["total_output_tokens"] == 30
    assert task["total_cached_tokens"] == 120
    assert task["total_tokens"] == 180
    assert task["mean_total_tokens"] == 90
    assert task["model_calls"] == 3
