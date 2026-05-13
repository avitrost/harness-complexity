from scripts.bootstrap_ci import bootstrap_ci


def test_bootstrap_ci_is_seedable_on_synthetic_data() -> None:
    records = [
        {"task": "a", "trial": 1, "reward": 1},
        {"task": "a", "trial": 2, "reward": 0},
        {"task": "b", "trial": 1, "reward": 1},
        {"task": "b", "trial": 2, "reward": 1},
    ]
    one = bootstrap_ci(records, samples=100, seed=7)
    two = bootstrap_ci(records, samples=100, seed=7)
    assert one == two
    assert one["q025"] <= one["q500"] <= one["q975"]
