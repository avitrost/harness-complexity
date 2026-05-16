from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def bootstrap_ci(
    records: list[dict[str, Any]],
    samples: int = 10000,
    seed: int | None = None,
) -> dict[str, float | int | None]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(record)
    tasks = sorted(by_task)
    if not tasks:
        return {"samples": samples, "seed": seed, "q025": None, "q500": None, "q975": None}
    estimates = []
    for _ in range(samples):
        task_means = []
        for task in (rng.choice(tasks) for _ in tasks):
            trials = by_task[task]
            rewards = [float(rng.choice(trials).get("reward", 0)) for _ in trials]
            task_means.append(mean(rewards))
        estimates.append(mean(task_means))
    estimates.sort()
    return {
        "samples": samples,
        "seed": seed,
        "q025": _quantile(estimates, 0.025),
        "q500": _quantile(estimates, 0.5),
        "q975": _quantile(estimates, 0.975),
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_json", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    records = json.loads(args.records_json.read_text(encoding="utf-8"))
    print(json.dumps(bootstrap_ci(records, args.samples, args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
