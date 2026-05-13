from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.run_val import run_split
from evaluator.splits import TEST_CONCURRENCY, TEST_TRIALS, get_test_tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--harbor-bin")
    args = parser.parse_args()
    summary = run_split(
        "test",
        args.candidate_dir,
        args.budget,
        args.out_dir,
        get_test_tasks(),
        TEST_TRIALS,
        TEST_CONCURRENCY,
        args.dry_run,
        args.harbor_bin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ran", True) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
