from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    pool = [row for row in rows if row.get("valid", True)] or rows
    return sorted(
        pool,
        key=lambda row: (
            -float(row.get("val_split_mean", 0)),
            int(row.get("actual_loc", 10**9)),
            float(row.get("crash_rate", 1)),
            float(row.get("mean_runtime", 10**9)),
            int(row.get("iteration", 10**9)),
        ),
    )[0]


def load_budget_rows(budget_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for iter_dir in sorted(budget_dir.glob("iter_*")):
        iteration = _iteration_number(iter_dir.name)
        validation = _read_json(iter_dir / "validation.json")
        summary = _read_json(iter_dir / "summary.json") or _read_json(iter_dir / "val_summary.json")
        count = _extract_count(validation)
        num_trials = int(summary.get("num_trials", 0) or 0)
        num_crashes = int(summary.get("num_crashes", 0) or 0)
        rows.append(
            {
                "budget": _budget_from_dir(budget_dir),
                "iteration": iteration,
                "candidate_dir": str(iter_dir / "workspace"),
                "val_split_mean": float(summary.get("split_mean", 0) or 0),
                "actual_loc": int(count.get("physical_loc", 10**9)),
                "crash_rate": num_crashes / num_trials if num_trials else 1.0,
                "mean_runtime": float(summary.get("mean_runtime") or 10**9),
                "valid": bool(validation.get("ok", False)),
            }
        )
    return rows


def write_selection(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any] | None:
    selected = select_best(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selected_candidates.json"
    csv_path = out_dir / "selected_candidates.csv"
    selections = _merge_selection(_read_existing_selections(json_path), selected)
    json_path.write_text(json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8")
    if selections:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "budget",
                    "iteration",
                    "candidate_dir",
                    "val_split_mean",
                    "actual_loc",
                    "crash_rate",
                    "mean_runtime",
                    "valid",
                ],
            )
            writer.writeheader()
            writer.writerows(selections)
    else:
        csv_path.write_text("", encoding="utf-8")
    return selected


def _read_existing_selections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if isinstance(payload, dict):
        return [payload] if payload else []
    return [item for item in payload if isinstance(item, dict)]


def _merge_selection(
    existing: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if selected is None:
        return sorted(existing, key=lambda row: int(row.get("budget", 0)))
    budget = int(selected.get("budget", 0))
    merged = [row for row in existing if int(row.get("budget", -1)) != budget]
    merged.append(selected)
    return sorted(merged, key=lambda row: int(row.get("budget", 0)))


def _extract_count(validation: dict[str, Any]) -> dict[str, Any]:
    for check in validation.get("checks", []):
        data = check.get("json")
        if isinstance(data, dict) and "physical_loc" in data:
            return data
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _iteration_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _budget_from_dir(path: Path) -> int:
    match = re.search(r"B(\d+)", path.name)
    return int(match.group(1)) if match else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    selected = write_selection(load_budget_rows(args.budget_dir), args.out_dir)
    print(json.dumps(selected or {}, indent=2, sort_keys=True))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
