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
    return sorted(pool, key=_selection_key)[0]


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [row for row in rows if row.get("valid", True)] or rows
    frontier = []
    for row in pool:
        if not any(_dominates(other, row) for other in pool if other is not row):
            frontier.append(row)
    return sorted(frontier, key=_selection_key)


def load_budget_rows(budget_dir: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for iter_dir in _iter_dirs(budget_dir, run_id):
        iteration, candidate = _iteration_candidate(iter_dir.name)
        validation = _read_json(iter_dir / "validation.json")
        summary = _read_json(iter_dir / "summary.json")
        if summary.get("dry_run"):
            continue
        count = _extract_count(validation)
        num_trials = int(summary.get("num_trials", 0) or 0)
        num_crashes = int(summary.get("num_crashes", 0) or 0)
        rows.append(
            {
                "budget": _budget_from_dir(budget_dir),
                "run_id": _run_id(iter_dir),
                "iteration": iteration,
                "candidate": candidate,
                "candidate_dir": str(iter_dir / "workspace"),
                "val_split_mean": float(summary.get("split_mean", 0) or 0),
                "actual_loc": int(
                    count.get("nonblank_noncomment_sloc", count.get("physical_loc", 10**9))
                ),
                "crash_rate": num_crashes / num_trials if num_trials else 1.0,
                "mean_runtime": float(summary.get("mean_runtime") or 10**9),
                "valid": bool(validation.get("ok", False)),
            }
        )
    return rows


def write_selection(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any] | None:
    selected = select_best(rows)
    frontier = pareto_frontier(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selected_candidates.json"
    csv_path = out_dir / "selected_candidates.csv"
    frontier_json_path = out_dir / "pareto_frontier.json"
    frontier_csv_path = out_dir / "pareto_frontier.csv"
    selections = _merge_selection(_read_existing_selections(json_path), selected)
    frontiers = _merge_frontier(_read_existing_selections(frontier_json_path), frontier, rows)
    json_path.write_text(json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8")
    frontier_json_path.write_text(json.dumps(frontiers, indent=2, sort_keys=True), encoding="utf-8")
    if selections:
        _write_csv(csv_path, selections)
    else:
        csv_path.write_text("", encoding="utf-8")
    if frontiers:
        _write_csv(frontier_csv_path, frontiers)
    else:
        frontier_csv_path.write_text("", encoding="utf-8")
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


def _merge_frontier(
    existing: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return sorted(existing, key=_selection_key)
    budgets = {int(row.get("budget", 0)) for row in rows}
    merged = [row for row in existing if int(row.get("budget", -1)) not in budgets]
    merged.extend(frontier)
    return sorted(merged, key=lambda row: (int(row.get("budget", 0)), _selection_key(row)))


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


def _iter_dirs(budget_dir: Path, run_id: str | None = None) -> list[Path]:
    if run_id:
        return sorted((budget_dir / f"run_{run_id}").glob("iter_*"))
    return sorted(budget_dir.glob("run_*/iter_*"))


def _run_id(iter_dir: Path) -> str:
    parent = iter_dir.parent.name
    return parent.removeprefix("run_") if parent.startswith("run_") else ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "budget",
                "run_id",
                "iteration",
                "candidate",
                "candidate_dir",
                "val_split_mean",
                "actual_loc",
                "crash_rate",
                "mean_runtime",
                "valid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _selection_key(row: dict[str, Any]) -> tuple[float, int, float, float, int, int]:
    return (
        -_float(row.get("val_split_mean"), 0),
        int(row.get("actual_loc") or 10**9),
        _float(row.get("crash_rate"), 1),
        _float(row.get("mean_runtime"), 10**9),
        int(row.get("iteration", 10**9)),
        int(row.get("candidate", 10**9)),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = _objective_values(left)
    right_values = _objective_values(right)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _objective_values(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _float(row.get("val_split_mean"), 0),
        -_float(row.get("actual_loc"), 10**9),
        -_float(row.get("crash_rate"), 1),
        -_float(row.get("mean_runtime"), 10**9),
    )


def _float(value: Any, default: float) -> float:
    return float(default if value is None else value)


def _iteration_candidate(name: str) -> tuple[int, int]:
    match = re.search(r"iter_(\d+)_cand_(\d+)$", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"iter_(\d+)_seed$", name)
    if match:
        return int(match.group(1)), 0
    return _iteration_number(name), 1


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
    parser.add_argument("--run-id")
    args = parser.parse_args()
    selected = write_selection(load_budget_rows(args.budget_dir, args.run_id), args.out_dir)
    print(json.dumps(selected or {}, indent=2, sort_keys=True))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
