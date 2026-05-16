from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_complexity_curve(
    selected_csv: Path, final_test_dir: Path, out_dir: Path, run_id: str | None = None
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = _plot_complexity(selected_csv, final_test_dir, out_dir)
    outputs.extend(_plot_cycle_metrics(Path("experience"), out_dir, run_id))
    return outputs


def _plot_complexity(selected_csv: Path, final_test_dir: Path, out_dir: Path) -> list[Path]:
    selected = (
        pd.read_csv(selected_csv)
        if selected_csv.exists() and selected_csv.stat().st_size
        else pd.DataFrame()
    )
    if selected.empty:
        return []
    rows = []
    for _, row in selected.iterrows():
        budget = int(row["budget"])
        summary = _read_json(final_test_dir / f"B{budget:04d}" / "summary.json")
        if summary and "estimated_full_score" in summary:
            rows.append({**row.to_dict(), **summary})
        else:
            rows.append(
                {
                    **row.to_dict(),
                    "split_mean": float(row["val_split_mean"]),
                    "estimated_full_score": float(row["val_split_mean"]),
                    "per_task": [],
                }
            )
    if not rows:
        return []
    data = pd.DataFrame(rows)
    path = out_dir / "complexity_curve.png"
    plt.figure()
    plt.errorbar(
        data["actual_loc"],
        data["estimated_full_score"],
        yerr=_bootstrap_errors(data, final_test_dir),
        fmt="o",
    )
    for _, item in data.iterrows():
        plt.annotate(str(int(item["budget"])), (item["actual_loc"], item["estimated_full_score"]))
    plt.xlabel("Formatted physical LOC")
    plt.ylabel("Optimization score")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    outputs = [path]
    heatmap = _plot_task_heatmap(data, out_dir)
    if heatmap:
        outputs.append(heatmap)
    return outputs


def _bootstrap_errors(data: pd.DataFrame, final_test_dir: Path) -> list[list[float]] | None:
    lower = []
    upper = []
    for _, item in data.iterrows():
        ci = _read_json(final_test_dir / f"B{int(item['budget']):04d}" / "bootstrap_ci.json")
        if not ci:
            return None
        y = float(item["estimated_full_score"])
        lower.append(max(0.0, y - float(ci["q025"])))
        upper.append(max(0.0, float(ci["q975"]) - y))
    return [lower, upper]


def _plot_cycle_metrics(
    experience_dir: Path, out_dir: Path, run_id: str | None = None
) -> list[Path]:
    rows = []
    for budget_dir in sorted(experience_dir.glob("B*")):
        for cycle, iter_dir in enumerate(_iter_dirs(budget_dir, run_id), start=1):
            summary = _read_json(iter_dir / "summary.json")
            if summary.get("dry_run"):
                continue
            validation = _read_json(iter_dir / "validation.json")
            rows.append(
                {
                    "budget": int(budget_dir.name[1:]),
                    "cycle": cycle,
                    "val_split_mean": float(summary.get("split_mean", 0) or 0),
                    "actual_loc": _loc_from_validation(validation),
                    "invalid": not bool(validation.get("ok", False)),
                }
            )
    if not rows:
        return []
    data = pd.DataFrame(rows)
    return [
        _line_plot(
            data,
            out_dir / "best_so_far_val.png",
            "val_split_mean",
            "Best-so-far validation mean",
            cumulative=True,
        ),
        _line_plot(
            data,
            out_dir / "actual_loc_vs_cycle.png",
            "actual_loc",
            "Formatted physical LOC",
            cumulative=False,
        ),
        _invalid_rate_plot(data, out_dir / "invalid_rate_by_budget.png"),
    ]


def _line_plot(
    data: pd.DataFrame,
    path: Path,
    column: str,
    ylabel: str,
    cumulative: bool,
) -> Path:
    plt.figure()
    for budget, group in data.groupby("budget"):
        ordered = group.sort_values("cycle")
        values = ordered[column].cummax() if cumulative else ordered[column]
        plt.plot(ordered["cycle"], values, marker="o", label=str(budget))
    plt.xlabel("Cycle")
    plt.ylabel(ylabel)
    plt.legend(title="Budget")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _invalid_rate_plot(data: pd.DataFrame, path: Path) -> Path:
    plt.figure()
    data.groupby("budget")["invalid"].mean().plot(kind="bar")
    plt.xlabel("Budget")
    plt.ylabel("Invalid candidate rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _iter_dirs(budget_dir: Path, run_id: str | None = None) -> list[Path]:
    if run_id:
        return sorted((budget_dir / f"run_{run_id}").glob("iter_*"))
    return sorted(budget_dir.glob("run_*/iter_*"))


def _plot_task_heatmap(data: pd.DataFrame, out_dir: Path) -> Path | None:
    rows = []
    for _, item in data.iterrows():
        for task in item.get("per_task", []) or []:
            rows.append({"budget": int(item["budget"]), "task": task["task"], "mean": task["mean"]})
    if not rows:
        return None
    matrix = pd.DataFrame(rows).pivot(index="budget", columns="task", values="mean")
    path = out_dir / "per_task_heatmap.png"
    plt.figure(figsize=(max(6, len(matrix.columns) * 0.45), 3))
    plt.imshow(matrix.fillna(0).to_numpy(), aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(label="Reward mean")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _loc_from_validation(validation: dict[str, Any]) -> int | None:
    for check in validation.get("checks", []):
        data = check.get("json") if isinstance(check, dict) else None
        if isinstance(data, dict) and "physical_loc" in data:
            return int(data["physical_loc"])
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-csv", type=Path, default=Path("results/selected_candidates.csv")
    )
    parser.add_argument("--final-test-dir", type=Path, default=Path("final_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    outputs = plot_complexity_curve(
        args.selected_csv, args.final_test_dir, args.out_dir, args.run_id
    )
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
