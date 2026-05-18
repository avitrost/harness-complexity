from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def write_history_artifacts(history_dir: Path, history_dirs: list[Path]) -> None:
    rows = [_candidate_row(path) for path in history_dirs]
    traces = [trace for path in history_dirs for trace in _candidate_traces(path)]
    _write_json(history_dir / "index.json", rows, sort_keys=True)
    _write_json(history_dir / "frontier.json", _frontier(rows), sort_keys=True)
    _write_json(history_dir / "trace_index.json", traces, sort_keys=True)
    _write_evolution_summary(history_dir / "evolution_summary.jsonl", rows)
    _write_failures_md(history_dir / "failures.md", rows, traces)
    _write_official_layout(history_dir)


def _write_official_layout(history_dir: Path) -> None:
    workspace = history_dir.parent
    logs_dir = workspace / "logs"
    reports_dir = logs_dir / "reports"
    jobs_dir = workspace / "jobs"
    logs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)
    logs_jobs = logs_dir / "jobs"
    if not logs_jobs.exists():
        _symlink_or_pointer(jobs_dir, logs_jobs)
    _copy_if_exists(history_dir / "evolution_summary.jsonl", logs_dir / "evolution_summary.jsonl")
    _copy_if_exists(history_dir / "frontier.json", logs_dir / "frontier_val.json")
    _copy_if_exists(history_dir / "index.json", logs_dir / "index.json")
    _copy_if_exists(history_dir / "trace_index.json", logs_dir / "trace_index.json")
    _copy_if_exists(history_dir / "failures.md", logs_dir / "failures.md")
    _copy_if_exists(history_dir / "failures.md", reports_dir / "failures.md")
    for candidate_dir in sorted(path for path in history_dir.iterdir() if path.is_dir()):
        link = jobs_dir / candidate_dir.name
        if not link.exists():
            _symlink_or_pointer(candidate_dir, link)


def _candidate_row(path: Path) -> dict[str, Any]:
    summary = _read_json(path / "summary.json")
    validation = _read_json(path / "validation.json")
    count = _extract_count(validation)
    per_task = {
        item["task"]: {
            "mean": item.get("mean"),
            "num_successes": item.get("num_successes"),
            "num_crashes": item.get("num_crashes"),
            "mean_runtime": item.get("mean_runtime"),
        }
        for item in summary.get("per_task") or []
        if isinstance(item, dict) and item.get("task")
    }
    return {
        "dir": path.name,
        "split_mean": summary.get("split_mean"),
        "estimated_full_score": summary.get("estimated_full_score"),
        "num_trials": summary.get("num_trials"),
        "num_successes": summary.get("num_successes"),
        "num_crashes": summary.get("num_crashes"),
        "mean_runtime": summary.get("mean_runtime"),
        "physical_loc": count.get("physical_loc"),
        "valid": validation.get("ok", False),
        "per_task": per_task,
    }


def _candidate_traces(candidate_dir: Path) -> list[dict[str, Any]]:
    rows = []
    seen_by_task: dict[str, int] = {}
    for result_path in sorted(candidate_dir.rglob("result.json")):
        trial_dir = result_path.parent
        if "__" not in trial_dir.name:
            continue
        payload = _read_json(result_path)
        task = _task_name(payload, trial_dir)
        seen_by_task[task] = seen_by_task.get(task, 0) + 1
        reward = _reward(payload)
        exception_path = trial_dir / "exception.txt"
        harness_result = trial_dir / "agent" / "harness-result.json"
        agent_result = payload.get("agent_result") or {}
        metadata = agent_result.get("metadata") or {}
        rows.append(
            {
                "candidate": candidate_dir.name,
                "task": task,
                "trial": seen_by_task[task],
                "reward": reward,
                "status": _status(payload, reward, exception_path),
                "runtime_sec": _runtime_sec(payload),
                "done": metadata.get("done"),
                "turns": metadata.get("turns"),
                "last_return_code": metadata.get("last_return_code"),
                "result": _history_path(candidate_dir, result_path),
                "trial_log": _path_if_exists(candidate_dir, trial_dir / "trial.log"),
                "harness_result": _path_if_exists(candidate_dir, harness_result),
                "turn_logs_glob": _glob_if_any(
                    candidate_dir, trial_dir / "agent", "harness-turn-*.json"
                ),
                "model_calls_glob": _glob_if_any(
                    candidate_dir, trial_dir / "agent", "model-call-*.json"
                ),
                "exception": _path_if_exists(candidate_dir, exception_path),
                "exception_summary": _exception_summary(exception_path),
            }
        )
    return rows


def _frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda row: row.get("split_mean") or -1, default=None)
    per_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        for task, data in row.get("per_task", {}).items():
            current = per_task.get(task)
            mean = data.get("mean")
            if current is None or (mean is not None and mean > (current.get("mean") or -1)):
                per_task[task] = {"candidate": row["dir"], **data}
    return {
        "best_overall": (
            None
            if best is None
            else {
                "candidate": best["dir"],
                "split_mean": best.get("split_mean"),
                "num_successes": best.get("num_successes"),
                "num_crashes": best.get("num_crashes"),
                "physical_loc": best.get("physical_loc"),
            }
        ),
        "per_task": per_task,
    }


def _write_evolution_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "candidate": row["dir"],
                    "split_mean": row.get("split_mean"),
                    "estimated_full_score": row.get("estimated_full_score"),
                    "num_successes": row.get("num_successes"),
                    "num_crashes": row.get("num_crashes"),
                    "physical_loc": row.get("physical_loc"),
                    "per_task": row.get("per_task", {}),
                },
                sort_keys=True,
            )
        )
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _write_failures_md(
    path: Path, rows: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> None:
    lines = ["# History Failures", ""]
    if rows:
        lines.extend(["## Candidates", ""])
        for row in sorted(rows, key=lambda item: item.get("split_mean") or 0, reverse=True):
            lines.append(
                "- {dir}: score={score} successes={successes} crashes={crashes} loc={loc}".format(
                    dir=row["dir"],
                    score=_fmt(row.get("split_mean")),
                    successes=row.get("num_successes"),
                    crashes=row.get("num_crashes"),
                    loc=row.get("physical_loc"),
                )
            )
        lines.append("")
    failures = [trace for trace in traces if trace.get("reward") != 1]
    failures.sort(key=_failure_sort_key)
    lines.extend(["## Traces To Inspect First", ""])
    for trace in failures[:80]:
        lines.append(
            "- {candidate} {task} trial {trial}: status={status} reward={reward} "
            "runtime={runtime}s turns={turns}".format(
                candidate=trace["candidate"],
                task=trace["task"],
                trial=trace["trial"],
                status=trace.get("status"),
                reward=trace.get("reward"),
                runtime=_fmt(trace.get("runtime_sec")),
                turns=trace.get("turns"),
            )
        )
        for key in (
            "result",
            "trial_log",
            "harness_result",
            "turn_logs_glob",
            "model_calls_glob",
            "exception",
        ):
            if trace.get(key):
                lines.append(f"  - {key}: `{trace[key]}`")
        if trace.get("exception_summary"):
            lines.append(f"  - exception_summary: {trace['exception_summary']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failure_sort_key(trace: dict[str, Any]) -> tuple[int, float]:
    status = str(trace.get("status") or "")
    crash_rank = 0 if status == "crash" else 1
    runtime = float(trace.get("runtime_sec") or 0)
    return crash_rank, -runtime


def _task_name(payload: dict[str, Any], trial_dir: Path) -> str:
    value = payload.get("task_name") or payload.get("task") or payload.get("task_id")
    if isinstance(value, dict):
        value = value.get("name") or value.get("path")
    if value:
        return str(value)
    return trial_dir.name.rsplit("__", 1)[0]


def _reward(payload: dict[str, Any]) -> int:
    reward = payload.get("reward", payload.get("score"))
    verifier = payload.get("verifier_result")
    if reward is None and isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if isinstance(rewards, dict):
            reward = rewards.get("reward", rewards.get("score"))
    try:
        return 1 if float(reward or 0) >= 1 else 0
    except (TypeError, ValueError):
        return 0


def _status(payload: dict[str, Any], reward: int, exception_path: Path) -> str:
    if payload.get("exception_info") or exception_path.exists():
        return "crash"
    return "success" if reward == 1 else "failure"


def _runtime_sec(payload: dict[str, Any]) -> float | None:
    runtime = payload.get("runtime_sec") or payload.get("duration_sec")
    if runtime is not None:
        return float(runtime)
    started = _parse_datetime(payload.get("started_at"))
    finished = _parse_datetime(payload.get("finished_at"))
    if started and finished:
        return round(max(0.0, (finished - started).total_seconds()), 3)
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _history_path(candidate_dir: Path, path: Path) -> str:
    return f"history/{candidate_dir.name}/{path.relative_to(candidate_dir).as_posix()}"


def _path_if_exists(candidate_dir: Path, path: Path) -> str | None:
    return _history_path(candidate_dir, path) if path.exists() else None


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def _symlink_or_pointer(target: Path, link: Path) -> None:
    try:
        link.symlink_to(Path(os.path.relpath(target, link.parent)), target_is_directory=True)
    except OSError:
        link.write_text(f"See ../history/{target.name}\n", encoding="utf-8")


def _glob_if_any(candidate_dir: Path, directory: Path, pattern: str) -> str | None:
    if not list(directory.glob(pattern)):
        return None
    relative_dir = directory.relative_to(candidate_dir).as_posix()
    return f"history/{candidate_dir.name}/{relative_dir}/{pattern}"


def _exception_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [
        line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    useful = [line for line in lines if line and not line.startswith(("File ", "...<"))]
    if not useful:
        return None
    return useful[-1][:240]


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _extract_count(validation: dict[str, Any]) -> dict[str, Any]:
    for check in validation.get("checks", []):
        data = check.get("json")
        if isinstance(data, dict) and "physical_loc" in data:
            return data
    return {}


def _write_json(path: Path, payload: Any, sort_keys: bool = False) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
