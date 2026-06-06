from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolEvent:
    model: str
    candidate: str
    task: str
    attempt: str
    tool_name: str
    return_code: int | None
    wall_time_seconds: float | None
    original_token_count: int | None
    stdout_chars: int
    stderr_chars: int
    terminal_command_count: int
    patch_bytes: int
    intercepted_apply_patch: bool
    tmux_event: bool

    @property
    def trial_key(self) -> tuple[str, str, str, str]:
        return (self.model, self.candidate, self.task, self.attempt)

    @property
    def failed(self) -> bool:
        return isinstance(self.return_code, int) and self.return_code != 0


def aggregate_tool_usage(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    model_names = _model_name_map(run_root)
    events = list(_iter_tool_events(run_root, model_names))
    return {
        "run_root": str(run_root),
        "total_tool_events": len(events),
        "total_trials_with_tools": len({event.trial_key for event in events}),
        "by_tool": _summaries(events, ("tool_name",)),
        "by_model": _summaries(events, ("model",)),
        "by_candidate": _summaries(events, ("candidate",)),
        "by_task": _summaries(events, ("task",)),
        "by_model_candidate_tool": _summaries(events, ("model", "candidate", "tool_name")),
        "by_task_tool": _summaries(events, ("task", "tool_name")),
    }


def write_tool_usage_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tool_usage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for key in ("by_tool", "by_model", "by_candidate", "by_task"):
        _write_csv(out_dir / f"tool_usage_{key.removeprefix('by_')}.csv", summary[key])
    _write_csv(out_dir / "tool_usage_model_candidate_tool.csv", summary["by_model_candidate_tool"])
    _write_csv(out_dir / "tool_usage_task_tool.csv", summary["by_task_tool"])


def _iter_tool_events(run_root: Path, model_names: dict[str, str]) -> list[ToolEvent]:
    for path in _turn_log_paths(run_root):
        dims = _path_dimensions(run_root, path, model_names)
        if dims is None:
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        unified = metadata.get("unified_exec")
        unified = unified if isinstance(unified, dict) else {}
        yield ToolEvent(
            model=dims["model"],
            candidate=dims["candidate"],
            task=dims["task"],
            attempt=dims["attempt"],
            tool_name=str(payload.get("tool_name") or "unknown"),
            return_code=_optional_int(payload.get("return_code")),
            wall_time_seconds=_optional_float(unified.get("wall_time_seconds")),
            original_token_count=_optional_int(unified.get("original_token_count")),
            stdout_chars=len(str(payload.get("stdout") or "")),
            stderr_chars=len(str(payload.get("stderr") or "")),
            terminal_command_count=_optional_int(metadata.get("terminal_command_count")) or 0,
            patch_bytes=_optional_int(metadata.get("patch_bytes")) or 0,
            intercepted_apply_patch=bool(metadata.get("intercepted_apply_patch")),
            tmux_event=str(metadata.get("backend") or "") == "tmux",
        )


def _turn_log_paths(run_root: Path) -> list[Path]:
    manifest = _read_json(run_root / "manifest.json")
    if not isinstance(manifest, dict):
        return list(run_root.rglob("harness-turn-*.json"))
    models = [item for item in manifest.get("models", []) if isinstance(item, str)]
    candidates = [
        item.get("name")
        for item in manifest.get("candidates", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    tasks = [item for item in manifest.get("tasks", []) if isinstance(item, str)]
    if not models or not candidates or not tasks:
        return list(run_root.rglob("harness-turn-*.json"))
    paths: list[Path] = []
    for model in models:
        model_dir = model.replace(".", "_").replace("-", "_")
        for candidate in candidates:
            for task in tasks:
                task_dir = run_root / model_dir / candidate / task
                for attempt_dir in task_dir.glob("attempt_*"):
                    paths.extend(attempt_dir.glob("*/*/agent/harness-turn-*.json"))
    return paths


def _path_dimensions(
    run_root: Path,
    path: Path,
    model_names: dict[str, str],
) -> dict[str, str] | None:
    try:
        parts = path.relative_to(run_root).parts
    except ValueError:
        return None
    if len(parts) < 8:
        return None
    try:
        agent_index = parts.index("agent")
    except ValueError:
        return None
    if agent_index < 4:
        return None
    model_dir = parts[0]
    return {
        "model": model_names.get(model_dir, model_dir),
        "candidate": parts[1],
        "task": parts[2],
        "attempt": parts[3],
    }


def _model_name_map(run_root: Path) -> dict[str, str]:
    manifest = _read_json(run_root / "manifest.json")
    models = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(models, list):
        return {}
    result = {}
    for model in models:
        if isinstance(model, str):
            result[model.replace(".", "_").replace("-", "_")] = model
    return result


def _summaries(events: list[ToolEvent], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[ToolEvent]] = defaultdict(list)
    for event in events:
        groups[tuple(getattr(event, field) for field in fields)].append(event)
    rows = []
    for key, items in groups.items():
        row = {field: value for field, value in zip(fields, key, strict=True)}
        row.update(_metrics(items))
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in fields))


def _metrics(events: list[ToolEvent]) -> dict[str, Any]:
    failures = sum(1 for event in events if event.failed)
    return_none = sum(1 for event in events if event.return_code is None)
    wall_times = [event.wall_time_seconds for event in events if event.wall_time_seconds is not None]
    original_tokens = [
        event.original_token_count for event in events if event.original_token_count is not None
    ]
    return {
        "events": len(events),
        "trials": len({event.trial_key for event in events}),
        "failures": failures,
        "failure_rate": failures / len(events) if events else 0.0,
        "return_none": return_none,
        "runtime_observations": len(wall_times),
        "total_wall_time_seconds": sum(wall_times) if wall_times else None,
        "mean_wall_time_seconds": sum(wall_times) / len(wall_times) if wall_times else None,
        "total_stdout_chars": sum(event.stdout_chars for event in events),
        "total_stderr_chars": sum(event.stderr_chars for event in events),
        "original_token_observations": len(original_tokens),
        "total_original_token_count": sum(original_tokens) if original_tokens else None,
        "mean_original_token_count": (
            sum(original_tokens) / len(original_tokens) if original_tokens else None
        ),
        "total_terminal_command_count": sum(event.terminal_command_count for event in events),
        "total_patch_bytes": sum(event.patch_bytes for event in events),
        "intercepted_apply_patch_events": sum(1 for event in events if event.intercepted_apply_patch),
        "tmux_events": sum(1 for event in events if event.tmux_event),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_root
    summary = aggregate_tool_usage(args.run_root)
    write_tool_usage_outputs(summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
