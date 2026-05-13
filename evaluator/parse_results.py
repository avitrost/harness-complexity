from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_records(
    out_dir: Path | None = None, records_json: Path | None = None
) -> list[dict[str, Any]]:
    if records_json:
        return _records_from_payload(_read_json(records_json))
    records: list[dict[str, Any]] = []
    if out_dir and out_dir.exists():
        for path in out_dir.rglob("*.json"):
            if path.name in {"command.json", "records.json", "summary.json", "validation.json"}:
                continue
            for record in _records_from_payload(_read_json(path)):
                record["_source"] = str(path)
                records.append(record)
    return _strip_internal_fields(_renumber_trials(_dedupe(records)))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_normalize(item) for item in payload if _looks_like_record(item)]
    if isinstance(payload, dict):
        if isinstance(payload.get("trial_results"), list):
            return [
                _normalize(item) for item in payload["trial_results"] if _looks_like_record(item)
            ]
        for key in ("records", "trials", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_normalize(item) for item in value if _looks_like_record(item)]
        if _looks_like_record(payload):
            return [_normalize(payload)]
    return []


def _looks_like_record(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if "task_name" in item and ("trial_name" in item or "verifier_result" in item):
        return True
    return any(key in item for key in ("task", "task_id", "name")) and any(
        key in item for key in ("reward", "score", "success", "status")
    )


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    reward = _extract_reward(item)
    status = item.get("status")
    if not status:
        if item.get("exception_info"):
            status = "crash"
        else:
            status = "success" if reward == 1 else "failure" if reward == 0 else "unknown"
    return {
        "task": _task_name(item),
        "trial": _trial_number(item),
        "reward": 1 if float(reward or 0) >= 1 else 0,
        "status": status,
        "runtime_sec": _runtime_sec(item),
    }


def _extract_reward(item: dict[str, Any]) -> float | int | None:
    reward = item.get("reward", item.get("score"))
    if reward is None and "success" in item:
        reward = 1 if item["success"] else 0
    verifier = item.get("verifier_result")
    if reward is None and isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if isinstance(rewards, dict):
            for key in ("reward", "score", "success"):
                if key in rewards:
                    return rewards[key]
            numeric = [value for value in rewards.values() if isinstance(value, int | float)]
            if numeric:
                return numeric[0]
    return reward


def _task_name(item: dict[str, Any]) -> str:
    value = item.get("task") or item.get("task_name") or item.get("task_id") or item.get("name")
    if isinstance(value, dict):
        return str(value.get("name") or value.get("path") or value)
    return str(value)


def _trial_number(item: dict[str, Any]) -> int:
    value = item.get("trial") or item.get("trial_id")
    if value is not None:
        return int(value)
    trial_name = str(item.get("trial_name") or "")
    match = re.search(r"(\d+)(?!.*\d)", trial_name)
    return int(match.group(1)) if match else 1


def _runtime_sec(item: dict[str, Any]) -> float | None:
    runtime = item.get("runtime_sec") or item.get("duration_sec") or item.get("runtime")
    if runtime is not None:
        return float(runtime)
    started = _parse_datetime(item.get("started_at"))
    finished = _parse_datetime(item.get("finished_at"))
    if started and finished:
        return max(0.0, (finished - started).total_seconds())
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for record in records:
        key = (
            str(record.get("_source"))
            if record.get("_source")
            else json.dumps(record, sort_keys=True)
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _renumber_trials(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for record in records:
        task = str(record["task"])
        counts[task] = counts.get(task, 0) + 1
        record["trial"] = counts[task]
    return records


def _strip_internal_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record.pop("_source", None)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--records-json", type=Path)
    args = parser.parse_args()
    print(json.dumps(parse_records(args.out_dir, args.records_json), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
