from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

HARNESS_INFO = {
    "seed_minimal_agent": ("minimal", "minimal-agent", 100),
    "seed_mini_swe_agent_barebones": ("mini_bare", "msa-fully-compressed", 149),
    "seed_codex_400": ("c400", "codex-400", 398),
    "seed_mini_swe_agent_barebones_v2": ("bare_v2", "msa-prompt-compressed", 408),
    "seed_mini_swe_agent_barebones_v2_persistent": (
        "bbv2_bash_persistent",
        "msa-prompt-compressed-hidden-persistent-bash",
        408,
    ),
    "seed_mini_swe_agent_barebones_v2_rich_terminal": (
        "bbv2_rich_terminal",
        "msa-prompt-compressed-rich-terminal",
        408,
    ),
    "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples": (
        "bbv2_rich_no_examples",
        "msa-prompt-compressed-rich-terminal-no-examples",
        408,
    ),
    "seed_mini_swe_agent_v2": ("mini_v2", "mini-swe-agent", 478),
    "seed_terminus_2_compressed": ("term2_comp", "terminus-2-compressed", 634),
    "seed_codex_700": ("c700", "codex-700", 700),
    "seed_codex_1000": ("c1000", "codex-1000", 1000),
    "seed_codex_1300": ("c1300", "codex-1300", 1300),
    "seed_codex_compressed": ("c_comp", "codex-compressed", 1660),
    "seed_codex_full": ("c_full", "codex-full", 2210),
}


@dataclass(frozen=True)
class AttemptRecord:
    provider: str
    model: str
    effort: str
    candidate: str
    task: str
    attempt: str
    reward: str
    status: str
    failure_class: str
    out_dir: Path
    selected_run_dir: Path | None
    source_run_id: str
    api_cached_tokens: str
    api_input_tokens: str
    api_output_tokens: str
    api_total_tokens: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.task, self.attempt)

    @property
    def success(self) -> bool:
        return _to_float(self.reward) == 1.0 or self.status == "success"


def analyze_attempts(
    attempts_csv: Path,
    out_dir: Path,
    provider: str | None,
    model: str | None,
    effort: str | None,
    candidates: set[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    attempts = list(_load_attempts(attempts_csv, provider, model, effort, candidates))
    if limit is not None:
        attempts = attempts[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    attempt_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []

    for record in attempts:
        attempt_row, events, snippets = _analyze_attempt(record)
        attempt_rows.append(attempt_row)
        command_rows.extend(events)
        qualitative_rows.extend(snippets)

    by_harness = _aggregate_attempts(attempt_rows, ("candidate",))
    by_task_harness = _aggregate_attempts(attempt_rows, ("task", "candidate"))
    by_success_harness = _aggregate_attempts(attempt_rows, ("candidate", "status"))
    contrasts = _paired_contrasts(attempt_rows)
    summary = {
        "attempts_csv": str(attempts_csv),
        "filters": {
            "provider": provider,
            "model": model,
            "effort": effort,
            "candidates": sorted(candidates) if candidates else None,
            "limit": limit,
        },
        "attempts": len(attempt_rows),
        "commands": len(command_rows),
        "harnesses": sorted({row["candidate"] for row in attempt_rows}),
        "tasks": sorted({row["task"] for row in attempt_rows}),
        "outputs": {
            "attempts": str(out_dir / "attempt_behavior.csv"),
            "commands": str(out_dir / "command_events.csv"),
            "by_harness": str(out_dir / "by_harness.csv"),
            "by_task_harness": str(out_dir / "by_task_harness.csv"),
            "by_success_harness": str(out_dir / "by_success_harness.csv"),
            "paired_contrasts": str(out_dir / "paired_contrasts.csv"),
            "qualitative_cases": str(out_dir / "qualitative_cases.csv"),
            "report": str(out_dir / "behavior_report.md"),
        },
    }

    _write_csv(out_dir / "attempt_behavior.csv", attempt_rows)
    _write_csv(out_dir / "command_events.csv", command_rows)
    _write_csv(out_dir / "by_harness.csv", by_harness)
    _write_csv(out_dir / "by_task_harness.csv", by_task_harness)
    _write_csv(out_dir / "by_success_harness.csv", by_success_harness)
    _write_csv(out_dir / "paired_contrasts.csv", contrasts)
    _write_csv(out_dir / "qualitative_cases.csv", qualitative_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "behavior_report.md").write_text(
        _report(summary, by_harness, by_task_harness, by_success_harness, contrasts),
        encoding="utf-8",
    )
    return summary


def _load_attempts(
    attempts_csv: Path,
    provider: str | None,
    model: str | None,
    effort: str | None,
    candidates: set[str] | None,
) -> list[AttemptRecord]:
    rows = []
    with attempts_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if provider and row.get("provider") != provider:
                continue
            if model and row.get("model") != model:
                continue
            if effort and row.get("effort") != effort:
                continue
            if candidates and row.get("candidate") not in candidates:
                continue
            rows.append(
                AttemptRecord(
                    provider=row.get("provider", ""),
                    model=row.get("model", ""),
                    effort=row.get("effort", ""),
                    candidate=row.get("candidate", ""),
                    task=row.get("task", ""),
                    attempt=row.get("attempt", ""),
                    reward=row.get("reward", ""),
                    status=row.get("status", ""),
                    failure_class=row.get("failure_class", ""),
                    out_dir=Path(row.get("out_dir", "")),
                    selected_run_dir=_path_or_none(row.get("selected_run_dir", "")),
                    source_run_id=row.get("source_run_id", ""),
                    api_cached_tokens=row.get("api_cached_tokens", ""),
                    api_input_tokens=row.get("api_input_tokens", ""),
                    api_output_tokens=row.get("api_output_tokens", ""),
                    api_total_tokens=row.get("api_total_tokens", ""),
                )
            )
    return rows


def _analyze_attempt(
    record: AttemptRecord,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_run_dir = (
        _selected_run_dir(record.selected_run_dir)
        if record.selected_run_dir
        else _selected_run_dir(record.out_dir)
    )
    agent_dir = _agent_dir(selected_run_dir) if selected_run_dir else None
    result = _read_json(agent_dir / "harness-result.json") if agent_dir else {}
    result = result if isinstance(result, dict) else {}
    accounting = (
        result.get("model_accounting") if isinstance(result.get("model_accounting"), dict) else {}
    )

    turn_paths = sorted(agent_dir.glob("harness-turn-*.json")) if agent_dir else []
    model_paths = sorted(agent_dir.glob("model-call-*.json")) if agent_dir else []
    turns = [_read_json(path) for path in turn_paths]
    turns = [turn for turn in turns if isinstance(turn, dict)]
    model_calls = [_read_json(path) for path in model_paths]
    model_calls = [call for call in model_calls if isinstance(call, dict)]

    command_rows = []
    category_totals: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    rc_counts: Counter[str] = Counter()
    failed_commands = 0
    none_return_commands = 0
    first_failure_turn: int | None = None
    stdout_chars = 0
    stderr_chars = 0
    max_stdout_chars = 0
    max_stderr_chars = 0
    assistant_content_chars = 0
    assistant_content_turns = 0
    response_item_turns = 0
    response_items_total = 0
    response_replay_turns = 0
    format_retry_turns = 0
    format_retry_total = 0
    format_retry_max = 0
    recovery_turns = 0
    output_only_turns = 0
    env_prefix_commands = 0
    session_commands = 0
    sequential_tool_call_turns = 0
    emitted_tool_calls_total = 0
    emitted_tool_calls_max = 0
    multi_call_turns = 0
    compacted_turns = 0
    compaction_summary_chars = 0
    pruned_items_total = 0
    prompt_estimated_tokens_total = 0
    prompt_estimated_tokens_max = 0
    prompt_stats_turns = 0

    for index, turn in enumerate(turns, start=1):
        metadata = _dict(turn.get("metadata"))
        command = str(turn.get("command") or "")
        command_clean = _clean_command(command)
        tool_name = str(turn.get("tool_name") or "unknown")
        return_code = _optional_int(turn.get("return_code"))
        stdout_len = len(str(turn.get("stdout") or ""))
        stderr_len = len(str(turn.get("stderr") or ""))
        stdout_chars += stdout_len
        stderr_chars += stderr_len
        max_stdout_chars = max(max_stdout_chars, stdout_len)
        max_stderr_chars = max(max_stderr_chars, stderr_len)
        tool_counts[tool_name] += 1
        rc_counts[_return_code_bucket(return_code)] += 1
        if return_code is None:
            none_return_commands += 1
        elif return_code != 0:
            failed_commands += 1
            if first_failure_turn is None:
                first_failure_turn = index

        categories = _command_categories(tool_name, command_clean, metadata)
        category_totals.update(categories)
        if metadata.get("assistant_content"):
            assistant_content_turns += 1
            assistant_content_chars += len(str(metadata.get("assistant_content") or ""))
        response_items = metadata.get("mini_swe_agent_v2_response_items") or metadata.get(
            "codex_response_items"
        )
        if isinstance(response_items, list) and response_items:
            response_item_turns += 1
            response_items_total += len(response_items)
        if metadata.get("mini_swe_agent_v2_messages") or metadata.get("codex_response_items"):
            response_replay_turns += 1
        retry_count = _optional_int(metadata.get("mini_swe_agent_v2_format_retries")) or 0
        if retry_count:
            format_retry_turns += 1
            format_retry_total += retry_count
            format_retry_max = max(format_retry_max, retry_count)
        if metadata.get("codex_recovery"):
            recovery_turns += 1
        if metadata.get("codex_output_only"):
            output_only_turns += 1
        if metadata.get("sequential_tool_calls"):
            sequential_tool_call_turns += 1
        emitted_tool_calls = _optional_int(metadata.get("codex_emitted_tool_calls"))
        if emitted_tool_calls is not None:
            emitted_tool_calls_total += emitted_tool_calls
            emitted_tool_calls_max = max(emitted_tool_calls_max, emitted_tool_calls)
            if emitted_tool_calls > 1:
                multi_call_turns += 1
        port_stats = _dict(metadata.get("codex_port_stats"))
        if port_stats:
            prompt_stats_turns += 1
            if port_stats.get("compacted"):
                compacted_turns += 1
            compaction_summary_chars += (
                _optional_int(port_stats.get("compaction_summary_chars")) or 0
            )
            pruned_items_total += _optional_int(port_stats.get("pruned_items")) or 0
            estimated_tokens = _optional_int(port_stats.get("estimated_tokens"))
            if estimated_tokens is not None:
                prompt_estimated_tokens_total += estimated_tokens
                prompt_estimated_tokens_max = max(prompt_estimated_tokens_max, estimated_tokens)
        if command.startswith("export PAGER=cat"):
            env_prefix_commands += 1
        unified = _dict(metadata.get("unified_exec"))
        if unified.get("session_id") is not None:
            session_commands += 1

        command_rows.append(
            {
                "provider": record.provider,
                "model": record.model,
                "effort": record.effort,
                "candidate": record.candidate,
                "task": record.task,
                "attempt": record.attempt,
                "reward": record.reward,
                "status": record.status,
                "turn_index": index,
                "tool_name": tool_name,
                "return_code": "" if return_code is None else return_code,
                "stdout_chars": stdout_len,
                "stderr_chars": stderr_len,
                "command_chars": len(command_clean),
                "command_hash": _hash(command_clean),
                "command_head": command_clean[:260].replace("\n", "\\n"),
                **{f"is_{category}": int(category in categories) for category in _all_categories()},
            }
        )

    model_usage = _model_usage(model_calls)
    model_call_no_tool = sum(1 for call in model_calls if not call.get("tool_calls"))
    model_call_response_chars = sum(len(str(call.get("response") or "")) for call in model_calls)
    commands_after_first_failure = (
        max(0, len(turns) - first_failure_turn) if first_failure_turn is not None else 0
    )
    alias, harness_name, harness_loc = _harness_info(record.candidate)

    attempt_row = {
        "provider": record.provider,
        "model": record.model,
        "effort": record.effort,
        "candidate": record.candidate,
        "harness_alias": alias,
        "harness_name": harness_name,
        "harness_loc": harness_loc,
        "task": record.task,
        "attempt": record.attempt,
        "reward": record.reward,
        "success": int(record.success),
        "status": record.status,
        "failure_class": record.failure_class,
        "source_run_id": record.source_run_id,
        "out_dir": str(record.out_dir),
        "selected_run_dir": str(selected_run_dir or ""),
        "agent_dir": str(agent_dir or ""),
        "trace_missing": int(agent_dir is None),
        "api_cached_tokens": record.api_cached_tokens,
        "api_input_tokens": record.api_input_tokens,
        "api_output_tokens": record.api_output_tokens,
        "api_total_tokens": record.api_total_tokens,
        "result_done": result.get("done", ""),
        "result_turns": result.get("turns", ""),
        "result_elapsed_sec": result.get("elapsed_sec", ""),
        "result_model_calls": accounting.get("model_calls", ""),
        "result_input_tokens": accounting.get("input_tokens", ""),
        "result_output_tokens": accounting.get("output_tokens", ""),
        "result_cached_tokens": accounting.get("cached_tokens", ""),
        "result_total_tokens": accounting.get("total_tokens", ""),
        "turn_logs": len(turns),
        "model_call_logs": len(model_calls),
        "model_call_no_tool": model_call_no_tool,
        "model_call_response_chars": model_call_response_chars,
        "model_call_input_tokens": model_usage["input_tokens"],
        "model_call_output_tokens": model_usage["output_tokens"],
        "model_call_cached_tokens": model_usage["cached_tokens"],
        "model_call_total_tokens": model_usage["total_tokens"],
        "commands": len(turns),
        "failed_commands": failed_commands,
        "none_return_commands": none_return_commands,
        "first_failure_turn": "" if first_failure_turn is None else first_failure_turn,
        "commands_after_first_failure": commands_after_first_failure,
        "stdout_chars": stdout_chars,
        "stderr_chars": stderr_chars,
        "max_stdout_chars": max_stdout_chars,
        "max_stderr_chars": max_stderr_chars,
        "assistant_content_turns": assistant_content_turns,
        "assistant_content_chars": assistant_content_chars,
        "response_item_turns": response_item_turns,
        "response_items_total": response_items_total,
        "response_replay_turns": response_replay_turns,
        "format_retry_turns": format_retry_turns,
        "format_retry_total": format_retry_total,
        "format_retry_max": format_retry_max,
        "recovery_turns": recovery_turns,
        "output_only_turns": output_only_turns,
        "env_prefix_commands": env_prefix_commands,
        "session_commands": session_commands,
        "sequential_tool_call_turns": sequential_tool_call_turns,
        "emitted_tool_calls_total": emitted_tool_calls_total,
        "emitted_tool_calls_max": emitted_tool_calls_max,
        "multi_call_turns": multi_call_turns,
        "compacted_turns": compacted_turns,
        "compaction_summary_chars": compaction_summary_chars,
        "pruned_items_total": pruned_items_total,
        "prompt_stats_turns": prompt_stats_turns,
        "prompt_estimated_tokens_mean": (
            prompt_estimated_tokens_total / prompt_stats_turns if prompt_stats_turns else 0
        ),
        "prompt_estimated_tokens_max": prompt_estimated_tokens_max,
        **{f"tool_{name}": tool_counts[name] for name in sorted(tool_counts)},
        **{f"rc_{name}": rc_counts[name] for name in sorted(rc_counts)},
        **{f"{category}_commands": category_totals[category] for category in _all_categories()},
    }

    snippets = _qualitative_snippets(record, attempt_row, command_rows)
    return attempt_row, command_rows, snippets


def _selected_run_dir(out_dir: Path) -> Path | None:
    if not out_dir:
        return None
    if (out_dir / "result.json").exists() and _agent_dir(out_dir):
        return out_dir
    candidates = []
    try:
        children = list(out_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        if (child / "result.json").exists() or _agent_dir(child):
            candidates.append(child)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.name, path.stat().st_mtime))[-1]


def _agent_dir(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    if (run_dir / "agent").is_dir():
        return run_dir / "agent"
    try:
        matches = sorted(path for path in run_dir.glob("*/agent") if path.is_dir())
    except OSError:
        return None
    return matches[0] if matches else None


def _command_categories(tool_name: str, command: str, metadata: dict[str, Any]) -> set[str]:
    lowered = command.lower()
    categories: set[str] = set()
    if SUBMIT_MARKER.lower() in lowered:
        categories.add("submit")
    if tool_name not in {"bash", "persistent_bash", "local_shell", "exec_command", "shell_command"}:
        categories.add("function")
    if tool_name in {"write_stdin"}:
        categories.add("interactive")
    if metadata.get("codex_recovery"):
        categories.add("recovery")
    if any(token in lowered for token in ("rg ", "grep ", "find ", "locate ")):
        categories.add("search")
    if _matches_any(
        lowered,
        (
            r"\bls\b",
            r"\bpwd\b",
            r"\bfind\b",
            r"\brg\b",
            r"\bgrep\b",
            r"\bcat\b",
            r"\bhead\b",
            r"\btail\b",
            r"\bnl\b",
            r"\bsed\s+-n\b",
            r"\bwc\b",
            r"\bstat\b",
            r"\bfile\b",
            r"\btree\b",
            r"\bdu\b",
            r"\bdf\b",
            r"\bwhich\b",
        ),
    ):
        categories.add("inspect")
    if _matches_any(lowered, (r"\bcat\b", r"\bhead\b", r"\btail\b", r"\bnl\b", r"\bsed\s+-n\b")):
        categories.add("read_file")
    if _matches_any(
        lowered,
        (
            r"\bpython(?:3)?\b",
            r"\brscript\b",
            r"\bnode\b",
            r"\bruby\b",
            r"\bperl\b",
            r"\bawk\b",
            r"\bbash\s+-c\b",
            r"\bsh\s+-c\b",
        ),
    ):
        categories.add("script")
    if _matches_any(
        lowered,
        (
            r"\bpytest\b",
            r"\bunittest\b",
            r"\bnpm\s+test\b",
            r"\bcargo\s+test\b",
            r"\bgo\s+test\b",
            r"\bmake\s+test\b",
            r"\bbats\b",
            r"\bctest\b",
            r"\bprove\b",
            r"\btest_",
            r"\bverify\b",
        ),
    ):
        categories.add("test")
    if _is_edit_command(lowered):
        categories.add("edit")
    if _matches_any(
        lowered,
        (
            r"\bpip\s+install\b",
            r"\buv\s+pip\s+install\b",
            r"\bapt(?:-get)?\s+install\b",
            r"\bnpm\s+install\b",
            r"\bconda\s+install\b",
        ),
    ):
        categories.add("install")
    if not categories:
        categories.add("other")
    return categories


def _is_edit_command(lowered: str) -> bool:
    if "apply_patch" in lowered or "sed -i" in lowered or "perl -pi" in lowered:
        return True
    if re.search(r"(?:^|[;&|]\s*)(?:cat|tee|printf|echo)\b[^|]*(?:>|>>)", lowered):
        return True
    if re.search(r"\b(?:mv|cp|chmod|mkdir|rm|touch)\b", lowered):
        return True
    if any(token in lowered for token in (".write(", "write_text(", "to_csv(", "json.dump(")):
        return True
    if re.search(r"open\([^)]*[\"']w", lowered):
        return True
    return False


def _all_categories() -> tuple[str, ...]:
    return (
        "edit",
        "function",
        "inspect",
        "install",
        "interactive",
        "other",
        "read_file",
        "recovery",
        "script",
        "search",
        "submit",
        "test",
    )


def _model_usage(model_calls: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    for call in model_calls:
        metadata = _dict(call.get("request_metadata"))
        usage = _dict(metadata.get("usage"))
        for key in totals:
            value = _optional_int(usage.get(key))
            if value is not None:
                totals[key] += value
    return totals


def _aggregate_attempts(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in fields)].append(row)

    result = []
    numeric_fields = (
        "success",
        "commands",
        "failed_commands",
        "commands_after_first_failure",
        "format_retry_total",
        "format_retry_turns",
        "response_replay_turns",
        "env_prefix_commands",
        "session_commands",
        "sequential_tool_call_turns",
        "emitted_tool_calls_total",
        "multi_call_turns",
        "compacted_turns",
        "pruned_items_total",
        "prompt_estimated_tokens_mean",
        "prompt_estimated_tokens_max",
        "inspect_commands",
        "search_commands",
        "read_file_commands",
        "script_commands",
        "edit_commands",
        "test_commands",
        "submit_commands",
        "stdout_chars",
        "stderr_chars",
        "api_input_tokens",
        "api_output_tokens",
        "api_total_tokens",
    )
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        row = {field: value for field, value in zip(fields, key, strict=True)}
        if "candidate" in fields:
            candidate = str(row["candidate"])
            alias, harness_name, harness_loc = _harness_info(candidate)
            row["harness_alias"] = alias
            row["harness_name"] = harness_name
            row["harness_loc"] = harness_loc
        row["attempts"] = len(items)
        row["successes"] = sum(_num(item.get("success")) for item in items)
        row["success_rate"] = _mean([_num(item.get("success")) for item in items])
        for field in numeric_fields:
            values = [_num(item.get(field)) for item in items]
            row[f"mean_{field}"] = _mean(values)
            row[f"sum_{field}"] = sum(values)
        result.append(row)
    return result


def _paired_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["task"]), str(row["attempt"]))][str(row["candidate"])] = row
    candidates = sorted({str(row["candidate"]) for row in rows})
    if len(candidates) != 2:
        return []
    left, right = candidates
    result = []
    metrics = (
        "success",
        "commands",
        "failed_commands",
        "commands_after_first_failure",
        "format_retry_total",
        "response_replay_turns",
        "inspect_commands",
        "search_commands",
        "script_commands",
        "edit_commands",
        "test_commands",
        "api_input_tokens",
        "api_output_tokens",
        "api_total_tokens",
    )
    for (task, attempt), group in sorted(by_key.items()):
        if left not in group or right not in group:
            continue
        lrow = group[left]
        rrow = group[right]
        row: dict[str, Any] = {
            "task": task,
            "attempt": attempt,
            "left_candidate": left,
            "right_candidate": right,
            "left_status": lrow["status"],
            "right_status": rrow["status"],
            "left_success": lrow["success"],
            "right_success": rrow["success"],
            "winner": (
                right
                if _num(rrow["success"]) > _num(lrow["success"])
                else left if _num(lrow["success"]) > _num(rrow["success"]) else "tie"
            ),
        }
        for metric in metrics:
            row[f"{left}.{metric}"] = lrow.get(metric, "")
            row[f"{right}.{metric}"] = rrow.get(metric, "")
            row[f"delta_{metric}_right_minus_left"] = _num(rrow.get(metric)) - _num(
                lrow.get(metric)
            )
        result.append(row)
    return result


def _qualitative_snippets(
    record: AttemptRecord,
    attempt_row: dict[str, Any],
    command_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snippets = []
    if (
        record.success
        or _num(attempt_row.get("format_retry_total"))
        or _num(attempt_row.get("failed_commands"))
    ):
        heads = [row["command_head"] for row in command_rows[:6]]
        tails = [row["command_head"] for row in command_rows[-3:]]
        snippets.append(
            {
                "provider": record.provider,
                "model": record.model,
                "effort": record.effort,
                "candidate": record.candidate,
                "task": record.task,
                "attempt": record.attempt,
                "success": int(record.success),
                "commands": attempt_row["commands"],
                "failed_commands": attempt_row["failed_commands"],
                "format_retry_total": attempt_row["format_retry_total"],
                "first_commands": " || ".join(heads),
                "last_commands": " || ".join(tails),
                "selected_run_dir": attempt_row["selected_run_dir"],
            }
        )
    return snippets


def _report(
    summary: dict[str, Any],
    by_harness: list[dict[str, Any]],
    by_task_harness: list[dict[str, Any]],
    by_success_harness: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> str:
    lines = [
        "# Trace Behavior Report",
        "",
        f"Attempts: {summary['attempts']}",
        f"Commands: {summary['commands']}",
        f"Harnesses: {', '.join(summary['harnesses'])}",
        "",
        "## By Harness",
        "",
        _markdown_table(
            by_harness,
            (
                "candidate",
                "harness_loc",
                "harness_name",
                "attempts",
                "successes",
                "success_rate",
                "mean_commands",
                "mean_failed_commands",
                "mean_session_commands",
                "mean_emitted_tool_calls_total",
                "mean_multi_call_turns",
                "mean_compacted_turns",
                "mean_format_retry_total",
                "mean_response_replay_turns",
                "mean_edit_commands",
                "mean_test_commands",
                "mean_api_input_tokens",
                "mean_api_output_tokens",
            ),
        ),
        "",
        "## Task Scores",
        "",
        _markdown_table(
            by_task_harness,
            ("task", "candidate", "attempts", "successes", "success_rate", "mean_commands"),
        ),
        "",
        "## Success-Conditioned Behavior",
        "",
        _markdown_table(
            by_success_harness,
            (
                "candidate",
                "status",
                "attempts",
                "success_rate",
                "mean_commands",
                "mean_failed_commands",
                "mean_edit_commands",
                "mean_test_commands",
            ),
        ),
        "",
        "## Paired Contrasts",
        "",
        _contrast_summary(contrasts),
        "",
        "## Files",
        "",
    ]
    for name, path in summary["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _contrast_summary(contrasts: list[dict[str, Any]]) -> str:
    if not contrasts:
        return "No paired contrasts were generated."
    winners = Counter(str(row["winner"]) for row in contrasts)
    lines = [
        f"Paired cells: {len(contrasts)}",
        "",
        *[f"- {winner}: {count}" for winner, count in sorted(winners.items())],
        "",
        "Different-outcome cells:",
        "",
    ]
    changed = [row for row in contrasts if row["winner"] != "tie"]
    for row in changed[:30]:
        lines.append(
            "- "
            f"{row['task']} attempt {row['attempt']}: {row['winner']} "
            f"(delta commands {row.get('delta_commands_right_minus_left')}, "
            f"delta edit {row.get('delta_edit_commands_right_minus_left')}, "
            f"delta retry {row.get('delta_format_retry_total_right_minus_left')})"
        )
    if len(changed) > 30:
        lines.append(f"- ... {len(changed) - 30} more")
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "(none)"
    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def _clean_command(command: str) -> str:
    cleaned = command.strip()
    prefix = "export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _return_code_bucket(value: int | None) -> str:
    if value is None:
        return "none"
    if value == 0:
        return "zero"
    if value in {1, 2, 127, 124}:
        return str(value)
    return "other_nonzero"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_or_none(value: str | None) -> Path | None:
    if value in {None, "", "N/A"}:
        return None
    return Path(str(value))


def _harness_info(candidate: str) -> tuple[str, str, int | str]:
    return HARNESS_INFO.get(candidate, (candidate, candidate, ""))


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "provider",
        "model",
        "effort",
        "candidate",
        "harness_alias",
        "harness_name",
        "harness_loc",
        "task",
        "attempt",
        "status",
        "reward",
        "success",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in {None, "", "N/A"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in {None, "", "N/A"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    parsed = _to_float(value)
    if parsed is None or math.isnan(parsed):
        return 0.0
    return parsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    summary = analyze_attempts(
        attempts_csv=args.attempts_csv,
        out_dir=args.out_dir,
        provider=args.provider,
        model=args.model,
        effort=args.effort,
        candidates=set(args.candidates) if args.candidates else None,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
