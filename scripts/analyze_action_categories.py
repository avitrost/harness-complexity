from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "behavior_gpt54_low_family_20260612"
)

DEFAULT_SOURCES = (
    (
        "tb2",
        "gpt-5.4-mini",
        "low",
        DEFAULT_BASE / "gpt54mini_low_tb2_allharnesses",
    ),
    ("tb2", "gpt-5.4", "low", DEFAULT_BASE / "gpt54_low_tb2_allharnesses"),
    (
        "tblite",
        "gpt-5.4-mini",
        "low",
        DEFAULT_BASE / "gpt54mini_low_tblite_barev2_vs_codexfull",
    ),
    (
        "tblite",
        "gpt-5.4",
        "low",
        DEFAULT_BASE / "gpt54_low_tblite_barev2_vs_codexfull",
    ),
)

CATEGORIES = (
    "test_execution",
    "file_edit",
    "dependency_setup",
    "git_operation",
    "search",
    "file_read",
    "script_execution",
    "navigation",
    "interrupt_abort",
    "reasoning_only",
)

SUBMISSION_MARKERS = (
    "complete_task_and_submit",
    "complete_task_and_submit_final_output",
    "mark_task_complete",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the HarnessBridge-style mutually-exclusive action category taxonomy "
            "to local trace CSVs."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BASE / "action_category_analysis")
    parser.add_argument(
        "--source",
        action="append",
        help="Source as benchmark:model:effort:/path/to/analysis_dir. Defaults to current gpt-5.4 low-family outputs.",
    )
    args = parser.parse_args()

    sources = _parse_sources(args.source) if args.source else list(DEFAULT_SOURCES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    action_events: list[dict[str, Any]] = []
    attempt_profiles: list[dict[str, Any]] = []
    for benchmark, model, effort, source_dir in sources:
        events, profiles = _load_source(benchmark, model, effort, source_dir)
        action_events.extend(events)
        attempt_profiles.extend(profiles)

    included = [row for row in action_events if row["included_in_category_analysis"] == "1"]
    by_model = _aggregate_categories(
        included,
        ("benchmark", "model", "effort", "action_category"),
        ("benchmark", "model", "effort"),
        attempt_profiles,
    )
    by_harness = _aggregate_categories(
        included,
        (
            "benchmark",
            "model",
            "effort",
            "candidate",
            "harness_alias",
            "harness_name",
            "harness_loc",
            "action_category",
        ),
        ("benchmark", "model", "effort", "candidate"),
        attempt_profiles,
    )
    by_success = _aggregate_categories(
        included,
        (
            "benchmark",
            "model",
            "effort",
            "candidate",
            "harness_alias",
            "harness_name",
            "harness_loc",
            "success",
            "action_category",
        ),
        ("benchmark", "model", "effort", "candidate", "success"),
        attempt_profiles,
    )
    matrix = _harness_matrix(included, attempt_profiles)
    summary = _summary(sources, action_events, included)

    _write_csv(args.out_dir / "action_events.csv", action_events)
    _write_csv(args.out_dir / "attempt_action_profiles.csv", attempt_profiles)
    _write_csv(args.out_dir / "action_category_by_model.csv", by_model)
    _write_csv(args.out_dir / "action_category_by_harness.csv", by_harness)
    _write_csv(args.out_dir / "action_category_by_harness_success.csv", by_success)
    _write_csv(args.out_dir / "harness_action_category_matrix.csv", matrix)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "action_category_report.md").write_text(
        _report(summary, by_model, matrix),
        encoding="utf-8",
    )

    print(json.dumps({"out_dir": str(args.out_dir), **summary}, indent=2))
    return 0


def _parse_sources(values: list[str]) -> list[tuple[str, str, str, Path]]:
    sources = []
    for value in values:
        parts = value.split(":", 3)
        if len(parts) != 4:
            raise SystemExit(f"Invalid --source {value!r}; expected benchmark:model:effort:path")
        benchmark, model, effort, path = parts
        sources.append((benchmark, model, effort, Path(path)))
    return sources


def _load_source(
    benchmark: str,
    model: str,
    effort: str,
    source_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts = _load_attempts(source_dir / "attempt_behavior.csv")
    events: list[dict[str, Any]] = []
    profile_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    submission_counts: Counter[tuple[str, str, str]] = Counter()

    with (source_dir / "command_events.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["candidate"], row["task"], str(row["attempt"]))
            attempt = attempts.get(key, {})
            category, reason = classify_action(row)
            is_submission = _is_submission(row)
            included = not is_submission
            profile_key = (row["candidate"], row["task"], str(row["attempt"]))
            if included:
                profile_counts[profile_key][category] += 1
            else:
                submission_counts[profile_key] += 1

            events.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "effort": effort,
                    "candidate": row["candidate"],
                    "harness_alias": attempt.get("harness_alias", ""),
                    "harness_name": attempt.get("harness_name", ""),
                    "harness_loc": attempt.get("harness_loc", ""),
                    "task": row["task"],
                    "attempt": row["attempt"],
                    "success": attempt.get("success", _success_from_row(row)),
                    "status": row.get("status", attempt.get("status", "")),
                    "reward": row.get("reward", attempt.get("reward", "")),
                    "turn_index": row.get("turn_index", ""),
                    "tool_name": row.get("tool_name", ""),
                    "return_code": row.get("return_code", ""),
                    "stdout_chars": row.get("stdout_chars", ""),
                    "stderr_chars": row.get("stderr_chars", ""),
                    "command_chars": row.get("command_chars", ""),
                    "command_hash": row.get("command_hash", ""),
                    "command_head": row.get("command_head", ""),
                    "action_category": category,
                    "category_match_reason": reason,
                    "is_submission": "1" if is_submission else "0",
                    "included_in_category_analysis": "1" if included else "0",
                }
            )

    profiles = []
    for key, attempt in sorted(attempts.items()):
        counts = profile_counts.get(key, Counter())
        total = sum(counts.values())
        row = {
            "benchmark": benchmark,
            "model": model,
            "effort": effort,
            "candidate": key[0],
            "harness_alias": attempt.get("harness_alias", ""),
            "harness_name": attempt.get("harness_name", ""),
            "harness_loc": attempt.get("harness_loc", ""),
            "task": key[1],
            "attempt": key[2],
            "success": attempt.get("success", ""),
            "status": attempt.get("status", ""),
            "included_actions": total,
            "submission_actions": submission_counts[key],
        }
        for category in CATEGORIES:
            count = counts[category]
            row[f"{category}_count"] = count
            row[f"{category}_pct"] = count / total if total else 0.0
        profiles.append(row)
    return events, profiles


def _load_attempts(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    attempts = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            attempts[(row["candidate"], row["task"], str(row["attempt"]))] = row
    return attempts


def classify_action(row: dict[str, str]) -> tuple[str, str]:
    command = _command(row)
    lowered = command.lower()
    if _is_submission(row):
        return "submission", "submission_marker"
    if not lowered:
        return "reasoning_only", "empty_command"
    if _matches(
        lowered,
        (
            r"\bpytest\b",
            r"\bpython3?\s+-m\s+pytest\b",
            r"\bmake\b[^\n;&|]*\btest\b",
            r"\bcargo\s+test\b",
            r"\bnpm\s+(?:run\s+)?test\b",
            r"\bgo\s+test\b",
            r"\bctest\b",
            r"\bbats\b",
            r"\bprove\b",
        ),
    ) or row.get("is_test") == "1":
        return "test_execution", "test_pattern"
    if row.get("tool_name") == "apply_patch" or _matches(
        lowered,
        (
            r"\bapply_patch\b",
            r"\bpatch\b",
            r"\bsed\s+-i\b",
            r"\bperl\s+-pi\b",
            r"(?:^|[;&|]\s*)cat\b[^|]*(?:>|>>)",
            r"\bcat\s+<<[^|]*(?:>|>>)",
            r"\btee\b",
            r"(?:^|[;&|]\s*)echo\b[^|]*(?:>|>>)",
            r"(?:^|[;&|]\s*)printf\b[^|]*(?:>|>>)",
        ),
    ) or row.get("is_edit") == "1":
        return "file_edit", "edit_pattern"
    if _matches(
        lowered,
        (
            r"\bpip\s+install\b",
            r"\buv\s+pip\s+install\b",
            r"\bapt(?:-get)?\s+install\b",
            r"\bnpm\s+install\b",
            r"\bcmake\b",
            r"\bgcc\b",
            r"\bg\+\+\b",
            r"\bmake\b",
            r"\bgit\s+clone\b",
        ),
    ) or row.get("is_install") == "1":
        return "dependency_setup", "dependency_pattern"
    if _matches(
        lowered,
        (
            r"(?:^|[;&|]\s*)git\s+diff\b",
            r"(?:^|[;&|]\s*)git\s+log\b",
            r"(?:^|[;&|]\s*)git\s+status\b",
            r"(?:^|[;&|]\s*)git\s+commit\b",
            r"(?:^|[;&|]\s*)git\s+add\b",
            r"(?:^|[;&|]\s*)git\s+show\b",
            r"(?:^|[;&|]\s*)git\s+blame\b",
        ),
    ):
        return "git_operation", "git_pattern"
    if _matches(
        lowered,
        (
            r"\bgrep\b",
            r"\brg\b",
            r"\bag\b",
            r"\back\b",
            r"\bfind\b[^\n;&|]*(?:-name|-type)\b",
        ),
    ) or row.get("is_search") == "1":
        return "search", "search_pattern"
    if _matches(
        lowered,
        (
            r"(?:^|[;&|]\s*)cat\b(?![^;&|]*(?:>|>>))",
            r"(?:^|[;&|]\s*)head\b",
            r"(?:^|[;&|]\s*)tail\b",
            r"(?:^|[;&|]\s*)sed\s+-n\b",
            r"(?:^|[;&|]\s*)less\b",
            r"(?:^|[;&|]\s*)more\b",
            r"(?:^|[;&|]\s*)nl\b",
        ),
    ) or row.get("is_read_file") == "1":
        return "file_read", "file_read_pattern"
    if _matches(
        lowered,
        (
            r"\bpython3?\b",
            r"\bnode\b",
            r"\bbash\b",
            r"\bsh\s+-c\b",
            r"\brscript\b",
            r"\bsqlite3\b",
            r"\bruby\b",
            r"\bperl\b",
            r"\bawk\b",
        ),
    ) or row.get("is_script") == "1":
        return "script_execution", "script_pattern"
    if _matches(
        lowered,
        (
            r"(?:^|[;&|]\s*)ls\b",
            r"(?:^|[;&|]\s*)pwd\b",
            r"(?:^|[;&|]\s*)tree\b",
            r"(?:^|[;&|]\s*)cd\b",
        ),
    ):
        return "navigation", "navigation_pattern"
    stripped = lowered.strip()
    if stripped in {"q", "exit"} or _matches(lowered, (r"\^c", r"\bkill\b", r"\bpkill\b")):
        return "interrupt_abort", "interrupt_pattern"
    if row.get("tool_name") == "update_plan":
        return "reasoning_only", "planning_tool"
    return "reasoning_only", "residual_no_table6_pattern"


def _is_submission(row: dict[str, str]) -> bool:
    if row.get("is_submit") == "1":
        return True
    lowered = _command(row).lower()
    return any(marker in lowered for marker in SUBMISSION_MARKERS)


def _command(row: dict[str, str]) -> str:
    command = str(row.get("command_head") or "").replace("\\n", "\n").strip()
    prefix = "export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
    if command.startswith(prefix):
        command = command[len(prefix) :].strip()
    return command


def _aggregate_categories(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    denominator_fields: tuple[str, ...],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    denom: Counter[tuple[Any, ...]] = Counter()
    attempts: dict[tuple[Any, ...], set[tuple[str, str, str]]] = defaultdict(set)
    successes: dict[tuple[Any, ...], set[tuple[str, str, str]]] = defaultdict(set)
    for profile in profile_rows:
        dkey = tuple(profile.get(field, "") for field in denominator_fields)
        akey = (str(profile["candidate"]), str(profile["task"]), str(profile["attempt"]))
        attempts[dkey].add(akey)
        if str(profile.get("success")) == "1":
            successes[dkey].add(akey)
    for row in rows:
        gkey = tuple(row.get(field, "") for field in fields)
        dkey = tuple(row.get(field, "") for field in denominator_fields)
        group[gkey].append(row)
        denom[dkey] += 1

    out = []
    for key, items in sorted(group.items(), key=lambda item: tuple(str(part) for part in item[0])):
        row = {field: value for field, value in zip(fields, key, strict=True)}
        dkey = tuple(row.get(field, "") for field in denominator_fields)
        turn_count = len(items)
        attempt_count = len(attempts[dkey])
        row["turns"] = turn_count
        row["pct_turns"] = turn_count / denom[dkey] if denom[dkey] else 0.0
        row["attempts"] = attempt_count
        row["turns_per_attempt"] = turn_count / attempt_count if attempt_count else 0.0
        row["attempt_success_rate"] = (
            len(successes[dkey]) / attempt_count if attempt_count else 0.0
        )
        row["mean_stdout_chars"] = _mean(_num(item.get("stdout_chars")) for item in items)
        row["mean_stderr_chars"] = _mean(_num(item.get("stderr_chars")) for item in items)
        out.append(row)
    return out


def _harness_matrix(
    rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    attempts: dict[tuple[Any, ...], set[tuple[str, str, str]]] = defaultdict(set)
    successes: dict[tuple[Any, ...], set[tuple[str, str, str]]] = defaultdict(set)
    profile_keys = set()
    for profile in profile_rows:
        key = (
            profile["benchmark"],
            profile["model"],
            profile["effort"],
            profile["candidate"],
            profile["harness_alias"],
            profile["harness_name"],
            profile["harness_loc"],
        )
        profile_keys.add(key)
        akey = (str(profile["candidate"]), str(profile["task"]), str(profile["attempt"]))
        attempts[key].add(akey)
        if str(profile.get("success")) == "1":
            successes[key].add(akey)
    for row in rows:
        key = (
            row["benchmark"],
            row["model"],
            row["effort"],
            row["candidate"],
            row["harness_alias"],
            row["harness_name"],
            row["harness_loc"],
        )
        groups[key].append(row)

    out = []
    for key in sorted(profile_keys, key=_matrix_key):
        items = groups.get(key, [])
        counts = Counter(str(item["action_category"]) for item in items)
        total = sum(counts.values())
        attempt_count = len(attempts[key])
        row = {
            "benchmark": key[0],
            "model": key[1],
            "effort": key[2],
            "candidate": key[3],
            "harness_alias": key[4],
            "harness_name": key[5],
            "harness_loc": key[6],
            "attempts": attempt_count,
            "successes": len(successes[key]),
            "success_rate": len(successes[key]) / attempt_count if attempt_count else 0.0,
            "included_turns": total,
            "turns_per_attempt": total / attempt_count if attempt_count else 0.0,
        }
        for category in CATEGORIES:
            row[f"{category}_turns"] = counts[category]
            row[f"{category}_pct"] = counts[category] / total if total else 0.0
            row[f"{category}_per_attempt"] = (
                counts[category] / attempt_count if attempt_count else 0.0
            )
        out.append(row)
    return out


def _matrix_key(key: tuple[Any, ...]) -> tuple[str, str, float, str]:
    try:
        loc = float(key[6])
    except (TypeError, ValueError):
        loc = math.inf
    return (str(key[0]), str(key[1]), loc, str(key[3]))


def _summary(
    sources: list[tuple[str, str, str, Path]],
    action_events: list[dict[str, Any]],
    included: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "sources": [
            {"benchmark": b, "model": m, "effort": e, "path": str(p)}
            for b, m, e, p in sources
        ],
        "events": len(action_events),
        "included_events": len(included),
        "submission_events": len(action_events) - len(included),
        "categories": list(CATEGORIES),
        "taxonomy": "HarnessBridge Table 6 style: mutually exclusive, highest-priority syntax match; submissions excluded from aggregates.",
    }


def _report(
    summary: dict[str, Any],
    by_model: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> str:
    lines = [
        "# Action Category Analysis",
        "",
        "Taxonomy: HarnessBridge Table 6 style, with mutually exclusive syntax categories and submission turns excluded from aggregate percentages.",
        "",
        f"Events: {summary['events']}; included: {summary['included_events']}; submissions excluded: {summary['submission_events']}.",
        "",
        "Caveat: Terminus compressed traces store many actions as `write_stdin(tmux_session=..., commands=N)` summaries rather than the underlying command text, so those turns are labeled `reasoning_only` by the syntax classifier.",
        "",
        "## Category Mix By Model",
        "",
        _markdown_table(
            by_model,
            (
                "benchmark",
                "model",
                "action_category",
                "turns",
                "pct_turns",
                "turns_per_attempt",
            ),
        ),
        "",
        "## Harness Matrix",
        "",
        _markdown_table(
            matrix,
            (
                "benchmark",
                "model",
                "harness_loc",
                "harness_name",
                "success_rate",
                "turns_per_attempt",
                "file_read_pct",
                "search_pct",
                "script_execution_pct",
                "file_edit_pct",
                "test_execution_pct",
                "dependency_setup_pct",
            ),
        ),
        "",
        "## Files",
        "",
        "- `action_events.csv`: event-level category labels, including excluded submissions.",
        "- `attempt_action_profiles.csv`: one row per attempt with count/pct for every category.",
        "- `action_category_by_harness.csv`: long-form category distribution by harness.",
        "- `action_category_by_harness_success.csv`: same distribution split by success/failure.",
        "- `harness_action_category_matrix.csv`: wide-form per-harness category percentages.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "(none)"
    lines = [
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
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "benchmark",
        "model",
        "effort",
        "candidate",
        "harness_alias",
        "harness_name",
        "harness_loc",
        "task",
        "attempt",
        "success",
        "status",
        "action_category",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def _success_from_row(row: dict[str, str]) -> str:
    if row.get("status") == "success":
        return "1"
    try:
        return "1" if float(row.get("reward", "0")) >= 1.0 else "0"
    except ValueError:
        return "0"


def _num(value: Any) -> float:
    if value in {None, "", "N/A"}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
