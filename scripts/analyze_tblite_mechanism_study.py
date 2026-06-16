from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.parse_results import parse_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    run_root = args.run_root
    manifest = _read_json(run_root / "manifest.json")
    if not isinstance(manifest, dict):
        raise SystemExit(f"missing manifest.json under {run_root}")
    out_dir = args.out_dir or run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {item["name"]: item for item in manifest.get("variants", [])}
    attempts = _attempt_rows(run_root, manifest, variants)
    aggregate = _aggregate_rows(attempts)
    contrasts = _paired_contrasts(attempts, baseline="bare_v2_r3")
    mechanism = _mechanism_rows(aggregate, contrasts, variants)

    _write_csv(out_dir / "attempts.csv", attempts)
    _write_csv(out_dir / "aggregate_by_variant.csv", aggregate)
    _write_csv(out_dir / "paired_contrasts_vs_bare_v2_r3.csv", contrasts)
    _write_csv(out_dir / "mechanism_summary.csv", mechanism)
    report = _html_report(manifest, aggregate, contrasts, mechanism)
    (out_dir / "mechanism_report.html").write_text(report, encoding="utf-8")
    summary = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "attempts": len(attempts),
        "variants": len(aggregate),
        "outputs": {
            "attempts": str(out_dir / "attempts.csv"),
            "aggregate_by_variant": str(out_dir / "aggregate_by_variant.csv"),
            "paired_contrasts": str(out_dir / "paired_contrasts_vs_bare_v2_r3.csv"),
            "mechanism_summary": str(out_dir / "mechanism_summary.csv"),
            "report": str(out_dir / "mechanism_report.html"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _attempt_rows(
    run_root: Path,
    manifest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = [str(item) for item in manifest.get("tasks", [])]
    trials = int(manifest.get("trials") or 1)
    for variant_name, variant in variants.items():
        for task in tasks:
            for attempt in range(1, trials + 1):
                out_dir = run_root / variant_name / task / f"attempt_{attempt:02d}"
                records = parse_records(out_dir)
                record = records[0] if records else {}
                trace = _trace_metrics(out_dir)
                usage = _api_usage(record, trace)
                reward = _reward(record)
                status = str(record.get("status") or _status_from_summary(out_dir))
                rows.append(
                    {
                        "variant": variant_name,
                        "base_candidate": variant.get("base_candidate", ""),
                        "mechanism": variant.get("mechanism", ""),
                        "model": manifest.get("model", ""),
                        "effort": manifest.get("reasoning_effort", ""),
                        "task": task,
                        "attempt": attempt,
                        "status": status,
                        "reward": reward if reward is not None else "N/A",
                        "success": "1" if reward == 1.0 or status == "success" else "0",
                        "out_dir": str(out_dir),
                        **usage,
                        **trace,
                    }
                )
    return rows


def _trace_metrics(out_dir: Path) -> dict[str, Any]:
    agent_dir = _agent_dir(out_dir)
    result = _read_json(agent_dir / "harness-result.json") if agent_dir else {}
    result = result if isinstance(result, dict) else {}
    turns = (
        [_read_json(path) for path in sorted(agent_dir.glob("harness-turn-*.json"))]
        if agent_dir
        else []
    )
    turns = [turn for turn in turns if isinstance(turn, dict)]
    tool_counts: Counter[str] = Counter()
    command_categories: Counter[str] = Counter()
    return_code_counts: Counter[str] = Counter()
    failed_commands = 0
    session_commands = 0
    apply_patch_commands = 0
    verification_commands = 0
    edit_commands = 0
    format_retry_total = 0
    format_retry_max = 0
    response_replay_turns = 0
    recovery_turns = 0
    compacted_turns = 0
    assistant_chars = 0
    stdout_chars = 0
    stderr_chars = 0
    for turn in turns:
        metadata = _dict(turn.get("metadata"))
        command = str(turn.get("command") or "")
        tool_name = str(turn.get("tool_name") or "")
        return_code = turn.get("return_code")
        tool_counts[tool_name or "unknown"] += 1
        return_code_counts[_return_code_bucket(return_code)] += 1
        if isinstance(return_code, int) and return_code != 0:
            failed_commands += 1
        if tool_name in {"write_stdin", "persistent_bash"} or "session_id" in json.dumps(metadata):
            session_commands += 1
        if tool_name == "apply_patch":
            apply_patch_commands += 1
        categories = _command_categories(tool_name, command)
        command_categories.update(categories)
        if "verify" in categories:
            verification_commands += 1
        if "edit" in categories:
            edit_commands += 1
        retry_count = _metadata_int(metadata, "mini_swe_agent_v2_format_retries")
        format_retry_total += retry_count
        format_retry_max = max(format_retry_max, retry_count)
        if metadata.get("mini_swe_agent_v2_response_items") or metadata.get("codex_response_items"):
            response_replay_turns += 1
        if metadata.get("codex_recovery") or metadata.get("unresolved_format_error"):
            recovery_turns += 1
        if metadata.get("terminus_compacted") or metadata.get("compacted"):
            compacted_turns += 1
        assistant_chars += len(str(metadata.get("assistant_content") or ""))
        stdout_chars += len(str(turn.get("stdout") or ""))
        stderr_chars += len(str(turn.get("stderr") or ""))
    accounting = _dict(result.get("model_accounting"))
    return {
        "agent_dir": str(agent_dir) if agent_dir else "",
        "done": str(result.get("done", "")),
        "termination_reason": result.get("termination_reason", ""),
        "turns": int(result.get("turns") or len(turns)),
        "commands_logged": len(turns),
        "failed_commands": failed_commands,
        "session_commands": session_commands,
        "apply_patch_commands": apply_patch_commands,
        "verification_commands": verification_commands,
        "edit_commands": edit_commands,
        "format_retry_total": format_retry_total,
        "format_retry_max": format_retry_max,
        "response_replay_turns": response_replay_turns,
        "recovery_turns": recovery_turns,
        "compacted_turns": compacted_turns,
        "assistant_chars": assistant_chars,
        "stdout_chars": stdout_chars,
        "stderr_chars": stderr_chars,
        "tool_counts": json.dumps(dict(sorted(tool_counts.items())), sort_keys=True),
        "command_categories": json.dumps(dict(sorted(command_categories.items())), sort_keys=True),
        "return_code_counts": json.dumps(dict(sorted(return_code_counts.items())), sort_keys=True),
        "model_calls": _csv_value(accounting.get("model_calls")),
    }


def _aggregate_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_variant[str(row["variant"])].append(row)
    rows = []
    for variant, items in sorted(by_variant.items()):
        successes = sum(1 for item in items if str(item.get("success")) == "1")
        n = len(items)
        lo, hi = _wilson(successes, n)
        rows.append(
            {
                "variant": variant,
                "base_candidate": items[0].get("base_candidate", ""),
                "mechanism": items[0].get("mechanism", ""),
                "N": n,
                "successes": successes,
                "score": _fmt(successes / n if n else 0.0),
                "ci_low": _fmt(lo),
                "ci_high": _fmt(hi),
                "mean_turns": _fmt(_mean(items, "turns")),
                "mean_failed_commands": _fmt(_mean(items, "failed_commands")),
                "mean_session_commands": _fmt(_mean(items, "session_commands")),
                "mean_apply_patch_commands": _fmt(_mean(items, "apply_patch_commands")),
                "mean_verification_commands": _fmt(_mean(items, "verification_commands")),
                "mean_format_retry_total": _fmt(_mean(items, "format_retry_total")),
                "mean_response_replay_turns": _fmt(_mean(items, "response_replay_turns")),
                "api_input_tokens": _sum_csv(items, "api_input_tokens"),
                "api_output_tokens": _sum_csv(items, "api_output_tokens"),
                "api_cached_tokens": _sum_csv(items, "api_cached_tokens"),
                "api_total_tokens": _sum_csv(items, "api_total_tokens"),
            }
        )
    return rows


def _paired_contrasts(attempts: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_variant_task: dict[tuple[str, str, str], dict[str, Any]] = {}
    variants = sorted({str(row["variant"]) for row in attempts})
    for row in attempts:
        by_variant_task[(str(row["variant"]), str(row["task"]), str(row["attempt"]))] = row
    rows = []
    for variant in variants:
        if variant == baseline:
            continue
        wins = losses = ties = compared = 0
        variant_successes = baseline_successes = 0
        for key, base_row in by_variant_task.items():
            base_variant, task, attempt = key
            if base_variant != baseline:
                continue
            row = by_variant_task.get((variant, task, attempt))
            if row is None:
                continue
            compared += 1
            base_success = int(str(base_row.get("success")) == "1")
            success = int(str(row.get("success")) == "1")
            variant_successes += success
            baseline_successes += base_success
            if success > base_success:
                wins += 1
            elif success < base_success:
                losses += 1
            else:
                ties += 1
        rows.append(
            {
                "variant": variant,
                "baseline": baseline,
                "compared": compared,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "variant_successes": variant_successes,
                "baseline_successes": baseline_successes,
                "paired_delta": _fmt(
                    (variant_successes - baseline_successes) / compared if compared else 0.0
                ),
                "sign_test_two_sided": _fmt(_sign_test_two_sided(wins, losses)),
            }
        )
    return rows


def _mechanism_rows(
    aggregate: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    variants: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contrast_by_variant = {row["variant"]: row for row in contrasts}
    rows = []
    for row in aggregate:
        name = str(row["variant"])
        variant = variants.get(name, {})
        contrast = contrast_by_variant.get(name, {})
        rows.append(
            {
                "mechanism": row.get("mechanism", ""),
                "variant": name,
                "hypothesis": variant.get("hypothesis", ""),
                "score": row.get("score", ""),
                "ci_low": row.get("ci_low", ""),
                "ci_high": row.get("ci_high", ""),
                "paired_delta_vs_bare_v2_r3": contrast.get("paired_delta", "N/A"),
                "wins_vs_bare_v2_r3": contrast.get("wins", "N/A"),
                "losses_vs_bare_v2_r3": contrast.get("losses", "N/A"),
                "mean_turns": row.get("mean_turns", ""),
                "mean_verification_commands": row.get("mean_verification_commands", ""),
                "mean_session_commands": row.get("mean_session_commands", ""),
                "mean_apply_patch_commands": row.get("mean_apply_patch_commands", ""),
                "mean_format_retry_total": row.get("mean_format_retry_total", ""),
            }
        )
    return rows


def _html_report(
    manifest: dict[str, Any],
    aggregate: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TBLite Mechanism Study</title>
<style>
body {{ font-family: ui-serif, Georgia, serif; color: #111; margin: 36px; line-height: 1.35; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 18px 0 30px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
th {{ border-bottom: 2px solid #222; }}
.sub {{ color: #555; max-width: 900px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
</style>
</head>
<body>
<h1>TBLite Mechanism Study</h1>
<p class="sub">Model fixed to <code>{manifest.get("model")}</code>, effort <code>{manifest.get("reasoning_effort")}</code>, split <code>{manifest.get("split")}</code>. This report is descriptive; conclusions require checking paired deltas and trace examples.</p>
<h2>Scores By Variant</h2>
{_table(aggregate)}
<h2>Paired Contrasts vs bare_v2_r3</h2>
{_table(contrasts)}
<h2>Mechanism Summary</h2>
{_table(mechanism)}
</body>
</html>
"""


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    fieldnames = list(rows[0])
    head = "".join(f"<th>{_html(key)}</th>" for key in fieldnames)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_html(row.get(key, ''))}</td>" for key in fieldnames) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _command_categories(tool_name: str, command: str) -> list[str]:
    text = command.lower()
    categories = []
    if tool_name in {"apply_patch"} or any(
        token in text for token in ("apply_patch", "sed -i", "perl -pi", "cat <<", "python - <<")
    ):
        categories.append("edit")
    if any(
        token in text
        for token in (
            "pytest",
            " test",
            "npm test",
            "cargo test",
            "go test",
            "python -m pytest",
            "diff ",
            "git diff",
        )
    ):
        categories.append("verify")
    if any(token in text for token in ("ls", "find", "rg", "grep", "sed -n", "cat ", "nl -ba")):
        categories.append("inspect")
    if any(token in text for token in ("pip install", "npm install", "apt-get", "conda install")):
        categories.append("setup")
    if "complete_task_and_submit_final_output" in text:
        categories.append("submit")
    if tool_name in {"write_stdin", "persistent_bash"}:
        categories.append("session")
    return categories or ["other"]


def _agent_dir(out_dir: Path) -> Path | None:
    matches = sorted(out_dir.rglob("agent/harness-result.json"))
    if not matches:
        matches = sorted(out_dir.rglob("harness-result.json"))
    return matches[-1].parent if matches else None


def _api_usage(record: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    del trace
    return {
        "api_input_tokens": _csv_value(record.get("input_tokens")),
        "api_output_tokens": _csv_value(record.get("output_tokens")),
        "api_cached_tokens": _csv_value(record.get("cached_tokens")),
        "api_total_tokens": _csv_value(record.get("total_tokens")),
    }


def _status_from_summary(out_dir: Path) -> str:
    summary = _read_json(out_dir / "summary.json")
    if isinstance(summary, dict) and summary.get("returncode"):
        return "crash"
    return "unknown"


def _reward(record: dict[str, Any]) -> float | None:
    value = record.get("reward")
    if value is None:
        return None
    try:
        return 1.0 if float(value) >= 1 else 0.0
    except (TypeError, ValueError):
        return None


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            pass
    return mean(values) if values else 0.0


def _sum_csv(rows: list[dict[str, Any]], key: str) -> str:
    values = []
    for row in rows:
        try:
            values.append(int(str(row.get(key) or "")))
        except ValueError:
            pass
    return str(sum(values)) if values else "N/A"


def _return_code_bucket(value: Any) -> str:
    if value is None:
        return "none"
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "0" if code == 0 else "nonzero"


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _csv_value(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "variant",
        "base_candidate",
        "mechanism",
        "model",
        "effort",
        "task",
        "attempt",
        "status",
        "reward",
        "success",
        "N",
        "successes",
        "score",
        "ci_low",
        "ci_high",
        "paired_delta",
        "wins",
        "losses",
        "ties",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _html(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
