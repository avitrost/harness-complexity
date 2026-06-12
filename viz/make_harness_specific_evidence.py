#!/usr/bin/env python3
"""Harness-specific trace evidence for gpt-5.4-mini low."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path("/home/trost/harness-complexity")
BASE = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "behavior_gpt54_low_family_20260612"
)
ATTEMPTS = BASE / "gpt54mini_low_tb2_allharnesses" / "attempt_behavior.csv"
OUT = ROOT / "viz" / "harness_specific_evidence.html"
CSV_OUT = BASE / "attribute_driver_analysis" / "harness_specific_evidence_turns.csv"


CASE_GROUPS = [
    {
        "id": "long_running_lste",
        "title": "Long-running file transform: persistent session plus polling",
        "strength": "strong trace evidence",
        "claim": (
            "The richer Codex harness can let a long Vim command keep running and then poll the "
            "same live process with write_stdin. The mini-SWE-style bash harness has a fixed "
            "30 second command timeout and no write_stdin tool, so the equivalent long command "
            "is killed rather than continued."
        ),
        "code_refs": [
            ("codex-full exposes write_stdin", "seeds/codex_full/harness.py:787"),
            ("codex-full exec returns a session ID for ongoing commands", "seeds/codex_full/harness.py:835"),
            ("mini-bare has only bash and a 30 second timeout", "seeds/mini_swe_agent_barebones/harness.py:12"),
            ("mini-bare non-persistent shell rule", "seeds/mini_swe_agent_barebones/harness.py:20"),
        ],
        "cases": [
            {
                "label": "c_full success",
                "alias": "c_full",
                "task": "large-scale-text-editing",
                "attempt": "7",
                "turns": [1, 3, 5, 6, 7, 8, 12, 14, 15, 16, 17, 18, 22, 23, 24],
                "note": (
                    "The trace creates the macro with apply_patch, starts Vim in a session, "
                    "polls repeatedly, and the final poll returns EXIT:0 CMP:0."
                ),
            },
            {
                "label": "mini_bare counterexample",
                "alias": "mini_bare",
                "task": "large-scale-text-editing",
                "attempt": "7",
                "turns": [1, 2, 3, 4, 5, 6],
                "note": (
                    "The same task attempt times out twice on the long Vim command. There is no "
                    "way for this harness to poll an already-running process; it submits after six turns and fails."
                ),
            },
        ],
    },
    {
        "id": "exhaustive_bn",
        "title": "Long computation result: poll instead of abandon",
        "strength": "strong trace evidence",
        "claim": (
            "On bn-fit-modify, the successful Codex run launches an exhaustive DAG search, yields "
            "while it is still running, then polls the session and receives the decisive result. "
            "The barebones mini-SWE run has no live-session continuation path and writes a wrong solution."
        ),
        "code_refs": [
            ("codex-1000 exposes write_stdin", "seeds/codex_1000/harness.py:513"),
            ("codex-1000 keeps 64 history items", "seeds/codex_1000/harness.py:456"),
            ("mini-bare flattens all history into user text", "seeds/mini_swe_agent_barebones/harness.py:94"),
        ],
        "cases": [
            {
                "label": "c1000 success",
                "alias": "c1000",
                "task": "bn-fit-modify",
                "attempt": "3",
                "turns": [1, 2, 3, 5, 8, 9, 10, 11, 12],
                "note": (
                    "The key turn is write_stdin on session 2: it returns 'checked DAGs 2240' "
                    "and the best edge set, which the model then writes to the required files."
                ),
            },
            {
                "label": "mini_bare counterexample",
                "alias": "mini_bare",
                "task": "bn-fit-modify",
                "attempt": "3",
                "turns": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "note": (
                    "It recovers from missing python/pandas/numpy, but has no session polling "
                    "or structured replay path. It writes files and submits, but the verifier rejects them."
                ),
            },
        ],
    },
    {
        "id": "password_deep_search",
        "title": "Deep search loop: replay plus multi-call breadth",
        "strength": "observational support",
        "claim": (
            "Password recovery is not mainly about a single long command. The successful compressed "
            "Codex trace keeps a long narrowing search loop alive with raw response-item replay, "
            "multi-call turns, and session polling. The shorter c400 run finds the right area but "
            "stops with a truncated password."
        ),
        "code_refs": [
            ("codex-compressed replays raw response items", "seeds/codex_compressed/harness.py:774"),
            ("codex-compressed stores codex_response_items", "seeds/codex_compressed/harness.py:1595"),
            ("c400 uses shorter recent history", "seeds/codex_400/harness.py:236"),
        ],
        "cases": [
            {
                "label": "c_comp success",
                "alias": "c_comp",
                "task": "password-recovery",
                "attempt": "3",
                "turns": [1, 2, 5, 18, 20, 24, 33, 35, 46, 47, 56, 57, 64, 65, 66, 67],
                "note": (
                    "This run has 67 commands, 20 replay turns, 15 multi-call turns, and ends "
                    "with the full 23-character recovered password."
                ),
            },
            {
                "label": "c400 counterexample",
                "alias": "c400",
                "task": "password-recovery",
                "attempt": "3",
                "turns": [1, 2, 3, 4, 5, 6, 7, 8],
                "note": (
                    "This run finds 'PASSWORD=8XDP5Q2RT9Z' in the binary but writes that truncated "
                    "string as the answer and fails."
                ),
            },
        ],
    },
    {
        "id": "limits",
        "title": "What this does not prove",
        "strength": "counterexample",
        "claim": (
            "The traces do not support a simple 'more turns' or 'more sessions' story. There are "
            "failures with many sessions and failures with rich replay. The specific useful signal "
            "is not complexity itself; it is whether the trace actually uses an affordance that "
            "changes the recovery loop."
        ),
        "code_refs": [],
        "cases": [
            {
                "label": "many sessions still fail",
                "alias": "c400",
                "task": "large-scale-text-editing",
                "attempt": "1",
                "turns": [1, 4, 20, 80, 120, 140, 141, 142, 143, 144],
                "note": (
                    "This has 144 commands and 26 session-backed commands, but no replay/multi-call "
                    "signal and still fails."
                ),
            },
            {
                "label": "rich mechanics still fail",
                "alias": "c_full",
                "task": "large-scale-text-editing",
                "attempt": "10",
                "turns": [1, 3, 5, 10, 20, 30, 45, 55, 58],
                "note": (
                    "This has many Codex mechanics but still fails, so the affordances are enabling "
                    "conditions rather than guarantees."
                ),
            },
        ],
    },
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def short(value: object, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def load_attempts() -> dict[tuple[str, str, str], dict[str, str]]:
    with ATTEMPTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (row["harness_alias"], row["task"], row["attempt"]): row
        for row in rows
        if row.get("model") == "gpt-5.4-mini" and row.get("effort") == "low"
    }


def load_turns(agent_dir: str) -> dict[int, dict[str, object]]:
    turns: dict[int, dict[str, object]] = {}
    for path in sorted(Path(agent_dir).glob("harness-turn-*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        idx = int(path.stem.rsplit("-", 1)[-1])
        data["_path"] = str(path)
        turns[idx] = data
    return turns


def turn_flags(turn: dict[str, object]) -> str:
    flags = []
    tool = str(turn.get("tool_name") or "")
    if tool:
        flags.append(tool)
    md = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
    unified = md.get("unified_exec") if isinstance(md.get("unified_exec"), dict) else {}
    if unified.get("session_id") is not None:
        flags.append(f"session={unified.get('session_id')}")
    if md.get("codex_response_items"):
        flags.append(f"codex_items={len(md.get('codex_response_items'))}")
    if md.get("mini_swe_agent_v2_messages"):
        flags.append("mini_msgs")
    if md.get("mini_swe_agent_v2_response_items"):
        flags.append(f"mini_items={len(md.get('mini_swe_agent_v2_response_items'))}")
    if md.get("codex_emitted_tool_calls"):
        flags.append(f"multi={md.get('codex_emitted_tool_calls')}")
    if md.get("codex_recovery"):
        flags.append(f"recovery={md.get('codex_recovery')}")
    return "; ".join(flags)


def output_head(turn: dict[str, object]) -> str:
    stdout = str(turn.get("stdout") or "")
    stderr = str(turn.get("stderr") or "")
    if stdout and stderr:
        return short(f"stdout: {stdout} stderr: {stderr}", 260)
    if stdout:
        return short(stdout, 260)
    if stderr:
        return short(stderr, 260)
    return ""


def attempt_summary(attempt: dict[str, str]) -> str:
    bits = [
        f"success={attempt.get('success')}",
        f"commands={attempt.get('commands')}",
        f"failed={attempt.get('failed_commands')}",
        f"write_stdin={attempt.get('tool_write_stdin') or 0}",
        f"sessions={attempt.get('session_commands') or 0}",
        f"replay={attempt.get('response_replay_turns') or 0}",
        f"multi={attempt.get('multi_call_turns') or 0}",
    ]
    return " | ".join(bits)


def turn_table(
    group_id: str,
    case: dict[str, object],
    attempt: dict[str, str],
    turns: dict[int, dict[str, object]],
    csv_rows: list[dict[str, object]],
) -> str:
    rows = []
    for idx in case["turns"]:
        turn = turns.get(int(idx))
        if not turn:
            continue
        flags = turn_flags(turn)
        command = short(turn.get("command"), 280)
        output = output_head(turn)
        rc = turn.get("return_code")
        row = {
            "group_id": group_id,
            "case_label": case["label"],
            "harness_alias": case["alias"],
            "task": case["task"],
            "attempt": case["attempt"],
            "success": attempt.get("success"),
            "turn": idx,
            "return_code": rc,
            "flags": flags,
            "command_head": command,
            "output_head": output,
            "turn_path": turn.get("_path"),
        }
        csv_rows.append(row)
        rc_class = "ok" if rc == 0 else "bad" if rc not in (None, "") else "unknown"
        rows.append(
            "<tr>"
            f"<td class='num'>{esc(idx)}</td>"
            f"<td class='{rc_class}'>{esc(rc)}</td>"
            f"<td>{esc(flags)}</td>"
            f"<td><code>{esc(command)}</code></td>"
            f"<td>{esc(output)}</td>"
            "</tr>"
        )
    return (
        "<table class='turns'><thead><tr>"
        "<th>Turn</th><th>rc</th><th>trace flags</th><th>command</th><th>output head</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_case(
    group_id: str,
    case: dict[str, object],
    attempts: dict[tuple[str, str, str], dict[str, str]],
    csv_rows: list[dict[str, object]],
) -> str:
    key = (str(case["alias"]), str(case["task"]), str(case["attempt"]))
    attempt = attempts[key]
    turns = load_turns(attempt["agent_dir"])
    return f"""
      <div class="tracecase">
        <h4>{esc(case["label"])}</h4>
        <p class="meta">{esc(case["alias"])} / {esc(case["task"])} / attempt {esc(case["attempt"])}<br>{esc(attempt_summary(attempt))}</p>
        <p>{esc(case["note"])}</p>
        {turn_table(group_id, case, attempt, turns, csv_rows)}
        <p class="path"><code>{esc(attempt["agent_dir"])}</code></p>
      </div>
    """


def render_group(group: dict[str, object], attempts, csv_rows) -> str:
    refs = "".join(
        f"<li>{esc(label)}: <code>{esc(path)}</code></li>"
        for label, path in group.get("code_refs", [])
    )
    cases = "\n".join(render_case(group["id"], case, attempts, csv_rows) for case in group["cases"])
    return f"""
    <section class="group">
      <div class="kicker">{esc(group["strength"])}</div>
      <h2>{esc(group["title"])}</h2>
      <p>{esc(group["claim"])}</p>
      {f'<ul class="refs">{refs}</ul>' if refs else ''}
      {cases}
    </section>
    """


def css() -> str:
    return """
    body { margin: 0; background: #fffffb; color: #1f1b16; font: 18px/1.5 Georgia, "Times New Roman", serif; }
    main { max-width: 1280px; margin: 0 auto; padding: 42px 34px 80px; }
    h1 { font-size: 42px; line-height: 1.05; font-weight: 400; margin: 0 0 12px; letter-spacing: 0; }
    h2 { font-size: 28px; line-height: 1.12; font-weight: 400; margin: 6px 0 10px; letter-spacing: 0; }
    h3 { font: 700 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.08em; margin: 34px 0 8px; }
    h4 { font: 700 16px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 18px 0 6px; }
    p { max-width: 930px; margin: 9px 0; }
    .deck { font-size: 21px; color: #3c352d; }
    .note { border-left: 3px solid #3d6f85; padding-left: 14px; color: #332f28; }
    .group { border-top: 1px solid #d8d1c3; margin-top: 34px; padding-top: 18px; }
    .kicker { color: #756f63; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }
    .refs { margin: 12px 0 16px; padding-left: 20px; color: #3c352d; }
    .tracecase { margin: 18px 0 28px; }
    .meta, .path { color: #756f63; font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow-wrap: anywhere; }
    table.turns { border-collapse: collapse; width: 100%; font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 12px 0 8px; background: #fffdf7; }
    table.turns th { text-align: left; border-bottom: 1px solid #cfc3ae; padding: 7px 8px; white-space: nowrap; background: #f2eadc; color: #312b24; }
    table.turns tr:nth-child(even) td { background: #fbf6eb; }
    table.turns td { border-bottom: 1px solid #eadfce; padding: 7px 8px; vertical-align: top; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .ok { color: #2f7f6f; font-weight: 700; }
    .bad { color: #b44b4b; font-weight: 700; }
    .unknown { color: #8f6d2a; font-weight: 700; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 11px; overflow-wrap: anywhere; white-space: normal; color: #20313a; background: #eef6f7; border: 1px solid #c9dfe2; border-radius: 4px; padding: 1px 4px; }
    table.turns code { background: #f7fbfb; color: #16252c; border-color: #d6e7e9; }
    @media (max-width: 900px) { main { padding: 28px 18px 56px; } h1 { font-size: 34px; } table.turns { font-size: 11px; } }
    """


def main() -> None:
    attempts = load_attempts()
    csv_rows: list[dict[str, object]] = []
    groups = "\n".join(render_group(group, attempts, csv_rows) for group in CASE_GROUPS)
    if csv_rows:
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        with CSV_OUT.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Harness-Specific Recovery Evidence</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <h1>Harness-Specific Recovery Evidence</h1>
  <p class="deck">Concrete gpt-5.4-mini low TB2 traces where the useful behavior is tied to a harness affordance, not just to a vague notion of complexity.</p>
  <p class="note">The strongest evidence is when the successful trace uses a tool or state path that the counterexample harness cannot express. The replay/multi-call case is weaker: it is observational support, not a randomized ablation.</p>
  <p class="path">Selected-turn CSV: <code>{esc(str(CSV_OUT))}</code></p>
  {groups}
</main>
</body>
</html>
"""
    OUT.write_text(html_text)
    print(OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
