#!/usr/bin/env python3
"""Concrete examples and counterexamples for state/session/replay claims."""
import csv
import html
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "behavior_gpt54_low_family_20260612"
)
ATTEMPTS = BASE / "gpt54mini_low_tb2_allharnesses" / "attempt_behavior.csv"
OUT = HERE / "stateful_session_examples.html"
CSV_OUT = BASE / "attribute_driver_analysis" / "stateful_examples_turns.csv"


CASES = [
    {
        "id": "c700_lste_success",
        "alias": "c700",
        "task": "large-scale-text-editing",
        "attempt": "1",
        "title": "Success: Codex-700 recovers from an early failure",
        "claim": (
            "This is the cleanest positive example. The first command fails because "
            "the search path/tooling is not right, but the harness keeps structured "
            "Codex response items around later turns, allows multi-call planning, "
            "uses several session-backed commands for long Vim checks, and ends in "
            "a successful full-file comparison/submission."
        ),
        "extra_turns": [1, 2, 3, 5, 7, 8, 25, 26, 27],
    },
    {
        "id": "mini_v2_replay_no_session",
        "alias": "mini_v2",
        "task": "bn-fit-modify",
        "attempt": "5",
        "title": "Success: mini-SWE has replay without persistent sessions",
        "claim": (
            "This separates replay from session state. Every turn has mini-SWE v2 "
            "message/response-item metadata, but none has a session id. It still "
            "succeeds after multiple command failures, so persistent shell sessions "
            "are not required for every recovery."
        ),
        "extra_turns": [1, 2, 3, 6, 7, 10, 11, 12],
    },
    {
        "id": "c400_session_no_replay_success",
        "alias": "c400",
        "task": "bn-fit-modify",
        "attempt": "1",
        "title": "Success: sessions without structured replay can still work",
        "claim": (
            "Codex-400 has session-backed commands but no Codex replay metadata "
            "and no multi-call turns. This attempt succeeds anyway. That is a "
            "counterexample to any claim that replay is strictly necessary."
        ),
        "extra_turns": [1, 2, 3, 4, 8, 12, 16, 17, 18],
    },
    {
        "id": "c400_session_no_replay_failure",
        "alias": "c400",
        "task": "large-scale-text-editing",
        "attempt": "1",
        "title": "Failure: many turns and sessions are not enough",
        "claim": (
            "This is the counterexample to 'more turns' or 'session commands' as "
            "the whole explanation. It runs 144 turns and 26 session-backed turns, "
            "but has no structured replay and no multi-call turns. It keeps probing "
            "and rewriting Vim macros without converging."
        ),
        "extra_turns": [1, 2, 3, 4, 5, 6, 20, 40, 80, 120, 140, 141, 142, 143, 144],
    },
    {
        "id": "c_full_rich_failure",
        "alias": "c_full",
        "task": "large-scale-text-editing",
        "attempt": "4",
        "title": "Failure: rich mechanics are helpful but not sufficient",
        "claim": (
            "This attempt has almost every thing we measured: many session turns, "
            "structured replay, and multi-call turns. It still fails. The metric is "
            "therefore a proxy for a useful loop, not a guarantee that the model "
            "understands the task."
        ),
        "extra_turns": [1, 2, 3, 4, 5, 20, 45, 70, 88, 89, 90, 91, 92, 93],
    },
    {
        "id": "c_comp_password_success",
        "alias": "c_comp",
        "task": "password-recovery",
        "attempt": "3",
        "title": "Success: long search loop with replay and multi-call turns",
        "claim": (
            "This is a positive example for search-heavy work. The harness lets the "
            "model keep narrowing the byte/string search over many turns, including "
            "multi-call/replay turns, until it writes the exact recovered password."
        ),
        "extra_turns": [1, 2, 3, 4, 5, 30, 50, 65, 66, 67],
    },
]


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def short(value, limit=210):
    value = " ".join(str(value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def load_attempts():
    with ATTEMPTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (r["harness_alias"], r["task"], r["attempt"]): r
        for r in rows
        if r.get("model") == "gpt-5.4-mini" and r.get("effort") == "low"
    }


def load_turns(agent_dir):
    turns = []
    for path in sorted(Path(agent_dir).glob("harness-turn-*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        idx = int(path.stem.rsplit("-", 1)[-1])
        md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        unified = md.get("unified_exec") if isinstance(md.get("unified_exec"), dict) else {}
        codex_items = md.get("codex_response_items") if isinstance(md.get("codex_response_items"), list) else []
        mini_items = (
            md.get("mini_swe_agent_v2_response_items")
            if isinstance(md.get("mini_swe_agent_v2_response_items"), list)
            else []
        )
        mini_msgs = md.get("mini_swe_agent_v2_messages")
        replay_bits = []
        if codex_items:
            replay_bits.append(f"codex_items={len(codex_items)}")
        if mini_msgs:
            replay_bits.append("mini_msgs")
        if mini_items:
            replay_bits.append(f"mini_items={len(mini_items)}")
        multi = md.get("codex_emitted_tool_calls")
        try:
            multi = int(multi) if multi is not None else 0
        except Exception:
            multi = 0
        turns.append(
            {
                "idx": idx,
                "path": str(path),
                "tool": data.get("tool_name") or "",
                "rc": data.get("return_code"),
                "session_id": unified.get("session_id"),
                "replay": "; ".join(replay_bits),
                "multi": multi,
                "seq": bool(md.get("sequential_tool_calls")),
                "cmd": str(data.get("command") or ""),
                "stdout": str(data.get("stdout") or ""),
                "stderr": str(data.get("stderr") or ""),
            }
        )
    return turns


def select_turns(turns, extras):
    wanted = set(extras)
    if turns:
        wanted.update([1, 2, 3, turns[-1]["idx"]])
    for t in turns:
        rc = t["rc"]
        if rc not in ("", None, 0, "0"):
            wanted.update([t["idx"], t["idx"] + 1, t["idx"] + 2])
            break
    for t in turns:
        if t["session_id"] is not None:
            wanted.update([t["idx"], t["idx"] + 1])
            break
    for t in turns:
        if t["replay"]:
            wanted.add(t["idx"])
            break
    for t in turns:
        if t["multi"] and t["multi"] > 1:
            wanted.add(t["idx"])
            break
    by_idx = {t["idx"]: t for t in turns}
    return [by_idx[i] for i in sorted(wanted) if i in by_idx]


def flags(t):
    bits = [t["tool"]]
    if t["session_id"] is not None:
        bits.append(f"session={t['session_id']}")
    if t["replay"]:
        bits.append(t["replay"])
    if t["multi"] > 1:
        bits.append(f"multi={t['multi']}")
    if t["seq"]:
        bits.append("sequential")
    return "; ".join(bits)


def rc_class(rc):
    if rc in ("", None):
        return "unknown"
    try:
        return "ok" if int(rc) == 0 else "bad"
    except Exception:
        return "unknown"


def turn_table(turns):
    rows = []
    for t in turns:
        out = ""
        if t["stderr"]:
            out = "stderr: " + short(t["stderr"], 180)
        elif t["stdout"]:
            out = "stdout: " + short(t["stdout"], 180)
        rows.append(
            "<tr>"
            f'<td class="num">{t["idx"]}</td>'
            f'<td class="{rc_class(t["rc"])}">{esc(t["rc"])}</td>'
            f"<td>{esc(flags(t))}</td>"
            f"<td><code>{esc(short(t['cmd'], 240))}</code></td>"
            f"<td>{esc(out)}</td>"
            "</tr>"
        )
    return (
        '<table class="turns"><thead><tr>'
        "<th>Turn</th><th>rc</th><th>metadata flags</th><th>command</th><th>output head</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def case_block(case, attempt, turns):
    selected = select_turns(turns, case["extra_turns"])
    status = "success" if attempt.get("success") == "1" else "failure"
    summary = [
        ("status", status),
        ("turns", attempt.get("commands")),
        ("failed", attempt.get("failed_commands")),
        ("post-failure", attempt.get("commands_after_first_failure")),
        ("session", attempt.get("session_commands")),
        ("replay", attempt.get("response_replay_turns")),
        ("multi", attempt.get("multi_call_turns")),
    ]
    chips = "".join(f"<span>{esc(k)}: <b>{esc(v)}</b></span>" for k, v in summary)
    return f"""
    <section class="case {status}">
      <h3>{esc(case["title"])}</h3>
      <p>{esc(case["claim"])}</p>
      <div class="chips">{chips}</div>
      <p class="path"><code>{esc(attempt.get("agent_dir"))}</code></p>
      {turn_table(selected)}
    </section>
    """


def css():
    return """
    body { margin: 0; background: #fffffb; color: #1d1a16; font: 18px/1.48 Georgia, "Times New Roman", serif; }
    main { max-width: 1240px; margin: 0 auto; padding: 42px 36px 76px; }
    h1 { font-weight: 400; font-size: 44px; line-height: 1.05; margin: 0 0 12px; letter-spacing: 0; }
    h2 { margin: 42px 0 10px; padding-top: 16px; border-top: 1px solid #d8d1c3; font: 700 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }
    h3 { font-size: 25px; line-height: 1.15; font-weight: 400; margin: 0 0 8px; }
    p { max-width: 900px; margin: 10px 0; }
    .deck { font-size: 21px; color: #3b352d; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 24px 0 12px; }
    .note { border-left: 2px solid #3d6f85; padding-left: 14px; color: #332f28; }
    .term { border-top: 3px solid #8fb2bd; padding-top: 9px; font-size: 15px; line-height: 1.38; }
    .term b { display: block; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .case { border-top: 1px solid #d8d1c3; margin-top: 30px; padding-top: 18px; }
    .case.success h3::after { content: " success"; color: #2f7f6f; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; margin-left: 8px; }
    .case.failure h3::after { content: " failure"; color: #b44b4b; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; margin-left: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 12px 0 6px; font: 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .chips span { border: 1px solid #d8d1c3; background: #fffefa; padding: 5px 7px; }
    .path { color: #756f63; font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow-wrap: anywhere; }
    table.turns { border-collapse: collapse; width: 100%; font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 12px 0 8px; background: #fffdf7; }
    table.turns th { text-align: left; border-bottom: 1px solid #cfc3ae; padding: 7px 8px; white-space: nowrap; vertical-align: bottom; background: #f2eadc; color: #312b24; }
    table.turns tr:nth-child(even) td { background: #fbf6eb; }
    table.turns td { border-bottom: 1px solid #eadfce; padding: 7px 8px; vertical-align: top; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .ok { color: #2f7f6f; font-weight: 700; }
    .bad { color: #b44b4b; font-weight: 700; }
    .unknown { color: #8f6d2a; font-weight: 700; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
      white-space: normal;
      color: #20313a;
      background: #eef6f7;
      border: 1px solid #c9dfe2;
      border-radius: 4px;
      padding: 1px 4px;
    }
    table.turns code { display: inline; background: #f7fbfb; color: #16252c; border-color: #d6e7e9; }
    @media (max-width: 900px) { main { padding: 28px 18px 56px; } h1 { font-size: 34px; } .grid { grid-template-columns: 1fr; } table.turns { font-size: 11px; } }
    """


def main():
    attempts = load_attempts()
    blocks = []
    csv_rows = []
    for case in CASES:
        key = (case["alias"], case["task"], case["attempt"])
        attempt = attempts[key]
        turns = load_turns(attempt["agent_dir"])
        selected = select_turns(turns, case["extra_turns"])
        blocks.append(case_block(case, attempt, turns))
        for t in selected:
            csv_rows.append(
                {
                    "case_id": case["id"],
                    "harness_alias": case["alias"],
                    "task": case["task"],
                    "attempt": case["attempt"],
                    "success": attempt.get("success"),
                    "turn": t["idx"],
                    "return_code": t["rc"],
                    "flags": flags(t),
                    "command_head": short(t["cmd"], 500),
                    "stdout_head": short(t["stdout"], 500),
                    "stderr_head": short(t["stderr"], 500),
                    "turn_path": t["path"],
                }
            )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stateful Session Examples</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <h1>Concrete examples of session, replay, and recovery mechanics</h1>
  <p class="deck">Real <code>gpt-5.4-mini low</code> TB2 trace snippets showing what I mean by
  persistent sessions, structured replay, multi-call turns, and post-failure continuation.
  The point is to show both examples and counterexamples, not to claim any one flag is causal.</p>
  <p class="path">Selected-turn CSV: <code>{esc(str(CSV_OUT))}</code></p>

  <h2>Definitions</h2>
  <div class="grid">
    <div class="term"><b>Session</b>
      A turn with <code>metadata.unified_exec.session_id</code>. This is process/shell
      continuity, not merely file persistence. Files under <code>/app</code> persist even
      without a session.</div>
    <div class="term"><b>Structured replay</b>
      A turn with <code>codex_response_items</code> or
      <code>mini_swe_agent_v2_messages</code>. The harness keeps assistant/tool-call
      history as structured items instead of only flattened text.</div>
    <div class="term"><b>Multi-call</b>
      A Codex turn where <code>codex_emitted_tool_calls &gt; 1</code>. One model response
      emitted a small sequence of tool calls.</div>
  </div>
  <p class="note">A useful recoverable loop is usually a combination: failed command
  outcome, structured observation, continued work, and sometimes session/process
  continuity. The counterexamples below are why I do not reduce it to raw turns or
  raw session count.</p>

  <h2>Examples And Counterexamples</h2>
  {''.join(blocks)}
</main>
</body>
</html>
"""
    OUT.write_text(html_text)
    print(OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
