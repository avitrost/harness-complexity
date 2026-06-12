#!/usr/bin/env python3
"""Generate a Tufte-style HTML report for gpt-5.4-mini low harness behavior.

The input CSVs are the compact trace-analysis artifacts produced under
/wbl-fast, not the raw run directories. This keeps generation light enough for
the login node.
"""
import csv
import html
import math
import os
from collections import defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = (
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "behavior_gpt54_low_family_20260612"
)
ACTION_DIR = os.path.join(BASE, "action_category_analysis")
OUT = os.path.join(HERE, "gpt54_low_action_category_report.html")

CAT_ORDER = [
    "file_read",
    "search",
    "script_execution",
    "file_edit",
    "test_execution",
    "dependency_setup",
    "git_operation",
    "navigation",
    "reasoning_only",
    "interrupt_abort",
]

CAT_LABEL = {
    "dependency_setup": "Deps",
    "file_edit": "Edit",
    "file_read": "Read",
    "git_operation": "Git",
    "interrupt_abort": "Abort",
    "navigation": "Nav",
    "reasoning_only": "Other",
    "script_execution": "Script",
    "search": "Search",
    "test_execution": "Test",
}

CAT_COLOR = {
    "file_read": "#4c78a8",
    "search": "#f58518",
    "script_execution": "#54a24b",
    "file_edit": "#e45756",
    "test_execution": "#72b7b2",
    "dependency_setup": "#b279a2",
    "git_operation": "#9d755d",
    "navigation": "#ff9da6",
    "reasoning_only": "#8f8f8f",
    "interrupt_abort": "#000000",
}

HARNESS_ORDER = [
    "minimal",
    "mini_bare",
    "c400",
    "mini_v2",
    "term2_comp",
    "c700",
    "c1000",
    "c1300",
    "c_comp",
    "c_full",
]

CASE_NOTES = [
    {
        "title": "Large scale text editing, attempt 1",
        "task": "large-scale-text-editing",
        "attempt": "1",
        "candidates": [
            "seed_minimal_agent",
            "seed_mini_swe_agent_v2",
            "seed_codex_700",
            "seed_codex_compressed",
            "seed_codex_full",
        ],
        "read": (
            "The weak harnesses burn turns on repeated patch/script attempts and "
            "submit without a convincing full-file comparison. Codex-700 and the "
            "larger Codex harnesses do a more recognizable loop: inspect names, "
            "patch, run the transformation, and compare output."
        ),
    },
    {
        "title": "Password recovery, attempt 3",
        "task": "password-recovery",
        "attempt": "3",
        "candidates": [
            "seed_mini_swe_agent_barebones",
            "seed_mini_swe_agent_v2",
            "seed_codex_700",
            "seed_codex_compressed",
        ],
        "read": (
            "The mini-SWE variants find plausible prefixes and candidate strings, "
            "then stop short. Codex-700 and Codex-compressed keep doing offset and "
            "byte-level searches until the full token is recovered."
        ),
    },
    {
        "title": "Bayes-net fit modify, attempt 5",
        "task": "bn-fit-modify",
        "attempt": "5",
        "candidates": [
            "seed_minimal_agent",
            "seed_mini_swe_agent_v2",
            "seed_codex_700",
            "seed_codex_compressed",
            "seed_codex_full",
        ],
        "read": (
            "This is the counterexample: a direct read/compute/write path works, "
            "while extra harness affordances can pull the small model into longer "
            "setup or exploratory behavior that does not improve the final files."
        ),
    },
    {
        "title": "SPARQL university, attempt 4",
        "task": "sparql-university",
        "attempt": "4",
        "candidates": [
            "seed_mini_swe_agent_v2",
            "seed_codex_700",
            "seed_codex_1000",
        ],
        "read": (
            "Codex-700 succeeds by reading enough of the RDF shape, writing the "
            "query, and executing an rdflib check. Codex-1000 has similar surface "
            "capabilities but does not close the loop here; harness size alone is "
            "not the mechanism."
        ),
    },
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def fnum(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt_pct(value, digits=1):
    return f"{100.0 * fnum(value):.{digits}f}%"


def fmt_float(value, digits=1):
    return f"{fnum(value):.{digits}f}"


def fmt_int(value):
    return str(int(round(fnum(value))))


def sort_harness_rows(rows):
    return sorted(
        rows,
        key=lambda r: HARNESS_ORDER.index(r["harness_alias"])
        if r["harness_alias"] in HARNESS_ORDER
        else 999,
    )


def cell_style_success(successes, attempts):
    attempts = max(1.0, fnum(attempts, 1.0))
    frac = max(0.0, min(1.0, fnum(successes) / attempts))
    # Tufte-ish muted green scale, readable text throughout.
    r0, g0, b0 = 247, 246, 239
    r1, g1, b1 = 43, 119, 106
    r = int(r0 + (r1 - r0) * frac)
    g = int(g0 + (g1 - g0) * frac)
    b = int(b0 + (b1 - b0) * frac)
    text = "#111" if frac < 0.65 else "#fff"
    return f"background: rgb({r},{g},{b}); color: {text};"


def stacked_bar(row, width=230, height=14):
    x = 0.0
    parts = []
    for cat in CAT_ORDER:
        pct = fnum(row.get(f"{cat}_pct"))
        if pct <= 0:
            continue
        w = max(0.0, pct * width)
        if w < 0.5:
            continue
        parts.append(
            f'<rect x="{x:.2f}" y="0" width="{w:.2f}" height="{height}" '
            f'fill="{CAT_COLOR[cat]}"><title>{esc(CAT_LABEL[cat])}: '
            f'{pct * 100:.1f}%</title></rect>'
        )
        x += w
    return (
        f'<svg class="stack" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="action category mix">{"".join(parts)}</svg>'
    )


def score_svg(mini_rows, full_rows):
    rows = sort_harness_rows(mini_rows)
    full_by = {r["harness_alias"]: r for r in full_rows}
    w, h = 760, 245
    ml, mr, mt, mb = 54, 18, 18, 54
    plot_w, plot_h = w - ml - mr, h - mt - mb
    y_max = 0.65

    def xy(idx, score):
        x = ml + idx * (plot_w / (len(rows) - 1))
        y = mt + (1.0 - min(y_max, score) / y_max) * plot_h
        return x, y

    def polyline(model_rows):
        by = {r["harness_alias"]: r for r in model_rows}
        pts = []
        for i, r in enumerate(rows):
            score = fnum(by.get(r["harness_alias"], {}).get("success_rate"))
            x, y = xy(i, score)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    grid = []
    for tick in [0, 0.15, 0.30, 0.45, 0.60]:
        y = mt + (1.0 - tick / y_max) * plot_h
        grid.append(
            f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#e3dfd2" stroke-width="1"/>'
            f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{tick:.2f}</text>'
        )
    labels = []
    dots = []
    for i, r in enumerate(rows):
        x, _ = xy(i, 0)
        labels.append(
            f'<text class="xlab" x="{x:.1f}" y="{h-23}" text-anchor="end" '
            f'transform="rotate(-35 {x:.1f} {h-23})">{esc(r["harness_name"])}</text>'
        )
        for model_rows, color in [(mini_rows, "#b44b4b"), (full_rows, "#2f6f8f")]:
            by = {rr["harness_alias"]: rr for rr in model_rows}
            score = fnum(by.get(r["harness_alias"], {}).get("success_rate"))
            dx, dy = xy(i, score)
            dots.append(
                f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="3.8" fill="{color}">'
                f'<title>{esc(r["harness_name"])}: {score:.3f}</title></circle>'
            )
    return f"""
    <svg class="linechart" viewBox="0 0 {w} {h}" role="img"
      aria-label="TB2 score by harness complexity">
      <rect x="0" y="0" width="{w}" height="{h}" fill="#fffffb"/>
      {''.join(grid)}
      <polyline points="{polyline(mini_rows)}" fill="none" stroke="#b44b4b" stroke-width="2.4"/>
      <polyline points="{polyline(full_rows)}" fill="none" stroke="#2f6f8f" stroke-width="2.4"/>
      {''.join(dots)}
      {''.join(labels)}
      <text x="{ml}" y="15" class="legendtext" fill="#b44b4b">gpt-5.4-mini low</text>
      <text x="{ml + 160}" y="15" class="legendtext" fill="#2f6f8f">gpt-5.4 low</text>
    </svg>
    """


def score_table(rows, mech_by_key):
    header = """
    <thead><tr>
      <th>LOC</th><th>Harness</th><th>Score</th><th>Actions/attempt</th>
      <th>Read+search</th><th>Edit</th><th>Script</th><th>Test</th>
      <th>Input tok</th><th>Output tok</th><th>Action mix</th>
    </tr></thead>
    """
    body = []
    for r in sort_harness_rows(rows):
        key = (r["benchmark"], r["model"], r["harness_alias"])
        m = mech_by_key.get(key, {})
        read_search = fnum(r.get("file_read_per_attempt")) + fnum(r.get("search_per_attempt"))
        body.append(
            "<tr>"
            f"<td>{esc(r['harness_loc'])}</td>"
            f"<td>{esc(r['harness_name'])}</td>"
            f"<td class=\"num strong\">{fmt_pct(r['success_rate'], 1)}</td>"
            f"<td class=\"num\">{fmt_float(r['turns_per_attempt'], 1)}</td>"
            f"<td class=\"num\">{read_search:.1f}</td>"
            f"<td class=\"num\">{fmt_float(r['file_edit_per_attempt'], 1)}</td>"
            f"<td class=\"num\">{fmt_float(r['script_execution_per_attempt'], 1)}</td>"
            f"<td class=\"num\">{fmt_float(r['test_execution_per_attempt'], 2)}</td>"
            f"<td class=\"num\">{fmt_int(m.get('mean_api_input_tokens'))}</td>"
            f"<td class=\"num\">{fmt_int(m.get('mean_api_output_tokens'))}</td>"
            f"<td>{stacked_bar(r)}</td>"
            "</tr>"
        )
    return f'<table class="data compact">{header}<tbody>{"".join(body)}</tbody></table>'


def task_heatmap(rows):
    by_task = defaultdict(dict)
    harness_meta = {}
    for r in rows:
        by_task[r["task"]][r["harness_alias"]] = r
        harness_meta[r["harness_alias"]] = r
    header = ["<th>Task</th>"]
    for h in HARNESS_ORDER:
        if h in harness_meta:
            header.append(f"<th>{esc(harness_meta[h]['harness_name'])}</th>")
    body = []
    for task in sorted(by_task):
        cells = [f"<th>{esc(task)}</th>"]
        for h in HARNESS_ORDER:
            if h not in harness_meta:
                continue
            r = by_task[task].get(h)
            if not r:
                cells.append("<td></td>")
                continue
            style = cell_style_success(r["successes"], r["attempts"])
            cells.append(
                f'<td class="num heat" style="{style}">'
                f'{fmt_int(r["successes"])}/{fmt_int(r["attempts"])}</td>'
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="data heatmap"><thead><tr>'
        + "".join(header)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def success_split_table(rows):
    selected = ["minimal", "mini_v2", "c700", "c1000", "c_comp", "c_full"]
    by_key = defaultdict(dict)
    meta = {}
    for r in rows:
        if (
            r["benchmark"] == "tb2"
            and r["model"] == "gpt-5.4-mini"
            and r["effort"] == "low"
            and r["harness_alias"] in selected
        ):
            by_key[(r["harness_alias"], r["success"])][r["action_category"]] = r
            meta[r["harness_alias"]] = r

    def v(alias, success, cat):
        return fnum(by_key.get((alias, success), {}).get(cat, {}).get("turns_per_attempt"))

    body = []
    for h in selected:
        if h not in meta:
            continue
        m = meta[h]
        succ = {
            "read_search": v(h, "1", "file_read") + v(h, "1", "search"),
            "edit": v(h, "1", "file_edit"),
            "script": v(h, "1", "script_execution"),
            "test": v(h, "1", "test_execution"),
            "other": v(h, "1", "reasoning_only"),
        }
        fail = {
            "read_search": v(h, "0", "file_read") + v(h, "0", "search"),
            "edit": v(h, "0", "file_edit"),
            "script": v(h, "0", "script_execution"),
            "test": v(h, "0", "test_execution"),
            "other": v(h, "0", "reasoning_only"),
        }
        body.append(
            "<tr>"
            f"<td>{esc(m['harness_name'])}</td>"
            f"<td class=\"num\">{fmt_pct(m['attempt_success_rate'], 1)}</td>"
            f"<td class=\"num\">{succ['read_search']:.1f}</td>"
            f"<td class=\"num\">{fail['read_search']:.1f}</td>"
            f"<td class=\"num\">{succ['edit']:.1f}</td>"
            f"<td class=\"num\">{fail['edit']:.1f}</td>"
            f"<td class=\"num\">{succ['script']:.1f}</td>"
            f"<td class=\"num\">{fail['script']:.1f}</td>"
            f"<td class=\"num\">{succ['test']:.2f}</td>"
            f"<td class=\"num\">{fail['test']:.2f}</td>"
            f"<td class=\"num\">{succ['other']:.1f}</td>"
            f"<td class=\"num\">{fail['other']:.1f}</td>"
            "</tr>"
        )
    return """
    <table class="data compact">
      <thead><tr>
        <th>Harness</th><th>Score</th>
        <th>Read+search<br>success</th><th>Read+search<br>failure</th>
        <th>Edit<br>success</th><th>Edit<br>failure</th>
        <th>Script<br>success</th><th>Script<br>failure</th>
        <th>Test<br>success</th><th>Test<br>failure</th>
        <th>Other<br>success</th><th>Other<br>failure</th>
      </tr></thead>
      <tbody>""" + "".join(body) + "</tbody></table>"


def tblite_table(rows):
    body = []
    for r in sort_harness_rows(rows):
        body.append(
            "<tr>"
            f"<td>{esc(r['model'])}</td>"
            f"<td>{esc(r['harness_name'])}</td>"
            f"<td class=\"num\">{fmt_pct(r['success_rate'], 1)}</td>"
            f"<td class=\"num\">{fmt_float(r['turns_per_attempt'], 1)}</td>"
            f"<td class=\"num\">{fmt_float(r['test_execution_per_attempt'], 2)}</td>"
            f"<td>{stacked_bar(r, width=180)}</td>"
            "</tr>"
        )
    return """
    <table class="data compact">
      <thead><tr><th>Model</th><th>Harness</th><th>Score</th>
      <th>Actions/attempt</th><th>Tests/attempt</th><th>Action mix</th></tr></thead>
      <tbody>""" + "".join(body) + "</tbody></table>"


def event_summary(events):
    counts = defaultdict(int)
    for e in events:
        if e["included_in_category_analysis"] == "1":
            counts[e["action_category"]] += 1
    shown = []
    for cat in CAT_ORDER:
        if counts[cat]:
            shown.append(f"{CAT_LABEL[cat]} {counts[cat]}")
    return ", ".join(shown) if shown else "no included actions"


def command_excerpt(command, limit=155):
    command = " ".join((command or "").split())
    if len(command) <= limit:
        return command
    return command[: limit - 1] + "..."


def case_cards(events, profiles):
    events_by = defaultdict(list)
    for e in events:
        if e["benchmark"] == "tb2" and e["model"] == "gpt-5.4-mini" and e["effort"] == "low":
            events_by[(e["task"], e["attempt"], e["candidate"])].append(e)
    profile_by = {}
    for p in profiles:
        if p["benchmark"] == "tb2" and p["model"] == "gpt-5.4-mini" and p["effort"] == "low":
            profile_by[(p["task"], p["attempt"], p["candidate"])] = p

    cards = []
    for case in CASE_NOTES:
        blocks = []
        for cand in case["candidates"]:
            key = (case["task"], case["attempt"], cand)
            evs = sorted(events_by.get(key, []), key=lambda e: int(e["turn_index"] or 0))
            prof = profile_by.get(key)
            if not prof:
                continue
            status = "success" if prof["success"] == "1" else "failure"
            visible = [e for e in evs if e["included_in_category_analysis"] == "1"]
            first = visible[:4]
            last = visible[-2:] if len(visible) > 6 else []
            rows = []
            for e in first + last:
                rows.append(
                    "<tr>"
                    f"<td>{esc(e['turn_index'])}</td>"
                    f"<td>{esc(CAT_LABEL.get(e['action_category'], e['action_category']))}</td>"
                    f"<td><code>{esc(command_excerpt(e['command_head']))}</code></td>"
                    "</tr>"
                )
            ellipsis = ""
            if last:
                ellipsis = (
                    f"<tr><td colspan=\"3\" class=\"muted\">... "
                    f"{len(visible) - len(first) - len(last)} intermediate actions omitted"
                    "</td></tr>"
                )
                rows = rows[: len(first)] + [ellipsis] + rows[len(first) :]
            blocks.append(
                f"""
                <div class="case-run {status}">
                  <div class="case-head">
                    <span>{esc(prof['harness_name'])}</span>
                    <b>{status}</b>
                  </div>
                  <div class="micro">{esc(event_summary(evs))}</div>
                  <table class="trace"><tbody>{''.join(rows)}</tbody></table>
                </div>
                """
            )
        cards.append(
            f"""
            <section class="case-card">
              <h3>{esc(case['title'])}</h3>
              <p>{esc(case['read'])}</p>
              <div class="case-grid">{''.join(blocks)}</div>
            </section>
            """
        )
    return "".join(cards)


def legend():
    items = []
    for cat in CAT_ORDER:
        items.append(
            f'<span><i style="background:{CAT_COLOR[cat]}"></i>{esc(CAT_LABEL[cat])}</span>'
        )
    return '<div class="legend">' + "".join(items) + "</div>"


def css():
    return """
    :root {
      --paper: #fffffb;
      --ink: #1d1a16;
      --muted: #756f63;
      --rule: #d8d1c3;
      --soft: #f3f0e6;
    }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 18px/1.48 Georgia, "Times New Roman", serif;
    }
    main {
      max-width: 1380px;
      margin: 0 auto;
      padding: 42px 42px 80px;
    }
    article {
      max-width: 1040px;
    }
    h1 {
      font-size: 48px;
      line-height: 1.02;
      font-weight: 400;
      letter-spacing: 0;
      margin: 0 0 10px;
    }
    h2 {
      margin: 48px 0 10px;
      padding-top: 18px;
      border-top: 1px solid var(--rule);
      font: 700 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    h3 {
      font-size: 23px;
      line-height: 1.2;
      font-weight: 400;
      margin: 0 0 8px;
    }
    p {
      margin: 12px 0;
      max-width: 880px;
    }
    .deck {
      max-width: 920px;
      color: #3b352d;
      font-size: 21px;
    }
    .run-meta, .micro, .caption, .muted {
      color: var(--muted);
      font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .keypoints {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 28px 0 12px;
      max-width: 1050px;
    }
    .point {
      border-top: 3px solid #222;
      padding-top: 10px;
      font-size: 16px;
      line-height: 1.35;
    }
    .point b {
      display: block;
      font: 700 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }
    .figure {
      margin: 26px 0 16px;
      overflow-x: auto;
    }
    .linechart {
      width: min(100%, 920px);
      min-width: 720px;
      height: auto;
    }
    .linechart text {
      font: 11px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      fill: #5d554c;
    }
    .linechart .legendtext {
      font-weight: 700;
    }
    .linechart .xlab {
      font-size: 10px;
    }
    table.data {
      border-collapse: collapse;
      width: 100%;
      font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 12px 0 8px;
    }
    table.data th {
      text-align: left;
      font-weight: 700;
      color: #342f28;
      border-bottom: 1px solid var(--rule);
      padding: 6px 7px;
      vertical-align: bottom;
      white-space: nowrap;
    }
    table.data td {
      border-bottom: 1px solid #eee8dc;
      padding: 6px 7px;
      vertical-align: middle;
    }
    table.compact td, table.compact th {
      padding: 5px 6px;
    }
    .num {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .strong {
      font-weight: 700;
    }
    .stack {
      display: block;
      width: 230px;
      height: 14px;
      outline: 1px solid #e6dfd2;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 15px;
      margin: 10px 0 12px;
      font: 12px/1.3 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #4d463d;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .legend i {
      width: 11px;
      height: 11px;
      display: inline-block;
    }
    .heatmap {
      min-width: 980px;
    }
    .heatmap th:first-child {
      min-width: 190px;
    }
    .heat {
      font-weight: 700;
      border: 1px solid #fffffb !important;
    }
    .note {
      border-left: 2px solid #3d6f85;
      padding-left: 16px;
      max-width: 900px;
      color: #332f28;
    }
    .case-card {
      margin: 24px 0 28px;
      padding-top: 18px;
      border-top: 1px solid var(--rule);
    }
    .case-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .case-run {
      border: 1px solid var(--rule);
      background: #fffefa;
      padding: 10px 12px 12px;
      min-width: 0;
    }
    .case-run.success {
      border-top: 4px solid #2f7f6f;
    }
    .case-run.failure {
      border-top: 4px solid #b44b4b;
    }
    .case-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font: 14px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin-bottom: 5px;
    }
    .case-head b {
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
    }
    table.trace {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    table.trace td {
      border-top: 1px solid #eee8dc;
      padding: 4px 3px;
      vertical-align: top;
    }
    table.trace td:first-child {
      width: 32px;
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    table.trace td:nth-child(2) {
      width: 58px;
      color: #4b443b;
      font-weight: 700;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .files code {
      font-size: 12px;
    }
    @media (max-width: 900px) {
      main { padding: 28px 18px 56px; }
      h1 { font-size: 36px; }
      .keypoints, .case-grid { grid-template-columns: 1fr; }
      body { font-size: 17px; }
    }
    @media print {
      body { background: white; }
      main { padding: 20px; }
      .figure { overflow: visible; }
      .case-grid { grid-template-columns: 1fr; }
    }
    """


def main():
    matrix = read_csv(os.path.join(ACTION_DIR, "harness_action_category_matrix.csv"))
    success_split = read_csv(os.path.join(ACTION_DIR, "action_category_by_harness_success.csv"))
    profiles = read_csv(os.path.join(ACTION_DIR, "attempt_action_profiles.csv"))
    events = read_csv(os.path.join(ACTION_DIR, "action_events.csv"))
    mechanisms = read_csv(os.path.join(BASE, "combined_by_harness_mechanisms.csv"))
    mini_task = read_csv(os.path.join(BASE, "gpt54mini_low_tb2_allharnesses", "by_task_harness.csv"))
    correlations = read_csv(os.path.join(BASE, "complexity_correlations.csv"))

    mech_by_key = {
        (r["benchmark"], r["model"], r["harness_alias"]): r for r in mechanisms
    }
    mini_tb2 = [
        r
        for r in matrix
        if r["benchmark"] == "tb2" and r["model"] == "gpt-5.4-mini" and r["effort"] == "low"
    ]
    gpt_tb2 = [
        r
        for r in matrix
        if r["benchmark"] == "tb2" and r["model"] == "gpt-5.4" and r["effort"] == "low"
    ]
    tblite = [
        r
        for r in matrix
        if r["benchmark"] == "tblite"
        and r["model"] in ("gpt-5.4-mini", "gpt-5.4")
        and r["effort"] == "low"
    ]
    corr_text = []
    for r in correlations:
        if r["benchmark"] == "tb2":
            corr_text.append(
                f"{r['model']}: Pearson LOC {fnum(r['pearson_loc_score']):.3f}, "
                f"log LOC {fnum(r['pearson_logloc_score']):.3f}"
            )

    mini_best = max(mini_tb2, key=lambda r: fnum(r["success_rate"]))
    mini_simple = next(r for r in mini_tb2 if r["harness_alias"] == "minimal")
    mini_bare = next(r for r in mini_tb2 if r["harness_alias"] == "mini_bare")
    full_best = max(gpt_tb2, key=lambda r: fnum(r["success_rate"]))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPT-5.4 Low Harness Behavior Report</title>
  <style>{css()}</style>
</head>
<body>
<main>
<article>
  <h1>How the small model uses richer terminal harnesses</h1>
  <p class="deck">Quantitative and qualitative analysis of the low-effort GPT-5.4 family runs,
  centered on <b>gpt-5.4-mini low</b> across the 9-task TB2 subset, with gpt-5.4 low
  as a comparison and TBLite as an appendix.</p>
  <p class="run-meta">Generated from compact trace artifacts in <code>{esc(BASE)}</code>.</p>

  <div class="keypoints">
    <div class="point"><b>Main result</b>
      gpt-5.4-mini low improves from {fmt_pct(mini_simple['success_rate'], 1)}
      on minimal-agent and {fmt_pct(mini_bare['success_rate'], 1)} on the 149-line
      mini-SWE compression to {fmt_pct(mini_best['success_rate'], 1)} on
      {esc(mini_best['harness_name'])}.</div>
    <div class="point"><b>Mechanism</b>
      The useful harnesses do not just increase command count. They reshape the
      loop toward inspect/search, edit, execute, and occasional verification,
      while keeping enough state to recover from local mistakes.</div>
    <div class="point"><b>Limit</b>
      The trend is positive but non-monotone for the small model: codex-700/1000
      and codex-compressed are the strongest region, while codex-full and
      codex-1300 are not uniformly better.</div>
  </div>

  <h2>Method</h2>
  <p>The action taxonomy follows the style of the HarnessBridge action-category
  analysis: deterministic syntax categories, one category per action, and
  submission turns excluded from category percentages. The categories are
  read, search, script execution, file edit, test execution, dependency setup,
  git operation, navigation, reasoning-only, and interrupt/abort.</p>
  <p>The token columns are API-reported usage totals from saved
  <code>agent/model-call-*.json</code> metadata, not post-hoc estimates. In the
  OpenAI traces, saved histories include encrypted <code>reasoning</code> items
  with empty visible content, and sampled calls show API output-token totals
  substantially above the visible assistant text plus tool-call JSON. The logs
  therefore support treating <code>Output tok</code> as an API total that likely
  includes hidden reasoning, but they do not expose a separate reasoning-token
  split.</p>
  <p class="note">Important caveat: Terminus compressed traces often store tmux
  batches as summarized <code>write_stdin(..., commands=N)</code> entries rather
  than the underlying command strings. Those events classify as reasoning-only
  here, so Terminus action mix is not directly comparable to the other harnesses.</p>

  <h2>Score Trend</h2>
  <p>On TB2, the small model benefits from added harness structure, but it does
  not get a clean monotonic lift from every extra line of harness code. The
  stronger gpt-5.4 low run shows a smoother relationship: {esc('; '.join(corr_text))}.</p>
  <div class="figure">{score_svg(mini_tb2, gpt_tb2)}
    <div class="caption">Scores are success rates across 9 tasks x 10 trials per harness.</div>
  </div>

  <h2>GPT-5.4-Mini Low, TB2</h2>
  <p>The winning region uses fewer actions than minimal-agent but with a more
  balanced action mix. Minimal-agent is very active, yet much of that activity is
  repeated reading/searching/script execution without reliable closing checks.</p>
  {legend()}
  <div class="figure">{score_table(mini_tb2, mech_by_key)}</div>

  <h2>GPT-5.4 Low Comparison</h2>
  <p>The larger model extracts value from both mini-SWE and Codex-style harnesses
  more consistently. It reaches {fmt_pct(full_best['success_rate'], 1)} on
  {esc(full_best['harness_name'])}, and the whole Codex family is relatively flat
  near the top.</p>
  {legend()}
  <div class="figure">{score_table(gpt_tb2, mech_by_key)}</div>

  <h2>Success-Conditioned Behavior</h2>
  <p>For gpt-5.4-mini low, successful Codex attempts generally perform more of
  every useful action type than failed Codex attempts. The biggest distinction is
  not raw tool volume alone; it is the complete loop: read/search enough context,
  edit, execute, and sometimes run test-like checks.</p>
  <div class="figure">{success_split_table(success_split)}</div>

  <h2>Task-Level Heatmap</h2>
  <p>The lift is task-specific. Large-scale text editing and password recovery
  benefit strongly from richer harness behavior. Bayes-net fitting is mixed, and
  several hard tasks remain unsolved for every gpt-5.4-mini low harness.</p>
  <div class="figure">{task_heatmap(mini_task)}</div>

  <h2>Qualitative Trace Reads</h2>
  <p>These paired cases are not meant as a full causal proof, but they show the
  repeated pattern behind the aggregate: richer harnesses help when they cause a
  small model to maintain a usable inspect-edit-verify loop, and hurt when the
  extra affordances become exploratory overhead.</p>
  {case_cards(events, profiles)}

  <h2>TBLite Appendix</h2>
  <p>TBLite is a separate 100-task, 1-trial-per-task benchmark and should not be
  pooled with TB2. It points in the same direction: codex-full beats
  msa-prompt-compressed by 20 percentage points for gpt-5.4-mini low and by 10
  points for gpt-5.4 low.</p>
  {legend()}
  <div class="figure">{tblite_table(tblite)}</div>

  <h2>Interpretation</h2>
  <p>The answer to "how is the small model making use of the harness?" is:
  the harness improves the <i>control loop</i>, not just the available command
  surface. For gpt-5.4-mini low, the best harnesses keep tool interaction
  organized enough that the model can iterate after partial failures, preserve
  context through replay/session mechanics, and spend more of its limited
  competence on concrete repository state rather than remembering what happened.</p>
  <p>That benefit is bounded. Very compact agents can terminate too early or
  lack recovery pressure. Very rich harnesses can add overhead, reasoning-only
  turns, or affordances the small model does not exploit reliably. The empirical
  sweet spot in this TB2 subset is the Codex 700/1000/compressed region, not the
  smallest or largest harness.</p>

  <h2>Files</h2>
  <p class="files">Source CSVs used here:</p>
  <ul class="files">
    <li><code>{esc(os.path.join(ACTION_DIR, 'harness_action_category_matrix.csv'))}</code></li>
    <li><code>{esc(os.path.join(ACTION_DIR, 'action_category_by_harness_success.csv'))}</code></li>
    <li><code>{esc(os.path.join(ACTION_DIR, 'attempt_action_profiles.csv'))}</code></li>
    <li><code>{esc(os.path.join(ACTION_DIR, 'action_events.csv'))}</code></li>
    <li><code>{esc(os.path.join(BASE, 'combined_by_harness_mechanisms.csv'))}</code></li>
    <li><code>{esc(os.path.join(BASE, 'gpt54mini_low_tb2_allharnesses', 'by_task_harness.csv'))}</code></li>
  </ul>
</article>
</main>
</body>
</html>
"""
    with open(OUT, "w") as f:
        f.write(html_text)
    print(OUT)


if __name__ == "__main__":
    main()
