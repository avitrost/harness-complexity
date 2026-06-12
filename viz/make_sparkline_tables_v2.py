#!/usr/bin/env python3
"""Generate Tufte-style sparkline tables from the merged provider-matrix aggregate.

v2 adds a third deliverable: the gpt-5.5/high *retry ablation* for the two
mini-swe-agent variants (mini-swe-agent and msa-prompt-compressed) across retry limits
0..3. It also keeps the original two deliverables unchanged in style:

  1. One table per reasoning effort (rows = harnesses by LOC, cols = model).
  2. The transpose: one table per model (rows = harnesses, cols = effort).
  3. Retry ablation: rows = {mini-swe-agent, msa-prompt-compressed}, cols = retry 0..3.

CI in every cell is the Wilson binomial interval over the 90 trials (per-trial
sampling noise on these exact 9 tasks), matching the original deliverable.

To drop Anthropic / TBLite gpt-5.5 in later: add their (model, display) entries
to FAMILIES, append their aggregate path to AGG_SOURCES, and rerun. Cells with
no data render as a dash + broken line; nothing is fabricated.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Data source is absolute so the script keeps working wherever it's moved;
# the HTML is written next to the script (HERE).
AGG_DIR = ("/wbl-fast/usrs/trost/harness-complexity/final_test/"
           "aggregate_openai_deepseek_20260609")
SRC = os.path.join(AGG_DIR, "aggregate_by_candidate_with_prior_openai10.csv")
# Anthropic "best complete run" manual aggregate (same schema: candidate,
# effort, model, num_successes, num_crashes, num_attempts=90 per cell). We
# recompute Wilson-over-90 from successes like every other provider, so the
# CSV's normal-approx ci columns are not used.
ANTH_SRC = ("/wbl-fast/usrs/trost/harness-complexity/final_test/"
            "manual_aggregate_anthropic_best_20260612_0028/aggregate_by_candidate.csv")
CANDIDATE_SOURCES = [SRC, ANTH_SRC]
OUT = os.path.join(HERE, "sparkline_tables_v2.html")

# Retry-ablation sources (gpt-5.5 / high, same 9 tasks).
RETRY_DIR = ("/wbl-fast/usrs/trost/harness-complexity/final_test/"
             "mini_swe_retry_matrix_gpt55_high_20260609_204939")
RETRY_SCORES = os.path.join(RETRY_DIR, "scores_by_harness.csv")

EXPECTED_TRIALS = 90  # 9 tasks x 10 trials

# ---- fixed scale (shared by the main sparklines) ----
SCALE_LO, SCALE_HI = 10.0, 70.0
# the retry ablation lives in a tight 59-77% band -> its own scale so the
# trend across retries is legible instead of clipped at the global ceiling.
RETRY_LO, RETRY_HI = 50.0, 80.0

# ---- harness order: ascending by LOC ----
# col key -> (display name, LOC)
HARNESS = [
    ("minimal",    ("minimal-agent",         100)),
    ("mini_bare",  ("msa-fully-compressed",          149)),
    ("c400",       ("codex-400",             398)),
    ("bare_v2",    ("msa-prompt-compressed",          408)),
    ("mini_v2",    ("mini-swe-agent",        478)),
    ("term2_comp", ("terminus-2-compressed", 634)),
    ("c700",       ("codex-700",             700)),
    ("c1000",      ("codex-1000",           1000)),
    ("c1300",      ("codex-1300",           1300)),
    ("c_comp",     ("codex-compressed",     1660)),
    ("c_full",     ("codex-full",           2210)),
]
HARNESS_ORDER = [k for k, _ in HARNESS]
HARNESS_META = dict(HARNESS)

CANDIDATE_TO_KEY = {
    "seed_codex_1000": "c1000",
    "seed_codex_1300": "c1300",
    "seed_codex_400": "c400",
    "seed_codex_700": "c700",
    "seed_codex_compressed": "c_comp",
    "seed_codex_full": "c_full",
    "seed_mini_swe_agent_barebones": "mini_bare",      # v1
    "seed_mini_swe_agent_barebones_v2": "bare_v2",     # v2
    "seed_mini_swe_agent_v2": "mini_v2",
    "seed_minimal_agent": "minimal",
    "seed_terminus_2_compressed": "term2_comp",
}

# ---- model families, small -> large ----
# family -> list of (model id, display name)
FAMILIES = [
    ("OpenAI", [
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt-5.4",      "gpt-5.4"),
        ("gpt-5.5",      "gpt-5.5"),
    ]),
    ("Anthropic", [
        ("claude-haiku-4-5",  "Haiku 4.5"),
        ("claude-sonnet-4-6", "Sonnet 4.6"),
        ("claude-opus-4-8",   "Opus 4.8"),
    ]),
    ("DeepSeek", [
        ("deepseek-v4-flash", "v4-flash"),
        ("deepseek-v4-pro",   "v4-pro"),
    ]),
]
MODEL_DISPLAY = {m: d for _, ms in FAMILIES for m, d in ms}
MODEL_FAMILY = {m: fam for fam, ms in FAMILIES for m, _ in ms}
MODEL_ORDER = [m for _, ms in FAMILIES for m, _ in ms]

# effort order per family (least -> most reasoning). Anthropic: haiku ran at
# "none" only; sonnet/opus at low/medium/high. Absent efforts are filtered out.
EFFORT_ORDER = {
    "OpenAI": ["low", "medium", "high"],
    "DeepSeek": ["none", "high", "max"],
    "Anthropic": ["none", "low", "medium", "high"],
}

Z = 1.96  # 95%


def wilson_half(succ, n):
    """Half-width (in pp) of the Wilson score interval for succ/n.

    Trial-level binomial CI on these exact tasks (per-trial sampling noise),
    not the task-level CI shipped in the source file. Wilson is used instead of
    Wald so cells at 0 or n successes still get a sensible, non-zero interval.
    """
    if n == 0:
        return None
    p = succ / n
    denom = 1.0 + Z * Z / n
    margin = (Z / denom) * math.sqrt(p * (1 - p) / n + Z * Z / (4.0 * n * n))
    return margin * 100.0


# ---------------------------------------------------------------- load
def load():
    """Return data[(model, effort, harness_key)] = cell dict, or absent if missing.

    Reads every candidate source (OpenAI+DeepSeek merge, then Anthropic). All use
    90 attempts/cell and the same columns; msa-prompt-compressed (Anthropic-only here) is
    dropped by CANDIDATE_TO_KEY so the proxy tables keep the canonical 10 rows.
    """
    data = {}
    for src in CANDIDATE_SOURCES:
        with open(src) as f:
            for r in csv.DictReader(f):
                hk = CANDIDATE_TO_KEY.get(r["candidate"])
                if hk is None:
                    continue
                succ = int(r["num_successes"])
                score = 100.0 * succ / EXPECTED_TRIALS  # crashes count as failures
                data[(r["model"], r["effort"], hk)] = {
                    "score": score,
                    "half": wilson_half(succ, EXPECTED_TRIALS),
                    "succ": succ,
                    "crashes": int(r["num_crashes"]),
                    "attempts": int(r["num_attempts"]),
                }
    return data


# ---- retry ablation: rows = harness variant, cols = retry limit 0..3 ----
# display label -> retry-sweep `harness` prefix (cells named <prefix>_r<N>)
RETRY_ROWS = [
    ("mini-swe-agent", "mini_v2"),
    ("msa-prompt-compressed",   "barebones_v2"),
]
RETRY_LIMITS = [0, 1, 2, 3]


def load_retry():
    """retry[(label, retry)] = cell dict.

    Cells come from the dedicated retry sweep. The mini-swe-agent r3 cell is
    absent from that sweep -- its r3 (=default 3-retry run) is the provider
    matrix datapoint already in SRC (gpt-5.5/high/seed_mini_swe_agent_v2).
    """
    by_harness = {}
    with open(RETRY_SCORES) as f:
        for r in csv.DictReader(f):
            succ = int(r["successes"])
            n = int(r["N"])
            by_harness[r["harness"]] = {
                "score": 100.0 * succ / n,
                "half": wilson_half(succ, n),
                "succ": succ,
                "attempts": n,
            }

    # mini-swe-agent r3 default, pulled from the main aggregate.
    default_r3 = None
    with open(SRC) as f:
        for r in csv.DictReader(f):
            if (r["model"], r["effort"], r["candidate"]) == (
                    "gpt-5.5", "high", "seed_mini_swe_agent_v2"):
                succ = int(r["num_successes"])
                default_r3 = {
                    "score": 100.0 * succ / EXPECTED_TRIALS,
                    "half": wilson_half(succ, EXPECTED_TRIALS),
                    "succ": succ,
                    "attempts": int(r["num_attempts"]),
                    "is_default": True,
                }

    retry = {}
    for label, prefix in RETRY_ROWS:
        for n in RETRY_LIMITS:
            cell = by_harness.get(f"{prefix}_r{n}")
            if cell is None and label == "mini-swe-agent" and n == 3:
                cell = default_r3
            if cell is not None:
                retry[(label, n)] = cell
    return retry


# ---- TBLite: a separate (broader) 100-task benchmark, 1 trial/task. Several runs
# across model/effort configs, shown as their own standalone tables on the right.
TBLITE_BASE = "/wbl-fast/usrs/trost/harness-complexity/final_test"
# (model, effort, run dir). The model/effort is not in the records -> taken here.
TBLITE_ROOTS = [
    ("gpt-5.5", "high", "tblite_gpt55_high_1x100_c200_20260611_190038"),
    ("gpt-5.5", "high", "tblite_gpt55_high_barebones_v1_c100_20260611_220336"),
    ("gpt-5.4", "low",  "tblite_gpt54_low_barebonesv2_vs_codexfull_c200_20260611_221815"),
    ("gpt-5.4-mini", "low",
     "tblite_gpt54mini_low_barebonesv2_vs_codexfull_c200_20260611_221815"),
]
# TBLite seed dir -> row key. barebones v1 (149 LOC) and v2 (408) are distinct.
TBLITE_SEED_TO_KEY = {
    "seed_codex_1000": "c1000", "seed_codex_1300": "c1300",
    "seed_codex_400": "c400", "seed_codex_700": "c700",
    "seed_codex_compressed": "c_comp", "seed_codex_full": "c_full",
    "seed_mini_swe_agent_v2": "mini_v2", "seed_minimal_agent": "minimal",
    "seed_terminus_2_compressed": "term2_comp",
    "seed_mini_swe_agent_barebones": "bare_v1",
    "seed_mini_swe_agent_barebones_v2": "bare_v2",
}
# TBLite's own harness order/labels (ascending LOC). Both barebones variants shown.
TBLITE_HARNESS = [
    ("minimal",    ("minimal-agent",         100)),
    ("bare_v1",    ("msa-fully-compressed",          149)),
    ("c400",       ("codex-400",             398)),
    ("bare_v2",    ("msa-prompt-compressed",          408)),
    ("mini_v2",    ("mini-swe-agent",        478)),
    ("term2_comp", ("terminus-2-compressed", 634)),
    ("c700",       ("codex-700",             700)),
    ("c1000",      ("codex-1000",           1000)),
    ("c1300",      ("codex-1300",           1300)),
    ("c_comp",     ("codex-compressed",     1660)),
    ("c_full",     ("codex-full",           2210)),
]
# msa-prompt-compressed vs codex-full across model strength (ascending capability).
TBLITE_CMP_ROWS = [("bare_v2", "msa-prompt-compressed"), ("c_full", "codex-full")]
TBLITE_CMP_COLS = [
    ("gpt-5.4-mini", "low",  "5.4-mini<br>low"),
    ("gpt-5.4",      "low",  "5.4<br>low"),
    ("gpt-5.5",      "high", "5.5<br>high"),
]


def load_tblite():
    """tblite[(model, effort, harness_key)] = cell (Wilson CI over the 100 trials)."""
    import json
    import os
    import re
    tbl = {}
    for model, effort, root in TBLITE_ROOTS:
        path = os.path.join(TBLITE_BASE, root, "summary.json")
        with open(path) as f:
            recs = json.load(f)
        for e in recs:
            cmd = " ".join(e["command"]) if isinstance(e.get("command"), list) else ""
            m = re.search(r"seed_[a-z0-9_]+", cmd)  # captures the full _v2 suffix
            hk = TBLITE_SEED_TO_KEY.get(m.group(0)) if m else None
            if hk is None:
                continue
            n = int(e["num_trials"])
            succ = int(e["num_successes"])  # crashes excluded -> counted as fails
            tbl[(model, effort, hk)] = {
                "score": 100.0 * succ / n,
                "half": wilson_half(succ, n),
                "succ": succ,
                "attempts": n,
            }
    return tbl


# ---------------------------------------------------------------- geometry
def y_of(score, top, plot_h, lo=SCALE_LO, hi=SCALE_HI):
    """Map a score% to a y coord on a [lo, hi] scale, clamped to that band."""
    s = max(lo, min(hi, score))
    frac = (s - lo) / (hi - lo)
    return top + (1.0 - frac) * plot_h


def spark_svg(scores, *, w, h, dot_color, heavy=False, pad=2.5, dot_r=2.4,
              lo=SCALE_LO, hi=SCALE_HI):
    """Build an SVG sparkline. `scores` is a list of float|None across columns.

    Missing points are skipped entirely: the line connects straight through the
    present points (no break, no dash), as if the missing column were not there.
    """
    n = len(scores)
    top, plot_h = pad, h - 2 * pad
    if n == 1:
        xs = [w / 2.0]
    else:
        left, right = pad + 1.0, w - pad - 1.0
        xs = [left + i * (right - left) / (n - 1) for i in range(n)]

    # present points only, kept in column order; gaps are simply dropped
    pts = [(xs[i], y_of(scores[i], top, plot_h, lo, hi))
           for i in range(n) if scores[i] is not None]

    sw = "1.5" if heavy else "1.1"
    parts = []
    if len(pts) == 1:
        x, y = pts[0]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.2" '
                     f'fill="var(--ink)" class="spark-pt"/>')
    elif len(pts) >= 2:
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="var(--line)" '
            f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # endpoint dot on the last present point (largest model / heaviest harness)
    if pts:
        x, y = pts[-1]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_r}" fill="{dot_color}"/>')

    cls = "spark tall" if heavy else "spark"
    return (f'<svg class="{cls}" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'preserveAspectRatio="none" role="img">{"".join(parts)}</svg>')


def cell_text(cell):
    if cell is None:
        return '<span class="miss">—</span>'
    s = f"{round(cell['score'])}"
    if cell["half"] is not None:
        s = f'{s}<span class="ci"> ±{round(cell["half"])}</span>'
    if cell.get("flag"):
        s += f'<span class="ci"> {cell["flag"]}</span>'
    return s


# ---------------------------------------------------------------- table builders
ROW_W, ROW_H = 88, 18      # row sparkline ("vs ...")
FOOT_W, FOOT_H = 88, 90    # footer sparkline ("vs harness")
CORAL = "#D85A30"


def build_table(*, columns, col_family, scores_for, caption, vs_label,
                col_header_fn, family_groups=None, highlight=None):
    """columns: list of column keys. scores_for(harness_key, col) -> cell|None."""
    out = ['<table class="grid">']

    # optional family-group header row
    if family_groups:
        out.append("<thead><tr><th class=\"corner\"></th>")
        for fam, span in family_groups:
            out.append(f'<th class="fam" colspan="{span}">{fam}</th>')
        out.append('<th class="corner"></th></tr>')
        hdr_open = "<tr>"
    else:
        out.append("<thead>")
        hdr_open = "<tr>"

    # which columns begin a new family (=> draw a stronger vertical divider)
    fam_start = []
    prev_fam = None
    for c in columns:
        fam = col_family.get(c)
        fam_start.append(prev_fam is not None and fam != prev_fam)
        prev_fam = fam

    def famcls(base, i):
        return f"{base} famstart" if fam_start[i] else base

    # column header row
    out.append(hdr_open)
    out.append('<th class="rowlab">harness <span class="ci">loc</span></th>')
    for i, c in enumerate(columns):
        out.append(f'<th class="{famcls("colh", i)}">{col_header_fn(c)}</th>')
    out.append(f'<th class="sparkh">{vs_label}</th></tr></thead><tbody>')

    # data rows
    for hk in HARNESS_ORDER:
        name, loc = HARNESS_META[hk]
        is_hi = (highlight == hk)
        row_scores = [scores_for(hk, c) for c in columns]
        lab_cls = "rowlab hl" if is_hi else "rowlab"
        out.append('<tr class="minrow">' if hk == "minimal" else "<tr>")
        out.append(
            f'<td class="{lab_cls}">{name} <span class="ci">{loc}</span></td>'
        )
        for i, cell in enumerate(row_scores):
            out.append(f'<td class="{famcls("num", i)}">{cell_text(cell)}</td>')
        spk = spark_svg([c["score"] if c else None for c in row_scores],
                        w=ROW_W, h=ROW_H, dot_color=CORAL, heavy=is_hi)
        out.append(f'<td class="sp">{spk}</td>')
        out.append("</tr>")

    # footer: one tall sparkline per column, across harnesses in LOC order.
    # minimal-agent is excluded -- it's a low outlier that distorts the profile.
    foot_harnesses = [hk for hk in HARNESS_ORDER if hk != "minimal"]
    out.append('</tbody><tfoot><tr>')
    out.append('<td class="rowlab foot">vs harness ↓ <span class="ci">loc→</span>'
               '<br><span class="ci">ex. minimal</span></td>')
    for i, c in enumerate(columns):
        col_scores = []
        for hk in foot_harnesses:
            cell = scores_for(hk, c)
            col_scores.append(cell["score"] if cell else None)
        spk = spark_svg(col_scores, w=FOOT_W, h=FOOT_H,
                        dot_color="var(--ink)", heavy=True, dot_r=3.0)
        out.append(f'<td class="{famcls("sp foot", i)}">{spk}</td>')
    out.append('<td class="sp foot blank"></td></tr></tfoot></table>')

    return (f'<figure><figcaption>{caption}</figcaption>'
            + "".join(out) + "</figure>")


def build_retry_table(retry, caption):
    """Retry ablation: rows = harness variant, cols = retry limit 0..3.

    Own scale (RETRY_LO..RETRY_HI). Row sparkline traces score vs retry count;
    the coral dot marks the highest retry budget. A dashed trailing segment +
    a default-cell marker flag the mini-swe-agent r3 = default-run cell.
    """
    out = ['<table class="grid">']
    out.append('<thead><tr>')
    out.append('<th class="rowlab">harness</th>')
    for n in RETRY_LIMITS:
        out.append(f'<th class="colh">r{n}</th>')
    out.append('<th class="sparkh">vs retries →</th></tr></thead><tbody>')

    for label, _prefix in RETRY_ROWS:
        cells = [retry.get((label, n)) for n in RETRY_LIMITS]
        out.append('<tr>')
        out.append(f'<td class="rowlab">{label}</td>')
        for n, cell in zip(RETRY_LIMITS, cells):
            txt = cell_text(cell)
            if cell is not None and cell.get("is_default"):
                txt += '<span class="ci"> †</span>'
            out.append(f'<td class="num">{txt}</td>')
        spk = spark_svg([c["score"] if c else None for c in cells],
                        w=ROW_W, h=ROW_H, dot_color=CORAL,
                        lo=RETRY_LO, hi=RETRY_HI)
        out.append(f'<td class="sp">{spk}</td>')
        out.append('</tr>')

    out.append('</tbody></table>')
    return (f'<figure><figcaption>{caption}</figcaption>'
            + "".join(out) + "</figure>")


def build_tblite_profile(tblite):
    """TBLite gpt-5.5/high harness profile: rows = harness (LOC ↑), one pass col.

    A separate 100-task benchmark (1 trial/task) shown apart from the 9-task
    proxy. A tall footer sparkline traces the harness profile. Both barebones
    variants are present (v1 @149, v2 @408).
    """
    cap = ('<b>TBLite · gpt-5.5 · high</b> · 100-task set, 1 trial/task · '
           'pass-rate % with 95% Wilson CI over 100 · rows by LOC ↑ '
           '<span class="note">· a broader benchmark than the 9-task proxy at '
           'left: compare the harness <i>profile</i>, not identical tasks.</span>')
    out = ['<table class="grid">']
    out.append('<thead><tr><th class="rowlab">harness <span class="ci">loc</span></th>'
               '<th class="colh">pass %</th></tr></thead><tbody>')
    for hk, (name, loc) in TBLITE_HARNESS:
        out.append('<tr class="minrow">' if hk == "minimal" else '<tr>')
        out.append(f'<td class="rowlab">{name} <span class="ci">{loc}</span></td>')
        out.append(f'<td class="num">{cell_text(tblite.get(("gpt-5.5", "high", hk)))}</td>')
        out.append('</tr>')
    col_scores = [(tblite.get(("gpt-5.5", "high", hk)) or {}).get("score")
                  for hk, _ in TBLITE_HARNESS if hk != "minimal"]
    spk = spark_svg(col_scores, w=FOOT_W, h=FOOT_H, dot_color="var(--ink)",
                    heavy=True, dot_r=3.0)
    out.append('</tbody><tfoot><tr><td class="rowlab foot">vs harness ↓ '
               '<span class="ci">loc→ · ex. minimal</span></td>'
               f'<td class="sp foot">{spk}</td></tr></tfoot>')
    out.append('</table>')
    return (f'<figure><figcaption>{cap}</figcaption>'
            + "".join(out) + "</figure>")


def build_tblite_compare(tblite):
    """TBLite: msa-prompt-compressed vs codex-full across model strength (weak→strong).

    Row sparkline traces each harness as the model gets stronger; the gap
    between the heavy and minimal harness is the story.
    """
    cap = ('<b>TBLite · msa-prompt-compressed vs codex-full</b> · pass-rate % with 95% '
           'Wilson CI over 100 · cols = model, weak→strong '
           '<span class="note">· how much the heavy harness helps as the model '
           'strengthens. Sparkline scale matches the profile above.</span>')
    out = ['<table class="grid cmp">']
    out.append('<thead><tr><th class="rowlab">harness</th>')
    for _m, _e, lab in TBLITE_CMP_COLS:
        out.append(f'<th class="colh">{lab}</th>')
    out.append('<th class="sparkh">vs model →</th></tr></thead><tbody>')
    for hk, name in TBLITE_CMP_ROWS:
        cells = [tblite.get((m, e, hk)) for m, e, _ in TBLITE_CMP_COLS]
        out.append('<tr>')
        out.append(f'<td class="rowlab">{name}</td>')
        for cell in cells:
            out.append(f'<td class="num">{cell_text(cell)}</td>')
        spk = spark_svg([c["score"] if c else None for c in cells],
                        w=ROW_W, h=ROW_H, dot_color=CORAL)
        out.append(f'<td class="sp">{spk}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return (f'<figure><figcaption>{cap}</figcaption>'
            + "".join(out) + "</figure>")


def build_harness_diff():
    """Quick prose summary of how the three msa-lineage harnesses differ."""
    return (
        '<p class="hdiff"><b>msa-prompt-compressed</b> and <b>mini-swe-agent</b> '
        'are the <i>same harness mechanically</i> — same loop, assistant/tool-call '
        'replay, format retries, env prefixing and logging. They differ only in the '
        'instance prompt: mini-swe-agent ships the full prompt (workflow guidance, '
        'reasoning-text instruction, an example, system info, command snippets); '
        'msa-prompt-compressed strips that down to command rules + submit format.</p>'
        '<p class="hdiff"><b>msa-fully-compressed</b> (the 149-line harness) is a '
        'different, much simpler agent — not just a shorter prompt. It flattens the '
        'whole history into one user message each turn (no assistant/tool-call '
        'replay), has no format retries (a malformed or no-tool response ends the '
        'task instead of being corrected), uses plain-text observations, and skips '
        'the env prefixing and rich metadata. It is closer to a minimal for-loop '
        'agent.</p>')


# ---------------------------------------------------------------- assemble
def main():
    data = load()
    retry = load_retry()
    tblite = load_tblite()

    # gpt msa-prompt-compressed was never run in the TB2 provider matrix; the only TB2
    # datapoint we have is gpt-5.5/high from the retry sweep r3 (= default
    # 3-retry config). Inject it (flagged ‡) so the msa-prompt-compressed row shows it.
    r3 = retry.get(("msa-prompt-compressed", 3))
    if r3 and ("gpt-5.5", "high", "bare_v2") not in data:
        data[("gpt-5.5", "high", "bare_v2")] = {
            "score": r3["score"], "half": r3["half"], "succ": r3["succ"],
            "crashes": 0, "attempts": r3["attempts"], "flag": "‡",
        }

    # ---- Deliverable 1: one table per (family, effort) ----
    # Never mix providers in one table: OpenAI "reasoning = high" and DeepSeek
    # "thinking = high" are different knobs, so each family gets its own tables.
    EFFORT_WORD = {"OpenAI": "reasoning", "DeepSeek": "thinking",
                   "Anthropic": "thinking"}
    sec1 = []
    for fam, fam_models in FAMILIES:
        fam_ids = [m for m, _ in fam_models]
        for eff in EFFORT_ORDER[fam]:
            models = [m for m in fam_ids
                      if any((m, eff, hk) in data for hk in HARNESS_ORDER)]
            if not models:
                continue
            prov = ""
            if fam == "OpenAI" and eff in ("low", "medium"):
                prov = (' <span class="note">· non-mini cells assembled from a prior '
                        '10-trial sweep + the provider matrix (full 90 trials).</span>')
            if fam == "Anthropic":
                prov = (' <span class="note">· manual “best complete run” aggregate; '
                        'CIs recomputed as Wilson-over-90 like the other providers.</span>')
            cap = (f'<b>{fam} · {EFFORT_WORD[fam]} = {eff}</b> · pass-rate % with '
                   f'95% CI half-width · rows by LOC ↑ · cols small→large{prov}')
            sec1.append(build_table(
                columns=models,
                col_family=MODEL_FAMILY,
                scores_for=lambda hk, m: data.get((m, eff, hk)),
                caption=cap,
                vs_label="vs model →",
                col_header_fn=lambda m: MODEL_DISPLAY[m],
            ))

    # ---- Deliverable 2: transpose, one table per model ----
    sec2 = []
    for m in MODEL_ORDER:
        fam = MODEL_FAMILY[m]
        efforts = [e for e in EFFORT_ORDER[fam]
                   if any((m, e, hk) in data for hk in HARNESS_ORDER)]
        if not efforts:
            continue
        prov = ""
        if fam == "OpenAI" and m != "gpt-5.4-mini":
            prov = (' <span class="note">· low/medium cells use a prior 10-trial sweep '
                    '+ provider matrix (full 90 trials).</span>')
        if fam == "Anthropic":
            prov = (' <span class="note">· manual “best complete run” aggregate.</span>')
        erange = efforts[0] if len(efforts) == 1 else f'{efforts[0]}→{efforts[-1]}'
        cap = (f'<b>{MODEL_DISPLAY[m]}</b> <span class="ci">({fam})</span> · '
               f'effort {erange} as columns · rows by LOC ↑{prov}')
        sec2.append(build_table(
            columns=efforts,
            col_family={e: fam for e in efforts},
            scores_for=lambda hk, e, _m=m: data.get((_m, e, hk)),
            caption=cap,
            vs_label="vs effort →",
            col_header_fn=lambda e: e,
        ))

    # ---- Deliverable 3: retry ablation ----
    retry_cap = (
        '<b>gpt-5.5 · high</b> · retry-limit ablation · pass-rate % with 95% CI '
        'half-width · cols = max retries (r0→r3) · '
        f'<span class="note">sparkline scale {int(RETRY_LO)}–{int(RETRY_HI)}% '
        '(tighter than the main tables) so the retry trend reads · '
        '† mini-swe-agent r3 is the default 3-retry provider-matrix run.</span>'
    )
    sec3 = build_retry_table(retry, retry_cap)

    # ---- Deliverable 4: TBLite, a separate benchmark, laid out on the right ----
    sec4 = build_tblite_profile(tblite) + "\n" + build_tblite_compare(tblite)

    # ---- Deliverable 5: harness mechanics diff (full width, very bottom) ----
    sec5 = build_harness_diff()

    html = TEMPLATE.format(
        scale_lo=int(SCALE_LO), scale_hi=int(SCALE_HI),
        retry_lo=int(RETRY_LO), retry_hi=int(RETRY_HI),
        sec1="\n".join(sec1), sec2="\n".join(sec2), sec3=sec3, sec4=sec4,
        sec5=sec5,
    )
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TB2-core pass rates · sparkline tables</title>
<style>
  :root {{
    --bg: #ffffff; --ink: #1a1a18; --tertiary: #8a8880;
    --line: #5F5E5A; --rule: #d9d7d0; --rule-strong: #b4b1a8;
    --hl-bg: #f3f1ea; --coral: #D85A30; --red: #c0392b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16161a; --ink: #e9e7e0; --tertiary: #9b988e;
      --line: #b9b6ad; --rule: #34343a; --rule-strong: #54545c;
      --hl-bg: #232228; --coral: #ef7a4f; --red: #ff6e63;
    }}
  }}
  html {{ background: var(--bg); color: var(--ink); }}
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 1380px;
         margin: 40px auto; padding: 0 24px 80px; line-height: 1.5; }}
  .page {{ display: flex; gap: 40px; align-items: flex-start; }}
  .main {{ flex: 1 1 auto; min-width: 0; }}
  .side {{ flex: 0 0 360px; width: 360px; }}
  .side h2 {{ margin-top: 0; }}
  .side figure {{ margin: 0 0 26px; }}
  @media (max-width: 1180px) {{
    .page {{ display: block; }}
    .side {{ flex: none; width: auto; margin-top: 20px; }}
  }}
  h1 {{ font-size: 21px; margin-bottom: 2px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .09em;
        color: var(--tertiary); font-family: system-ui, sans-serif;
        margin: 44px 0 6px; }}
  .sub {{ color: var(--tertiary); font-size: 13px; margin-bottom: 8px;
          font-family: system-ui, sans-serif; }}
  .legend {{ font-family: system-ui, sans-serif; font-size: 12px;
             color: var(--tertiary); margin: 10px 0 26px; line-height: 1.7; }}
  .legend b {{ color: var(--ink); }}
  .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%;
             vertical-align:baseline; margin: 0 1px; }}
  figure {{ margin: 0 0 30px; }}
  figcaption {{ font-family: system-ui, sans-serif; font-size: 12.5px;
                color: var(--ink); margin-bottom: 7px; }}
  figcaption .note {{ color: var(--tertiary); font-style: italic; }}
  figcaption .ci {{ color: var(--tertiary); }}
  table.grid {{ border-collapse: collapse; font-variant-numeric: tabular-nums;
                font-family: system-ui, sans-serif; }}
  .grid th, .grid td {{ border: 0.5px solid var(--rule); padding: 3px 8px;
                        font-size: 12px; text-align: right; white-space: nowrap; }}
  .grid thead th {{ font-weight: 600; color: var(--ink); vertical-align: bottom; }}
  .grid th.fam {{ text-align: center; font-size: 11px; letter-spacing: .05em;
                  text-transform: uppercase; color: var(--tertiary);
                  border-bottom: 0.5px solid var(--rule-strong); }}
  .grid th.corner {{ border: none; }}
  .grid th.rowlab, .grid td.rowlab {{ text-align: left; font-weight: 400; }}
  .grid td.rowlab {{ color: var(--ink); }}
  .grid td.rowlab.hl {{ font-weight: 700; }}
  .grid .ci {{ color: var(--tertiary); font-size: 10.5px; }}
  .grid td.num {{ color: var(--ink); }}
  .grid .miss {{ color: var(--tertiary); }}
  .grid th.colh {{ text-align: right; }}
  .grid th.sparkh {{ color: var(--tertiary); font-weight: 400; font-size: 11px; }}
  .grid td.sp {{ padding: 0 4px; }}
  .grid .famstart {{ border-left: 1.5px solid var(--rule-strong); }}
  .grid td.sp.blank {{ border: none; }}
  .grid tfoot td.foot {{ border-top: 0.5px solid var(--rule-strong);
                         vertical-align: bottom; }}
  .grid td.rowlab.foot {{ color: var(--tertiary); font-style: italic;
                          vertical-align: bottom; }}
  svg.spark {{ width: 88px; height: 18px; display: block; }}
  svg.spark.tall {{ height: 90px; }}
  .grid.cmp td.sp svg.spark {{ width: 62px; }}
  .grid tr.minrow td.rowlab,
  .grid tr.minrow td.num,
  .grid tr.minrow .ci {{ color: var(--red); }}
  p.hdiff {{ font-family: system-ui, sans-serif; font-size: 12.5px;
             color: var(--ink); line-height: 1.6; margin: 4px 0 10px;
             max-width: 860px; }}
  p.hdiff code {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace;
                  font-size: 11px; }}
  .page + h2 {{ clear: both; }}
</style>
</head>
<body>

<h1>TerminalBench-2 core · harness × model pass rates</h1>
<div class="sub">9-task TB2-core proxy set, 10 trials/task = 90 trials per cell.
Crashes counted as failures (score = successes / 90).</div>

<div class="legend">
  Each cell: <b>pass-rate %</b> with the 95% CI half-width in
  <span class="ci">smaller tertiary type (±pp)</span>.
  The CI is the <b>Wilson binomial interval over the 90 trials</b> — per-trial
  sampling noise on <i>these exact 9 tasks</i>, not the (much wider) task-level CI.
  Main-table sparklines share one fixed score scale, <b>{scale_lo}–{scale_hi}%</b> — height
  is comparable everywhere and a flat line means within-noise. Scores outside the band
  rest on the floor/ceiling.
  Row sparklines (18px) trace the row across columns;
  the <span class="swatch" style="background:var(--coral)"></span> coral dot marks the
  largest-model / highest-effort endpoint.
  Footer sparklines (tall, 90px) trace each column down the harnesses in LOC order
  (even-spaced, not to LOC scale); the
  <span class="swatch" style="background:var(--ink)"></span> ink dot marks the
  heaviest harness (codex-full). <b style="color:var(--red)">minimal-agent</b> rows
  are drawn in red and excluded from these vs-harness sparklines as a low outlier.
  <b>‡</b> = msa-prompt-compressed for gpt-5.5/high is the retry-sweep r3 (default config);
  the TB2 provider matrix did not run msa-prompt-compressed for OpenAI/DeepSeek (those cells
  are blank). Anthropic ran both msa-fully-compressed and msa-prompt-compressed.
  No data → “—” in the cell; the sparkline skips that point and connects straight
  through its neighbours (the missing column is treated as absent, not zero).
</div>

<div class="page">
<div class="main">

<h2>I · One table per model</h2>
<div class="sub">Rows = harnesses (LOC ↑). Columns = reasoning effort (least→most).</div>
{sec2}

<h2>II · Transpose — one table per provider × effort</h2>
<div class="sub">Rows = harnesses (LOC ↑). Columns = model (small→large). Each table is a single
provider: OpenAI “reasoning” and DeepSeek “thinking” levels are different knobs and are
never placed in the same table.</div>
{sec1}

<h2>III · Retry-limit ablation</h2>
<div class="sub">gpt-5.5 · high. Rows = mini-swe-agent vs msa-prompt-compressed.
Columns = max retries (r0→r3). Sparkline scale {retry_lo}–{retry_hi}% — local to this
section so the retry trend is visible; do not compare its heights to the tables above.</div>
{sec3}

</div><!-- /.main -->

<aside class="side">
<h2>IV · TBLite benchmark</h2>
<div class="sub">A separate, broader benchmark — 100 tasks × 1 trial. Not the
9-task proxy used at left.</div>
{sec4}
</aside>

</div><!-- /.page -->

<h2>V · Harness mechanics — msa-fully-compressed vs msa-prompt-compressed vs mini-swe-agent</h2>
<div class="sub">Why the rows differ in behavior, not just score. msa-prompt-compressed and
mini-swe-agent share machinery (prompt-only diff); msa-fully-compressed is a much simpler agent.</div>
{sec5}

</body>
</html>
"""

if __name__ == "__main__":
    main()
