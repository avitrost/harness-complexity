#!/usr/bin/env python3
"""Attribute-driver analysis for gpt-5.4-mini low harness behavior.

This is intentionally small-data exploratory analysis. It excludes
minimal-agent and terminus-2-compressed, then compares the remaining harnesses
with several complementary views:

  * harness-level correlations
  * task-fixed-effect single-feature regressions
  * leave-one-task-out checks for feature groups
  * paired deltas versus msa-fully-compressed
  * attempt-level success/failure behavior contrasts
"""
import csv
import html
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "behavior_gpt54_low_family_20260612"
)
OUT = HERE / "gpt54_low_attribute_drivers.html"
ARTIFACT_DIR = BASE / "attribute_driver_analysis"
EXCLUDE = {"minimal", "term2_comp"}


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def num(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def pct(value):
    try:
        return f"{100.0 * float(value):.1f}%"
    except Exception:
        return ""


def corr(a, b, method="pearson"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if method == "spearman":
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_task_rows():
    task = pd.read_csv(BASE / "gpt54mini_low_tb2_allharnesses" / "by_task_harness.csv")
    task = task[~task.harness_alias.isin(EXCLUDE)].copy()
    for col in task.columns:
        if col.startswith("mean_") or col in {
            "success_rate",
            "successes",
            "attempts",
            "harness_loc",
        }:
            task[col] = pd.to_numeric(task[col], errors="coerce").fillna(0)
    task["read_search_commands"] = task["mean_read_file_commands"] + task["mean_search_commands"]
    task["stateful_replay"] = (
        task["mean_session_commands"]
        + task["mean_multi_call_turns"]
        + task["mean_response_replay_turns"]
    )
    task["codex_tooling"] = task["mean_session_commands"] + task["mean_multi_call_turns"]
    task["loop_actions"] = (
        task["mean_read_file_commands"]
        + task["mean_search_commands"]
        + task["mean_edit_commands"]
        + task["mean_script_commands"]
        + task["mean_test_commands"]
    )
    task["edit_verify"] = task["mean_edit_commands"] + task["mean_test_commands"]
    task["verify_plus_script"] = task["mean_script_commands"] + task["mean_test_commands"]
    task["search_read_per_cmd"] = task["read_search_commands"] / task[
        "mean_commands"
    ].replace(0, np.nan)
    task["edit_per_cmd"] = task["mean_edit_commands"] / task["mean_commands"].replace(0, np.nan)
    task["test_per_cmd"] = task["mean_test_commands"] / task["mean_commands"].replace(0, np.nan)
    task = task.replace([np.inf, -np.inf], np.nan).fillna(0)
    classes = []
    for alias in task.harness_alias:
        if alias == "mini_bare":
            classes.append("none")
        elif alias == "mini_v2":
            classes.append("replay_only")
        elif alias == "c400":
            classes.append("session_only")
        elif alias in {"c700", "c1000", "c1300", "c_comp", "c_full"}:
            classes.append("session_multi_replay")
        else:
            classes.append("other")
    task["mechanics_class"] = classes
    return task


FEATURES = [
    "harness_loc",
    "mean_commands",
    "mean_failed_commands",
    "mean_edit_commands",
    "mean_test_commands",
    "mean_inspect_commands",
    "mean_read_file_commands",
    "mean_search_commands",
    "read_search_commands",
    "mean_script_commands",
    "mean_session_commands",
    "mean_multi_call_turns",
    "mean_response_replay_turns",
    "stateful_replay",
    "codex_tooling",
    "mean_api_input_tokens",
    "mean_api_output_tokens",
    "mean_stdout_chars",
    "mean_format_retry_total",
    "loop_actions",
    "edit_verify",
    "verify_plus_script",
    "search_read_per_cmd",
    "edit_per_cmd",
    "test_per_cmd",
]


FEATURE_LABEL = {
    "stateful_replay": "session + multi-call + replay",
    "codex_tooling": "session + multi-call",
    "mean_response_replay_turns": "response replay turns",
    "mean_multi_call_turns": "multi-call turns",
    "mean_session_commands": "session commands",
    "read_search_commands": "read + search commands",
    "mean_test_commands": "test-like commands",
    "mean_commands": "total commands",
    "mean_api_input_tokens": "API input tokens",
    "mean_api_output_tokens": "API output tokens",
    "mean_edit_commands": "edit commands",
    "mean_script_commands": "script commands",
}


def harness_summary(task):
    agg_map = {"success_rate": "mean"}
    for feature in FEATURES:
        agg_map[feature] = "mean"
    return (
        task.groupby(["harness_alias", "harness_name", "harness_loc"], as_index=False)
        .agg(agg_map)
        .sort_values("harness_loc")
    )


def harness_correlations(harness):
    rows = []
    for feature in FEATURES:
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABEL.get(feature, feature),
                "pearson": corr(harness[feature], harness.success_rate),
                "spearman": corr(harness[feature], harness.success_rate, "spearman"),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False)


def task_fe_single(task):
    y = task["success_rate"] - task.groupby("task")["success_rate"].transform("mean")
    rows = []
    for feature in FEATURES:
        x = task[feature] - task.groupby("task")[feature].transform("mean")
        xs = x.to_numpy(dtype=float)
        ys = y.to_numpy(dtype=float)
        sd = xs.std(ddof=1)
        if sd == 0:
            continue
        xs = xs / sd
        beta = float(xs.dot(ys) / xs.dot(xs))
        pred = xs * beta
        ssr = float(((ys - pred) ** 2).sum())
        sst = float((ys**2).sum())
        r2 = 1 - ssr / sst if sst else np.nan
        sigma2 = ssr / (len(ys) - 1)
        se = math.sqrt(sigma2 / xs.dot(xs)) if xs.dot(xs) > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABEL.get(feature, feature),
                "beta_pp_per_sd": beta * 100,
                "r2": r2,
                "t_stat": beta / se if se else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "beta_pp_per_sd", key=lambda s: s.abs(), ascending=False
    )


def residualize_by_task(task, series, train_mask, test_mask):
    arr = series.to_numpy(dtype=float)
    out = np.zeros_like(arr, dtype=float)
    tasks = task.task.to_numpy()
    for name in sorted(set(tasks)):
        idx = tasks == name
        if (idx & train_mask).any():
            out[idx & train_mask] = arr[idx & train_mask] - arr[idx & train_mask].mean()
        if (idx & test_mask).any():
            out[idx & test_mask] = arr[idx & test_mask] - arr[idx & test_mask].mean()
    return out


def leave_task_out(task):
    sets = {
        "stateful_replay": ["stateful_replay"],
        "codex_tooling": ["codex_tooling"],
        "actions": [
            "read_search_commands",
            "mean_edit_commands",
            "mean_script_commands",
            "mean_test_commands",
        ],
        "actions+mechanics": [
            "read_search_commands",
            "mean_edit_commands",
            "mean_script_commands",
            "mean_test_commands",
            "mean_session_commands",
            "mean_multi_call_turns",
            "mean_response_replay_turns",
        ],
        "tokens+loc": ["harness_loc", "mean_api_input_tokens", "mean_api_output_tokens"],
    }
    tasks = task.task.to_numpy()
    y_raw = task.success_rate.to_numpy(dtype=float)
    rows = []
    for name, features in sets.items():
        pred_all = np.full(len(task), np.nan)
        yres_all = np.full(len(task), np.nan)
        for held in sorted(set(tasks)):
            train = tasks != held
            test = tasks == held
            yres = residualize_by_task(task, task.success_rate, train, test)
            xs_train = []
            xs_test = []
            for feature in features:
                xres = residualize_by_task(task, task[feature], train, test)
                mu = xres[train].mean()
                sd = xres[train].std(ddof=1)
                if sd == 0:
                    continue
                xs_train.append((xres[train] - mu) / sd)
                xs_test.append((xres[test] - mu) / sd)
            if xs_train:
                xtr = np.vstack(xs_train).T
                xte = np.vstack(xs_test).T
                beta = np.linalg.solve(xtr.T @ xtr + np.eye(xtr.shape[1]), xtr.T @ yres[train])
                pred = xte @ beta
            else:
                pred = np.zeros(test.sum())
            pred_all[test] = pred
            yres_all[test] = yres[test]
        r2 = 1 - float(((yres_all - pred_all) ** 2).sum()) / float((yres_all**2).sum())
        rows.append({"feature_set": name, "features": ", ".join(features), "cv_r2": r2})
    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False)


def paired_deltas(task):
    base = task[task.harness_alias == "mini_bare"].set_index("task")
    delta_rows = []
    for _, row in task.iterrows():
        if row.harness_alias == "mini_bare":
            continue
        b = base.loc[row.task]
        out = {
            "task": row.task,
            "harness_alias": row.harness_alias,
            "harness_name": row.harness_name,
            "delta_success": row.success_rate - b.success_rate,
        }
        for feature in FEATURES:
            out[f"d_{feature}"] = row[feature] - b[feature]
        delta_rows.append(out)
    delta = pd.DataFrame(delta_rows)
    corr_rows = []
    for feature in FEATURES:
        corr_rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABEL.get(feature, feature),
                "pearson": corr(delta[f"d_{feature}"], delta.delta_success),
                "spearman": corr(delta[f"d_{feature}"], delta.delta_success, "spearman"),
            }
        )
    return delta, pd.DataFrame(corr_rows).sort_values(
        "pearson", key=lambda s: s.abs(), ascending=False
    )


def mechanics_class_summary(task):
    return (
        task.groupby(["mechanics_class", "harness_alias", "harness_name"], as_index=False)
        .agg(
            score=("success_rate", "mean"),
            attempts=("attempts", "sum"),
            stateful_replay=("stateful_replay", "mean"),
            tests=("mean_test_commands", "mean"),
            read_search=("read_search_commands", "mean"),
            edits=("mean_edit_commands", "mean"),
        )
        .sort_values(["mechanics_class", "score"], ascending=[True, False])
    )


def success_failure_contrasts():
    prof = pd.read_csv(BASE / "action_category_analysis" / "attempt_action_profiles.csv")
    prof = prof[
        (prof.benchmark == "tb2")
        & (prof.model == "gpt-5.4-mini")
        & (prof.effort == "low")
        & (~prof.harness_alias.isin(EXCLUDE))
    ].copy()
    beh = pd.read_csv(BASE / "gpt54mini_low_tb2_allharnesses" / "attempt_behavior.csv")
    beh = beh[~beh.harness_alias.isin(EXCLUDE)].copy()
    numeric = {
        "success",
        "included_actions",
        "file_read_count",
        "search_count",
        "file_edit_count",
        "script_execution_count",
        "test_execution_count",
        "reasoning_only_count",
        "commands",
        "failed_commands",
        "session_commands",
        "multi_call_turns",
        "response_replay_turns",
        "model_call_input_tokens",
        "model_call_output_tokens",
    }
    for frame in (prof, beh):
        for col in frame.columns:
            if col in numeric:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    merged = prof.merge(
        beh[
            [
                "candidate",
                "task",
                "attempt",
                "commands",
                "failed_commands",
                "session_commands",
                "multi_call_turns",
                "response_replay_turns",
                "model_call_output_tokens",
                "model_call_input_tokens",
            ]
        ],
        on=["candidate", "task", "attempt"],
        how="left",
    )
    fields = [
        "included_actions",
        "file_read_count",
        "search_count",
        "file_edit_count",
        "script_execution_count",
        "test_execution_count",
        "reasoning_only_count",
        "commands",
        "failed_commands",
        "session_commands",
        "multi_call_turns",
        "response_replay_turns",
        "model_call_input_tokens",
        "model_call_output_tokens",
    ]
    rows = []
    for field in fields:
        succ = merged[merged.success == 1][field].mean()
        fail = merged[merged.success == 0][field].mean()
        rows.append({"feature": field, "success_mean": succ, "failure_mean": fail, "diff": succ - fail})
    return pd.DataFrame(rows)


def fine_session_metrics():
    path = ARTIFACT_DIR / "fine_session_attempt_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    fields = [
        "session_turns_raw",
        "unique_sessions",
        "session_reuse_turns",
        "write_stdin_turns",
        "post_failure_turns_raw",
        "post_failure_session_turns",
        "post_failure_replay_turns",
        "replay_turns_raw",
        "multi_turns_raw",
        "multi_tool_calls_raw",
        "timeout_turns",
    ]
    for col in ["success", *fields]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    rows = []
    for field in fields:
        succ = frame[frame.success == 1][field].mean()
        fail = frame[frame.success == 0][field].mean()
        rows.append(
            {
                "metric": field,
                "success_mean": succ,
                "failure_mean": fail,
                "diff": succ - fail,
            }
        )
    return pd.DataFrame(rows)


def write_csvs(outputs):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False)


def table(frame, columns, limit=None):
    shown = frame.head(limit) if limit else frame
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "pct":
                rendered = pct(value)
            elif kind == "num":
                rendered = num(value, 2)
            elif kind == "num3":
                rendered = num(value, 3)
            else:
                rendered = esc(value)
            cls = ' class="num"' if kind in {"pct", "num", "num3"} else ""
            cells.append(f"<td{cls}>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def css():
    return """
    body { margin: 0; background: #fffffb; color: #1d1a16; font: 18px/1.48 Georgia, "Times New Roman", serif; }
    main { max-width: 1180px; margin: 0 auto; padding: 42px 36px 76px; }
    h1 { font-weight: 400; font-size: 43px; line-height: 1.05; margin: 0 0 12px; letter-spacing: 0; }
    h2 { margin: 42px 0 10px; padding-top: 16px; border-top: 1px solid #d8d1c3; font: 700 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }
    p { max-width: 890px; margin: 12px 0; }
    .deck { font-size: 21px; color: #3b352d; }
    .note { border-left: 2px solid #3d6f85; padding-left: 14px; color: #332f28; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 24px 0 12px; }
    .point { border-top: 3px solid #222; padding-top: 9px; font-size: 16px; line-height: 1.35; }
    .point b { display: block; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .glossary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 14px 0 8px; max-width: 1040px; }
    .term { border-top: 1px solid #d8d1c3; padding-top: 8px; font-size: 15px; line-height: 1.4; }
    .term b { display: block; font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    table.data { border-collapse: collapse; width: 100%; font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 12px 0 8px; }
    table.data th { text-align: left; border-bottom: 1px solid #d8d1c3; padding: 6px 7px; white-space: nowrap; vertical-align: bottom; }
    table.data td { border-bottom: 1px solid #eee8dc; padding: 6px 7px; vertical-align: top; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; }
    .small { color: #756f63; font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    @media (max-width: 850px) { main { padding: 28px 18px 56px; } h1 { font-size: 34px; } .grid, .glossary { grid-template-columns: 1fr; } }
    """


def main():
    task = load_task_rows()
    harness = harness_summary(task)
    hcorr = harness_correlations(harness)
    single = task_fe_single(task)
    cv = leave_task_out(task)
    deltas, delta_corr = paired_deltas(task)
    mechanics = mechanics_class_summary(task)
    contrasts = success_failure_contrasts()
    fine_sessions = fine_session_metrics()
    outputs = {
        "harness_summary": harness,
        "harness_correlations": hcorr,
        "task_fe_single_feature": single,
        "leave_task_out_feature_sets": cv,
        "paired_delta_vs_mini_bare": deltas,
        "paired_delta_correlations": delta_corr,
        "mechanics_class_summary": mechanics,
        "success_failure_contrasts": contrasts,
    }
    write_csvs(outputs)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPT-5.4-Mini Low Attribute Drivers</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <h1>Which harness attributes seem to drive the small-model lift?</h1>
  <p class="deck">Exploratory analysis for <b>gpt-5.4-mini low</b> on the TB2
  9-task subset, excluding <b>minimal-agent</b> and
  <b>terminus-2-compressed</b>.</p>
  <p class="small">CSV outputs: <code>{esc(str(ARTIFACT_DIR))}</code></p>

  <div class="grid">
    <div class="point"><b>Most stable signal</b>
      Session handling plus multi-call/replay mechanics is the only feature
      family that survives the task-fixed and leave-one-task-out checks.</div>
    <div class="point"><b>Secondary signal</b>
      Read/search plus a bounded edit/execute loop helps. Repeated editing or
      scripting by itself looks more like struggle than capability.</div>
    <div class="point"><b>Weak signal</b>
      LOC, input tokens, output tokens, and total command count are weaker and
      less stable than the mechanics features.</div>
  </div>

  <h2>Caveat</h2>
  <p class="note">This is not causal identification. The sample is small, many
  harness features are correlated, and action counts are partly downstream of
  the model's behavior. The goal is to narrow plausible mechanisms, not to prove
  one variable independently causes the whole lift.</p>

  <h2>What The Terms Mean</h2>
  <p>By <b>recoverable stateful interaction</b>, I mean a harness that gives the
  model a durable enough interaction loop to recover from a bad command: it can
  see the command output or error, keep track of prior tool calls, adjust the
  next action, and continue without having to reconstruct the whole state from
  scratch. In these logs this is not one primitive; it is a bundle of mechanics
  that make the next step better informed after each shell action.</p>
  <div class="glossary">
    <div class="term"><b>Session commands</b>
      Counted when a shell turn has <code>metadata.unified_exec.session_id</code>.
      Practically, this means the harness is using a persistent execution
      session rather than only isolated one-shot shell calls. That can preserve
      useful process/session state such as current shell context, active
      subprocesses, or interactive continuity.</div>
    <div class="term"><b>Multi-call turns</b>
      Counted when Codex metadata reports <code>codex_emitted_tool_calls &gt; 1</code>.
      This means one model response produced multiple tool calls instead of a
      single command. It is a proxy for a harness that lets the model express a
      small sequence of operations in one reasoning step.</div>
    <div class="term"><b>Response replay turns</b>
      Counted when a turn includes <code>mini_swe_agent_v2_messages</code> or
      <code>codex_response_items</code> metadata. This means the harness is
      preserving or replaying structured assistant/tool-call history around the
      command, rather than logging only the raw shell command. It helps the next
      model call retain the action/observation chain.</div>
    <div class="term"><b>Session + multi-call + replay</b>
      The combined metric used in the regressions:
      <code>mean_session_commands + mean_multi_call_turns + mean_response_replay_turns</code>.
      It should be read as a proxy for interaction scaffolding, not as a literal
      causal variable. It captures whether the harness supports continuity,
      batched/planned actions, and trace replay.</div>
  </div>
  <p>So the claim is not "state" in the philosophical sense, and not just "the
  model has a longer prompt." The concrete claim is: the small model does better
  when the harness makes the action/observation loop recoverable enough that
  failed commands and partial results become useful information for the next
  action.</p>
  <p>More explicitly: most files written to <code>/app</code> persist even in a
  non-persistent shell, so the likely benefit is not just "the file system keeps
  state." The helpful pieces appear to be: keeping the command/observation
  history structured, letting the model continue after errors, allowing multiple
  related tool calls in one response, and, when needed, preserving a live shell
  session for ongoing processes or interactive continuations.</p>

  <h2>Mechanics Class</h2>
  <p>The coarse class view is the clearest: replay-only and session-only each
  move the score from 13.3% to 16.7%, while the session+multi-call+replay Codex
  family averages 22.9%. The outlier is codex-1300, which has the mechanics but
  underperforms, so mechanics are necessary-looking but not sufficient.</p>
  {table(mechanics, [
      ("mechanics_class", "Class", "text"),
      ("harness_name", "Harness", "text"),
      ("score", "Score", "pct"),
      ("stateful_replay", "Stateful+replay", "num"),
      ("tests", "Tests/attempt", "num"),
      ("read_search", "Read+search", "num"),
      ("edits", "Edits", "num"),
  ])}

  <h2>Task-Fixed Regressions</h2>
  <p>These regressions subtract each task's mean score first, then ask which
  within-task feature variation predicts above-task-average performance. The
  coefficient is percentage points of score for a one-standard-deviation
  increase in the feature residual.</p>
  {table(single, [
      ("label", "Feature", "text"),
      ("beta_pp_per_sd", "Beta pp / SD", "num"),
      ("r2", "R2", "num3"),
      ("t_stat", "t", "num"),
  ], limit=14)}

  <h2>Out-of-Task Check</h2>
  <p>Leave-one-task-out prediction is harsh with only 9 tasks. The mechanics
  feature sets remain positive; action-only and token/LOC sets do not.</p>
  {table(cv, [
      ("feature_set", "Feature set", "text"),
      ("cv_r2", "LOTO CV R2", "num3"),
      ("features", "Features", "text"),
  ])}

  <h2>Paired Delta Against MSA-Fully-Compressed</h2>
  <p>For every task, each harness is compared to the 149-line compressed
  mini-SWE baseline. The largest positive correlations with score deltas are
  again session/multi/replay features.</p>
  {table(delta_corr, [
      ("label", "Feature delta", "text"),
      ("pearson", "Pearson", "num3"),
      ("spearman", "Spearman", "num3"),
  ], limit=14)}

  <h2>Success Versus Failure</h2>
  <p>Successful attempts are more active, but the important distinction is the
  complete loop. Successes have more read/search, edits, scripts, tests, session
  commands, and replay turns. Output tokens move in the opposite direction,
  suggesting long hidden/visible output often marks struggle rather than
  success.</p>
  {table(contrasts, [
      ("feature", "Feature", "text"),
      ("success_mean", "Success mean", "num"),
      ("failure_mean", "Failure mean", "num"),
      ("diff", "Diff", "num"),
  ])}

  <h2>Finer Session Metrics</h2>
  <p>I added a finer metadata pass over the raw <code>harness-turn-*.json</code>
  files. This separates raw session use from session reuse, post-failure
  continuation, replay, and multi-call behavior. The signal is strongest for
  <b>continuing after a failure with structured replay</b>, not for session reuse
  alone.</p>
  {table(fine_sessions, [
      ("metric", "Metric", "text"),
      ("success_mean", "Success mean", "num"),
      ("failure_mean", "Failure mean", "num"),
      ("diff", "Diff", "num"),
  ]) if not fine_sessions.empty else '<p class="small">Fine session metrics CSV not found.</p>'}

  <h2>Working Interpretation</h2>
  <p>The most likely driver is <b>recoverable stateful interaction</b>: the model
  benefits when the harness preserves enough interaction structure for it to
  recover from partial failures and continue an inspect-edit-execute loop. The
  second driver is <b>context acquisition without thrash</b>: read/search is
  useful, while high edit/script volume without the stateful loop often marks
  wandering. Test-like checks help when present, but they are not the whole
  story because some high-scoring Codex rows have few explicit test commands.</p>
</main>
</body>
</html>
"""
    OUT.write_text(html_text)
    print(OUT)
    print(ARTIFACT_DIR)


if __name__ == "__main__":
    main()
