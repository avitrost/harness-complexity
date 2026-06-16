# Mechanism Goal Coordination Summary

Date: 2026-06-16

Shared run:
`/wbl-fast/usrs/trost/harness-complexity/final_test/tblite_mechanism_gpt54mini_low_20260616_182830`

Slurm jobs:
- Mechanism grid: `473388`
- Dependent analyzer: `473415`

Scope: dedicated goal-oriented subagents investigated each proposed harness mechanism for
`gpt-5.4-mini` low on TBLite. Most current conclusions are bounded because the shared
100-task grid is still running; bounded results below identify the exact post-run
analysis or follow-up experiment needed.

## Prompt And Context Prior

Status: inconclusive pending full paired grid.

Partial evidence: `bare_v2_codex_prompt_r3` was positive versus
`bare_v2_context_only_r3` in early overlap, but the confidence interval crossed zero and
there was at least one concrete regression.

Next step: after analyzer `473415` completes, compare `bare_v2_r3`,
`bare_v2_context_only_r3`, `bare_v2_codex_prompt_r3`,
`codex_full_minimal_prompt`, and `codex_full` on the full 100 paired tasks. If
uncertain, run 2 more trials only for `bare_v2_r3`,
`bare_v2_context_only_r3`, and `bare_v2_codex_prompt_r3`.

## Persistent Shell And Sessions

Status: inconclusive; direct persistence is not supported as the main factor in partial
data.

Partial evidence: `persistent_exec_only` was not outperforming prompt-only in the sparse
overlap, and `codex_full` versus `codex_full_no_write_stdin` was mixed.

Next step: after full run, compute paired estimands:
`persistent_bash - bare`, `prompt_only - bare`, `exec_only - bare`,
`exec_only - prompt_only`, `persistent_bash - prompt_only`, and
`codex_full - codex_full_no_write_stdin`, stratified by actual session reuse and polling.

## Structured Exec And Error Feedback

Status: current ablation is contaminated for the intended causal claim.

Reason: `codex_full_no_unified_output` still exposes wall time, exit code, and framed
output through fallback formatting, so it does not isolate exit/status/session metadata.

Next experiment: one paired Codex loop with identical prompt, tools, model, and tasks,
varying only model-visible observation text:
full unified output, exit-code-only, raw stdout/stderr only, no `session_id`, and no
exit code.

## Recovery, Retries, Parser Tolerance, Verification

Status: unlikely to be the main driver, but final read awaits full grid.

Partial evidence: `bare_v2_r3` versus `bare_v2_r0` was flat in early paired overlap, and
observed Codex recovery wins did not involve recorded recovery events.

Next step: full paired audit after analyzer output. If parser tolerance remains a target,
run a strict-parser Codex ablation disabling response-item fallback, alias normalization,
and custom/local shell parsing.

## Patch And Editing Affordance

Status: current no-patch ablation is contaminated.

Reason: `codex_full_no_patch_tool` hides the custom `apply_patch` tool, but
`exec_command` still intercepts shell `apply_patch` commands and applies them through
Harbor patch machinery.

Next experiment: create a true `codex_full_no_patch_affordance` profile that disables the
patch tool, removes/conditions model-visible patch prompt guidance, and disables
`exec_command` patch interception or returns a hard blocked-patch failure. Run it paired
against `codex_full` on the same TBLite tasks.

## History Replay, Response Replay, Context Management

Status: source-level ablations are model-visible, but causal read awaits full grid.

Findings: `no_history_replay` removes prior assistant/tool/tool-output state;
`no_response_replay` can lose exact response-item shape; `no_compaction` only disables
summary replacement; `no_context_manager` is broader than compaction.

Next step: full paired analysis after analyzer output. If ambiguous, run 5 trials on
tasks with multi-request history, parallel tool calls, or compaction, comparing
`codex_full`, `no_history_replay`, `no_response_replay`, and a synthetic-history variant
that strips raw response IDs while preserving all tool calls and outputs.

## Terminus And Tmux Semantics

Status: inconclusive. Evidence shows real lossy terminal observation, but not that it is
the dominant cause.

Partial evidence: all observed Terminus turns had `return_code=null`; traces showed
continuation ambiguity in here-doc style commands. Counterexamples were ordinary
solution errors rather than terminal failures.

Next analysis: after analyzer output, build the discordant set where Terminus fails and
at least two structured controls succeed, plus the reverse. Label each failure as
`terminal-continuation`, `missing-exit/status`, `premature-task-complete`,
`raw-output-noise`, `ordinary-solution-error`, or `other`.

Next experiment if still ambiguous: add one Terminus variant with the same prompt/parser
but a sentinelized persistent shell that emits a unique end marker plus `$?`, waits for
that marker, and exposes exit code, running/completion state, and clean stdout. Run only
on the final discordant subset.

## Tool Surface And Action Categories

Status: no final causal conclusion before analyzer output.

Next analysis: after `473415`, use `analysis/aggregate_by_variant.csv`,
`analysis/paired_contrasts_vs_bare_v2_r3.csv`, and `analysis/mechanism_summary.csv`, then
run an action-category pass over completed traces using
`scripts/analyze_action_categories.py`, focusing on paired win/loss cells for
`codex_full`, `codex_full_exec_only_tools`, `codex_full_no_patch_tool`,
`codex_full_no_write_stdin`, `codex_full_minimal_surfaces`, and
`codex_full_minimal_loop`.

Next experiment if ambiguous: rerun only discordant task cells for `codex_full`,
`codex_full_exec_only_tools`, `codex_full_no_patch_tool`, and `bare_v2_r3` with 2-3
extra paired trials.

## Current Synthesis

The current evidence does not support a single-factor explanation. The safest working
account is a bundle:

- prompt/context prior improves task framing and action choice;
- tool surface, editing affordances, and observation feedback support diagnose/edit/test
  loops;
- history/context machinery may preserve multi-step state;
- persistent sessions and tmux semantics matter for specific stateful/interactive traces;
- format retries/recovery are not looking like the main driver.

Important invalid or contaminated treatments:
- `codex_full_no_patch_tool` is not a true no-patch-affordance ablation.
- `codex_full_no_unified_output` is not a true no-exit/no-status/raw-output ablation.

