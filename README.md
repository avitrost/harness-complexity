# Harness Complexity

Minimal scaffold for a controlled TerminalBench 2.0 experiment measuring how the
maximum formatted physical line count of `candidate/harness.py` affects performance.

## Codebase Map

This repository separates the counted experimental artifact from the uncounted
mechanics needed to run it.

- `candidate/harness.py` is the only counted file. It contains the starter harness
  and is the file future optimization cycles edit.
- `plumbing/` is uncounted support code: OpenAI API calls, the Harbor custom-agent
  adapter, candidate loading, small shared types, logging helpers, and secrets loading.
- `evaluator/` is uncounted orchestration: split definitions, validation, Harbor
  run wrappers, result parsing, aggregation, budget optimization, and full experiment
  sequencing.
- `scripts/` contains uncounted command-line helpers for line counting, static audit,
  workspace creation, candidate selection, bootstrap confidence intervals, and plots.
- `seeds/seed_minimal.py` is the starter harness copied into new optimization
  workspaces.
- `experience/`, `final_test/`, and `results/` are artifact directories. Generated
  contents are ignored by git; only `.gitkeep` placeholders are tracked.
- `tests/` contains local unit tests. They use mocks/fakes and do not run OpenAI calls
  or TerminalBench jobs.

## Counted Harness

The barebones starter harness is intentionally small. It imports only allowed plumbing,
formats recent terminal observations, calls `plumbing.openai_client.call_terminal_model(...)`,
and returns the next command or `DONE`.

The terminal-solving model is not named in `candidate/harness.py`. It is frozen in
`plumbing/openai_client.py` as `gpt-5.4-nano`, outside the line-counted file. Future
optimization cycles may change harness behavior only inside `candidate/harness.py`;
prompt text in that file counts toward the budget.

Budget optimization invokes Codex in a temporary isolated workspace containing only
`candidate/harness.py`, `proposal.md`, and local workspace instructions. After Codex
returns, those files are copied into `experience/Bxxxx/iter_NNN/workspace` for
validation and record keeping.

## Plumbing Boundary

Uncounted plumbing must stay mechanical. It may load `OPENAI_API_KEY`, call the fixed
terminal model, adapt the harness to Harbor, validate line count and static constraints,
parse Harbor outputs, aggregate scores, and plot results. It must not contain
task-solving strategy that would change benchmark behavior.

The Harbor adapter exposes `plumbing.harbor_adapter:HarborHarnessAgent`. Harbor calls
that class, the adapter loads the candidate harness from the selected workspace, passes
terminal history to the counted harness, executes the returned command, and records
observations. The adapter does not choose the model and does not add task-specific
behavior.

## Install

```bash
python -m pip install -e .
```

Set the terminal model API key:

```bash
export OPENAI_API_KEY=...
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
```

Authenticate the Codex CLI separately using your local Codex setup.

Install Harbor as a CLI tool. On this machine it was installed with `uv tool install harbor
--python 3.12`, which places `harbor.exe` in `C:\Users\trost\.local\bin`. That directory
has been added to the user PATH. If a shell cannot find it yet, open a new shell or run:

```powershell
$env:PATH = "$HOME\.local\bin;$env:PATH"
```

## Common Commands

Validate the starter harness:

```bash
python -m evaluator.validate_candidate candidate/harness.py --max-lines 64
```

Dry-run one budget:

```bash
python -m evaluator.optimize_budget --budget 128 --cycles 10 --dry-run
```

Run one real budget:

```bash
python -m evaluator.optimize_budget --budget 128 --cycles 10 --codex-model gpt-5.5 --codex-reasoning-effort medium
```

On Windows the optimizer resolves `codex.cmd` explicitly. If Codex is installed in a
non-standard location, pass:

```powershell
python -m evaluator.optimize_budget --budget 128 --cycles 10 --codex-bin C:\path\to\codex.cmd
```

For ChatGPT-authenticated Codex CLI accounts, model names are slugs such as `gpt-5.5`;
reasoning effort is separate.

On Windows, Codex `workspace-write` sandboxing can fail with
`CreateProcessWithLogonW failed: 1056`. The optimizer therefore runs Codex in a
temporary isolated workspace with `--sandbox danger-full-access`, then copies back only
the candidate workspace artifacts.

Dry-run validation Harbor command construction:

```bash
python -m evaluator.run_val --candidate-dir . --budget 128 --out-dir experience/B0128/iter_001 --dry-run
```

The Harbor command shape is:

```text
harbor run --dataset terminal-bench@2.0 --include-task-name <task> --n-attempts <trials> --n-concurrent <concurrency> --jobs-dir <out-dir> --agent-import-path plumbing.harbor_adapter:HarborHarnessAgent --agent-kwarg candidate_dir=<candidate-dir>
```

Run final test for a selected candidate:

```bash
python -m evaluator.run_test --candidate-dir path/to/workspace --budget 128 --out-dir final_test/B0128
```

Regenerate plots:

```bash
python scripts/plot_complexity_curve.py
```

## Pre-Registered Design

- Meta-optimizer: Codex GPT-5.5 Medium.
- Terminal-solving model: GPT-5.4 Nano, fixed in uncounted plumbing.
- Counted file: `candidate/harness.py` only.
- Independent variable: Black-formatted physical lines, including comments and blank lines.
- Budgets: 64, 128, 256, 512.
- Optimization budget: 10 evaluated candidates per budget.
- One evaluated candidate per budget per cycle.
- Each budget has independent search history.
- No cross-budget sharing in the primary experiment.
- No test feedback may be used during optimization.
- Validation split: 5 listed tasks, N=4 trials per task, concurrency 8.
- Final test split: 15 listed tasks, N=5 trials per task, concurrency 8.

Validation monitoring score:

```text
estimated_full_score = 0.361193 * val_split_mean + 0.295842
```

Held-out test score:

```text
estimated_full_score = 0.510101 * test_split_mean + 0.108900
```

Do not expose final-test results to optimization cycles. Keep `final_test/` and `results/`
outside any candidate runtime path.

## Harbor Integration

This scaffold targets Harbor `0.6.6` and the remote dataset `terminal-bench@2.0`.
The run wrappers use `--include-task-name` filters for the registered val/test tasks,
`--n-attempts` for repeated trials, and `--n-concurrent` for concurrency. Use
`--dry-run` first to inspect the exact command without launching benchmarks.
