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
- `tree/main/candidate` is the canonical seed copied into new optimization
  workspaces.
- `experience/`, `final_test/`, and `results/` are artifact directories. Generated
  contents are ignored by git; only `.gitkeep` placeholders are tracked.
- `tests/` contains local unit tests. They use mocks/fakes and do not run OpenAI calls
  or TerminalBench jobs.

## Counted Harness

The barebones starter harness is intentionally small. It imports only allowed plumbing,
formats recent terminal observations, calls `plumbing.openai_client.call_terminal_model(...)`,
and returns a JSON action: `run` with the next command or `done` when complete.

The terminal-solving model and reasoning effort are not named in
`candidate/harness.py`. They are frozen in `plumbing/openai_client.py`, outside the
line-counted file. Future
optimization cycles may change harness behavior only inside `candidate/harness.py`;
prompt text in that file counts toward the budget.

Budget optimization follows the Meta-Harness loop: the canonical seed is first
validated and evaluated as `iter_000_seed`, then each iteration asks Codex for `k`
new candidate harnesses. Codex runs in a temporary isolated workspace containing
`candidate/harness.py`, `proposal.md`, local workspace instructions, and a
`history/` snapshot of prior valid candidates from the same budget. After Codex
returns, the candidate is copied into
`experience/Bxxxx/run_YYYYMMDD_HHMMSS/iter_NNN_cand_KK/workspace` for validation,
evaluation, and record keeping; the `history/` snapshot is stripped before validation
so candidate runtime code cannot read prior traces.

The proposer workspace treats `history/` as a queryable filesystem, not a prompt
summary. It also exposes official-style `logs/` and `jobs/` aliases so the proposer
sees the same shape as the Meta-Harness TerminalBench reference. In addition to the
copied raw candidate directories, the optimizer writes:
`history/index.json` for candidate-level scores, `history/frontier.json` for the
best prior candidate overall and by task, `history/evolution_summary.jsonl` for an
append-only compact run history, `history/trace_index.json` for machine-readable
paths to raw trial logs, and `history/failures.md` as a short table of contents for
failed/crashed traces worth inspecting first. The raw `trial.log`,
`harness-turn-*.json`, `model-call-*.json`, `result.json`, and `exception.txt` files
remain the source of truth.

Proposer workspaces edit `agents/baseline_kira.py`, matching the official reference
parent name. That file starts as this repo's <128-line baseline harness. Before
validation, the optimizer copies it back to `candidate/harness.py`, which remains
the counted and evaluated artifact. Workspaces also include
`references/terminus_kira.py` plus `references/open_source_harnesses.md` and
small source snapshots for Codex, opencode, gemini-cli, and qwen-code. These
alias/reference directories are excluded from copied-back candidate workspaces
and cannot be imported at runtime.

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

The adapter imposes no independent harness turn cap; it loops until the candidate
returns `done`, returns an empty command, or Harbor stops the run.
It logs per-turn command observations plus `model-call-XX.json` prompt/response traces.

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

For local smoke runs, the plumbing can also use your existing Codex CLI ChatGPT
login instead of a Platform API key:

```powershell
$env:OPENAI_AUTH_MODE = "codex"
```

This reads `~/.codex/auth.json` locally and calls the Codex backend directly. It is
intended for local experimentation only; use the Platform API path for preregistered
benchmark runs.

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
python -m evaluator.validate_candidate candidate/harness.py --max-lines 128
```

Dry-run one budget:

```bash
python -m evaluator.optimize_budget --budget 128 --cycles 10 --k 2 --dry-run
```

Run one real budget:

```bash
python -m evaluator.optimize_budget --budget 128 --cycles 10 --k 2 --codex-model gpt-5.5 --codex-reasoning-effort medium
```

Resume an existing budget run and set Harbor concurrency explicitly:

```bash
python -m evaluator.optimize_budget \
  --budget 128 \
  --run-id overnight_YYYYMMDD \
  --cycles 10 \
  --resume \
  --concurrency 160 \
  --backend slurm-pyxis
```

During optimization, `--concurrency` is the per-iteration validation target. With
`k=2` and the default `--concurrency 160`, both candidates in an iteration validate
at the same time with Harbor concurrency 80 each, then the next optimizer iteration
starts after both finish.

Run the overnight budget set under one run id:

```bash
python scripts/run_overnight.py \
  --run-id overnight_YYYYMMDD \
  --iterations 10 \
  --codex-model gpt-5.5 \
  --terminal-model gpt-5.4-mini
```

On Windows the optimizer resolves `codex.cmd` explicitly. If Codex is installed in a
non-standard location, pass:

```powershell
python -m evaluator.optimize_budget --budget 128 --cycles 10 --k 2 --codex-bin C:\path\to\codex.cmd
```

For ChatGPT-authenticated Codex CLI accounts, model names are slugs such as `gpt-5.5`;
reasoning effort is separate.

## Slurm Container Runs

Local validation uses Harbor's Docker environment by default. CPU Slurm nodes do
not run Docker directly, so use the explicit Slurm/Pyxis backend there:

```bash
OPENAI_AUTH_MODE=codex python -m evaluator.run_val \
  --candidate-dir . \
  --budget 128 \
  --out-dir experience/B0128/iter_001 \
  --backend slurm-pyxis
```

The backend is a Harbor custom environment
(`plumbing.slurm_pyxis_environment:SlurmPyxisEnvironment`). It keeps Harbor's
task, agent, verifier, and result flow, but runs each task attempt as one
persistent `srun --container-image=<task>.sqsh` job. Cached Docker archives are
read from `/wbl-fast/usrs/ee/agent-collab/docker-image-cache`; converted images
and Slurm staging live under `/wbl-fast/usrs/trost`. Direct
`--container-image=<cached>.tar` fails with Enroot `Invalid image format`.

The Slurm/Pyxis wrapper stages a small stdlib HTTP exec server instead of
Harbor's FastAPI/uvicorn bootstrap, so compute nodes do not need pip, network
package downloads, or `asciinema` setup just to start the control plane. It also
uses a private Enroot config without the host `/etc/localtime` bind mount so
`tzdata` package setup does not poison the container dpkg state. These changes
are for Harbor's control server only; the TerminalBench task image, working
directory, task files, verifier, and result flow stay under Harbor.

The optimizer runs Codex inside a Bubblewrap filesystem namespace, rooted at a
temporary workspace, and also passes Codex
`--sandbox workspace-write --cd <workspace>`. The namespace mounts system tooling,
DNS, the temp workspace, and only the Codex auth file needed for login; repo parent
directories and prior experiment runs are not mounted. The optimizer then copies
back only the explicit outputs: `candidate/harness.py`, `proposal.md`, and the
workspace instructions for recordkeeping. Symlinked output paths are rejected before
reading or copying. The only prior-run material exposed to Codex is the run-local
`history/` snapshot created by the optimizer; test results, stale runs, and other
budget histories are not included.

Dry-run validation Harbor command construction:

```bash
python -m evaluator.run_val --candidate-dir . --budget 128 --out-dir experience/B0128/iter_001 --dry-run
```

The Harbor command shape is:

```text
harbor run --dataset terminal-bench@2.0 --include-task-name <task> --n-attempts <trials> --n-concurrent <concurrency> --jobs-dir <out-dir> --agent-import-path plumbing.harbor_adapter:HarborHarnessAgent --agent-kwarg candidate_dir=<candidate-dir>
```

Run final test for a selected candidate after a held-out task list has been configured:

```bash
python -m evaluator.run_test --candidate-dir path/to/workspace --budget 128 --out-dir final_test/B0128
```

The default experiment command does not run a final test. Pass `--run-final-test`
only when you intentionally want that extra manual evaluation.

Regenerate plots:

```bash
python scripts/plot_complexity_curve.py
```

## Pre-Registered Design

- Meta-optimizer: Codex GPT-5.5 Medium.
- Terminal-solving model: GPT-5.4 Mini with medium reasoning, fixed in uncounted plumbing.
- Counted file: `candidate/harness.py` only.
- Independent variable: Black-formatted physical lines, including comments and blank lines.
- LOC buckets: 1-128, 129-256, 257-512, 513-1024, 1025-2048.
- Optimization iterations: 10 per budget by default.
- Default proposal batch size: `k=2` candidates per iteration, matching the explicit
  candidate count reported for Meta-Harness search runs in the paper.
- The canonical seed is evaluated once as the initial population before proposals; for
  higher buckets, the seed workspace is padded with comments to satisfy the bucket floor.
- Each budget has independent search history.
- No cross-budget sharing in the primary experiment.
- Optimization split: the prior validation and test task lists are combined into
  one 20-task set, N=4 trials per task.
- There is no automatic held-out test set in this phase; final evaluation is run
  manually later with a separate task set.

Validation monitoring score:

```text
optimization_score = split_mean
```

Keep `final_test/` and `results/` outside any candidate runtime path. Candidate selection writes both
`results/selected_candidates.*` for the single representative per budget and
`results/pareto_frontier.*` for the non-dominated validation frontier.

## Harbor Integration

This scaffold targets Harbor `0.6.6` and the remote dataset `terminal-bench@2.0`.
The run wrappers use `--include-task-name` filters for the registered val/test tasks,
`--n-attempts` for repeated trials, and `--n-concurrent` for concurrency. Use
`--dry-run` first to inspect the exact command without launching benchmarks.
