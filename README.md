# Harness Complexity

Utilities and harness seeds for regular TerminalBench-style evaluation on Harbor,
with Slurm/Pyxis support for running task attempts on the CPU cluster.

The meta-harness optimization loop has been moved to the
`legacy/meta-optimization` branch. `main` is now for evaluating fixed harnesses and
official baselines, then aggregating score, runtime, crash, and token metrics.

## Codebase Map

- `seeds/` contains the evaluated harnesses:
  - `codex_full`: readable Codex-port baseline.
  - `codex_compressed`: standalone compressed Codex port.
  - `codex_1300`, `codex_1000`, `codex_700`, `codex_400`: line-budget Codex
    compression seeds.
  - `minimal_agent`: minimal one-tool shell baseline.
  - `terminus_2_compressed`: standalone compressed Terminus2 port.
- `plumbing/` contains mechanical adapter and transport code: OpenAI/Codex auth,
  Harbor custom-agent glue, Slurm/Pyxis execution, tmux terminal support, and
  result logging.
- `evaluator/` contains split definitions, Harbor command construction, parsing,
  aggregation, and official Codex/Terminus2 baseline wrappers.
- `scripts/` contains runnable evaluation helpers. The primary helper is
  `scripts/run_tb2_core.py`.
- `final_test/`, `results/`, `experience/`, and `external_datasets/` are generated
  artifact/cache locations and are ignored except for placeholders.
- `tests/` contains local unit tests with fakes/mocks. They do not launch real
  benchmark jobs.

## Primary Eval: TB2 Core

The default evaluation target is the fixed 9-task TB2 core proxy set, 10 trials per
task:

1. `bn-fit-modify`
2. `circuit-fibsqrt`
3. `polyglot-c-py`
4. `sparql-university`
5. `mteb-retrieve`
6. `cobol-modernization`
7. `password-recovery`
8. `model-extraction-relu-logits`
9. `large-scale-text-editing`

Run one harness on Slurm/Pyxis:

```bash
OPENAI_AUTH_MODE=codex python scripts/run_tb2_core.py \
  --candidate seed_codex_400 \
  --concurrency 45 \
  --backend slurm-pyxis
```

Run official Codex CLI on the same subset:

```bash
OPENAI_AUTH_MODE=codex python scripts/run_tb2_core.py \
  --candidate codex_cli \
  --codex-model gpt-5.4-mini \
  --codex-reasoning-effort medium \
  --concurrency 45 \
  --backend slurm-pyxis
```

Run several candidates in parallel by raising `--max-candidate-workers`. The
`--concurrency` value is per candidate, so the total Slurm fanout is approximately
`concurrency * max_candidate_workers`.

```bash
OPENAI_AUTH_MODE=codex python scripts/run_tb2_core.py \
  --candidate seed_codex_compressed \
  --candidate seed_terminus_2_compressed \
  --candidate codex_cli \
  --concurrency 45 \
  --max-candidate-workers 3 \
  --backend slurm-pyxis
```

Run the default Codex-backend model sweep over seed harnesses only. This excludes
`gpt-5.4-nano` because the Codex/ChatGPT backend rejects it for this account.
The sweep uses a global attempt pool, so `--concurrency` is the approximate total
Slurm fanout across models, harnesses, tasks, and trials.

```bash
OPENAI_AUTH_MODE=codex python scripts/run_tb2_model_sweep.py \
  --backend slurm-pyxis
```

The run directory contains:

- `manifest.json`: tasks, trials, candidates, backend, and concurrency policy.
- `run_tb2_core.py` writes one subdirectory per candidate.
- `run_tb2_model_sweep.py` writes one subdirectory per model, then candidate,
  task, and attempt.
- candidate subdirectories contain `records.json`, `summary.json`, and
  `per_task.csv`; attempt subdirectories contain the raw Harbor command/output.

## Other Eval Modes

Direct custom-harness eval:

```bash
OPENAI_AUTH_MODE=codex python -m evaluator.run_val \
  --candidate-dir seeds/codex_compressed \
  --budget 1660 \
  --out-dir final_test/manual_codex_compressed \
  --backend slurm-pyxis
```

Official Codex CLI wrapper:

```bash
OPENAI_AUTH_MODE=codex python -m evaluator.run_codex_cli \
  --split tb2-core \
  --out-dir final_test/codex_cli_medium \
  --codex-model gpt-5.4-mini \
  --codex-reasoning-effort medium \
  --backend slurm-pyxis
```

Official Terminus2 wrapper:

```bash
OPENAI_AUTH_MODE=codex python -m evaluator.run_terminus_2 \
  --split tb2-core \
  --out-dir final_test/terminus_2 \
  --terminus-model gpt-5.4-mini \
  --backend slurm-pyxis
```

OpenThoughts-TBLite remains available as a secondary dataset:

```bash
OPENAI_AUTH_MODE=codex python scripts/run_tblite.py \
  --candidate seed_codex_compressed \
  --candidate codex_cli \
  --backend slurm-pyxis
```

## Metrics

`records.json` is the normalized trial table. `summary.json` and `per_task.csv`
aggregate:

- reward score and successes
- crashes
- mean runtime
- input, output, cached, and total tokens
- mean total tokens per trial
- cost when Harbor/provider data supplies it
- model-call count for custom harnesses when available

Token accounting is supported for both evaluation paths:

- Official Harbor agents such as Codex CLI expose token fields in each trial's
  `agent_result`.
- Custom harnesses run through `plumbing.harbor_adapter:HarborHarnessAgent`, which
  writes `agent/harness-result.json` with `model_accounting` derived from
  `model-call-*.json` traces.

If provider usage is unavailable for a custom trace, the adapter falls back to an
approximate token count from logged prompt/response text. Missing accounting stays
`null`; it is not silently counted as zero.

## Slurm/Pyxis Backend

Use `--backend slurm-pyxis` for cluster runs. The backend is
`plumbing.slurm_pyxis_environment:SlurmPyxisEnvironment`. It runs each task attempt
as a persistent `srun --container-image=<task>.sqsh` job, using cached Docker
archives from `/wbl-fast/usrs/ee/agent-collab/docker-image-cache` and converted
SQSH images under `/wbl-fast/usrs/trost`.

The Slurm wrapper stages a small stdlib HTTP exec server inside the task container.
This avoids needing package downloads or Harbor's normal FastAPI/uvicorn bootstrap
on compute nodes, while preserving Harbor's task, agent, verifier, and result flow.

Set `HARBOR_SLURM_PYXIS_PARTITION` to override the partition:

```bash
export HARBOR_SLURM_PYXIS_PARTITION=m7i-cpu2
```

## Validation

Run local validation before pushing changes:

```bash
python -m pytest -q
python -m ruff check .
```

Validate a single harness file:

```bash
python -m evaluator.validate_candidate seeds/codex_400/harness.py --max-lines 400 --min-lines 390
```

Dry-run the primary eval command without launching Harbor:

```bash
python scripts/run_tb2_core.py --candidate seed_codex_400 --dry-run
```
