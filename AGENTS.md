# AGENTS.MD

  ## Slurm Cluster Rules

  This repo is used on a shared Slurm cluster. Login nodes are only for light interactive work.

  Do not run heavy, long-lived, or highly concurrent work on login nodes. This includes:

  - `harbor` / `harbor run`
  - terminal-bench runs
  - model evals or sweep controllers
  - scripts that spawn many Slurm, Pyxis, Docker, container, or eval jobs
  - benchmarks, large test suites, data preprocessing, or file-watcher-heavy tools
  - VS Code/Codex tasks that launch many subprocesses

  All Harbor jobs and eval orchestration must run inside Slurm. This includes the top-level Python/controller process, not only the worker tasks it submits.

  ## Required Preflight

  Before running any expensive command, check that it is inside a Slurm allocation:

  ```bash
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Refusing to run heavy work outside Slurm. Use sbatch, salloc, or srun first." >&2
    exit 2
  fi

  If the command includes harbor, terminal-bench, run_tb2_model_sweep.py, vllm, pyxis, or high-concurrency evaluation code, this check is mandatory.

  ## Acceptable Login Node Work

  Login nodes may be used for:

  - editing files
  - git operations
  - small rg, ls, tail, head, and config inspection
  - checking squeue, sacct, scancel
  - submitting Slurm jobs
  - short commands that use little CPU and finish quickly

  ## Running Harbor Or Evals

  Use Slurm for the whole run. Prefer sbatch for non-interactive work:

  sbatch \
    --partition=m7i-cpu2 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=04:00:00 \

  For interactive debugging, first enter an allocation:

  srun --partition=m7i-cpu2 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

  Then run Harbor/eval commands only from inside that allocated shell.

  ## Concurrency

  Keep controller concurrency conservative unless the Slurm allocation is sized for it. Do not start dozens of local controller processes on a login node. If a script has --concurrency, --n-concurrent, or
  similar, verify that the controller itself is running under Slurm before launching it.

  ## If Cleanup Is Needed

  To inspect your processes on a login node:

  ps -u "$USER" -o pid,ppid,stat,pcpu,pmem,comm,args --sort=-pcpu | head -40

  To stop your Harbor processes:

  pkill -u "$USER" -TERM -x harbor
  sleep 5
  pkill -u "$USER" -KILL -x harbor

  If Harbor processes respawn, stop the parent sweep/controller process instead of repeatedly killing children.
