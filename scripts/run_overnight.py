from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime

DEFAULT_BUDGETS = "128,256,512,1024,2048"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("overnight_%Y%m%d_%H%M%S"))
    parser.add_argument("--iterations", "--cycles", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--codex-model", default="gpt-5.5")
    parser.add_argument("--terminal-model", default="gpt-5.4-mini")
    parser.add_argument("--codex-reasoning-effort", default="medium")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = {**os.environ, "OPENAI_AUTH_MODE": os.environ.get("OPENAI_AUTH_MODE", "codex")}
    command = [
        sys.executable,
        "-m",
        "evaluator.run_experiment",
        "--budgets",
        args.budgets,
        "--run-id",
        args.run_id,
        "--cycles",
        str(args.iterations),
        "--k",
        str(args.k),
        "--backend",
        args.backend,
        "--codex-model",
        args.codex_model,
        "--terminal-model",
        args.terminal_model,
        "--codex-reasoning-effort",
        args.codex_reasoning_effort,
        *(("--dry-run",) if args.dry_run else ()),
    ]
    print(" ".join(command), flush=True)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
