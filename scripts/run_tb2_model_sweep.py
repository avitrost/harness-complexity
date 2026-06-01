from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_TB2_CORE = ROOT / "scripts" / "run_tb2_core.py"
SUPPORTED_CODEX_BACKEND_MODELS = ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=Path("final_test"))
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=45)
    parser.add_argument("--max-candidate-workers", type=int, default=2)
    parser.add_argument("--slurm-partition", default="m7i-cpu2")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_candidate_workers < 1:
        raise ValueError("--max-candidate-workers must be >= 1")

    models = tuple(args.models or SUPPORTED_CODEX_BACKEND_MODELS)
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args, models)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries = []
    for model in models:
        command = _tb2_core_command(args, root, model)
        summaries.append(_run_model(root, model, command, args))
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(root), "summaries": summaries}, indent=2))
    return 0 if all(item["returncode"] == 0 for item in summaries) else 1


def _manifest(args: argparse.Namespace, models: tuple[str, ...]) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "models": list(models),
        "reasoning_effort": args.reasoning_effort,
        "trials": args.trials,
        "concurrency_per_candidate": args.concurrency,
        "max_candidate_workers": args.max_candidate_workers,
        "effective_max_in_flight": args.concurrency * args.max_candidate_workers,
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
        "include_codex_cli": False,
        "include_terminus_2": False,
        "candidates": args.candidates,
    }


def _tb2_core_command(args: argparse.Namespace, root: Path, model: str) -> list[str]:
    command = [
        sys.executable,
        str(RUN_TB2_CORE),
        "--run-id",
        _model_run_id(model),
        "--out-root",
        str(root),
        "--backend",
        args.backend,
        "--trials",
        str(args.trials),
        "--concurrency",
        str(args.concurrency),
        "--max-candidate-workers",
        str(args.max_candidate_workers),
        "--codex-model",
        model,
        "--codex-reasoning-effort",
        args.reasoning_effort,
        "--terminus-model",
        model,
        "--terminus-reasoning-effort",
        args.reasoning_effort,
        "--no-include-codex-cli",
        "--no-include-terminus-2",
    ]
    if args.harbor_bin:
        command.extend(["--harbor-bin", args.harbor_bin])
    if args.dry_run:
        command.append("--dry-run")
    for candidate in args.candidates or ():
        command.extend(["--candidate", candidate])
    return command


def _run_model(
    root: Path,
    model: str,
    command: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("OPENAI_AUTH_MODE", "codex")
    if args.backend == "slurm-pyxis" and args.slurm_partition:
        env["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition
    print(f"[tb2-model-sweep] starting {model}", flush=True)
    result = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
    model_dir = root / _model_run_id(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "launcher-command.json").write_text(
        json.dumps({"command": command}, indent=2),
        encoding="utf-8",
    )
    (model_dir / "launcher-stdout.log").write_text(result.stdout, encoding="utf-8")
    (model_dir / "launcher-stderr.log").write_text(result.stderr, encoding="utf-8")
    print(f"[tb2-model-sweep] finished {model} rc={result.returncode}", flush=True)
    return {
        "model": model,
        "run_id": _model_run_id(model),
        "returncode": result.returncode,
        "command": command,
    }


def _model_run_id(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tb2_core_model_sweep_medium10_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
