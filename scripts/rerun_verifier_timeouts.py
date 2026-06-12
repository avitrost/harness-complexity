from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.models.trial.result import TrialResult  # noqa: E402
from harbor.models.verifier.result import VerifierResult  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402
from scripts.run_provider_matrix import _refresh_classifier_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        print(
            "Refusing to run verifier reruns outside Slurm. Use sbatch, salloc, or srun first.",
            file=sys.stderr,
        )
        return 2

    targets = _verifier_timeout_rows(args.run_root)
    if args.limit is not None:
        targets = targets[: args.limit]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit_dir = args.run_root / "_reruns" / f"verifier_retry_{stamp}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_json(
            audit_dir / "summary.json",
            {"run_root": str(args.run_root), "targets": len(targets), "dry_run": True},
        )
        _write_json(audit_dir / "targets.json", targets)
        print(json.dumps({"targets": len(targets), "audit_dir": str(audit_dir)}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(_rerun_one, row, audit_dir) for row in targets]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[verifier-rerun] completed {index}/{len(futures)} "
                f"{result['task']} {result['attempt']} -> {result['outcome']} "
                f"reward={result.get('reward')}",
                flush=True,
            )

    _write_json(audit_dir / "results.json", results)
    summary = {
        "run_root": str(args.run_root),
        "targets": len(targets),
        "audit_dir": str(audit_dir),
        "outcome_counts": _counts(result["outcome"] for result in results),
        "reward_counts": _counts(str(result.get("reward")) for result in results),
    }
    _write_json(audit_dir / "summary.json", summary)

    refresh_args = argparse.Namespace(skip_aggregate_tables=True)
    aggregate = _refresh_classifier_outputs(args.run_root, refresh_args)
    summary["post_refresh"] = aggregate
    _write_json(audit_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _verifier_timeout_rows(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "attempts.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("failure_class") == "verifier_timeout"]


def _rerun_one(row: dict[str, Any], audit_dir: Path) -> dict[str, Any]:
    out_dir = Path(str(row["out_dir"]))
    run_dir = _latest_run_dir(out_dir)
    trial_dir = _trial_dir(run_dir)
    trial_id = _attempt_id(row)
    backup_dir = audit_dir / "backups" / trial_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    _backup_trial_files(trial_dir, backup_dir)

    result: dict[str, Any] = {
        "config_id": row.get("config_id"),
        "candidate": row.get("candidate"),
        "task": row.get("task"),
        "attempt": row.get("attempt"),
        "out_dir": str(out_dir),
        "run_dir": str(run_dir),
        "trial_dir": str(trial_dir),
        "backup_dir": str(backup_dir),
    }

    try:
        reward = asyncio.run(_rerun_trial_verifier(trial_dir))
        result.update({"outcome": "verified", "reward": reward})
    except Exception as exc:
        _mark_trial_failed(trial_dir, exc)
        result.update(
            {
                "outcome": "verification_failed_marked_failure",
                "reward": 0,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


async def _rerun_trial_verifier(trial_dir: Path) -> float | int:
    config = TrialConfig.model_validate_json((trial_dir / "config.json").read_text())
    trial = await Trial.create(config)
    trial._result = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    _clear_verifier_dir(trial_dir)

    try:
        await trial._setup_environment()
        await trial._environment.run_healthcheck()
        trial._environment.default_user = trial._task.config.verifier.user
        try:
            await trial._maybe_upload_agent_logs()
            await trial._run_verification()
        finally:
            trial._environment.default_user = None
    finally:
        await trial._cleanup_and_finalize()
        trial._close_logger_handler()

    if trial.result.verifier_result is None or not trial.result.verifier_result.rewards:
        raise RuntimeError("Verifier completed without rewards")
    trial.result.exception_info = None
    trial.result.finished_at = datetime.now(timezone.utc)
    (trial_dir / "result.json").write_text(trial.result.model_dump_json(indent=4), encoding="utf-8")
    (trial_dir / "exception.txt").unlink(missing_ok=True)
    (trial_dir / "verifier-rerun-failed.txt").unlink(missing_ok=True)
    return _reward_value(trial.result.verifier_result.rewards)


def _mark_trial_failed(trial_dir: Path, exc: BaseException) -> None:
    payload = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    payload.verifier_result = VerifierResult(rewards={"reward": 0})
    payload.exception_info = None
    payload.finished_at = datetime.now(timezone.utc)
    (trial_dir / "result.json").write_text(payload.model_dump_json(indent=4), encoding="utf-8")
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    (trial_dir / "verifier-rerun-failed.txt").write_text(
        "".join(traceback.format_exception(exc)),
        encoding="utf-8",
    )
    (trial_dir / "exception.txt").unlink(missing_ok=True)


def _clear_verifier_dir(trial_dir: Path) -> None:
    verifier_dir = trial_dir / "verifier"
    if verifier_dir.exists():
        shutil.rmtree(verifier_dir)
    verifier_dir.mkdir(parents=True, exist_ok=True)


def _backup_trial_files(trial_dir: Path, backup_dir: Path) -> None:
    for name in ("result.json", "exception.txt", "verifier-rerun-failed.txt"):
        source = trial_dir / name
        if source.exists():
            target = backup_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
    verifier = trial_dir / "verifier"
    if verifier.exists():
        shutil.copytree(verifier, backup_dir / "verifier", dirs_exist_ok=True)


def _latest_run_dir(out_dir: Path) -> Path:
    dirs = sorted(path for path in out_dir.iterdir() if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No Harbor run directory under {out_dir}")
    return dirs[-1]


def _trial_dir(run_dir: Path) -> Path:
    trial_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    if len(trial_dirs) != 1:
        raise RuntimeError(f"Expected one trial directory under {run_dir}, found {len(trial_dirs)}")
    return trial_dirs[0]


def _reward_value(rewards: dict[str, float | int]) -> float | int:
    for key in ("reward", "score", "success"):
        if key in rewards:
            return rewards[key]
    numeric = [value for value in rewards.values() if isinstance(value, int | float)]
    if not numeric:
        raise RuntimeError(f"Verifier rewards had no numeric value: {rewards}")
    return numeric[0]


def _attempt_id(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("config_id", "config")),
        str(row.get("candidate", "candidate")),
        str(row.get("task", "task")),
        f"attempt_{int(row.get('attempt') or 0):02d}",
    ]
    return "__".join(_safe(part) for part in parts)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
