from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.run_val import run_split  # noqa: E402
from evaluator.tblite import (  # noqa: E402
    TBLITE_DATASET_ID,
    TBLITE_REVISION,
    TBLITE_SPLIT,
    discover_tblite_tasks,
    materialize_tblite,
    select_tblite_tasks,
)
from plumbing.harbor_adapter import detect_harbor_executable, harbor_help  # noqa: E402

DEFAULT_OUT_ROOT = Path("/wbl-fast/usrs/trost/harness-complexity/final_test")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_PARTITION = "m7i-cpu2"
DEFAULT_CONCURRENCY = 200
DEFAULT_TBLITE_MAX_RETRIES = 2
DEFAULT_VERIFIER_TIMEOUT_MULTIPLIER = 3.0
DEFAULT_RETRY_EXCLUDE = ("VerifierTimeoutError",)


@dataclass(frozen=True)
class MechanismVariant:
    name: str
    base_candidate: str
    candidate_dir: Path
    loc: int
    mechanism: str
    hypothesis: str
    env: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptSpec:
    variant: MechanismVariant
    task: str
    attempt: int


VARIANTS: tuple[MechanismVariant, ...] = (
    MechanismVariant(
        "minimal_agent",
        "seed_minimal_agent",
        Path("seeds/minimal_agent"),
        100,
        "negative_control",
        "Minimal single-shell-tool behavior should remain a low-control baseline.",
    ),
    MechanismVariant(
        "msa_fully_compressed",
        "seed_mini_swe_agent_barebones",
        Path("seeds/mini_swe_agent_barebones"),
        149,
        "minimal_loop",
        "Flattened history plus no format retries should underperform structured mini-SWE v2.",
    ),
    MechanismVariant(
        "bare_v2_r0",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        408,
        "format_retries",
        "Removing format retries from the same v2 loop should reduce successful recovery.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=0",),
    ),
    MechanismVariant(
        "bare_v2_r3",
        "seed_mini_swe_agent_barebones_v2",
        Path("seeds/mini_swe_agent_barebones_v2"),
        408,
        "control",
        "Default barebones v2 provides the matched control for retry, prompt, and terminal variants.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_context_only_r3",
        "seed_mini_swe_agent_barebones_v2_context_only",
        Path("seeds/mini_swe_agent_barebones_v2_context_only"),
        408,
        "context_control",
        "Mini-SWE prompt plus Codex-style cwd/AGENTS context controls for the Codex-prompt graft.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "mini_v2_r3",
        "seed_mini_swe_agent_v2",
        Path("seeds/mini_swe_agent_v2"),
        478,
        "prompt_prior",
        "The original mini-SWE prompt should improve behavior if workflow guidance matters.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_codex_prompt_r3",
        "seed_mini_swe_agent_barebones_v2_codex_prompt",
        Path("seeds/mini_swe_agent_barebones_v2_codex_prompt"),
        408,
        "prompt_prior",
        "A Codex-like prompt on mini-SWE mechanics tests prompt prior without Codex tools.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_persistent_bash_r3",
        "seed_mini_swe_agent_barebones_v2_persistent",
        Path("seeds/mini_swe_agent_barebones_v2_persistent"),
        408,
        "persistent_shell",
        "A hidden persistent bash session tests state retention without exposing rich tools.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_persistent_prompt_only_r3",
        "seed_mini_swe_agent_barebones_v2_persistent_prompt_only",
        Path("seeds/mini_swe_agent_barebones_v2_persistent_prompt_only"),
        408,
        "persistent_shell_instruction_control",
        "Prompt says persistent while execution is nonpersistent, isolating instruction effects.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_persistent_exec_only_r3",
        "seed_mini_swe_agent_barebones_v2_persistent_exec_only",
        Path("seeds/mini_swe_agent_barebones_v2_persistent_exec_only"),
        408,
        "persistent_shell_execution_control",
        "Prompt says nonpersistent while execution is persistent, isolating actual shell state.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_rich_terminal_no_examples_r3",
        "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples",
        Path("seeds/mini_swe_agent_barebones_v2_rich_terminal_no_examples"),
        408,
        "structured_tools",
        "Exposing exec/write_stdin/session primitives tests structured terminal affordances.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "bare_v2_rich_terminal_r3",
        "seed_mini_swe_agent_barebones_v2_rich_terminal",
        Path("seeds/mini_swe_agent_barebones_v2_rich_terminal"),
        408,
        "structured_tools_prompted",
        "Adding rich-terminal examples tests whether the weak model needs usage demonstrations.",
        ("MINI_SWE_FORMAT_RETRY_LIMIT=3",),
    ),
    MechanismVariant(
        "codex_400",
        "seed_codex_400",
        Path("seeds/codex_400"),
        398,
        "codex_compressed_control",
        "A compact Codex port tests whether Codex-native tool semantics work without full prompt mass.",
    ),
    MechanismVariant(
        "codex_full_minimal_prompt",
        "seed_codex_full_minimal_prompt",
        Path("seeds/codex_full_minimal_prompt"),
        2210,
        "prompt_prior",
        "Codex mechanics with a compressed prompt tests whether mechanics alone preserve performance.",
    ),
    MechanismVariant(
        "codex_full_minimal_surfaces",
        "seed_codex_full_minimal_surfaces",
        Path("seeds/codex_full_minimal_surfaces"),
        2210,
        "prompt_surface",
        "Codex mechanics with compressed prompt surfaces tests how much non-core surface text matters.",
    ),
    MechanismVariant(
        "codex_full",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "positive_control",
        "Full Codex port is the high-performing positive control for weak-model harness assistance.",
        ("CODEX_HARNESS_PROFILE=codex_full",),
    ),
    MechanismVariant(
        "codex_full_loo_apply_patch",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "tool_leave_one_out",
        "Remove explicit apply_patch tool, prompt guidance, and shell interception while preserving shell editing.",
        ("CODEX_HARNESS_PROFILE=loo_apply_patch",),
    ),
    MechanismVariant(
        "codex_full_loo_update_plan",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "tool_leave_one_out",
        "Remove explicit update_plan tool and planning prompt guidance while preserving ordinary text planning.",
        ("CODEX_HARNESS_PROFILE=loo_update_plan",),
    ),
    MechanismVariant(
        "codex_full_loo_write_stdin",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "tool_leave_one_out",
        "Remove explicit write_stdin tool and session continuation guidance while preserving bash fallbacks.",
        ("CODEX_HARNESS_PROFILE=loo_write_stdin",),
    ),
    MechanismVariant(
        "codex_full_renamed_shell",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "tool_name_prior",
        "Rename exec_command to shell while preserving the same shell execution semantics.",
        ("CODEX_HARNESS_PROFILE=renamed_shell",),
    ),
    MechanismVariant(
        "codex_full_no_recovery",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "recovery_policy",
        "Disabling Codex recovery policy tests empty/malformed response recovery as a causal factor.",
        ("CODEX_HARNESS_PROFILE=no_recovery",),
    ),
    MechanismVariant(
        "codex_full_no_patch_tool",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "editing_affordance",
        "Removing apply_patch tests whether structured edits drive the Codex advantage.",
        ("CODEX_HARNESS_PROFILE=no_patch_tool",),
    ),
    MechanismVariant(
        "codex_full_no_patch_affordance",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "editing_affordance",
        "Removing apply_patch tool, prompt guidance, and shell interception tests editing affordance cleanly.",
        ("CODEX_HARNESS_PROFILE=no_patch_affordance",),
    ),
    MechanismVariant(
        "codex_full_no_write_stdin",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "persistent_process_control",
        "Removing write_stdin tests whether polling/interactive session continuation matters.",
        ("CODEX_HARNESS_PROFILE=no_write_stdin_tool",),
    ),
    MechanismVariant(
        "codex_full_no_compaction",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "context_retention",
        "Disabling compaction tests whether summarization/context retention drives the lift.",
        ("CODEX_HARNESS_PROFILE=no_compaction",),
    ),
    MechanismVariant(
        "codex_full_no_context_manager",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "context_retention",
        "Disabling Codex context management tests whether pruning/budgeting machinery is causal.",
        ("CODEX_HARNESS_PROFILE=no_context_manager",),
    ),
    MechanismVariant(
        "codex_full_no_history_replay",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "history_replay",
        "Disabling Codex structured history replay tests whether replay shape is causal.",
        ("CODEX_HARNESS_PROFILE=no_history_replay",),
    ),
    MechanismVariant(
        "codex_full_no_response_replay",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "history_replay",
        "Disabling raw model response item replay tests whether exact assistant/tool replay matters.",
        ("CODEX_HARNESS_PROFILE=no_model_response_item_replay",),
    ),
    MechanismVariant(
        "codex_full_no_unified_output",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "error_feedback",
        "Disabling structured unified exec output tests whether exit/session metadata is causal.",
        ("CODEX_HARNESS_PROFILE=no_unified_exec_output",),
    ),
    MechanismVariant(
        "codex_full_raw_exec_output",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "error_feedback",
        "Rendering only raw stdout/stderr tests whether exit/status/timing observation metadata is causal.",
        ("CODEX_HARNESS_PROFILE=raw_exec_output",),
    ),
    MechanismVariant(
        "codex_full_exec_only_tools",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "tool_surface",
        "Keeping only exec_command tests the combined value of patch, plan, and write_stdin.",
        ("CODEX_HARNESS_PROFILE=exec_only_tools",),
    ),
    MechanismVariant(
        "codex_full_minimal_loop",
        "seed_codex_full",
        Path("seeds/codex_full"),
        2210,
        "codex_minimal_loop",
        "Disabling most Codex mechanics tests whether the advantage collapses toward minimal loops.",
        ("CODEX_HARNESS_PROFILE=minimal_loop",),
    ),
    MechanismVariant(
        "terminus_2_compressed",
        "seed_terminus_2_compressed",
        Path("seeds/terminus_2_compressed"),
        634,
        "terminus_completion",
        "Terminus compressed tests tmux/raw-terminal completion and polling semantics.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("external_datasets"))
    parser.add_argument("--revision", default=TBLITE_REVISION)
    parser.add_argument("--download-method", choices=("git", "snapshot"), default="git")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--variant", action="append", dest="variant_names")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--slurm-partition", default=DEFAULT_PARTITION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--harbor-bin")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_TBLITE_MAX_RETRIES)
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=DEFAULT_VERIFIER_TIMEOUT_MULTIPLIER,
    )
    parser.add_argument("--retry-include", action="append", default=[])
    parser.add_argument("--retry-exclude", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        print(json.dumps([_variant_manifest(item) for item in VARIANTS], indent=2))
        return 0
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if args.backend == "slurm-pyxis" and not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to run Harbor/evals outside Slurm. Submit with sbatch/salloc/srun."
        )

    dataset_path = args.dataset_path or materialize_tblite(
        cache_root=args.cache_root,
        revision=args.revision,
        local_files_only=args.local_files_only,
        download_method=args.download_method,
    )
    args.dataset_path = dataset_path
    tasks = select_tblite_tasks(dataset_path, args.tasks)
    variants = _select_variants(args.variant_names)
    all_specs = _attempt_specs(variants, tasks, args.trials)
    specs = _shard_specs(all_specs, args.shard_count, args.shard_index)
    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    retry_exclude = tuple(
        args.retry_exclude if args.retry_exclude is not None else DEFAULT_RETRY_EXCLUDE
    )
    manifest = {
        "run_id": args.run_id,
        "split": TBLITE_SPLIT,
        "dataset": TBLITE_DATASET_ID,
        "revision": args.revision,
        "dataset_path": str(dataset_path),
        "available_task_count": len(discover_tblite_tasks(dataset_path)),
        "tasks": tasks,
        "task_count": len(tasks),
        "trials": args.trials,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "backend": args.backend,
        "slurm_partition": args.slurm_partition if args.backend == "slurm-pyxis" else None,
        "concurrency": args.concurrency,
        "attempts": len(all_specs),
        "selected_attempts": len(specs),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "max_retries": args.max_retries,
        "verifier_timeout_multiplier": args.verifier_timeout_multiplier,
        "retry_include": list(args.retry_include),
        "retry_exclude": list(retry_exclude),
        "variants": [_variant_manifest(item) for item in variants],
    }
    if args.shard_index == 0 or not (root / "manifest.json").exists():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shard_dir = root / "_shards" / f"shard_{args.shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    os.environ.setdefault("OPENAI_AUTH_MODE", "codex")
    os.environ["TERMINAL_MODEL_PROVIDER"] = "openai"
    if args.backend == "slurm-pyxis":
        os.environ["HARBOR_SLURM_PYXIS_PARTITION"] = args.slurm_partition
    harbor_help_text = _harbor_help_text(args.harbor_bin)
    started = time.monotonic()
    print(
        f"[tblite-mechanism] starting {len(specs)} attempts "
        f"of {len(all_specs)} total "
        f"({len(variants)} variants x {len(tasks)} tasks x {args.trials} trials, "
        f"shard {args.shard_index}/{args.shard_count}) at concurrency {args.concurrency}",
        flush=True,
    )
    summaries = _run_attempt_pool(root, args, specs, harbor_help_text, retry_exclude)
    (shard_dir / "attempt_summaries.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    failed = sum(1 for item in summaries if int(item.get("returncode", 0) or 0) != 0)
    summary = {
        "run_id": args.run_id,
        "out_root": str(root),
        "attempts": len(all_specs),
        "selected_attempts": len(specs),
        "attempts_completed": len(summaries),
        "attempts_failed_controller": failed,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "dry_run": args.dry_run,
        "returncode": 0 if failed == 0 else 1,
    }
    (shard_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.shard_count == 1:
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["returncode"])


def _select_variants(names: list[str] | None) -> list[MechanismVariant]:
    if not names:
        return list(VARIANTS)
    by_name = {variant.name: variant for variant in VARIANTS}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown mechanism variant(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def _attempt_specs(
    variants: list[MechanismVariant],
    tasks: list[str],
    trials: int,
) -> list[AttemptSpec]:
    return [
        AttemptSpec(variant=variant, task=task, attempt=attempt)
        for task in tasks
        for variant in variants
        for attempt in range(1, trials + 1)
    ]


def _shard_specs(
    specs: list[AttemptSpec],
    shard_count: int,
    shard_index: int,
) -> list[AttemptSpec]:
    return [spec for index, spec in enumerate(specs) if index % shard_count == shard_index]


def _run_attempt_pool(
    root: Path,
    args: argparse.Namespace,
    specs: list[AttemptSpec],
    harbor_help_text: str | None,
    retry_exclude: tuple[str, ...],
) -> list[dict[str, Any]]:
    worker_count = min(args.concurrency, len(specs))
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_attempt, root, args, spec, harbor_help_text, retry_exclude)
            for spec in specs
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            summaries.append(future.result())
            if index == len(futures) or index % max(1, min(args.concurrency, 100)) == 0:
                elapsed = max(time.monotonic() - started, 1.0)
                print(
                    f"[tblite-mechanism] completed {index}/{len(specs)} "
                    f"({index / elapsed:.3f} attempts/sec)",
                    flush=True,
                )
    return summaries


def _run_attempt(
    root: Path,
    args: argparse.Namespace,
    spec: AttemptSpec,
    harbor_help_text: str | None,
    retry_exclude: tuple[str, ...],
) -> dict[str, Any]:
    out_dir = _attempt_dir(root, spec)
    try:
        summary = run_split(
            split=TBLITE_SPLIT,
            candidate_dir=ROOT / spec.variant.candidate_dir,
            budget=spec.variant.loc,
            out_dir=out_dir,
            tasks=[spec.task],
            trials=1,
            concurrency=1,
            dry_run=args.dry_run,
            harbor_bin=args.harbor_bin,
            harbor_help_text=harbor_help_text,
            backend=args.backend,
            dataset=TBLITE_DATASET_ID,
            dataset_path=args.dataset_path,
            max_retries=args.max_retries,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            retry_include=tuple(args.retry_include),
            retry_exclude=retry_exclude,
            agent_env=_agent_env(args, spec.variant),
        )
        return {
            **_attempt_identity(spec, out_dir),
            "returncode": int(summary.get("returncode", 0) or 0),
            "ran": bool(summary.get("ran", True)),
            "dry_run": bool(summary.get("dry_run", False)),
        }
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            **_attempt_identity(spec, out_dir),
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (out_dir / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure


def _agent_env(args: argparse.Namespace, variant: MechanismVariant) -> tuple[str, ...]:
    return (
        "TERMINAL_MODEL_PROVIDER=openai",
        f"OPENAI_TERMINAL_MODEL={args.model}",
        f"OPENAI_TERMINAL_REASONING_EFFORT={args.reasoning_effort}",
        "OPENAI_AUTH_MODE=codex",
        *variant.env,
    )


def _attempt_dir(root: Path, spec: AttemptSpec) -> Path:
    return root / spec.variant.name / spec.task / f"attempt_{spec.attempt:02d}"


def _attempt_identity(spec: AttemptSpec, out_dir: Path) -> dict[str, Any]:
    return {
        "variant": spec.variant.name,
        "base_candidate": spec.variant.base_candidate,
        "mechanism": spec.variant.mechanism,
        "task": spec.task,
        "attempt": spec.attempt,
        "out_dir": str(out_dir),
    }


def _variant_manifest(variant: MechanismVariant) -> dict[str, Any]:
    payload = asdict(variant)
    payload["candidate_dir"] = str(variant.candidate_dir)
    payload["env"] = list(variant.env)
    return payload


def _harbor_help_text(harbor_bin: str | None) -> str | None:
    executable = harbor_bin or detect_harbor_executable() or "harbor"
    return harbor_help(executable, "run")


def _default_run_id() -> str:
    return "tblite_mechanism_gpt54mini_low_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
