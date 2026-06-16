from __future__ import annotations

import json
from pathlib import Path

from scripts import run_provider_matrix
from scripts.run_tb2_core import SEED_CANDIDATES


def test_provider_matrix_default_attempt_count() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(
        tuple(candidate.name for candidate in candidates)
    )
    attempts = run_provider_matrix._attempts(
        configs,
        candidates,
        ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
        trials=10,
    )

    assert len(attempts) == 28260
    assert sum(1 for item in attempts if item.provider == "openai") == 8370
    assert sum(1 for item in attempts if item.provider == "anthropic") == 10710
    assert sum(1 for item in attempts if item.provider == "deepseek") == 9180


def test_provider_matrix_openai_backfills_only_mini_candidates() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(
        tuple(candidate.name for candidate in candidates)
    )
    medium = [
        config
        for config in configs
        if config.provider == "openai"
        and config.model == "gpt-5.4-mini"
        and config.effort == "medium"
    ][0]
    high = [
        config
        for config in configs
        if config.provider == "openai"
        and config.model == "gpt-5.4-mini"
        and config.effort == "high"
    ][0]

    assert set(medium.candidates) == {
        "seed_mini_swe_agent_barebones",
        "seed_mini_swe_agent_barebones_v2",
        "seed_mini_swe_agent_barebones_v2_codex_prompt",
        "seed_mini_swe_agent_barebones_v2_persistent",
        "seed_mini_swe_agent_barebones_v2_rich_terminal",
        "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples",
        "seed_mini_swe_agent_v2",
    }
    assert len(high.candidates) == len(candidates)


def test_provider_matrix_agent_env_maps_deepseek_reasoning() -> None:
    candidate = SEED_CANDIDATES[0]
    attempt = run_provider_matrix.MatrixAttempt(
        provider="deepseek",
        config_id="deepseek_flash_max",
        model="deepseek-v4-flash",
        effort="max",
        candidate=candidate,
        task="bn-fit-modify",
        attempt=1,
    )

    assert run_provider_matrix._agent_env(attempt) == (
        "TERMINAL_MODEL_PROVIDER=deepseek",
        "OPENAI_TERMINAL_MODEL=deepseek-v4-flash",
        "OPENAI_TERMINAL_REASONING_EFFORT=max",
    )


def test_provider_matrix_haiku_omits_unsupported_effort_sweep() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(
        tuple(candidate.name for candidate in candidates)
    )
    haiku_configs = [
        config for config in configs if config.provider == "anthropic" and "haiku" in config.model
    ]

    assert [config.effort for config in haiku_configs] == ["none"]


def test_provider_matrix_shards_are_disjoint_and_complete() -> None:
    candidates = list(SEED_CANDIDATES[:2])
    configs = run_provider_matrix._provider_configs(
        tuple(candidate.name for candidate in candidates)
    )[:3]
    attempts = run_provider_matrix._attempts(configs, candidates, ["t1", "t2"], trials=2)

    shards = [run_provider_matrix._shard_attempts(attempts, 4, index) for index in range(4)]
    flattened = [attempt for shard in shards for attempt in shard]

    assert len(flattened) == len(attempts)
    assert set(flattened) == set(attempts)
    assert sum(len(shard) for shard in shards) == len(attempts)


def test_provider_matrix_shard_start_is_provider_interleaved() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(
        tuple(candidate.name for candidate in candidates)
    )
    attempts = run_provider_matrix._attempts(
        configs,
        candidates,
        ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
        trials=10,
    )

    first_local_batch = run_provider_matrix._shard_attempts(attempts, 150, 0)[:16]

    assert {attempt.provider for attempt in first_local_batch} == {
        "anthropic",
        "deepseek",
        "openai",
    }


def test_provider_matrix_api_usage_ignores_missing_usage(tmp_path: Path) -> None:
    trace_dir = tmp_path / "attempt" / "agent"
    trace_dir.mkdir(parents=True)
    (trace_dir / "model-call-01.json").write_text(
        json.dumps(
            {
                "request_metadata": {
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 3,
                        "cached_tokens": 5,
                        "total_tokens": 14,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "model-call-02.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "do not estimate"}]}),
        encoding="utf-8",
    )

    assert run_provider_matrix._api_usage(tmp_path / "attempt") == {
        "input_tokens": 11,
        "output_tokens": 3,
        "cached_tokens": 5,
        "total_tokens": 14,
    }


def test_provider_matrix_failure_class_ignores_benign_slurm_startup(tmp_path: Path) -> None:
    (tmp_path / "stderr.log").write_text(
        "Starting Slurm/Pyxis environment\n",
        encoding="utf-8",
    )
    (tmp_path / "stdout.log").write_text(
        "terminal-bench completed; mean reward 0.0; exceptions 0\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "failure", "seed_minimal_agent")
        == "task_failure"
    )


def test_provider_matrix_failure_class_ignores_task_file_exists_output(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "harness-result.json").write_text(
        json.dumps({"stdout": "File /app/stolen_A1.npy exists and is loadable."}),
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "failure", "seed_minimal_agent")
        == "task_failure"
    )


def test_provider_matrix_failure_class_detects_harbor_cache_race(tmp_path: Path) -> None:
    (tmp_path / "stderr.log").write_text(
        "FileExistsError: [Errno 17] File exists: "
        "'/home/trost/.cache/harbor/tasks/model-extraction-relu-logits'\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "crash", "seed_minimal_agent")
        == "harbor_cache"
    )


def test_provider_matrix_failure_class_detects_fatal_slurm_error(tmp_path: Path) -> None:
    (tmp_path / "stderr.log").write_text(
        "Starting Slurm/Pyxis environment\n"
        "srun: error: nid00123: task 0: Exited with exit code 1\n",
        encoding="utf-8",
    )

    assert run_provider_matrix._failure_class(tmp_path, "crash", "seed_minimal_agent") == "infra"


def test_provider_matrix_failure_class_ignores_completed_cleanup_srun(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-06-08__16-46-26"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "job.log").write_text(
        "Starting Slurm/Pyxis environment\n"
        "__HARBOR_PYXIS_READY__host:12345\n"
        "srun: sending Ctrl-C to StepId=313782.2\n"
        "srun: forcing job termination\n"
        "srun: Job step aborted: Waiting up to 602 seconds for job step to finish.\n"
        "slurmstepd: error: *** STEP 313782.2 ON host CANCELLED AT 2026-06-08T17:24:12 ***\n"
        "slurmstepd: error: pyxis: child 565266 terminated with signal 9\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "failure", "seed_minimal_agent")
        == "task_failure"
    )


def test_provider_matrix_failure_class_prefers_final_verifier_reward_over_stale_exception(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "2026-06-08__19-10-06"
    task_dir = run_dir / "polyglot-c-py__abc"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "polyglot-c-py",
                "trial_name": "polyglot-c-py__abc",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "exception.txt").write_text(
        "harbor.trial.trial.VerifierTimeoutError: "
        "Verifier execution timed out after 900.0 seconds\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "failure", "seed_minimal_agent")
        == "task_failure"
    )


def test_provider_matrix_failure_class_prefers_context_over_transport_for_mini_swe(
    tmp_path: Path,
) -> None:
    (tmp_path / "stderr.log").write_text(
        "API call failed: context_length_exceeded: maximum context length exceeded\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "crash", "seed_mini_swe_agent_v2")
        == "context_overflow_no_compaction"
    )


def test_provider_matrix_failure_class_detects_verifier_timeout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    task_dir = run_dir / "task__abc"
    task_dir.mkdir(parents=True)
    (task_dir / "exception.txt").write_text(
        "harbor.trial.trial.VerifierTimeoutError: "
        "Verifier execution timed out after 900.0 seconds\n",
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "crash", "seed_minimal_agent")
        == "verifier_timeout"
    )
    assert "verifier_timeout" not in run_provider_matrix.CORRUPTED_CLASSES


def test_provider_matrix_failure_class_uses_latest_run_dir(tmp_path: Path) -> None:
    old_task_dir = tmp_path / "2026-06-07__01-00-00" / "task__old"
    old_task_dir.mkdir(parents=True)
    (old_task_dir / "trial.log").write_text(
        "srun: error: Unable to create step for job 1: Protocol authentication error\n",
        encoding="utf-8",
    )
    new_task_dir = tmp_path / "2026-06-07__02-00-00" / "task__new"
    new_task_dir.mkdir(parents=True)
    (new_task_dir / "harness-result.json").write_text(
        json.dumps({"status": "failure", "reward": 0}),
        encoding="utf-8",
    )

    assert (
        run_provider_matrix._failure_class(tmp_path, "failure", "seed_minimal_agent")
        == "task_failure"
    )
