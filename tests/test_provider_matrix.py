from __future__ import annotations

import json
from pathlib import Path

from scripts import run_provider_matrix
from scripts.run_tb2_core import SEED_CANDIDATES


def test_provider_matrix_default_attempt_count() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(tuple(candidate.name for candidate in candidates))
    attempts = run_provider_matrix._attempts(
        configs,
        candidates,
        ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
        trials=10,
    )

    assert len(attempts) == 15480
    assert sum(1 for item in attempts if item.provider == "openai") == 3780
    assert sum(1 for item in attempts if item.provider == "anthropic") == 6300
    assert sum(1 for item in attempts if item.provider == "deepseek") == 5400


def test_provider_matrix_openai_backfills_only_mini_candidates() -> None:
    candidates = list(SEED_CANDIDATES)
    configs = run_provider_matrix._provider_configs(tuple(candidate.name for candidate in candidates))
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
    configs = run_provider_matrix._provider_configs(tuple(candidate.name for candidate in candidates))
    haiku_configs = [
        config for config in configs if config.provider == "anthropic" and "haiku" in config.model
    ]

    assert [config.effort for config in haiku_configs] == ["none"]


def test_provider_matrix_shards_are_disjoint_and_complete() -> None:
    candidates = list(SEED_CANDIDATES[:2])
    configs = run_provider_matrix._provider_configs(tuple(candidate.name for candidate in candidates))[:3]
    attempts = run_provider_matrix._attempts(configs, candidates, ["t1", "t2"], trials=2)

    shards = [run_provider_matrix._shard_attempts(attempts, 4, index) for index in range(4)]
    flattened = [attempt for shard in shards for attempt in shard]

    assert len(flattened) == len(attempts)
    assert set(flattened) == set(attempts)
    assert sum(len(shard) for shard in shards) == len(attempts)


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
