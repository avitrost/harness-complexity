from __future__ import annotations

import asyncio
import json
from pathlib import Path

from evaluator.run_terminus_2 import run_terminus_2_split
from plumbing.openai_client import CodexBackendError
from plumbing.openai_client import ToolModelResult
from plumbing.terminus_2_agent import (
    CodexAuthLLM,
    TERMINUS_2_AGENT_IMPORT_PATH,
    TERMINUS_2_CODEX_AUTH_AGENT_IMPORT_PATH,
)

TASKS = ["fix-git", "sqlite-db-truncate", "pytorch-model-cli"]


def test_run_terminus_2_split_builds_harbor_command(tmp_path: Path) -> None:
    summary = run_terminus_2_split(
        split="val",
        out_dir=tmp_path / "out",
        tasks=["fix-git"],
        trials=1,
        concurrency=1,
        backend="slurm-pyxis",
        terminus_model="gpt-test",
        parser_name="json",
        reasoning_effort="none",
        record_terminal_session=False,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    command = summary["command"]
    assert TERMINUS_2_AGENT_IMPORT_PATH in command
    assert command[command.index("--model") + 1] == "gpt-test"
    assert "model_name=gpt-test" not in command
    assert "parser_name=json" in command
    assert "reasoning_effort=none" in command
    assert "record_terminal_session=false" in command
    assert "--environment-import-path" in command
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["dry_run"] is True


def test_run_terminus_2_split_includes_multiple_tasks(tmp_path: Path) -> None:
    summary = run_terminus_2_split(
        split="val",
        out_dir=tmp_path / "out",
        tasks=TASKS,
        trials=2,
        concurrency=3,
        backend="slurm-pyxis",
        terminus_model="gpt-test",
        parser_name="json",
        reasoning_effort=None,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    command = summary["command"]
    assert command.count("--include-task-name") == len(TASKS)
    assert "reasoning_effort=none" not in command
    for task in TASKS:
        index = command.index(task)
        assert command[index - 1] == "--include-task-name"
    assert command[command.index("--n-attempts") + 1] == "2"
    assert command[command.index("--n-concurrent") + 1] == "3"


def test_run_terminus_2_split_accepts_local_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "OpenThoughts-TBLite"
    summary = run_terminus_2_split(
        split="tblite",
        out_dir=tmp_path / "out",
        tasks=["acl-permissions-inheritance"],
        trials=5,
        concurrency=7,
        backend="slurm-pyxis",
        terminus_model="gpt-test",
        parser_name="json",
        reasoning_effort="none",
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--path --include-task-name --n-attempts --n-concurrent",
        dataset="open-thoughts/OpenThoughts-TBLite",
        dataset_path=dataset_path,
    )

    command = summary["command"]
    assert "--path" in command
    assert str(dataset_path) in command
    assert "--dataset" not in command
    assert summary["dataset_path"] == str(dataset_path)


def test_run_terminus_2_split_uses_codex_auth_transport(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_AUTH_MODE", "codex")
    summary = run_terminus_2_split(
        split="val",
        out_dir=tmp_path / "out",
        tasks=["fix-git"],
        trials=1,
        concurrency=1,
        backend="slurm-pyxis",
        terminus_model="gpt-test",
        parser_name="json",
        reasoning_effort="none",
        record_terminal_session=False,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    command = summary["command"]
    assert TERMINUS_2_CODEX_AUTH_AGENT_IMPORT_PATH in command
    assert TERMINUS_2_AGENT_IMPORT_PATH not in command


def test_codex_auth_llm_maps_chat_history_and_usage(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_call(messages):
        seen["messages"] = messages
        return ToolModelResult(
            content='{"analysis":"ok","commands":[],"task_complete":true}',
            tool_calls=[],
            request_metadata={"usage": {"input_tokens": 11}},
            response_id="resp_123",
            usage={"input_tokens": 11, "output_tokens": 7, "cached_tokens": 3},
        )

    monkeypatch.setattr("plumbing.openai_client._call_codex_backend_result", fake_call)
    llm = CodexAuthLLM("gpt-test")
    logging_path = tmp_path / "debug.json"
    response = asyncio.run(
        llm.call(
            "next",
            message_history=[{"role": "user", "content": "first"}],
            logging_path=logging_path,
        )
    )

    assert seen["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "next"},
    ]
    assert response.content.startswith('{"analysis"')
    assert response.response_id == "resp_123"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.cache_tokens == 3
    assert json.loads(logging_path.read_text())["model"] == "gpt-test"


def test_codex_auth_llm_retries_rate_limits(monkeypatch) -> None:
    calls = 0

    def fake_call(_messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CodexBackendError(429, '{"detail":"Rate limit exceeded"}', {})
        return ToolModelResult("{}", [])

    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setattr("plumbing.openai_client._call_codex_backend_result", fake_call)
    monkeypatch.setattr("plumbing.terminus_2_agent.time.sleep", lambda _seconds: None)

    response = asyncio.run(CodexAuthLLM("gpt-test").call("next"))

    assert response.content == "{}"
    assert calls == 2
