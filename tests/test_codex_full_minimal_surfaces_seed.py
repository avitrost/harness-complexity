from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from plumbing.openai_client import set_client_factory
from plumbing.types import CommandResult, TaskContext
from scripts.audit_candidate import audit_candidate

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "codex_full_minimal_surfaces" / "harness.py"


def test_codex_full_minimal_surfaces_seed_validates() -> None:
    result = audit_candidate(SEED_PATH)

    assert result["ok"], result["errors"]


def test_codex_full_minimal_surfaces_reduces_prompt_surfaces(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    monkeypatch.setenv("CODEX_CURRENT_DATE", "2030-01-02")
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("Fix the task.", working_dir="/app/task"), []
        )
    finally:
        set_client_factory(None)

    messages = fake.calls[0]["input"]
    tools = fake.calls[0]["tools"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are a coding agent working in a terminal.")
    assert "Codex CLI is an open source project" not in messages[0]["content"]
    assert [message.get("role") for message in messages] == ["system", "user"]
    assert '<env cwd="/app/task" shell="bash" date="2030-01-02"' in messages[1]["content"]
    assert "<environment_context>" not in messages[1]["content"]
    assert [tool["name"] for tool in tools] == [
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
    ]
    exec_tool = tools[0]
    assert exec_tool["description"] == "Run shell command; return output or session_id."
    assert "sandbox_permissions" not in exec_tool["parameters"]["properties"]
    assert tools[3]["description"] == "Apply a Begin/End Patch diff."
    assert turn.done is True


def test_codex_full_minimal_surfaces_compacts_output_replay(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    set_client_factory(lambda: fake)
    try:
        module.create_agent().next_command(
            TaskContext("Continue.", working_dir="/app"),
            [
                CommandResult(
                    command="ls",
                    return_code=0,
                    stdout="file.txt\n",
                    stderr="",
                    tool_name="exec_command",
                    tool_call_id="call_1",
                    metadata={
                        "arguments": {"cmd": "ls"},
                        "unified_exec": {
                            "chunk_id": "abc123",
                            "wall_time_seconds": 0.25,
                            "exit_code": 0,
                            "session_id": 9,
                            "original_token_count": 3,
                        },
                    },
                )
            ],
        )
    finally:
        set_client_factory(None)

    messages = fake.calls[0]["input"]
    outputs = [item for item in messages if item.get("type") == "function_call_output"]
    assert outputs
    output = outputs[0]["output"]
    assert output.startswith("chunk=abc123 wall=0.2500s exit=0 session=9 original_tokens=3")
    assert "Output:" not in output
    assert "output:\nfile.txt" in output


class RecordingToolOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _load_seed() -> Any:
    spec = importlib.util.spec_from_file_location("codex_full_minimal_surfaces", SEED_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_full_minimal_surfaces"] = module
    spec.loader.exec_module(module)
    return module
