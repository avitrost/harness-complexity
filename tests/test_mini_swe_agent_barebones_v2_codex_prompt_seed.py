from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from evaluator.validate_candidate import validate_candidate
from plumbing import mini_swe_barebones_v2_codex_prompt as impl
from plumbing.openai_client import ModelToolCall, ToolModelResult
from plumbing.types import TaskContext

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "mini_swe_agent_barebones_v2_codex_prompt" / "harness.py"


def test_codex_prompt_seed_validates() -> None:
    result = validate_candidate(SEED_PATH, max_lines=10, min_lines=2)

    assert result["ok"], result


def test_codex_prompt_changes_prompt_only(monkeypatch) -> None:
    _load_seed()
    base = impl._base()
    fake = RecordingTerminalModel([_result("Inspecting.", _call("pwd", "call_1"))])
    monkeypatch.setattr(base, "call_terminal_model_with_tools", fake)

    agent = impl.create_agent()
    turn = agent.next_command(
        TaskContext(
            "List files.",
            working_dir="/repo",
            metadata={"agents_md": [{"path": "/repo/AGENTS.md", "content": "Use pytest."}]},
        ),
        [],
    )

    messages = fake.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "developer", "user"]
    assert fake.calls[0]["tools"] == [base.BASH_TOOL]
    assert fake.calls[0]["kwargs"] == {}
    assert "You are Codex" in messages[0]["content"]
    assert "You do not have `apply_patch`" in messages[0]["content"]
    assert "<cwd>/repo</cwd>" in messages[2]["content"]
    assert "<agents_md path='/repo/AGENTS.md'>" in messages[2]["content"]
    assert "fresh subshell" in messages[2]["content"]
    assert turn.tool_calls[0].name == "bash"
    assert turn.tool_calls[0].arguments == {
        "command": "export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1;\npwd",
        "timeout_sec": base.DEFAULT_COMMAND_TIMEOUT_SEC,
    }
    assert getattr(agent, "wants_environment_context") is True
    assert getattr(agent, "wants_agents_context") is True


class RecordingTerminalModel:
    def __init__(self, responses: list[ToolModelResult]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ToolModelResult:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("no fake response queued")
        return self.responses.pop(0)


def _result(content: str, call: ModelToolCall) -> ToolModelResult:
    return ToolModelResult(content=content, tool_calls=[call])


def _call(command: str, call_id: str = "") -> ModelToolCall:
    arguments = {"command": command}
    return ModelToolCall(
        name="bash",
        arguments=arguments,
        arguments_text=json.dumps(arguments),
        call_id=call_id,
    )


def _load_seed() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mini_swe_agent_barebones_v2_codex_prompt", SEED_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
