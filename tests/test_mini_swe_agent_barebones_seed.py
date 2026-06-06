from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import ModelToolCall, ToolModelResult
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "mini_swe_agent_barebones" / "harness.py"


def test_mini_swe_agent_barebones_seed_validates() -> None:
    result = validate_candidate(SEED_PATH, max_lines=170, min_lines=120)

    assert result["ok"], result


def test_mini_swe_agent_barebones_prompt_is_minimal() -> None:
    module = _load_seed()
    messages = module._messages(TaskContext("Fix the bug."), [])
    prompt = messages[1]["content"]

    assert (
        messages[0]["content"] == "You are a helpful assistant that can interact with a computer."
    )
    assert "Task:\nFix the bug." in prompt
    assert "Every response must include at least one bash tool call." in prompt
    assert 'Arguments: {"command": "your_command_here"}' in prompt
    assert module.SUBMIT_COMMAND in prompt
    assert "non-persistent subshell" in prompt
    assert "Recommended Workflow" not in prompt
    assert "Example of a CORRECT response" not in prompt
    assert "sed -i" not in prompt
    assert "Reasoning text" not in prompt


def test_mini_swe_agent_barebones_returns_bash_tool_call(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel([_result("Running pwd.", _call("pwd", "call_1"))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_agent().next_command(TaskContext("List files."), [])

    assert fake.calls[0]["tools"] == [module.BASH_TOOL]
    assert fake.calls[0]["kwargs"] == {}
    assert turn.tool_calls[0].name == "bash"
    assert turn.tool_calls[0].arguments == {
        "command": "pwd",
        "timeout_sec": module.DEFAULT_COMMAND_TIMEOUT_SEC,
    }
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.metadata == {"sequential_tool_calls": True}


def test_mini_swe_agent_barebones_formats_history_as_plain_text(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel([_result("Next.", _call("echo ok", "call_2"))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)
    history = [
        CommandResult(
            command="pwd",
            return_code=0,
            stdout="/repo\n",
            tool_name="bash",
            tool_call_id="call_1",
        )
    ]

    turn = module.create_agent().next_command(TaskContext("List files."), history)

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "$ pwd" in prompt
    assert "returncode: 0" in prompt
    assert "output:\n/repo\n" in prompt
    assert turn.tool_calls[0].arguments["command"] == "echo ok"


def test_mini_swe_agent_barebones_stops_after_submit_marker(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel([])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_agent().next_command(
        TaskContext("Finish."),
        [
            CommandResult(
                command=module.SUBMIT_COMMAND,
                return_code=0,
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfinal answer\n",
                tool_name="bash",
                tool_call_id="call_done",
            )
        ],
    )

    assert turn.done is True
    assert turn.assistant_content == "final answer\n"
    assert fake.calls == []


def test_mini_swe_agent_barebones_stops_when_model_returns_no_bash(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel([ToolModelResult(content="done", tool_calls=[])])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_agent().next_command(TaskContext("List files."), [])

    assert turn.done is True
    assert turn.assistant_content == "done"


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


def _load_seed():
    spec = importlib.util.spec_from_file_location("mini_swe_agent_barebones_harness", SEED_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
