from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import ModelToolCall, ToolModelResult
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "mini_swe_agent_v2" / "harness.py"


def test_mini_swe_agent_v2_seed_validates() -> None:
    result = validate_candidate(SEED_PATH, max_lines=500, min_lines=430)

    assert result["ok"], result


def test_mini_swe_agent_v2_embeds_upstream_mini_yaml_prompts() -> None:
    module = _load_seed()

    assert module.UPSTREAM_COMMIT == "2afd0fb81bacbf0aacfac9ded6f093c5acd0bf7c"
    assert _sha256(module.MINI_SYSTEM_TEMPLATE) == (
        "6fb54145bbb1724ce77430ff3852887acbd4a5cce10c86cd8dfbf4c7d55f1091"
    )
    assert _sha256(module.MINI_INSTANCE_TEMPLATE) == (
        "546a89156d7823eb34eb49c5b31a3703df4d27639d034a6d13f0162488d70821"
    )
    assert _sha256(module.MINI_OBSERVATION_TEMPLATE) == (
        "bf5a29fe56e297588a40b3a8df70c8eb4d4664d205a930fa62e72560382b2916"
    )
    assert _sha256(module.MINI_FORMAT_ERROR_TEMPLATE) == (
        "7bab62857f7862c904a03716ec17fde8d7da7be5262cc65eff57878ce57ec4eb"
    )


def test_mini_swe_agent_v2_returns_bash_tool_call(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel([_result("I will inspect.", _call("pwd", "call_1"))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_agent().next_command(TaskContext("List files.", "/repo"), [])

    assert fake.calls[0]["kwargs"] == {}
    assert fake.calls[0]["tools"] == [module.BASH_TOOL]
    assert fake.calls[0]["messages"][0] == {
        "role": "system",
        "content": module._render(module.MINI_SYSTEM_TEMPLATE, {}),
    }
    assert "Please solve this issue: List files." in fake.calls[0]["messages"][1]["content"]
    assert "<system_information>" in fake.calls[0]["messages"][1]["content"]
    assert turn.assistant_content == "I will inspect."
    assert turn.tool_calls[0].name == "bash"
    assert turn.tool_calls[0].arguments["command"].endswith("\npwd")
    assert "PAGER=cat" in turn.tool_calls[0].arguments["command"]
    assert turn.tool_calls[0].arguments["timeout_sec"] == module.DEFAULT_COMMAND_TIMEOUT_SEC
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.metadata["sequential_tool_calls"] is True
    assert turn.metadata["mini_swe_agent_v2_response_items"][-1]["name"] == "bash"
    assert '"command": "pwd"' in turn.metadata["mini_swe_agent_v2_response_items"][-1]["arguments"]


def test_mini_swe_agent_v2_replays_tool_call_and_observation(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel(
        [
            _result("I will inspect.", _call("pwd", "call_1")),
            _result("Now list files.", _call("ls -la", "call_2")),
        ]
    )
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)
    agent = module.create_agent()
    first = agent.next_command(TaskContext("List files.", "/repo"), [])
    history = [
        CommandResult(
            command="pwd",
            return_code=0,
            stdout="/repo\n",
            tool_name="bash",
            tool_call_id=first.tool_calls[0].call_id,
            metadata=first.metadata,
        )
    ]

    second = agent.next_command(TaskContext("List files.", "/repo"), history)

    replay = fake.calls[1]["messages"]
    assert any(
        item.get("type") == "function_call" and item.get("call_id") == "call_1" for item in replay
    )
    observation = next(item for item in replay if item.get("type") == "function_call_output")
    assert observation["call_id"] == "call_1"
    assert '"returncode": 0' in observation["output"]
    assert '"output": "/repo\\n"' in observation["output"]
    assert second.tool_calls[0].arguments["command"].endswith("\nls -la")


def test_mini_swe_agent_v2_retries_when_response_has_no_bash_tool(monkeypatch) -> None:
    module = _load_seed()
    fake = RecordingTerminalModel(
        [
            ToolModelResult(
                content="I should inspect first.",
                tool_calls=[],
                response_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I should inspect first."}],
                    }
                ],
            ),
            _result("Running pwd.", _call("pwd", "call_1")),
        ]
    )
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_agent().next_command(TaskContext("List files.", "/repo"), [])

    assert len(fake.calls) == 2
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "Tool call error:" in retry_messages[-1]["content"]
    assert "No tool calls found in the response" in retry_messages[-1]["content"]
    assert turn.tool_calls[0].arguments["command"].endswith("\npwd")
    assert turn.metadata["mini_swe_agent_v2_format_retries"] == 1
    assert any(
        "Tool call error:" in str(item) for item in turn.metadata["mini_swe_agent_v2_messages"]
    )


def test_mini_swe_agent_v2_stops_after_submit_marker(monkeypatch) -> None:
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
    assert turn.metadata["exit_status"] == "Submitted"


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
    return ToolModelResult(
        content=content,
        tool_calls=[call],
        response_items=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            },
            {
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments_text,
                "call_id": call.call_id,
            },
        ],
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _call(command: str, call_id: str = "") -> ModelToolCall:
    arguments = {"command": command}
    return ModelToolCall(
        name="bash",
        arguments=arguments,
        arguments_text=json.dumps(arguments),
        call_id=call_id,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_seed():
    spec = importlib.util.spec_from_file_location("mini_swe_agent_v2_harness", SEED_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
