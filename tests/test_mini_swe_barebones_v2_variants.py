from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import ModelToolCall, ToolModelResult
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
VARIANT_MODULE = "plumbing.mini_swe_barebones_v2_variants"
SEED_PATHS = {
    "persistent": ROOT / "seeds/mini_swe_agent_barebones_v2_persistent/harness.py",
    "persistent_prompt_only": (
        ROOT / "seeds/mini_swe_agent_barebones_v2_persistent_prompt_only/harness.py"
    ),
    "persistent_exec_only": (
        ROOT / "seeds/mini_swe_agent_barebones_v2_persistent_exec_only/harness.py"
    ),
    "rich": ROOT / "seeds/mini_swe_agent_barebones_v2_rich_terminal/harness.py",
    "rich_no_examples": (
        ROOT / "seeds/mini_swe_agent_barebones_v2_rich_terminal_no_examples/harness.py"
    ),
}


@pytest.mark.parametrize("path", SEED_PATHS.values(), ids=SEED_PATHS)
def test_barebones_v2_variant_seed_validates(path: Path) -> None:
    result = validate_candidate(path, max_lines=20, min_lines=5)

    assert result["ok"], result


def test_hidden_persistent_variant_exposes_only_bash(monkeypatch) -> None:
    module = _variant_module()
    fake = RecordingTerminalModel([_result("Inspecting.", _call("bash", {"command": "pwd"}))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_bash_persistent_agent().next_command(
        TaskContext(
            "List files.",
            metadata={"persistent_terminal": {"available": True, "session_id": 7}},
        ),
        [],
    )

    prompt = fake.calls[0]["messages"][1]["content"]
    assert fake.calls[0]["tools"] == [module.BASH_TOOL]
    assert "one persistent interactive shell" in prompt
    assert "Every action is executed in a new subshell" not in prompt
    assert "write_stdin" not in prompt
    assert turn.tool_calls[0].name == "persistent_bash"
    assert turn.tool_calls[0].arguments["session_id"] == 7
    assert turn.tool_calls[0].arguments["command"].endswith("\npwd")


def test_persistent_prompt_only_prompts_persistent_but_runs_nonpersistent(monkeypatch) -> None:
    module = _variant_module()
    fake = RecordingTerminalModel([_result("Inspecting.", _call("bash", {"command": "pwd"}))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_bash_persistent_prompt_only_agent().next_command(TaskContext("List."), [])

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "one persistent interactive shell" in prompt
    assert turn.tool_calls[0].name == "local_shell"
    assert "session_id" not in turn.tool_calls[0].arguments


def test_persistent_exec_only_prompts_nonpersistent_but_runs_persistent(monkeypatch) -> None:
    module = _variant_module()
    fake = RecordingTerminalModel([_result("Inspecting.", _call("bash", {"command": "pwd"}))])
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_bash_persistent_exec_only_agent().next_command(
        TaskContext(
            "List.",
            metadata={"persistent_terminal": {"available": True, "session_id": 7}},
        ),
        [],
    )

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Every action is executed in a new subshell" in prompt
    assert turn.tool_calls[0].name == "persistent_bash"
    assert turn.tool_calls[0].arguments["session_id"] == 7


def test_rich_terminal_variant_exposes_exec_and_write_stdin_with_examples(monkeypatch) -> None:
    module = _variant_module()
    fake = RecordingTerminalModel(
        [_result("Inspecting.", _call("exec_command", {"cmd": "pwd"}, "call_exec"))]
    )
    monkeypatch.setattr(module, "call_terminal_model_with_tools", fake)

    turn = module.create_rich_terminal_agent().next_command(
        TaskContext(
            "List files.",
            metadata={"persistent_terminal": {"available": True, "session_name": "tmux-test"}},
        ),
        [],
    )

    prompt = fake.calls[0]["messages"][1]["content"]
    tool_names = [tool["name"] for tool in fake.calls[0]["tools"]]
    assert tool_names == ["exec_command", "write_stdin"]
    assert 'tmux_session="tmux-test"' in prompt
    assert "Examples:" in prompt
    assert "Run or poll a long-running command" in prompt
    assert turn.tool_calls[0].name == "exec_command"
    assert turn.tool_calls[0].arguments["cmd"] == "pwd"
    assert turn.tool_calls[0].arguments["timeout_sec"] == module.DEFAULT_COMMAND_TIMEOUT_SEC


def test_rich_terminal_no_examples_differs_only_by_examples(monkeypatch) -> None:
    module = _variant_module()
    monkeypatch.setattr(
        module,
        "call_terminal_model_with_tools",
        RecordingTerminalModel(
            [
                _result("Inspecting.", _call("exec_command", {"cmd": "pwd"}, "call_1")),
                _result("Inspecting.", _call("exec_command", {"cmd": "pwd"}, "call_2")),
            ]
        ),
    )
    task = TaskContext(
        "List files.",
        metadata={"persistent_terminal": {"available": True, "session_name": "tmux-test"}},
    )

    module.create_rich_terminal_agent().next_command(task, [])
    module.create_rich_terminal_no_examples_agent().next_command(task, [])
    rich_prompt = module.call_terminal_model_with_tools.calls[0]["messages"][1]["content"]
    plain_prompt = module.call_terminal_model_with_tools.calls[1]["messages"][1]["content"]

    assert "Examples:" in rich_prompt
    assert "Examples:" not in plain_prompt
    rendered_examples = module.RICH_TERMINAL_EXAMPLES.replace("{{tmux_session}}", "tmux-test")
    assert rich_prompt.replace("\n\n" + rendered_examples, "") == plain_prompt


def test_rich_terminal_submit_exec_command_finishes() -> None:
    module = _variant_module()

    turn = module.create_rich_terminal_agent().next_command(
        TaskContext("Finish."),
        [
            CommandResult(
                command=module.SUBMIT_COMMAND,
                return_code=0,
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                tool_name="exec_command",
                tool_call_id="call_done",
            )
        ],
    )

    assert turn.done is True
    assert turn.assistant_content == ""


def test_persistent_submit_with_shell_prompt_finishes() -> None:
    module = _variant_module()

    turn = module.create_bash_persistent_agent().next_command(
        TaskContext(
            "Finish.",
            metadata={"persistent_terminal": {"available": True, "session_id": 7}},
        ),
        [
            CommandResult(
                command=(
                    "export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off "
                    "TQDM_DISABLE=1;\n"
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                ),
                return_code=0,
                stdout=(
                    "$ export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off "
                    "TQDM_DISABLE=1;\n"
                    "$ echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                    "$ __hc_persistent_bash_status=$?\n"
                ),
                tool_name="persistent_bash",
                tool_call_id="call_done",
            )
        ],
    )

    assert turn.done is True
    assert turn.assistant_content == ""


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


def _call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> ModelToolCall:
    return ModelToolCall(
        name=name,
        arguments=arguments,
        arguments_text=json.dumps(arguments),
        call_id=call_id,
    )


def _variant_module():
    return importlib.import_module(VARIANT_MODULE)
