from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from plumbing.openai_client import set_client_factory
from plumbing.types import CommandResult, TaskContext


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "codex_full" / "harness.py"
TASK_CASES = [
    ("fix-git", "Recover the lost git changes and make the tests pass."),
    ("sqlite-db-truncate", "Repair the SQLite truncation logic and verify the database."),
    ("pytorch-model-cli", "Fix the PyTorch model CLI so inference works end to end."),
]


def test_codex_full_seed_embeds_pinned_prompt_and_grammar() -> None:
    module = _load_seed()

    assert module.CODEX_BASE_INSTRUCTIONS == (
        ROOT / "references" / "codex_port" / "base_instructions_default.md"
    ).read_text(encoding="utf-8").rstrip()
    assert module.APPLY_PATCH_GRAMMAR == (
        ROOT / "references" / "codex_port" / "apply_patch.lark"
    ).read_text(encoding="utf-8").rstrip()


def test_codex_full_seed_uses_codex_tool_specs() -> None:
    module = _load_seed()
    tools = module._built_tools()

    assert [tool["name"] for tool in tools] == ["exec_command", "apply_patch"]
    assert tools[0]["description"] == (
        "Runs a command in a PTY, returning output or a session ID for ongoing interaction."
    )
    assert tools[0]["parameters"]["required"] == ["cmd"]
    assert tools[0]["parameters"]["properties"]["cmd"]["description"] == "Shell command to execute."
    assert tools[0]["output_schema"]["required"] == ["wall_time_seconds", "output"]
    assert tools[1]["type"] == "custom"
    assert tools[1]["format"]["definition"] == module.APPLY_PATCH_GRAMMAR


def test_codex_full_seed_returns_model_tool_call(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI(
        [
            SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="exec_command",
                        arguments='{"cmd":"pwd","yield_time_ms":1000}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("List files."), [])
    finally:
        set_client_factory(None)

    assert fake.calls[0]["tool_choice"] == "auto"
    assert fake.calls[0]["parallel_tool_calls"] is True
    assert [tool["name"] for tool in fake.calls[0]["tools"]] == ["exec_command", "apply_patch"]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "exec_command"
    assert turn.tool_calls[0].arguments["cmd"] == "pwd"
    assert turn.tool_calls[0].call_id == "call_1"


@pytest.mark.parametrize(("task_name", "instruction"), TASK_CASES)
def test_codex_full_seed_preserves_codex_prompt_shape_across_tasks(
    monkeypatch, task_name: str, instruction: str
) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext(instruction=instruction, working_dir=f"/app/{task_name}"), []
        )
    finally:
        set_client_factory(None)

    messages = fake.calls[0]["input"]
    assert messages[0] == {"role": "system", "content": module.CODEX_BASE_INSTRUCTIONS}
    assert messages[1]["role"] == "user"
    assert f"<cwd>/app/{task_name}</cwd>" in messages[1]["content"]
    assert messages[1]["content"].endswith(instruction)
    assert fake.calls[0]["tools"] == module._built_tools()
    assert fake.calls[0]["tool_choice"] == "auto"
    assert fake.calls[0]["parallel_tool_calls"] is True
    assert turn.done is True


def test_codex_full_seed_returns_apply_patch_custom_tool(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    patch = (
        "*** Begin Patch\n"
        "*** Update File: hello.txt\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )
    fake = RecordingToolOpenAI(
        [
            SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(
                        type="custom_tool_call",
                        name="apply_patch",
                        input=patch,
                        call_id="call_patch",
                    )
                ],
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Edit the file."), [])
    finally:
        set_client_factory(None)

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "apply_patch"
    assert turn.tool_calls[0].arguments == {"patch": patch}
    assert turn.tool_calls[0].call_id == "call_patch"


def test_codex_full_seed_replays_history_as_response_items(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    set_client_factory(lambda: fake)
    history = [
        CommandResult(
            command="pwd",
            return_code=0,
            stdout="/workspace\n",
            tool_name="exec_command",
            tool_call_id="call_1",
            metadata={"arguments": {"cmd": "pwd"}},
        )
    ]
    try:
        turn = module.create_agent().next_command(TaskContext("List files."), history)
    finally:
        set_client_factory(None)

    input_items = fake.calls[0]["input"]
    assert input_items[2]["type"] == "function_call"
    assert input_items[2]["name"] == "exec_command"
    assert input_items[3]["type"] == "function_call_output"
    assert input_items[3]["output"].startswith("Wall time: 0.0000 seconds")
    assert turn.done is True


def _load_seed():
    name = "codex_full_seed_under_test"
    spec = importlib.util.spec_from_file_location(name, SEED_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {SEED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingToolOpenAI:
    def __init__(self, responses: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)
