from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import set_client_factory
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {
    "codex_1300": (1300, 1280),
    "codex_1000": (1000, 980),
    "codex_700": (700, 680),
    "codex_400": (400, 390),
}


@pytest.mark.parametrize(("seed", "limits"), SEEDS.items())
def test_codex_budget_seed_validates(seed: str, limits: tuple[int, int]) -> None:
    max_lines, min_lines = limits
    result = validate_candidate(_seed_path(seed), max_lines=max_lines, min_lines=min_lines)

    assert result["ok"], result


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_returns_exec_command(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI(
        [
            _response(
                [
                    SimpleNamespace(
                        type="function_call",
                        name="exec_command",
                        arguments='{"cmd":"pwd","yield_time_ms":1000}',
                        call_id="call_exec",
                    )
                ]
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("List files.", "/repo", metadata=_agents_metadata()),
            [],
        )
    finally:
        set_client_factory(None)

    assert fake.calls[0]["tool_choice"] == "auto"
    assert fake.calls[0]["parallel_tool_calls"] is True
    assert fake.calls[0]["input"][0]["content"].startswith(
        "You are a coding agent running in the Codex CLI"
    )
    assert "AGENTS.md" in _input_text(fake.calls[0]["input"])
    assert turn.tool_calls[0].name == "exec_command"
    assert turn.tool_calls[0].arguments["cmd"] == "pwd"
    assert turn.tool_calls[0].arguments["yield_time_ms"] == 1000
    assert turn.tool_calls[0].call_id == "call_exec"
    assert turn.done is False


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_maps_apply_patch(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    patch = "*** Begin Patch\n*** Add File: note.txt\n+hello\n*** End Patch"
    fake = RecordingToolOpenAI(
        [
            _response(
                [
                    SimpleNamespace(
                        type="custom_tool_call",
                        name="apply_patch",
                        input=patch,
                        call_id="call_patch",
                    )
                ]
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Edit a file.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert turn.tool_calls[0].name == "apply_patch"
    assert turn.tool_calls[0].arguments == {"patch": patch}
    assert turn.tool_calls[0].call_id == "call_patch"


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_maps_write_stdin(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI(
        [
            _response(
                [
                    SimpleNamespace(
                        type="function_call",
                        name="write_stdin",
                        arguments='{"process_id":7,"chars":"\\n"}',
                        call_id="call_write",
                    )
                ]
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Continue process.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert turn.tool_calls[0].name == "write_stdin"
    assert turn.tool_calls[0].arguments["session_id"] == 7
    assert turn.tool_calls[0].arguments["chars"] == "\n"


@pytest.mark.parametrize("seed", ("codex_1300", "codex_1000", "codex_700"))
def test_codex_budget_seed_maps_update_plan(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI(
        [
            _response(
                [
                    SimpleNamespace(
                        type="function_call",
                        name="update_plan",
                        arguments='{"plan":[{"step":"inspect","status":"in_progress"}]}',
                        call_id="call_plan",
                    )
                ]
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Plan work.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert turn.tool_calls[0].name == "update_plan"
    assert turn.tool_calls[0].arguments["plan"][0]["step"] == "inspect"


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_finishes_on_text(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    history = [CommandResult(command="pytest -q", return_code=0, stdout="1 passed\n")]
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Run tests.", "/repo"), history)
    finally:
        set_client_factory(None)

    assert "pytest -q" in json.dumps(fake.calls[0]["input"], sort_keys=True)
    assert turn.tool_calls == ()
    assert turn.assistant_content == "done"
    assert turn.done is True


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_recovers_on_preamble_without_tool(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="I'll inspect the repo.", output=[])])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Inspect project.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert turn.done is False
    assert turn.tool_calls[0].name == "exec_command"
    assert "find ." in turn.tool_calls[0].arguments["cmd"]


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_recovers_on_empty_model_turn(seed: str, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed(seed)
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="", output=[])])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Inspect project.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert turn.done is False
    assert turn.tool_calls[0].name == "exec_command"
    assert "find ." in turn.tool_calls[0].arguments["cmd"]


def test_codex_400_surfaces_repeated_command_history(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed("codex_400")
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    history = [
        CommandResult(command="sed -n '1,220p' app.py", return_code=0, stdout=f"out {index}")
        for index in range(5)
    ]
    history.append(CommandResult(command="pytest -q", return_code=1, stderr="failed"))
    set_client_factory(lambda: fake)
    try:
        module.create_agent().next_command(TaskContext("Fix it.", "/repo"), history)
    finally:
        set_client_factory(None)

    items = fake.calls[0]["input"]
    calls = [item for item in items if item.get("type") == "function_call"]
    outputs = [item for item in items if item.get("type") == "function_call_output"]
    assert len(calls) == 6
    assert len(outputs) == 6
    assert calls[0]["name"] == "exec_command"
    assert json.loads(calls[0]["arguments"])["cmd"] == "sed -n '1,220p' app.py"
    assert "out 4" in outputs[4]["output"]
    assert "STDERR:\nfailed" in outputs[-1]["output"]


def test_codex_400_replays_response_items_with_reasoning_content() -> None:
    module = _load_seed("codex_400")
    response_items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [],
            "reasoning_content": "hidden chain",
        },
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
            "call_id": "call_reasoning",
        },
    ]
    record = CommandResult(
        command="pwd",
        return_code=0,
        stdout="/app",
        tool_call_id="call_reasoning",
        metadata={"codex_response_items": response_items, "arguments": {"cmd": "ignored"}},
    )

    items = module._record_items(1, record)

    assert items[0]["reasoning_content"] == "hidden chain"
    assert items[1]["call_id"] == "call_reasoning"
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_reasoning"


def test_codex_400_ignores_response_items_when_call_id_mismatches() -> None:
    module = _load_seed("codex_400")
    record = CommandResult(
        command="pwd",
        return_code=0,
        stdout="/app",
        tool_call_id="recovery_status",
        metadata={
            "codex_response_items": [
                {
                    "type": "function_call",
                    "name": "update_plan",
                    "arguments": "{}",
                    "call_id": "call_plan",
                }
            ]
        },
    )

    items = module._record_items(1, record)

    assert items[0]["call_id"] == "recovery_status"
    assert items[0]["name"] == "exec_command"
    assert items[1]["call_id"] == "recovery_status"


@pytest.mark.parametrize("seed", SEEDS)
def test_codex_budget_seed_prefers_response_items_to_avoid_duplicate_tool_calls(
    seed: str,
) -> None:
    module = _load_seed(seed)
    response_items = [
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
            "call_id": "call_dup",
        }
    ]
    result = SimpleNamespace(
        response_items=response_items,
        tool_calls=[
            SimpleNamespace(
                name="exec_command",
                arguments={"cmd": "pwd"},
                call_id="call_dup",
                arguments_text='{"cmd":"pwd"}',
            )
        ],
    )

    assert module._model_items(result) == response_items


def _seed_path(seed: str) -> Path:
    return ROOT / "seeds" / seed / "harness.py"


def _load_seed(seed: str):
    name = f"{seed}_under_test"
    spec = importlib.util.spec_from_file_location(name, _seed_path(seed))
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {seed}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _response(output: list[object]) -> SimpleNamespace:
    return SimpleNamespace(output_text="", output=output)


def _agents_metadata() -> dict[str, list[dict[str, str]]]:
    return {"agents_md": [{"path": "/repo/AGENTS.md", "content": "Run tests before final."}]}


def _input_text(items: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)


class RecordingToolOpenAI:
    def __init__(self, responses: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)
