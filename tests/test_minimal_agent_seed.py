from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import set_client_factory
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "minimal_agent" / "harness.py"


def test_minimal_agent_seed_validates() -> None:
    result = validate_candidate(SEED_PATH, max_lines=120)

    assert result["ok"], result


def test_minimal_agent_returns_exec_command_tool_call(monkeypatch) -> None:
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
        turn = module.create_agent().next_command(TaskContext("List files.", "/repo"), [])
    finally:
        set_client_factory(None)

    assert fake.calls[0]["tools"] == module.TOOLS
    assert fake.calls[0]["tool_choice"] == "auto"
    assert fake.calls[0]["parallel_tool_calls"] is True
    assert "<cwd>/repo</cwd>" in fake.calls[0]["input"][1]["content"]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "exec_command"
    assert turn.tool_calls[0].arguments["cmd"] == "pwd"
    assert turn.tool_calls[0].arguments["yield_time_ms"] == 1000
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.done is False


def test_minimal_agent_finishes_on_text_without_tool_call(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    history = [CommandResult(command="pytest -q", return_code=0, stdout="1 passed\n")]
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Run tests."), history)
    finally:
        set_client_factory(None)

    assert "pytest -q" in fake.calls[0]["input"][1]["content"]
    assert turn.tool_calls == ()
    assert turn.assistant_content == "done"
    assert turn.done is True


def _load_seed():
    name = "minimal_agent_seed_under_test"
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
