from __future__ import annotations

import json
from types import SimpleNamespace

from plumbing.openai_client import (
    _codex_body,
    call_terminal_model_with_tools,
    call_terminal_model,
    check_terminal_model_available,
    reset_trace_dir,
    set_client_factory,
    set_trace_dir,
)


def test_terminal_model_calls_use_no_reasoning(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = RecordingOpenAI("ok")
    set_client_factory(lambda: fake)
    try:
        assert call_terminal_model([{"role": "user", "content": "next?"}]) == "ok"
    finally:
        set_client_factory(None)

    assert fake.calls[0]["reasoning"] == {"effort": "none"}


def test_terminal_model_preflight_uses_no_reasoning(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = RecordingOpenAI("OK")
    set_client_factory(lambda: fake)
    try:
        check_terminal_model_available()
    finally:
        set_client_factory(None)

    assert fake.calls[0]["reasoning"] == {"effort": "none"}


def test_codex_backend_body_uses_no_reasoning() -> None:
    body = _codex_body([{"role": "user", "content": "next?"}])

    assert body["reasoning"] == {"effort": "none"}


def test_codex_backend_body_can_include_tools() -> None:
    tools = [{"type": "function", "name": "execute_commands", "parameters": {}}]
    body = _codex_body([{"role": "user", "content": "next?"}], tools)

    assert body["tools"] == tools


def test_terminal_model_tool_calls_are_extracted(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = RecordingToolOpenAI()
    set_client_factory(lambda: fake)
    tools = [{"type": "function", "name": "execute_commands", "parameters": {}}]
    try:
        result = call_terminal_model_with_tools([{"role": "user", "content": "next?"}], tools)
    finally:
        set_client_factory(None)

    assert fake.calls[0]["tools"] == tools
    assert result.content == "thinking"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "execute_commands"
    assert result.tool_calls[0].arguments == {"commands": [{"keystrokes": "pwd\n"}]}


def test_model_trace_records_reasoning_effort(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = RecordingOpenAI("ok")
    set_client_factory(lambda: fake)
    token = set_trace_dir(tmp_path)
    try:
        call_terminal_model([{"role": "user", "content": "next?"}])
    finally:
        reset_trace_dir(token)
        set_client_factory(None)

    trace = json.loads((tmp_path / "model-call-01.json").read_text(encoding="utf-8"))
    assert trace["reasoning_effort"] == "none"


class RecordingOpenAI:
    def __init__(self, text: str) -> None:
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)
        self.text = text

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class RecordingToolOpenAI:
    def __init__(self) -> None:
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="thinking",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="execute_commands",
                    arguments='{"commands":[{"keystrokes":"pwd\\n"}]}',
                    call_id="call_1",
                )
            ],
        )
