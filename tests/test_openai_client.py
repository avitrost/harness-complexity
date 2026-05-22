from __future__ import annotations

import json
from types import SimpleNamespace

from plumbing.openai_client import (
    _codex_body,
    _extract_response_dict_result,
    _extract_sse_result,
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


def test_codex_backend_body_can_enable_parallel_tool_calls() -> None:
    tools = [{"type": "function", "name": "execute_commands", "parameters": {}}]
    body = _codex_body(
        [{"role": "user", "content": "next?"}],
        tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True


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


def test_codex_response_items_extract_custom_and_local_shell_calls() -> None:
    result = _extract_response_dict_result(
        {
            "output": [
                {"type": "custom_tool_call", "name": "apply_patch", "input": "diff", "call_id": "c1"},
                {
                    "type": "local_shell_call",
                    "action": {"command": "pytest -q", "timeout_ms": 1500},
                    "call_id": "c2",
                },
            ]
        }
    )

    assert [(call.name, call.call_id) for call in result.tool_calls] == [
        ("apply_patch", "c1"),
        ("local_shell", "c2"),
    ]
    assert result.tool_calls[0].arguments == {"input": "diff"}
    assert result.tool_calls[0].arguments_text == "diff"
    assert result.tool_calls[1].arguments == {"command": "pytest -q", "timeout_ms": 1500}


def test_codex_sse_extracts_streamed_tool_call_items() -> None:
    result = _extract_sse_result(
        'data: {"type":"response.output_item.done","item":{"type":"local_shell_call",'
        '"action":{"command":"ls"},"call_id":"c1"}}\n\n'
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "local_shell"
    assert result.tool_calls[0].arguments == {"command": "ls"}


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
