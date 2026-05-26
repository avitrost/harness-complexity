from __future__ import annotations

import json
from types import SimpleNamespace

from plumbing.openai_client import (
    CodexBackendError,
    _codex_body,
    _codex_headers,
    _extract_response_dict_result,
    _extract_sse_result,
    _retry_delay,
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
    assert body["include"] == ["reasoning.encrypted_content"]


def test_codex_backend_body_can_include_tools() -> None:
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "parameters": {},
            "output_schema": {"type": "object"},
        }
    ]
    body = _codex_body([{"role": "user", "content": "next?"}], tools)

    assert body["tools"] == [{"type": "function", "name": "exec_command", "parameters": {}}]
    assert body["tool_choice"] == "auto"


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


def test_codex_backend_body_preserves_response_items() -> None:
    item = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    body = _codex_body(
        [
            {"role": "system", "content": "sys"},
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": "task"},
            item,
        ]
    )

    assert body["instructions"] == "sys"
    assert body["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "dev"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "task"}],
        },
        item,
    ]


def test_codex_backend_body_sanitizes_response_items_like_codex() -> None:
    body = _codex_body(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hi",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "name": "exec_command",
                "arguments": "{}",
                "call_id": "call_1",
            },
            {
                "type": "custom_tool_call",
                "id": "ct_1",
                "status": "completed",
                "name": "apply_patch",
                "input": "patch",
                "call_id": "call_2",
            },
        ]
    )

    assert body["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "hi"}],
        },
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": "{}",
            "call_id": "call_1",
        },
        {
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_2",
            "name": "apply_patch",
            "input": "patch",
        },
    ]


def test_codex_backend_body_uses_trace_scoped_prompt_cache_key(tmp_path) -> None:
    token = set_trace_dir(tmp_path / "trial")
    try:
        body = _codex_body([{"role": "user", "content": "next?"}])
        headers = _codex_headers("token", "account")
    finally:
        reset_trace_dir(token)

    assert body["prompt_cache_key"] == headers["thread-id"]
    assert headers["session-id"]
    assert headers["thread-id"]
    assert headers["x-client-request-id"] == headers["thread-id"]
    assert headers["x-codex-window-id"] == f"{headers['thread-id']}:0"
    assert "x-codex-installation-id" not in headers
    assert body["client_metadata"]["x-codex-installation-id"].startswith("harness-")


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
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "input": "diff",
                    "call_id": "c1",
                },
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
        '"id":"item_1","action":{"command":"ls"},"call_id":"c1"}}\n\n'
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "local_shell"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.response_items == [
        {"type": "local_shell_call", "call_id": "c1", "action": {"command": "ls"}}
    ]


def test_codex_sse_prefers_completed_tool_call_arguments() -> None:
    result = _extract_sse_result(
        'data: {"type":"response.output_item.added","item":{"type":"local_shell_call",'
        '"call_id":"c1"}}\n\n'
        'data: {"type":"response.output_item.done","item":{"type":"local_shell_call",'
        '"action":{"command":"ls"},"call_id":"c1"}}\n\n'
        'data: {"type":"response.output_item.added","item":{"type":"custom_tool_call",'
        '"name":"apply_patch","call_id":"c2","input":""}}\n\n'
        'data: {"type":"response.output_item.done","item":{"type":"custom_tool_call",'
        '"name":"apply_patch","call_id":"c2","input":"diff"}}\n\n'
    )

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "local_shell"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.tool_calls[1].name == "apply_patch"
    assert result.tool_calls[1].arguments == {"input": "diff"}


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


def test_codex_backend_retry_delay_uses_retry_after_header() -> None:
    exc = CodexBackendError(429, '{"detail":"Rate limit exceeded"}', {"retry-after": "7"})

    assert _retry_delay(exc, 0) == 7.0


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
