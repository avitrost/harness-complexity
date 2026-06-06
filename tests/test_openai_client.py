from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
    terminal_provider,
    terminal_reasoning_effort,
    using_codex_auth,
)


@pytest.fixture(autouse=True)
def clear_terminal_provider_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HARNESS_SECRETS_FILE", str(tmp_path / "missing-secrets.env"))
    for name in (
        "TERMINAL_MODEL_PROVIDER",
        "OPENAI_TERMINAL_PROVIDER",
        "MODEL_PROVIDER",
        "TERMINAL_MODEL",
        "OPENAI_TERMINAL_MODEL",
        "TERMINAL_REASONING_EFFORT",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BETA",
        "ANTHROPIC_MESSAGES_URL",
        "ANTHROPIC_THINKING_BUDGET_TOKENS",
        "ANTHROPIC_THINKING_TYPE",
        "ANTHROPIC_VERSION",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_TERMINAL_REASONING_EFFORT", raising=False)


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


def test_terminal_reasoning_effort_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_TERMINAL_REASONING_EFFORT", "medium")
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = RecordingOpenAI("ok")
    set_client_factory(lambda: fake)
    try:
        assert terminal_reasoning_effort() == "medium"
        assert call_terminal_model([{"role": "user", "content": "next?"}]) == "ok"
    finally:
        set_client_factory(None)

    assert fake.calls[0]["reasoning"] == {"effort": "medium"}
    assert _codex_body([{"role": "user", "content": "next?"}])["reasoning"] == {"effort": "medium"}


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


def test_anthropic_provider_uses_messages_api_payload(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return JsonResponse(
            {
                "id": "msg_1",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "exec_command",
                        "input": {"cmd": "pwd"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        )

    monkeypatch.setenv("TERMINAL_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_AUTH_MODE", "codex")
    monkeypatch.setenv("OPENAI_TERMINAL_MODEL", "claude-sonnet-test")
    monkeypatch.setenv("OPENAI_TERMINAL_REASONING_EFFORT", "medium")
    monkeypatch.setattr("plumbing.openai_client.urlopen", fake_urlopen)

    result = call_terminal_model_with_tools(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "next?"},
        ],
        [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ],
        tool_choice="auto",
    )

    assert terminal_provider() == "anthropic"
    assert using_codex_auth() is False
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-test"
    assert captured["body"]["system"] == "system instructions"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "next?"}]}
    ]
    assert captured["body"]["tools"][0]["input_schema"]["required"] == ["cmd"]
    assert captured["body"]["tool_choice"] == {"type": "auto"}
    assert captured["body"]["output_config"] == {"effort": "medium"}
    assert result.content == "checking"
    assert result.tool_calls[0].name == "exec_command"
    assert result.tool_calls[0].arguments == {"cmd": "pwd"}
    assert result.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_tokens": 0,
    }
    assert result.request_metadata["provider"] == "anthropic"


def test_anthropic_provider_replays_tool_results(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return JsonResponse({"id": "msg_2", "content": [{"type": "text", "text": "done"}]})

    monkeypatch.setenv("TERMINAL_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr("plumbing.openai_client.urlopen", fake_urlopen)

    assert (
        call_terminal_model(
            [
                {"role": "user", "content": "run pwd"},
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                    "call_id": "call_1",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "/tmp"},
            ]
        )
        == "done"
    )

    assert captured["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "run pwd"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "exec_command",
                    "input": {"cmd": "pwd"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "/tmp"}],
        },
    ]


def test_deepseek_provider_uses_chat_completions_payload(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return JsonResponse(
            {
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "working",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"ls"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "prompt_cache_hit_tokens": 3,
                },
            }
        )

    monkeypatch.setenv("TERMINAL_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_TERMINAL_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr("plumbing.openai_client.urlopen", fake_urlopen)

    result = call_terminal_model_with_tools(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "edit"},
        ],
        [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch.",
                "format": {"type": "grammar"},
            }
        ],
        tool_choice="auto",
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer deepseek-key"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "edit"},
    ]
    assert captured["body"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a patch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Raw input for this freeform tool.",
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured["body"]
    assert result.content == "working"
    assert result.tool_calls[0].name == "exec_command"
    assert result.tool_calls[0].arguments == {"cmd": "ls"}
    assert result.usage == {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
        "cached_tokens": 3,
    }
    assert result.request_metadata["provider"] == "deepseek"


def test_deepseek_provider_maps_non_none_effort_to_thinking(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return JsonResponse({"id": "chatcmpl_1", "choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("TERMINAL_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_TERMINAL_REASONING_EFFORT", "medium")
    monkeypatch.setattr("plumbing.openai_client.urlopen", fake_urlopen)

    assert call_terminal_model([{"role": "user", "content": "next?"}]) == "ok"

    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "high"


def test_deepseek_provider_replays_tool_results(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return JsonResponse({"id": "chatcmpl_2", "choices": [{"message": {"content": "done"}}]})

    monkeypatch.setenv("TERMINAL_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr("plumbing.openai_client.urlopen", fake_urlopen)

    assert (
        call_terminal_model(
            [
                {"role": "user", "content": "run pwd"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "using the tool"}],
                    "reasoning_content": "I need the current directory.",
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                    "call_id": "call_1",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "/tmp"},
            ]
        )
        == "done"
    )

    assert captured["body"]["messages"] == [
        {"role": "user", "content": "run pwd"},
        {
            "role": "assistant",
            "content": "using the tool",
            "reasoning_content": "I need the current directory.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
    ]


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


def test_model_trace_records_response_usage(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    fake = UsageOpenAI()
    set_client_factory(lambda: fake)
    token = set_trace_dir(tmp_path)
    try:
        call_terminal_model([{"role": "user", "content": "next?"}])
    finally:
        reset_trace_dir(token)
        set_client_factory(None)

    trace = json.loads((tmp_path / "model-call-01.json").read_text(encoding="utf-8"))
    assert trace["request_metadata"]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "cached_tokens": 4,
    }


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


class UsageOpenAI:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        return SimpleNamespace(
            output_text="ok",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
                input_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        )


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


class JsonResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")
