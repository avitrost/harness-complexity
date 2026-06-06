from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openai import OpenAI

from plumbing.secrets import (
    require_anthropic_api_key,
    require_deepseek_api_key,
    require_openai_api_key,
)

TERMINAL_PROVIDER = "openai"
TERMINAL_MODEL = "gpt-5.4-mini"
ANTHROPIC_TERMINAL_MODEL = "claude-sonnet-4-6"
DEEPSEEK_TERMINAL_MODEL = "deepseek-v4-flash"
TERMINAL_REASONING_EFFORT = "none"
TIMEOUT_SEC = 120.0
DEFAULT_MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 4096
PREFLIGHT_OUTPUT_TOKENS = 16
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

_client_factory: Callable[[], OpenAI] | None = None
_last_request_at = 0.0
_trace_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "terminal_model_trace_dir", default=None
)
_trace_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "terminal_model_trace_count", default=0
)


@dataclass(frozen=True)
class ModelToolCall:
    name: str
    arguments: dict[str, Any]
    arguments_text: str = ""
    call_id: str = ""


@dataclass(frozen=True)
class ToolModelResult:
    content: str
    tool_calls: list[ModelToolCall]
    request_metadata: dict[str, Any] | None = None
    response_items: list[dict[str, Any]] = field(default_factory=list)
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodexBackendSession:
    session_id: str
    thread_id: str
    window_id: str
    installation_id: str


_codex_session: contextvars.ContextVar[CodexBackendSession | None] = contextvars.ContextVar(
    "codex_backend_session", default=None
)


def set_client_factory(factory: Callable[[], OpenAI] | None) -> None:
    global _client_factory
    _client_factory = factory


def set_trace_dir(path: Path | str | None) -> object:
    _trace_count.set(0)
    trace_path = Path(path) if path is not None else None
    trace_token = _trace_dir.set(trace_path)
    session_token = _codex_session.set(_new_codex_session(trace_path))
    return trace_token, session_token


def reset_trace_dir(token: object) -> None:
    if isinstance(token, tuple) and len(token) == 2:
        _trace_dir.reset(token[0])  # type: ignore[arg-type]
        _codex_session.reset(token[1])  # type: ignore[arg-type]
    else:
        _trace_dir.reset(token)  # type: ignore[arg-type]
    _trace_count.set(0)


def _make_client() -> OpenAI:
    if _client_factory is not None:
        return _client_factory()
    return OpenAI(
        api_key=require_openai_api_key(),
        timeout=TIMEOUT_SEC,
        max_retries=_max_retries(),
    )


def terminal_provider() -> str:
    configured = (
        os.getenv("TERMINAL_MODEL_PROVIDER")
        or os.getenv("OPENAI_TERMINAL_PROVIDER")
        or os.getenv("MODEL_PROVIDER")
        or TERMINAL_PROVIDER
    )
    provider = configured.strip().lower()
    aliases = {
        "openai": "openai",
        "codex": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "deepseek": "deepseek",
    }
    if provider not in aliases:
        raise RuntimeError(f"unsupported terminal model provider: {configured}")
    return aliases[provider]


def using_codex_auth() -> bool:
    return terminal_provider() == "openai" and os.getenv("OPENAI_AUTH_MODE", "").lower() == "codex"


def terminal_model() -> str:
    configured = os.getenv("TERMINAL_MODEL") or os.getenv("OPENAI_TERMINAL_MODEL")
    if configured:
        return configured
    provider = terminal_provider()
    if provider == "anthropic":
        return ANTHROPIC_TERMINAL_MODEL
    if provider == "deepseek":
        return DEEPSEEK_TERMINAL_MODEL
    return TERMINAL_MODEL


def terminal_reasoning_effort() -> str:
    return (
        os.getenv("TERMINAL_REASONING_EFFORT")
        or os.getenv("OPENAI_TERMINAL_REASONING_EFFORT")
        or TERMINAL_REASONING_EFFORT
    )


def call_terminal_model(messages: list[dict[str, Any]]) -> str:
    last_error: Exception | None = None
    max_retries = _max_retries()
    for attempt in range(max_retries + 1):
        try:
            _throttle_if_requested()
            started_at = time.monotonic()
            provider = terminal_provider()
            if using_codex_auth():
                result = _call_codex_backend_result(messages)
                _write_model_trace(
                    messages,
                    result.content,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result.content
            if provider == "anthropic":
                result = _call_anthropic_messages_result(messages)
                _write_model_trace(
                    messages,
                    result.content,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result.content
            if provider == "deepseek":
                result = _call_deepseek_chat_result(messages)
                _write_model_trace(
                    messages,
                    result.content,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result.content
            response = _make_client().responses.create(
                model=terminal_model(),
                input=messages,
                reasoning={"effort": terminal_reasoning_effort()},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=TIMEOUT_SEC,
            )
            result = _extract_result(response)
            _write_model_trace(
                messages,
                result.content,
                duration_sec=time.monotonic() - started_at,
                request_metadata={"usage": result.usage} if result.usage else None,
            )
            text = result.content
            return text
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= max_retries:
                _write_model_trace(messages, "", error=str(exc))
                break
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("terminal model call failed") from last_error


def call_terminal_model_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
) -> ToolModelResult:
    last_error: Exception | None = None
    max_retries = _max_retries()
    for attempt in range(max_retries + 1):
        try:
            _throttle_if_requested()
            started_at = time.monotonic()
            provider = terminal_provider()
            if using_codex_auth():
                result = _call_codex_backend_result(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                _write_model_trace(
                    messages,
                    result.content,
                    tool_calls=result.tool_calls,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result
            if provider == "anthropic":
                result = _call_anthropic_messages_result(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                _write_model_trace(
                    messages,
                    result.content,
                    tool_calls=result.tool_calls,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result
            if provider == "deepseek":
                result = _call_deepseek_chat_result(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                _write_model_trace(
                    messages,
                    result.content,
                    tool_calls=result.tool_calls,
                    duration_sec=time.monotonic() - started_at,
                    request_metadata=result.request_metadata,
                )
                return result
            kwargs: dict[str, Any] = dict(
                model=terminal_model(),
                input=messages,
                reasoning={"effort": terminal_reasoning_effort()},
                tools=tools,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=TIMEOUT_SEC,
            )
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls
            response = _make_client().responses.create(**kwargs)
            result = _extract_result(response)
            _write_model_trace(
                messages,
                result.content,
                tool_calls=result.tool_calls,
                duration_sec=time.monotonic() - started_at,
                request_metadata={"usage": result.usage} if result.usage else None,
            )
            return result
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= max_retries:
                _write_model_trace(messages, "", error=str(exc))
                break
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("terminal model tool call failed") from last_error


def check_terminal_model_available() -> None:
    _throttle_if_requested()
    provider = terminal_provider()
    if using_codex_auth():
        _call_codex_backend([{"role": "user", "content": "Reply OK."}])
        return
    if provider == "anthropic":
        _call_anthropic_messages_result(
            [{"role": "user", "content": "Reply OK."}],
            max_tokens=PREFLIGHT_OUTPUT_TOKENS,
        )
        return
    if provider == "deepseek":
        _call_deepseek_chat_result(
            [{"role": "user", "content": "Reply OK."}],
            max_tokens=PREFLIGHT_OUTPUT_TOKENS,
        )
        return
    response = _make_client().responses.create(
        model=terminal_model(),
        input=[{"role": "user", "content": "Reply OK."}],
        reasoning={"effort": terminal_reasoning_effort()},
        max_output_tokens=PREFLIGHT_OUTPUT_TOKENS,
        timeout=TIMEOUT_SEC,
    )
    _extract_text(response)


def _throttle_if_requested() -> None:
    interval = float(os.getenv("OPENAI_MIN_REQUEST_INTERVAL_SEC", "0") or 0)
    if interval <= 0:
        return
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_at = time.monotonic()


def _max_retries() -> int:
    try:
        return max(0, int(os.getenv("OPENAI_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)) or 0))
    except ValueError:
        return DEFAULT_MAX_RETRIES


def _retry_delay(exc: Exception, attempt: int) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        headers = getattr(exc, "headers", {}) or {}
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    if "rate limit" in str(exc).lower() or "429" in str(exc):
        return 65.0
    return float(2**attempt)


class ModelProviderError(RuntimeError):
    def __init__(self, provider: str, status_code: int, detail: str, headers: Any) -> None:
        self.provider = provider
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
        super().__init__(f"{provider} API call failed: {status_code} {detail}")


def _call_anthropic_messages_result(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> ToolModelResult:
    del parallel_tool_calls
    system, api_messages = _anthropic_messages_and_system(messages)
    body: dict[str, Any] = {
        "model": terminal_model(),
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system:
        body["system"] = system
    anthropic_tools = _anthropic_tools(tools)
    if anthropic_tools:
        body["tools"] = anthropic_tools
    converted_tool_choice = _anthropic_tool_choice(tool_choice)
    if converted_tool_choice is not None:
        body["tool_choice"] = converted_tool_choice
    effort = _anthropic_effort()
    if effort is not None:
        body["output_config"] = {"effort": effort}
    thinking = _anthropic_thinking(max_tokens)
    if thinking is not None:
        body["thinking"] = thinking
    headers = {
        "content-type": "application/json",
        "x-api-key": require_anthropic_api_key(),
        "anthropic-version": os.getenv("ANTHROPIC_VERSION", ANTHROPIC_VERSION),
    }
    if beta := os.getenv("ANTHROPIC_BETA"):
        headers["anthropic-beta"] = beta
    response = _post_json(
        os.getenv("ANTHROPIC_MESSAGES_URL", ANTHROPIC_MESSAGES_URL),
        body,
        headers,
        "anthropic",
    )
    result = _extract_anthropic_result(response)
    return ToolModelResult(
        result.content,
        result.tool_calls,
        _provider_request_metadata("anthropic", body, result),
        result.response_items,
        result.response_id,
        result.usage,
    )


def _call_deepseek_chat_result(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> ToolModelResult:
    del parallel_tool_calls
    body: dict[str, Any] = {
        "model": terminal_model(),
        "messages": _deepseek_messages(messages),
        "max_tokens": max_tokens,
        "stream": False,
    }
    deepseek_tools = _deepseek_tools(tools)
    if deepseek_tools:
        body["tools"] = deepseek_tools
    converted_tool_choice = _deepseek_tool_choice(tool_choice)
    if converted_tool_choice is not None:
        body["tool_choice"] = converted_tool_choice
    thinking, reasoning_effort = _deepseek_thinking_and_effort()
    body["thinking"] = thinking
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    response = _post_json(
        _deepseek_chat_url(),
        body,
        {
            "authorization": f"Bearer {require_deepseek_api_key()}",
            "content-type": "application/json",
        },
        "deepseek",
    )
    result = _extract_deepseek_result(response)
    return ToolModelResult(
        result.content,
        result.tool_calls,
        _provider_request_metadata("deepseek", body, result),
        result.response_items,
        result.response_id,
        result.usage,
    )


def _post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    provider: str,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SEC) as response:
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelProviderError(provider, exc.code, detail[:500], exc.headers) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider} API returned invalid JSON: {text[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{provider} API returned unexpected JSON: {text[:500]}")
    return payload


def _provider_request_metadata(
    provider: str,
    body: dict[str, Any],
    result: ToolModelResult,
) -> dict[str, Any]:
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    metadata = {
        "provider": provider,
        "model": body.get("model"),
        "tool_choice": body.get("tool_choice"),
        "tool_count": len(tools),
        "tool_names": [_provider_tool_name(tool) for tool in tools],
        "input_count": len(messages),
        "response_id": result.response_id,
        "response_item_count": len(result.response_items),
    }
    if result.usage:
        metadata["usage"] = result.usage
    if body.get("output_config") is not None:
        metadata["output_config"] = body["output_config"]
    if body.get("thinking") is not None:
        metadata["thinking"] = body["thinking"]
    return metadata


def _provider_tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("function"), dict):
        return tool["function"].get("name")
    return tool.get("name")


def _anthropic_effort() -> str | None:
    effort = terminal_reasoning_effort().strip().lower()
    if not effort or effort == "none":
        return None
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise RuntimeError(f"unsupported Anthropic effort: {effort}")
    return effort


def _anthropic_thinking(max_tokens: int) -> dict[str, Any] | None:
    thinking_type = os.getenv("ANTHROPIC_THINKING_TYPE", "").strip().lower()
    budget = os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS")
    if not thinking_type and not budget:
        return None
    if thinking_type in {"", "enabled"}:
        if not budget:
            raise RuntimeError("ANTHROPIC_THINKING_BUDGET_TOKENS is required")
        try:
            budget_tokens = int(budget)
        except ValueError as exc:
            raise RuntimeError("ANTHROPIC_THINKING_BUDGET_TOKENS must be an integer") from exc
        if budget_tokens >= max_tokens:
            raise RuntimeError("ANTHROPIC_THINKING_BUDGET_TOKENS must be less than max_tokens")
        return {"type": "enabled", "budget_tokens": budget_tokens}
    if thinking_type == "adaptive":
        return {"type": "adaptive"}
    if thinking_type == "disabled":
        return {"type": "disabled"}
    raise RuntimeError(f"unsupported ANTHROPIC_THINKING_TYPE: {thinking_type}")


def _anthropic_messages_and_system(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        role = str(item.get("role") or "")
        if role in {"system", "developer"} and item_type != "message":
            system_parts.append(_plain_text(item.get("content", "")))
            continue
        if item_type == "message":
            message_role = str(item.get("role") or "assistant")
            if message_role in {"system", "developer"}:
                system_parts.append(_plain_text(item.get("content", "")))
                continue
            _anthropic_append_message(
                api_messages,
                _anthropic_role(message_role),
                _anthropic_text_blocks(item.get("content", "")),
            )
            continue
        if item_type in {"function_call", "custom_tool_call", "local_shell_call", "tool_call"}:
            call = _tool_call_from_mapping(item)
            if call is not None:
                _anthropic_append_message(api_messages, "assistant", [_anthropic_tool_use(call)])
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            block = _anthropic_tool_result(item)
            if block is not None:
                _anthropic_append_message(api_messages, "user", [block])
            continue
        if item_type == "reasoning":
            continue
        if role in {"system", "developer"}:
            system_parts.append(_plain_text(item.get("content", "")))
            continue
        _anthropic_append_message(
            api_messages,
            _anthropic_role(role),
            _anthropic_text_blocks(item.get("content", "")),
        )
    if not api_messages:
        api_messages.append({"role": "user", "content": [{"type": "text", "text": ""}]})
    return "\n\n".join(part for part in system_parts if part), api_messages


def _anthropic_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def _anthropic_append_message(
    messages: list[dict[str, Any]],
    role: str,
    blocks: list[dict[str, Any]],
) -> None:
    if not blocks:
        return
    starts_with_tool_result = blocks[0].get("type") == "tool_result"
    if messages and messages[-1].get("role") == role:
        existing = messages[-1].get("content")
        if not isinstance(existing, list):
            existing = []
            messages[-1]["content"] = existing
        if not starts_with_tool_result or all(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in existing
        ):
            existing.extend(blocks)
            return
    messages.append({"role": role, "content": blocks})


def _anthropic_text_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                blocks.append({"type": "text", "text": str(item.get("text", ""))})
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                block = _anthropic_tool_result(item)
                if block is not None:
                    blocks.append(block)
            elif isinstance(item, dict) and item.get("type") == "tool_use":
                name = str(item.get("name") or "")
                call_id = str(item.get("id") or item.get("tool_use_id") or "")
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": _coerce_tool_args(item.get("input")),
                    }
                )
            else:
                text = _plain_text(item)
                if text:
                    blocks.append({"type": "text", "text": text})
        return blocks
    return [{"type": "text", "text": _plain_text(content)}]


def _anthropic_tool_use(call: ModelToolCall) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": call.call_id or f"toolu_{uuid.uuid4().hex}",
        "name": call.name,
        "input": call.arguments,
    }


def _anthropic_tool_result(item: dict[str, Any]) -> dict[str, Any] | None:
    call_id = str(item.get("call_id") or item.get("tool_use_id") or item.get("tool_call_id") or "")
    if not call_id:
        return None
    output = item.get("output", item.get("content", ""))
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": _plain_text(output),
    }
    if item.get("is_error") is not None:
        block["is_error"] = bool(item["is_error"])
    return block


def _anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for tool in tools or []:
        name = _openai_style_tool_name(tool)
        if not name:
            continue
        item = {
            "name": name,
            "description": str(tool.get("description") or ""),
            "input_schema": _tool_schema(tool),
        }
        if tool.get("strict") is not None:
            item["strict"] = bool(tool["strict"])
        cleaned.append(item)
    return cleaned


def _anthropic_tool_choice(tool_choice: Any | None) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in {"auto", "any", "none"}:
            return {"type": normalized}
        if normalized:
            return {"type": "tool", "name": tool_choice}
        return None
    if not isinstance(tool_choice, dict):
        return None
    choice_type = str(tool_choice.get("type") or "").lower()
    if choice_type in {"auto", "any", "none"}:
        return {"type": choice_type}
    if choice_type == "tool":
        name = tool_choice.get("name")
        return {"type": "tool", "name": str(name)} if name else None
    if choice_type == "function":
        name = tool_choice.get("name")
        function = tool_choice.get("function")
        if not name and isinstance(function, dict):
            name = function.get("name")
        return {"type": "tool", "name": str(name)} if name else None
    return None


def _extract_anthropic_result(response: dict[str, Any]) -> ToolModelResult:
    text_chunks: list[str] = []
    calls: list[ModelToolCall] = []
    response_items: list[dict[str, Any]] = []
    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
            continue
        if block_type == "tool_use":
            call = _make_tool_call(
                str(block.get("name") or ""),
                block.get("input", {}),
                str(block.get("id") or ""),
            )
            if call is not None:
                calls.append(call)
    content = "\n".join(text_chunks)
    if content:
        response_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    response_items.extend(_function_call_items(calls))
    return ToolModelResult(
        content,
        calls,
        response_items=response_items,
        response_id=_response_id_from_response(response),
        usage=_usage_from_anthropic_response(response),
    )


def _usage_from_anthropic_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    cache_creation = _int_or_zero(usage.get("cache_creation_input_tokens"))
    cache_read = _int_or_zero(usage.get("cache_read_input_tokens"))
    input_tokens = _int_or_zero(usage.get("input_tokens")) + cache_creation + cache_read
    output_tokens = usage.get("output_tokens")
    return _clean_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + _int_or_zero(output_tokens),
        cached_tokens=cache_creation + cache_read,
    )


def _deepseek_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        role = str(item.get("role") or "")
        if role in {"system", "developer"} and item_type != "message":
            system_parts.append(_plain_text(item.get("content", "")))
            continue
        if item_type == "message":
            message_role = str(item.get("role") or "assistant")
            if message_role in {"system", "developer"}:
                system_parts.append(_plain_text(item.get("content", "")))
            elif message_role == "assistant" and item.get("reasoning_content") is not None:
                _deepseek_append_assistant_message(
                    api_messages,
                    _plain_text(item.get("content", "")),
                    reasoning_content=str(item.get("reasoning_content") or ""),
                )
            else:
                _deepseek_append_text_message(
                    api_messages,
                    _deepseek_role(message_role),
                    _plain_text(item.get("content", "")),
                )
            continue
        if item_type in {"function_call", "custom_tool_call", "local_shell_call", "tool_call"}:
            call = _tool_call_from_mapping(item)
            if call is not None:
                _deepseek_append_assistant_tool_call(api_messages, call)
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = str(item.get("call_id") or item.get("tool_call_id") or "")
            if call_id:
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _plain_text(item.get("output", item.get("content", ""))),
                    }
                )
            continue
        if item_type == "reasoning":
            continue
        if role in {"system", "developer"}:
            system_parts.append(_plain_text(item.get("content", "")))
            continue
        _deepseek_append_text_message(
            api_messages,
            _deepseek_role(role),
            _plain_text(item.get("content", "")),
        )
    if system_parts:
        api_messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not api_messages:
        api_messages.append({"role": "user", "content": ""})
    return api_messages


def _deepseek_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def _deepseek_thinking_and_effort() -> tuple[dict[str, str], str | None]:
    effort = terminal_reasoning_effort().strip().lower()
    if not effort or effort == "none":
        return {"type": "disabled"}, None
    if effort in {"low", "medium", "high"}:
        return {"type": "enabled"}, "high"
    if effort in {"xhigh", "max"}:
        return {"type": "enabled"}, "max"
    raise RuntimeError(f"unsupported DeepSeek reasoning effort: {effort}")


def _deepseek_append_text_message(
    messages: list[dict[str, Any]],
    role: str,
    content: str,
) -> None:
    if (
        messages
        and messages[-1].get("role") == role
        and "tool_calls" not in messages[-1]
        and role in {"system", "user", "assistant"}
    ):
        existing = str(messages[-1].get("content") or "")
        messages[-1]["content"] = f"{existing}\n\n{content}" if existing else content
        return
    messages.append({"role": role, "content": content})


def _deepseek_append_assistant_message(
    messages: list[dict[str, Any]],
    content: str,
    *,
    reasoning_content: str = "",
) -> None:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    messages.append(message)


def _deepseek_append_assistant_tool_call(
    messages: list[dict[str, Any]],
    call: ModelToolCall,
) -> None:
    tool_call = {
        "id": call.call_id or f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments_text or "{}"},
    }
    if messages and messages[-1].get("role") == "assistant":
        messages[-1].setdefault("tool_calls", []).append(tool_call)
        if messages[-1].get("content") == "":
            messages[-1]["content"] = None
        return
    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})


def _deepseek_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for tool in tools or []:
        name = _openai_style_tool_name(tool)
        if not name:
            continue
        cleaned.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": _tool_schema(tool),
                },
            }
        )
    return cleaned


def _deepseek_tool_choice(tool_choice: Any | None) -> Any | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in {"auto", "none", "required"}:
            return normalized
        if normalized == "any":
            return "required"
        if normalized:
            return {"type": "function", "function": {"name": tool_choice}}
        return None
    if not isinstance(tool_choice, dict):
        return None
    choice_type = str(tool_choice.get("type") or "").lower()
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "function":
        name = tool_choice.get("name")
        function = tool_choice.get("function")
        if not name and isinstance(function, dict):
            name = function.get("name")
        return {"type": "function", "function": {"name": str(name)}} if name else None
    if choice_type == "tool":
        name = tool_choice.get("name")
        return {"type": "function", "function": {"name": str(name)}} if name else None
    return None


def _deepseek_chat_url() -> str:
    return f"{os.getenv('DEEPSEEK_BASE_URL', DEEPSEEK_BASE_URL).rstrip('/')}/chat/completions"


def _extract_deepseek_result(response: dict[str, Any]) -> ToolModelResult:
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content") or ""
    text = str(content) if content is not None else ""
    reasoning_content = message.get("reasoning_content")
    calls = [
        call
        for item in message.get("tool_calls") or []
        if isinstance(item, dict)
        for call in [_tool_call_from_mapping(item)]
        if call is not None
    ]
    response_items: list[dict[str, Any]] = []
    if text or reasoning_content:
        message_item: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}] if text else [],
        }
        if reasoning_content is not None:
            message_item["reasoning_content"] = str(reasoning_content)
        response_items.append(message_item)
    response_items.extend(_function_call_items(calls))
    return ToolModelResult(
        text,
        calls,
        response_items=response_items,
        response_id=_response_id_from_response(response),
        usage=_usage_from_deepseek_response(response),
    )


def _usage_from_deepseek_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return _clean_usage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cached_tokens=usage.get("prompt_cache_hit_tokens"),
    )


def _openai_style_tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(tool.get("name") or function.get("name") or "")


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = tool.get("input_schema") or tool.get("parameters") or function.get("parameters")
    if isinstance(schema, dict):
        return dict(schema)
    if tool.get("type") == "custom":
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw input for this freeform tool.",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        }
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _function_call_items(calls: list[ModelToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function_call",
            "name": call.name,
            "arguments": call.arguments_text or json.dumps(call.arguments, sort_keys=True),
            "call_id": call.call_id,
        }
        for call in calls
    ]


def _plain_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = _plain_text_from_item(item)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    if isinstance(content, dict):
        return _plain_text_from_item(content)
    return str(content)


def _plain_text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    if isinstance(item.get("text"), str):
        return item["text"]
    if isinstance(item.get("content"), str):
        return item["content"]
    if isinstance(item.get("output"), str):
        return item["output"]
    if item.get("type") == "tool_result":
        return _plain_text(item.get("content", ""))
    return json.dumps(item, sort_keys=True)


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class CodexBackendError(RuntimeError):
    def __init__(self, status_code: int, detail: str, headers: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
        super().__init__(f"codex backend call failed: {status_code} {detail}")


def _write_model_trace(
    messages: list[dict[str, Any]],
    response_text: str,
    error: str | None = None,
    tool_calls: list[ModelToolCall] | None = None,
    duration_sec: float | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> None:
    trace_dir = _trace_dir.get()
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    index = _trace_count.get() + 1
    _trace_count.set(index)
    payload: dict[str, Any] = {
        "provider": _terminal_provider_for_trace(),
        "model": _terminal_model_for_trace(),
        "reasoning_effort": terminal_reasoning_effort(),
        "messages": messages,
        "response": response_text,
    }
    if duration_sec is not None:
        payload["duration_sec"] = duration_sec
    if request_metadata is not None:
        payload["request_metadata"] = request_metadata
    if error:
        payload["error"] = error
    if tool_calls is not None:
        payload["tool_calls"] = [asdict(call) for call in tool_calls]
    (trace_dir / f"model-call-{index:02d}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _terminal_provider_for_trace() -> str:
    try:
        return terminal_provider()
    except RuntimeError:
        return (
            os.getenv("TERMINAL_MODEL_PROVIDER")
            or os.getenv("OPENAI_TERMINAL_PROVIDER")
            or os.getenv("MODEL_PROVIDER")
            or TERMINAL_PROVIDER
        )


def _terminal_model_for_trace() -> str:
    try:
        return terminal_model()
    except RuntimeError:
        return os.getenv("TERMINAL_MODEL") or os.getenv("OPENAI_TERMINAL_MODEL") or TERMINAL_MODEL


def _call_codex_backend(messages: list[dict[str, Any]]) -> str:
    return _call_codex_backend_result(messages).content


def _call_codex_backend_result(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
) -> ToolModelResult:
    auth = _load_codex_auth()
    access_token = _fresh_codex_access_token(auth)
    account_id = _codex_account_id(access_token, auth)
    body = _codex_body(
        messages,
        tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    headers = _codex_headers(access_token, account_id)
    request = Request(
        CODEX_BASE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SEC) as response:
            result = _extract_sse_result(response.read().decode("utf-8", errors="replace"))
            return ToolModelResult(
                result.content,
                result.tool_calls,
                _codex_request_metadata(body, headers, result),
                result.response_items,
                result.response_id,
            )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CodexBackendError(exc.code, detail[:500], exc.headers) from exc


def _codex_body(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    instructions = "\n\n".join(
        item.get("content", "")
        for item in messages
        if item.get("role") == "system" and item.get("content")
    )
    input_messages = [_codex_input_item(item) for item in messages if not _is_instruction(item)]
    reasoning = {"effort": terminal_reasoning_effort()}
    body: dict[str, Any] = {
        "model": terminal_model(),
        "reasoning": reasoning,
        "instructions": instructions or "You are a concise assistant.",
        "input": input_messages or [_codex_input_item({"role": "user", "content": ""})],
        "tools": _codex_tools(tools),
        "tool_choice": "auto",
        "stream": True,
        "store": False,
        "include": _codex_include(reasoning),
        "prompt_cache_key": _codex_prompt_cache_key(),
        "client_metadata": {
            "x-codex-installation-id": _codex_current_installation_id(),
        },
    }
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls
    return body


def _codex_headers(access_token: str, account_id: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    session = _codex_session.get()
    if isinstance(session, CodexBackendSession):
        headers.update(
            {
                "session-id": session.session_id,
                "thread-id": session.thread_id,
                "x-client-request-id": session.thread_id,
                "x-codex-window-id": session.window_id,
            }
        )
    return headers


def _codex_request_metadata(
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    result: ToolModelResult | None = None,
) -> dict[str, Any]:
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    input_items = body.get("input") if isinstance(body.get("input"), list) else []
    metadata = {
        "prompt_cache_key": body.get("prompt_cache_key"),
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
        "tool_names": [tool.get("name") for tool in tools if isinstance(tool, dict)],
        "tool_count": len(tools),
        "tools_have_output_schema": any(
            isinstance(tool, dict) and "output_schema" in tool for tool in tools
        ),
        "input_count": len(input_items),
    }
    session = _codex_session.get()
    if isinstance(session, CodexBackendSession):
        metadata.update(
            {
                "session_id": session.session_id,
                "thread_id": session.thread_id,
                "window_id": session.window_id,
            }
        )
    if headers is not None:
        metadata["codex_header_names"] = sorted(
            name
            for name in headers
            if name
            not in {
                "Authorization",
                "chatgpt-account-id",
            }
        )
    if result is not None:
        metadata["response_id"] = result.response_id
        metadata["response_item_count"] = len(result.response_items)
        if result.usage:
            metadata["usage"] = result.usage
    return metadata


def _is_instruction(item: dict[str, Any]) -> bool:
    return item.get("role") == "system" and bool(item.get("content"))


def _codex_include(reasoning: dict[str, Any] | None) -> list[str]:
    if reasoning is None:
        return []
    return ["reasoning.encrypted_content"]


def _codex_input_item(item: dict[str, Any]) -> dict[str, Any]:
    if "type" in item:
        return _sanitize_response_item(item)
    role = str(item.get("role", "user"))
    return {
        "type": "message",
        "role": role,
        "content": _message_content_items(item.get("content", ""), role),
    }


def _message_content_items(content: Any, role: str) -> list[dict[str, Any]]:
    item_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": item_type, "text": content}]
    if isinstance(content, list):
        items: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                items.append(_sanitize_content_item(item, default_type=item_type))
            else:
                items.append({"type": item_type, "text": str(item)})
        return items
    return [{"type": item_type, "text": str(content)}]


def _codex_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for tool in tools or []:
        item = dict(tool)
        item.pop("output_schema", None)
        cleaned.append(item)
    return cleaned


def _codex_prompt_cache_key() -> str:
    override = os.getenv("CODEX_PROMPT_CACHE_KEY")
    if override:
        return override
    session = _codex_session.get()
    if isinstance(session, CodexBackendSession):
        return session.thread_id
    trace_dir = _trace_dir.get()
    if trace_dir is not None:
        digest = hashlib.sha256(str(trace_dir).encode("utf-8")).hexdigest()[:32]
        return f"harness-{digest}"
    return "harness-complexity"


def _codex_current_installation_id() -> str:
    session = _codex_session.get()
    if isinstance(session, CodexBackendSession):
        return session.installation_id
    return _codex_installation_id()


def _codex_installation_id() -> str:
    override = os.getenv("CODEX_INSTALLATION_ID")
    if override:
        return override
    auth_path = str(_codex_auth_path())
    digest = hashlib.sha256(auth_path.encode("utf-8")).hexdigest()[:32]
    return f"harness-{digest}"


def _new_codex_session(trace_path: Path | None) -> CodexBackendSession | None:
    if trace_path is None:
        return None
    base = str(trace_path.resolve())
    session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"harness-session:{base}"))
    thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"harness-thread:{base}"))
    return CodexBackendSession(
        session_id=session_id,
        thread_id=thread_id,
        window_id=f"{thread_id}:0",
        installation_id=_codex_installation_id(),
    )


def _codex_auth_path() -> Path:
    return Path(os.getenv("CODEX_AUTH_JSON_PATH", Path.home() / ".codex" / "auth.json"))


def _load_codex_auth() -> dict[str, Any]:
    path = _codex_auth_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Codex auth file not found: {path}") from exc


def _fresh_codex_access_token(auth: dict[str, Any]) -> str:
    tokens = auth.get("tokens") or {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    if not isinstance(access, str) or not isinstance(refresh, str):
        raise RuntimeError("Codex auth.json does not contain OAuth tokens")
    expires_at = float(_jwt_payload(access).get("exp", 0))
    if expires_at - time.time() > 60:
        return access
    refreshed = _refresh_codex_token(refresh)
    tokens.update(refreshed)
    auth["tokens"] = tokens
    _codex_auth_path().write_text(json.dumps(auth, indent=2), encoding="utf-8")
    return str(refreshed["access_token"])


def _refresh_codex_token(refresh_token: str) -> dict[str, Any]:
    data = urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_CLIENT_ID,
        }
    ).encode("utf-8")
    request = Request(
        CODEX_TOKEN_URL,
        data=data,
        headers={"content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=TIMEOUT_SEC) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("access_token") or not payload.get("refresh_token"):
        raise RuntimeError("Codex token refresh returned incomplete credentials")
    return payload


def _codex_account_id(access_token: str, auth: dict[str, Any]) -> str:
    claims = _jwt_payload(access_token)
    account_id = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    fallback = (auth.get("tokens") or {}).get("account_id")
    if isinstance(fallback, str) and fallback:
        return fallback
    raise RuntimeError("Codex access token does not contain a ChatGPT account id")


def _jwt_payload(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as exc:
        raise RuntimeError("Failed to decode Codex access token") from exc


def _extract_sse_text(text: str) -> str:
    return _extract_sse_result(text).content


def _extract_sse_result(text: str) -> ToolModelResult:
    chunks: list[str] = []
    streamed_calls: list[ModelToolCall] = []
    streamed_items: list[dict[str, Any]] = []
    completed: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            chunks.append(str(event.get("delta", "")))
        elif event_type in {"response.output_item.done", "response.output_item.added"}:
            item = event.get("item")
            if isinstance(item, dict):
                if event_type == "response.output_item.done":
                    streamed_items.append(_sanitize_response_item(item))
                call = _tool_call_from_mapping(item)
                if call is not None:
                    streamed_calls.append(call)
        elif event_type == "response.completed":
            completed = event.get("response")
    if chunks:
        content = "".join(chunks)
        calls = _merge_tool_calls(
            streamed_calls, _extract_response_dict_tool_calls(completed or {})
        )
        response_items = _response_items_from_response(completed) or streamed_items
        return ToolModelResult(
            content,
            calls,
            response_items=response_items,
            response_id=_response_id_from_response(completed),
            usage=_usage_from_response_dict(completed),
        )
    if completed:
        result = _extract_response_dict_result(completed)
        return ToolModelResult(
            result.content,
            _merge_tool_calls(streamed_calls, result.tool_calls),
            response_items=result.response_items,
            response_id=result.response_id,
            usage=result.usage,
        )
    return ToolModelResult("", _merge_tool_calls(streamed_calls), response_items=streamed_items)


def _extract_response_dict_text(response: dict[str, Any]) -> str:
    return _extract_response_dict_result(response).content


def _extract_response_dict_result(response: dict[str, Any]) -> ToolModelResult:
    chunks: list[str] = []
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return ToolModelResult(
        "\n".join(chunks),
        _extract_response_dict_tool_calls(response),
        response_items=_response_items_from_response(response),
        response_id=_response_id_from_response(response),
        usage=_usage_from_response_dict(response),
    )


def _extract_text(response: Any) -> str:
    return _extract_result(response).content


def _extract_result(response: Any) -> ToolModelResult:
    text = getattr(response, "output_text", None)
    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    if chunks:
        text = "\n".join(chunks)
    return ToolModelResult(
        text if isinstance(text, str) else str(response),
        _extract_tool_calls(response),
        response_items=_response_items_from_obj(response),
        response_id=str(getattr(response, "id", "") or ""),
        usage=_usage_from_response_obj(response),
    )


def _response_id_from_response(response: dict[str, Any] | None) -> str:
    if not isinstance(response, dict):
        return ""
    response_id = response.get("id") or response.get("response_id")
    return str(response_id) if response_id else ""


def _usage_from_response_dict(response: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    return _clean_usage(
        input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
        output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cached_tokens=input_details.get("cached_tokens") or usage.get("cached_tokens"),
    )


def _usage_from_response_obj(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    input_details = getattr(usage, "input_tokens_details", None)
    return _clean_usage(
        input_tokens=getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cached_tokens=(
            getattr(input_details, "cached_tokens", None)
            if input_details is not None
            else getattr(usage, "cached_tokens", None)
        ),
    )


def _clean_usage(
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
    cached_tokens: Any = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for key, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("cached_tokens", cached_tokens),
    ):
        if isinstance(value, bool) or value is None:
            continue
        try:
            usage[key] = int(value)
        except (TypeError, ValueError):
            continue
    return usage


def _response_items_from_response(response: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    return _sanitize_response_items(response.get("output"))


def _response_items_from_obj(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if hasattr(item, "model_dump"):
            dumped = item.model_dump(exclude_none=True)
        elif hasattr(item, "dict"):
            dumped = item.dict(exclude_none=True)
        else:
            dumped = {}
        if isinstance(dumped, dict):
            items.append(_sanitize_response_item(dumped))
    return items


def _sanitize_response_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [_sanitize_response_item(item) for item in items if isinstance(item, dict)]


def _sanitize_response_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "message":
        role = str(item.get("role", "assistant"))
        cleaned = {
            "type": "message",
            "role": role,
            "content": _message_content_items(item.get("content", []), role),
        }
        if item.get("phase") is not None:
            cleaned["phase"] = item["phase"]
        if item.get("reasoning_content") is not None:
            cleaned["reasoning_content"] = str(item.get("reasoning_content") or "")
        return cleaned
    if item_type == "function_call":
        cleaned = {
            "type": "function_call",
            "name": item.get("name", ""),
            "arguments": item.get("arguments", ""),
            "call_id": item.get("call_id", ""),
        }
        if item.get("namespace") is not None:
            cleaned["namespace"] = item["namespace"]
        return cleaned
    if item_type == "local_shell_call":
        cleaned = {"type": "local_shell_call"}
        _copy_present(cleaned, item, ("call_id", "status", "action"))
        return cleaned
    if item_type == "custom_tool_call":
        cleaned = {"type": "custom_tool_call"}
        _copy_present(cleaned, item, ("status", "call_id", "name", "input"))
        return cleaned
    if item_type == "reasoning":
        cleaned = {"type": "reasoning"}
        _copy_present(cleaned, item, ("summary", "content", "encrypted_content"))
        return cleaned
    if item_type == "function_call_output":
        cleaned = {"type": "function_call_output"}
        _copy_present(cleaned, item, ("call_id", "output"))
        return cleaned
    if item_type == "custom_tool_call_output":
        cleaned = {"type": item_type}
        _copy_present(cleaned, item, ("call_id", "name", "output"))
        return cleaned
    cleaned = dict(item)
    cleaned.pop("id", None)
    return cleaned


def _sanitize_content_item(
    item: dict[str, Any], default_type: str = "input_text"
) -> dict[str, Any]:
    item_type = item.get("type") or default_type
    if item_type in {"input_text", "output_text"}:
        return {"type": item_type, "text": item.get("text", "")}
    if item_type == "input_image":
        cleaned = {"type": "input_image", "image_url": item.get("image_url", "")}
        if item.get("detail") is not None:
            cleaned["detail"] = item["detail"]
        return cleaned
    cleaned = dict(item)
    for key in ("id", "status", "annotations", "logprobs"):
        cleaned.pop(key, None)
    return cleaned


def _copy_present(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in source and source[key] is not None:
            target[key] = source[key]


def _extract_tool_calls(response: Any) -> list[ModelToolCall]:
    calls: list[ModelToolCall] = []
    for item in getattr(response, "output", None) or []:
        call = _tool_call_from_obj(item)
        if call is not None:
            calls.append(call)
    return calls


def _extract_response_dict_tool_calls(response: dict[str, Any]) -> list[ModelToolCall]:
    calls: list[ModelToolCall] = []
    for item in response.get("output") or []:
        call = _tool_call_from_mapping(item)
        if call is not None:
            calls.append(call)
    return calls


def _merge_tool_calls(*groups: list[ModelToolCall]) -> list[ModelToolCall]:
    merged: list[ModelToolCall] = []
    seen_with_ids: dict[tuple[str, str], int] = {}
    seen_without_ids: set[tuple[str, str, str]] = set()
    for group in groups:
        for call in group:
            if not call.call_id:
                key = (call.call_id, call.name, call.arguments_text)
                if key in seen_without_ids:
                    continue
                seen_without_ids.add(key)
                merged.append(call)
                continue
            key = (call.call_id, call.name)
            existing_index = seen_with_ids.get(key)
            if existing_index is not None:
                existing = merged[existing_index]
                if _tool_call_detail_score(call) > _tool_call_detail_score(existing):
                    merged[existing_index] = call
                continue
            seen_with_ids[key] = len(merged)
            merged.append(call)
    return merged


def _tool_call_detail_score(call: ModelToolCall) -> tuple[int, int]:
    return (int(bool(call.arguments)), len(call.arguments_text or ""))


def _tool_call_from_obj(item: Any) -> ModelToolCall | None:
    item_type = getattr(item, "type", "")
    call_id = getattr(item, "call_id", None) or getattr(item, "id", "") or ""
    if item_type == "local_shell_call":
        return _make_tool_call(
            "local_shell",
            _local_shell_args_from_obj(item),
            call_id,
        )
    if item_type == "custom_tool_call":
        name = getattr(item, "name", "")
        arguments = getattr(item, "input", None)
        if arguments is None:
            arguments = getattr(item, "arguments", "")
        return _make_custom_tool_call(name, arguments, call_id)
    if item_type not in {"function_call", "tool_call"} and not hasattr(item, "arguments"):
        return None
    name = getattr(item, "name", "")
    arguments_text = getattr(item, "arguments", "") or ""
    return _make_tool_call(name, arguments_text, call_id)


def _tool_call_from_mapping(item: dict[str, Any]) -> ModelToolCall | None:
    item_type = item.get("type", "")
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    call_id = item.get("call_id") or item.get("id") or ""
    if item_type == "local_shell_call":
        return _make_tool_call("local_shell", _local_shell_args_from_mapping(item), str(call_id))
    if item_type == "custom_tool_call":
        name = item.get("name") or ""
        arguments = item["input"] if "input" in item else item.get("arguments", "")
        return _make_custom_tool_call(str(name), arguments, str(call_id))
    if item_type not in {"function_call", "tool_call"} and not function:
        return None
    name = item.get("name") or function.get("name") or ""
    arguments = item.get("arguments")
    if arguments is None:
        arguments = function.get("arguments", "")
    return _make_tool_call(str(name), arguments, str(call_id))


def _local_shell_args_from_obj(item: Any) -> dict[str, Any]:
    args = _coerce_tool_args(getattr(item, "action", None))
    for key in ("command", "cmd", "timeout_sec", "timeout_ms", "duration"):
        value = getattr(item, key, None)
        if value is not None and key not in args:
            args[key] = value
    return args


def _local_shell_args_from_mapping(item: dict[str, Any]) -> dict[str, Any]:
    args = _coerce_tool_args(item.get("action"))
    for key in ("command", "cmd", "timeout_sec", "timeout_ms", "duration"):
        if key in item and key not in args:
            args[key] = item[key]
    return args


def _coerce_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    args: dict[str, Any] = {}
    for key in ("command", "cmd", "timeout_sec", "timeout_ms", "duration"):
        item = getattr(value, key, None)
        if item is not None:
            args[key] = item
    if args:
        return args
    return {"command": str(value)}


def _make_custom_tool_call(name: str, arguments: Any, call_id: str) -> ModelToolCall | None:
    if not name:
        return None
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return ModelToolCall(
                name=name,
                arguments={"input": arguments},
                arguments_text=arguments,
                call_id=call_id,
            )
        return ModelToolCall(
            name=name,
            arguments=parsed if isinstance(parsed, dict) else {"input": parsed},
            arguments_text=arguments,
            call_id=call_id,
        )
    return _make_tool_call(name, arguments, call_id)


def _make_tool_call(name: str, arguments: Any, call_id: str) -> ModelToolCall | None:
    if not name:
        return None
    if isinstance(arguments, str):
        arguments_text = arguments
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed = {}
    else:
        parsed = arguments if isinstance(arguments, dict) else {}
        arguments_text = json.dumps(parsed, sort_keys=True)
    return ModelToolCall(
        name=name, arguments=parsed, arguments_text=arguments_text, call_id=call_id
    )
