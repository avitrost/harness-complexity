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

from plumbing.secrets import require_openai_api_key

TERMINAL_MODEL = "gpt-5.4-mini"
TERMINAL_REASONING_EFFORT = "none"
TIMEOUT_SEC = 120.0
MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 4096
PREFLIGHT_OUTPUT_TOKENS = 16
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
    return OpenAI(api_key=require_openai_api_key(), timeout=TIMEOUT_SEC, max_retries=MAX_RETRIES)


def using_codex_auth() -> bool:
    return os.getenv("OPENAI_AUTH_MODE", "").lower() == "codex"


def terminal_model() -> str:
    return os.getenv("OPENAI_TERMINAL_MODEL", TERMINAL_MODEL)


def terminal_reasoning_effort() -> str:
    return TERMINAL_REASONING_EFFORT


def call_terminal_model(messages: list[dict[str, Any]]) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle_if_requested()
            started_at = time.monotonic()
            if using_codex_auth():
                result = _call_codex_backend_result(messages)
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
            text = _extract_text(response)
            _write_model_trace(messages, text, duration_sec=time.monotonic() - started_at)
            return text
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= MAX_RETRIES:
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
    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle_if_requested()
            started_at = time.monotonic()
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
            )
            return result
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= MAX_RETRIES:
                _write_model_trace(messages, "", error=str(exc))
                break
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("terminal model tool call failed") from last_error


def check_terminal_model_available() -> None:
    _throttle_if_requested()
    if using_codex_auth():
        _call_codex_backend([{"role": "user", "content": "Reply OK."}])
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


def _retry_delay(exc: Exception, attempt: int) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    if "rate limit" in str(exc).lower() or "429" in str(exc):
        return 65.0
    return float(2**attempt)


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
        "model": terminal_model(),
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
        raise RuntimeError(f"codex backend call failed: {exc.code} {detail[:500]}") from exc


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
        if item.get("role") in {"system", "developer"} and item.get("content")
    )
    input_messages = [_codex_input_item(item) for item in messages if not _is_instruction(item)]
    body: dict[str, Any] = {
        "model": terminal_model(),
        "reasoning": {"effort": terminal_reasoning_effort()},
        "instructions": instructions or "You are a concise assistant.",
        "input": input_messages or [_codex_input_item({"role": "user", "content": ""})],
        "tools": _codex_tools(tools),
        "tool_choice": "auto",
        "stream": True,
        "store": False,
        "include": [],
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
    return metadata


def _is_instruction(item: dict[str, Any]) -> bool:
    return item.get("role") in {"system", "developer"} and bool(item.get("content"))


def _codex_input_item(item: dict[str, Any]) -> dict[str, Any]:
    if "type" in item:
        return dict(item)
    role = item.get("role", "user")
    content = item.get("content", "")
    if isinstance(content, str):
        item_type = "output_text" if role == "assistant" else "input_text"
        content = [{"type": item_type, "text": content}]
    return {"role": role, "content": content}


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
        )
    if completed:
        result = _extract_response_dict_result(completed)
        return ToolModelResult(
            result.content,
            _merge_tool_calls(streamed_calls, result.tool_calls),
            response_items=result.response_items,
            response_id=result.response_id,
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
    )


def _response_id_from_response(response: dict[str, Any] | None) -> str:
    if not isinstance(response, dict):
        return ""
    response_id = response.get("id") or response.get("response_id")
    return str(response_id) if response_id else ""


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
    cleaned = dict(item)
    cleaned.pop("id", None)
    return cleaned


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
