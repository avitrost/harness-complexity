from __future__ import annotations

import base64
import contextvars
import json
import os
import time
from collections.abc import Callable
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


def set_client_factory(factory: Callable[[], OpenAI] | None) -> None:
    global _client_factory
    _client_factory = factory


def set_trace_dir(path: Path | str | None) -> object:
    _trace_count.set(0)
    return _trace_dir.set(Path(path) if path is not None else None)


def reset_trace_dir(token: object) -> None:
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


def call_terminal_model(messages: list[dict[str, str]]) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle_if_requested()
            if using_codex_auth():
                text = _call_codex_backend(messages)
                _write_model_trace(messages, text)
                return text
            response = _make_client().responses.create(
                model=terminal_model(),
                input=messages,
                reasoning={"effort": terminal_reasoning_effort()},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=TIMEOUT_SEC,
            )
            text = _extract_text(response)
            _write_model_trace(messages, text)
            return text
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= MAX_RETRIES:
                _write_model_trace(messages, "", error=str(exc))
                break
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("terminal model call failed") from last_error


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
    messages: list[dict[str, str]],
    response_text: str,
    error: str | None = None,
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
    if error:
        payload["error"] = error
    (trace_dir / f"model-call-{index:02d}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _call_codex_backend(messages: list[dict[str, str]]) -> str:
    auth = _load_codex_auth()
    access_token = _fresh_codex_access_token(auth)
    account_id = _codex_account_id(access_token, auth)
    body = _codex_body(messages)
    request = Request(
        CODEX_BASE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "accept": "text/event-stream",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SEC) as response:
            return _extract_sse_text(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"codex backend call failed: {exc.code} {detail[:500]}") from exc


def _codex_body(messages: list[dict[str, str]]) -> dict[str, Any]:
    instructions = "\n\n".join(
        item.get("content", "")
        for item in messages
        if item.get("role") in {"system", "developer"} and item.get("content")
    )
    input_messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in messages
        if item.get("role") not in {"system", "developer"}
    ]
    return {
        "model": terminal_model(),
        "reasoning": {"effort": terminal_reasoning_effort()},
        "instructions": instructions or "You are a concise assistant.",
        "input": input_messages or [{"role": "user", "content": ""}],
        "stream": True,
        "store": False,
    }


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
    chunks: list[str] = []
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
        elif event_type == "response.completed":
            completed = event.get("response")
    if chunks:
        return "".join(chunks)
    if completed:
        return _extract_response_dict_text(completed)
    return ""


def _extract_response_dict_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    if chunks:
        return "\n".join(chunks)
    return str(response)
