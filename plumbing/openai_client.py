from __future__ import annotations

import time
from collections.abc import Callable
import os
from typing import Any

from openai import OpenAI

from plumbing.secrets import require_openai_api_key

TERMINAL_MODEL = "gpt-5.4-mini"
TIMEOUT_SEC = 120.0
MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 4096
PREFLIGHT_OUTPUT_TOKENS = 16

_client_factory: Callable[[], OpenAI] | None = None
_last_request_at = 0.0


def set_client_factory(factory: Callable[[], OpenAI] | None) -> None:
    global _client_factory
    _client_factory = factory


def _make_client() -> OpenAI:
    if _client_factory is not None:
        return _client_factory()
    return OpenAI(api_key=require_openai_api_key(), timeout=TIMEOUT_SEC, max_retries=MAX_RETRIES)


def call_terminal_model(messages: list[dict[str, str]]) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle_if_requested()
            response = _make_client().responses.create(
                model=TERMINAL_MODEL,
                input=messages,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=TIMEOUT_SEC,
            )
            return _extract_text(response)
        except Exception as exc:  # pragma: no cover - real API path
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("terminal model call failed") from last_error


def check_terminal_model_available() -> None:
    _throttle_if_requested()
    response = _make_client().responses.create(
        model=TERMINAL_MODEL,
        input=[{"role": "user", "content": "Reply OK."}],
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
