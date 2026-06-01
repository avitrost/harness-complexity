from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse
from harbor.models.metric import UsageInfo

import plumbing.openai_client as openai_client
from plumbing.openai_client import CodexBackendError, using_codex_auth

TERMINUS_2_AGENT_IMPORT_PATH = "harbor.agents.terminus_2:Terminus2"
TERMINUS_2_CODEX_AUTH_AGENT_IMPORT_PATH = "plumbing.terminus_2_agent:CodexAuthTerminus2"
DEFAULT_TERMINUS_2_PARSER_NAME = "json"
DEFAULT_TERMINUS_2_REASONING_EFFORT = "none"


class CodexAuthTerminus2(Terminus2):
    def _init_llm(self, **kwargs: Any) -> BaseLLM:
        return CodexAuthLLM(
            model_name=kwargs["model_name"],
            model_info=kwargs.get("model_info"),
        )


class CodexAuthLLM(BaseLLM):
    def __init__(self, model_name: str, model_info: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_info = model_info or {}

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Any] | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = _messages(message_history or [])
        messages.append({"role": "user", "content": prompt})
        try:
            result = await asyncio.to_thread(self._call, messages)
        except CodexBackendError as exc:
            if _looks_like_context_error(str(exc.detail)):
                raise ContextLengthExceededError from exc
            raise
        usage = _usage(result.usage)
        if logging_path is not None:
            logging_path.write_text(
                json.dumps(
                    {
                        "model": self.model_name,
                        "messages": messages,
                        "response": result.content,
                        "request_metadata": result.request_metadata,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return LLMResponse(
            content=result.content,
            model_name=self.model_name,
            usage=usage,
            response_id=result.response_id or None,
            extra={"codex_response_items": len(result.response_items)},
        )

    def _call(self, messages: list[dict[str, Any]]) -> openai_client.ToolModelResult:
        max_retries = openai_client._max_retries()
        for attempt in range(max_retries + 1):
            try:
                with _temporary_env("OPENAI_TERMINAL_MODEL", self.model_name):
                    return openai_client._call_codex_backend_result(messages)
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                time.sleep(openai_client._retry_delay(exc, attempt))
        raise RuntimeError("unreachable")

    def get_model_context_limit(self) -> int:
        value = self.model_info.get("max_input_tokens") or self.model_info.get("max_tokens")
        return _positive_int(value) or 1_000_000

    def get_model_output_limit(self) -> int | None:
        return _positive_int(self.model_info.get("max_output_tokens"))


def terminus_2_agent_import_path() -> str:
    if using_codex_auth():
        return TERMINUS_2_CODEX_AUTH_AGENT_IMPORT_PATH
    return TERMINUS_2_AGENT_IMPORT_PATH


def _messages(history: list[dict[str, Any] | Any]) -> list[dict[str, Any]]:
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = item.get("content", "")
        else:
            role = str(getattr(item, "role", "user"))
            content = getattr(item, "content", "")
        messages.append({"role": role, "content": content})
    return messages


def _usage(data: dict[str, Any]) -> UsageInfo | None:
    if not data:
        return None
    return UsageInfo(
        prompt_tokens=_positive_int(data.get("input_tokens")) or 0,
        completion_tokens=_positive_int(data.get("output_tokens")) or 0,
        cache_tokens=_positive_int(data.get("cached_tokens")) or 0,
        cost_usd=0.0,
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _looks_like_context_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "context length",
            "context_length",
            "maximum context",
            "too many tokens",
            "input is too long",
        )
    )


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
