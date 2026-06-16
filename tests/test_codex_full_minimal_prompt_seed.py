from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from plumbing.openai_client import set_client_factory
from plumbing.types import TaskContext
from scripts.audit_candidate import audit_candidate

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "codex_full_minimal_prompt" / "harness.py"


def test_codex_full_minimal_prompt_seed_validates() -> None:
    result = audit_candidate(SEED_PATH)

    assert result["ok"], result["errors"]


def test_codex_full_minimal_prompt_changes_only_base_instructions(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_seed()
    fake = RecordingToolOpenAI([SimpleNamespace(output_text="done", output=[])])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("Fix the task.", working_dir="/app/task"), []
        )
    finally:
        set_client_factory(None)

    messages = fake.calls[0]["input"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are a coding agent working in a terminal.")
    assert "## Task execution" in messages[0]["content"]
    assert "expected to be precise, safe, and helpful" not in messages[0]["content"]
    assert messages[1]["role"] == "developer"
    assert messages[2]["role"] == "user"
    assert fake.calls[0]["parallel_tool_calls"] is True
    assert [tool["name"] for tool in fake.calls[0]["tools"]] == [
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
    ]
    assert turn.done is True


class RecordingToolOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _load_seed() -> Any:
    spec = importlib.util.spec_from_file_location("codex_full_minimal_prompt", SEED_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_full_minimal_prompt"] = module
    spec.loader.exec_module(module)
    return module
