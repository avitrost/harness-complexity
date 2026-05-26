from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import set_client_factory
from plumbing.types import TaskContext

ROOT = Path(__file__).resolve().parents[1]
COMPRESSED_PATH = ROOT / "seeds" / "codex_compressed" / "harness.py"
FULL_PATH = ROOT / "seeds" / "codex_full" / "harness.py"


def test_codex_compressed_seed_is_standalone_source() -> None:
    text = COMPRESSED_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert "seeds.codex_full" not in text
    assert "from seeds" not in text
    assert len(lines) > 100
    assert len(lines) < len(FULL_PATH.read_text(encoding="utf-8").splitlines())
    assert max(map(len, lines)) <= 100


def test_codex_compressed_seed_is_black_formatted_and_validates() -> None:
    formatted = subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--check",
            "--line-length",
            "100",
            str(COMPRESSED_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert formatted.returncode == 0, formatted.stdout + formatted.stderr

    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E501",
            str(COMPRESSED_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr

    result = validate_candidate(COMPRESSED_PATH, max_lines=1660)

    assert result["ok"], result


def test_codex_compressed_matches_codex_full_default_behavior(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    compressed = _load_module(COMPRESSED_PATH)
    full = _load_module(FULL_PATH)

    assert compressed.CODEX_BASE_INSTRUCTIONS == full.CODEX_BASE_INSTRUCTIONS
    assert compressed.APPLY_PATCH_GRAMMAR == full.APPLY_PATCH_GRAMMAR
    assert compressed.SUMMARIZATION_PROMPT == full.SUMMARIZATION_PROMPT
    assert compressed.SUMMARY_PREFIX == full.SUMMARY_PREFIX
    assert compressed.PORT_PARITY_MANIFEST == full.PORT_PARITY_MANIFEST
    assert compressed._built_tools() == full._built_tools()

    compressed_call, compressed_turn = _run_single_turn(compressed)
    full_call, full_turn = _run_single_turn(full)

    assert compressed_call == full_call
    assert compressed_turn.tool_calls == full_turn.tool_calls
    assert compressed_turn.metadata == full_turn.metadata
    assert compressed_turn.done == full_turn.done


def test_codex_compressed_and_full_sanitize_parent_find(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    compressed = _load_module(COMPRESSED_PATH)
    full = _load_module(FULL_PATH)

    compressed_turn = _run_command_turn(compressed, "pwd && find .. -name AGENTS.md -print")
    full_turn = _run_command_turn(full, "pwd && find .. -name AGENTS.md -print")

    assert compressed_turn.tool_calls[0].arguments["cmd"] == "pwd && find . -name AGENTS.md -print"
    assert (
        full_turn.tool_calls[0].arguments["cmd"] == compressed_turn.tool_calls[0].arguments["cmd"]
    )


def test_codex_compressed_and_full_recover_on_preamble_without_tool(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    compressed = _load_module(COMPRESSED_PATH)
    full = _load_module(FULL_PATH)

    compressed_turn = _run_text_turn(compressed, "I'll inspect the repo first.")
    full_turn = _run_text_turn(full, "I'll inspect the repo first.")

    assert compressed_turn.done is False
    assert full_turn.done is False
    assert compressed_turn.tool_calls[0].name == "exec_command"
    assert full_turn.tool_calls == compressed_turn.tool_calls


def _run_single_turn(module):
    return _run_command_turn(module, "pwd", include_call=True)


def _run_text_turn(module, text: str):
    fake = RecordingToolOpenAI([SimpleNamespace(output_text=text, output=[])])
    set_client_factory(lambda: fake)
    try:
        return module.create_agent().next_command(
            TaskContext("List files.", working_dir="/app"), []
        )
    finally:
        set_client_factory(None)


def _run_command_turn(module, command: str, include_call: bool = False):
    fake = RecordingToolOpenAI(
        [
            SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="exec_command",
                        arguments=json.dumps({"cmd": command, "yield_time_ms": 1000}),
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("List files.", working_dir="/app/task"),
            [],
        )
    finally:
        set_client_factory(None)
    if include_call:
        return fake.calls[0], turn
    return turn


def _load_module(path: Path):
    name = f"seed_{path.parent.name}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingToolOpenAI:
    def __init__(self, responses: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)
