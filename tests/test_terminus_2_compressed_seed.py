from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from evaluator.validate_candidate import validate_candidate
from plumbing.openai_client import set_client_factory
from plumbing.types import CommandResult, TaskContext

ROOT = Path(__file__).resolve().parents[1]
COMPRESSED_PATH = ROOT / "seeds" / "terminus_2_compressed" / "harness.py"


def test_terminus_2_compressed_seed_is_standalone_source() -> None:
    text = COMPRESSED_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert "harbor.agents.terminus_2" not in text
    assert "references.terminus_kira" not in text
    assert len(lines) < 650
    assert max(map(len, lines)) <= 100


def test_terminus_2_compressed_seed_is_black_formatted_and_validates() -> None:
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

    result = validate_candidate(COMPRESSED_PATH, max_lines=650)

    assert result["ok"], result


def test_terminus_2_compressed_uses_official_prompt_templates() -> None:
    module = _load_module(COMPRESSED_PATH)
    from harbor.agents.terminus_2 import Terminus2

    root = Path(inspect.getfile(Terminus2)).parent / "templates"
    assert module.TERMINUS_JSON_PLAIN_TEMPLATE == (root / "terminus-json-plain.txt").read_text()
    assert module.TERMINUS_TIMEOUT_TEMPLATE == (root / "timeout.txt").read_text()


def test_terminus_2_compressed_parser_matches_official_parser() -> None:
    module = _load_module(COMPRESSED_PATH)
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    responses = [
        '{"analysis":"state","plan":"run","commands":[{"keystrokes":"pwd\\n","duration":0.1}]}',
        'preface {"analysis":"done","plan":"grade","commands":[],"task_complete":"true"} tail',
        '{"analysis":"x","plan":"y","commands":[{"keystrokes":"pytest -q\\n"}]}',
    ]
    official = TerminusJSONPlainParser()
    local = module.TerminusJSONPlainParser()
    for response in responses:
        expected = official.parse_response(response)
        actual = local.parse_response(response)
        assert [(c.keystrokes, c.duration) for c in actual.commands] == [
            (c.keystrokes, c.duration) for c in expected.commands
        ]
        assert actual.is_task_complete == expected.is_task_complete
        assert actual.error == expected.error
        assert actual.warning == expected.warning
        assert actual.analysis == expected.analysis
        assert actual.plan == expected.plan


def test_terminus_2_compressed_turn_uses_official_initial_prompt(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_module(COMPRESSED_PATH)
    response = json.dumps(
        {
            "analysis": "Need inspect.",
            "plan": "Run pwd.",
            "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
        }
    )
    fake = RecordingOpenAI([response])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("List files.", working_dir="/app"),
            [],
        )
    finally:
        set_client_factory(None)

    call = fake.calls[0]
    assert call["input"] == [
        {
            "role": "user",
            "content": module.TERMINUS_JSON_PLAIN_TEMPLATE.format(
                instruction="List files.",
                terminal_state="$ pwd\n/app\n",
            ),
        }
    ]
    assert turn.done is False
    assert len(turn.tool_calls) == 1
    tool = turn.tool_calls[0]
    assert tool.name == "exec_command"
    assert tool.arguments["cmd"] == "pwd"
    assert tool.arguments["workdir"] == "/app"
    assert tool.arguments["yield_time_ms"] == 100
    assert tool.arguments["tty"] is True
    assert turn.metadata["terminus_analysis"] == "Need inspect."


def test_terminus_2_compressed_uses_persistent_terminal_when_available(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_module(COMPRESSED_PATH)
    response = json.dumps(
        {
            "analysis": "Need inspect.",
            "plan": "Use the pane.",
            "commands": [
                {"keystrokes": "cd src\n", "duration": 0.1},
                {"keystrokes": "pwd\n", "duration": 0.2},
            ],
        }
    )
    fake = RecordingOpenAI([response])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext(
                "List files.",
                working_dir="/app",
                metadata={
                    "persistent_terminal": {
                        "available": True,
                        "session_id": 42,
                        "initial_output": "$ ",
                    }
                },
            ),
            [],
        )
    finally:
        set_client_factory(None)

    assert fake.calls[0]["input"][0]["content"] == module.TERMINUS_JSON_PLAIN_TEMPLATE.format(
        instruction="List files.",
        terminal_state="$ ",
    )
    tool = turn.tool_calls[0]
    assert tool.name == "write_stdin"
    assert tool.arguments["session_id"] == 42
    assert tool.arguments["commands"] == [
        {"chars": "cd src\n", "yield_time_ms": 100},
        {"chars": "pwd\n", "yield_time_ms": 200},
    ]


def test_terminus_2_compressed_prefers_tmux_terminal_when_available(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_module(COMPRESSED_PATH)
    response = json.dumps(
        {
            "analysis": "Need inspect.",
            "plan": "Use tmux.",
            "commands": [{"keystrokes": "C-c", "duration": 0.1}],
        }
    )
    fake = RecordingOpenAI([response])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext(
                "Interrupt.",
                metadata={
                    "persistent_terminal": {
                        "available": True,
                        "backend": "tmux",
                        "session_name": "term-1",
                        "initial_output": "Current Terminal Screen:\n$ ",
                    }
                },
            ),
            [],
        )
    finally:
        set_client_factory(None)

    assert module.create_agent().wants_persistent_terminal == "tmux"
    tool = turn.tool_calls[0]
    assert tool.name == "write_stdin"
    assert tool.arguments["session_name"] == "term-1"
    assert tool.arguments["commands"] == [{"chars": "C-c", "yield_time_ms": 100}]


def test_terminus_2_compressed_confirms_completion_inside_turn(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_module(COMPRESSED_PATH)
    done = json.dumps(
        {
            "analysis": "Solved.",
            "plan": "Finish.",
            "commands": [],
            "task_complete": True,
        }
    )
    fake = RecordingOpenAI([done, done])
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(TaskContext("Fix it.", working_dir="/app"), [])
    finally:
        set_client_factory(None)

    assert turn.done is True
    assert len(fake.calls) == 2
    assert fake.calls[1]["input"][1] == {"role": "assistant", "content": done}
    assert (
        "Are you sure you want to mark the task as complete?"
        in fake.calls[1]["input"][2]["content"]
    )


def test_terminus_2_compressed_replays_pending_completion(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    module = _load_module(COMPRESSED_PATH)
    done = json.dumps(
        {
            "analysis": "Verified.",
            "plan": "Confirm.",
            "commands": [],
            "task_complete": True,
        }
    )
    fake = RecordingOpenAI([done])
    history = [
        CommandResult(
            command="pytest -q",
            return_code=0,
            stdout="1 passed\n",
            metadata={
                "assistant_content": '{"analysis":"ok","plan":"verify","commands":[{"keystrokes":"pytest -q\\n","duration":1}],"task_complete":true}',
                "terminus_task_complete_requested": True,
            },
        )
    ]
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext("Fix it.", working_dir="/app"), history
        )
    finally:
        set_client_factory(None)

    assert turn.done is True
    assert (
        "Are you sure you want to mark the task as complete?"
        in fake.calls[0]["input"][-1]["content"]
    )


def test_terminus_2_compressed_uses_terminus_style_summarization(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    monkeypatch.setenv("TERMINUS_CONTEXT_LIMIT_TOKENS", "20")
    monkeypatch.setenv("TERMINUS_PROACTIVE_SUMMARIZATION_THRESHOLD", "10")
    module = _load_module(COMPRESSED_PATH)
    main = json.dumps(
        {
            "analysis": "Continue.",
            "plan": "Inspect.",
            "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
        }
    )
    fake = RecordingOpenAI(["summary text", "questions text", "answers text", main])
    history = [
        CommandResult(
            command="write_stdin(tmux_session=s, commands=1)",
            return_code=None,
            stdout="New Terminal Output:\n" + "x" * 400,
            tool_name="write_stdin",
            metadata={"assistant_content": '{"analysis":"old","plan":"old","commands":[]}'},
        )
    ]
    set_client_factory(lambda: fake)
    try:
        turn = module.create_agent().next_command(
            TaskContext(
                "Fix it.",
                metadata={
                    "persistent_terminal": {
                        "available": True,
                        "backend": "tmux",
                        "session_name": "term-1",
                        "initial_output": "Current Terminal Screen:\n$ ",
                    }
                },
            ),
            history,
        )
    finally:
        set_client_factory(None)

    assert len(fake.calls) == 4
    assert "You are about to hand off your work" in fake.calls[0]["input"][-1]["content"]
    assert "Please begin by asking several questions" in fake.calls[1]["input"][0]["content"]
    assert "The next agent has a few questions" in fake.calls[2]["input"][-1]["content"]
    assert "Here are the answers the other agent provided" in fake.calls[3]["input"][-1]["content"]
    assert turn.metadata["terminus_summarization_count"] == 1
    assert turn.metadata["terminus_compacted"] is True
    assert turn.tool_calls[0].arguments["session_name"] == "term-1"


def _load_module(path: Path):
    name = f"seed_{path.parent.name}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingOpenAI:
    def __init__(self, responses: list[str]) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self._responses.pop(0))
