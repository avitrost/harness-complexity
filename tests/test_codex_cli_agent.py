from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.run_codex_cli import run_codex_cli_split
from plumbing.codex_cli_agent import CODEX_CLI_AGENT_IMPORT_PATH, CodexCliAgent

TASK_CASES = [
    ("fix-git", "Recover the lost git changes and make the tests pass."),
    ("sqlite-db-truncate", "Repair the SQLite truncation logic and verify the database."),
    ("pytorch-model-cli", "Fix the PyTorch model CLI so inference works end to end."),
]


def test_codex_cli_agent_uploads_auth_and_runs_codex(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens":{}}\n', encoding="utf-8")
    env = FakeEnvironment()
    context = SimpleNamespace(metadata=None)
    agent = CodexCliAgent(
        logs_dir=tmp_path / "logs",
        codex_model="gpt-test",
        codex_reasoning_effort="none",
        host_codex_auth_path=str(auth),
        timeout_sec=123,
    )

    asyncio.run(agent.setup(env))
    asyncio.run(agent.run("Solve the task.", env, context))

    assert (str(auth), "/root/.codex/auth.json") in env.uploads
    assert (str(tmp_path / "logs" / "prompt.txt"), "/tmp/codex-task-prompt.txt") in env.uploads
    codex_exec = [call for call in env.execs if "codex exec" in call.command][0]
    assert "--model gpt-test" in codex_exec.command
    assert "--dangerously-bypass-approvals-and-sandbox" in codex_exec.command
    assert 'model_reasoning_effort="none"' in codex_exec.command
    assert "setsid bash -lc" in codex_exec.command
    assert "kill -TERM" in codex_exec.command
    assert codex_exec.env["CODEX_HOME"] == "/root/.codex"
    assert codex_exec.timeout_sec == 123
    assert context.metadata["return_code"] == 0


@pytest.mark.parametrize(("task_name", "instruction"), TASK_CASES)
def test_codex_cli_agent_passes_task_prompt_verbatim(
    tmp_path: Path, task_name: str, instruction: str
) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens":{}}\n', encoding="utf-8")
    env = FakeEnvironment()
    agent = CodexCliAgent(
        logs_dir=tmp_path / task_name,
        codex_model="gpt-test",
        host_codex_auth_path=str(auth),
    )

    asyncio.run(agent.setup(env))
    asyncio.run(agent.run(instruction, env, SimpleNamespace(metadata=None)))

    assert env.uploaded_text["/tmp/codex-task-prompt.txt"] == instruction
    codex_exec = [call for call in env.execs if "codex exec" in call.command][0]
    assert "< /tmp/codex-task-prompt.txt" in codex_exec.command
    assert codex_exec.env["HOME"] == "/root/.codex"


def test_run_codex_cli_split_builds_harbor_command(tmp_path: Path) -> None:
    summary = run_codex_cli_split(
        split="val",
        out_dir=tmp_path / "out",
        tasks=["fix-git"],
        trials=1,
        concurrency=1,
        backend="slurm-pyxis",
        codex_model="gpt-test",
        codex_reasoning_effort="none",
        timeout_sec=99,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    command = summary["command"]
    assert CODEX_CLI_AGENT_IMPORT_PATH in command
    assert "codex_model=gpt-test" in command
    assert "codex_reasoning_effort=none" in command
    assert "timeout_sec=99" in command
    assert "--environment-import-path" in command
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["dry_run"] is True


def test_run_codex_cli_split_includes_multiple_tasks(tmp_path: Path) -> None:
    tasks = [task for task, _ in TASK_CASES]
    summary = run_codex_cli_split(
        split="val",
        out_dir=tmp_path / "out",
        tasks=tasks,
        trials=2,
        concurrency=3,
        backend="slurm-pyxis",
        codex_model="gpt-test",
        codex_reasoning_effort="none",
        timeout_sec=99,
        dry_run=True,
        harbor_bin="harbor",
        harbor_help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    command = summary["command"]
    assert command.count("--include-task-name") == len(tasks)
    for task in tasks:
        index = command.index(task)
        assert command[index - 1] == "--include-task-name"
    assert command[command.index("--n-attempts") + 1] == "2"
    assert command[command.index("--n-concurrent") + 1] == "3"


class FakeExecCall(SimpleNamespace):
    command: str
    env: dict[str, str]
    timeout_sec: int | None


class FakeEnvironment:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.execs: list[FakeExecCall] = []
        self.uploaded_text: dict[str, str] = {}

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append((str(source_path), target_path))
        self.uploaded_text[target_path] = Path(source_path).read_text(encoding="utf-8")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        self.execs.append(FakeExecCall(command=command, env=env or {}, timeout_sec=timeout_sec))
        if command.startswith("cat "):
            return SimpleNamespace(stdout="final answer", stderr="", return_code=0)
        return SimpleNamespace(stdout="ok", stderr="", return_code=0)
