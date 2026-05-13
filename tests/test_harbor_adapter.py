from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from plumbing.base_agent import load_harness
from plumbing.harbor_adapter import HarborHarnessAgent


def test_load_harness_from_candidate_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "class H(BaseHarness):\n"
        "    def solve(self, task):\n"
        "        return 'echo workspace'\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    assert load_harness(candidate_dir=workspace).solve(SimpleNamespace()) == "echo workspace"


def test_harbor_agent_executes_candidate_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "class H(BaseHarness):\n"
        "    def solve(self, task):\n"
        "        return 'echo ok'\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    context = SimpleNamespace(metadata=None)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, context))
    assert env.commands == ["echo ok"]
    assert context.metadata["return_code"] == 0
    assert (tmp_path / "logs" / "harness-result.json").exists()


class FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str):
        self.commands.append(command)
        return SimpleNamespace(stdout="ok\n", stderr="", return_code=0)
