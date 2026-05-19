from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from plumbing.base_agent import load_harness
from plumbing.harbor_adapter import (
    HarborHarnessAgent,
    HarborRunSpec,
    build_harbor_command,
)
from plumbing.openai_client import set_client_factory


def test_load_harness_from_candidate_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(command='echo workspace')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    turn = load_harness(candidate_dir=workspace).next_command(SimpleNamespace(), [])
    assert turn.command == "echo workspace"


def test_harbor_agent_executes_candidate_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(command='echo ok')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    context = SimpleNamespace(metadata=None)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, context))
    assert env.commands == ["echo ok"]
    assert context.metadata["last_return_code"] == 0
    assert context.metadata["turns"] == 1
    assert (tmp_path / "logs" / "harness-result.json").exists()


def test_harbor_agent_forwards_candidate_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(command='sleep 10', timeout_sec=3)\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))
    assert env.commands == ["sleep 10"]
    assert env.timeouts == [3]


def test_harbor_agent_logs_model_call_traces(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    set_client_factory(lambda: FakeOpenAI("ok"))
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.openai_client import call_terminal_model\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        text = call_terminal_model([{'role': 'user', 'content': 'next?'}])\n"
        "        return HarnessTurn(command=f'echo {text}')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    try:
        agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
        asyncio.run(agent.run("instruction", FakeEnvironment(), SimpleNamespace(metadata=None)))
    finally:
        set_client_factory(None)
    trace = (tmp_path / "logs" / "model-call-01.json").read_text(encoding="utf-8")
    assert '"content": "next?"' in trace
    assert '"response": "ok"' in trace


def test_harbor_agent_writes_partial_result_on_exec_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(command='sleep forever')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)
    try:
        asyncio.run(agent.run("instruction", FailingEnvironment(), context))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected exec failure")

    assert (tmp_path / "logs" / "harness-result.json").exists()
    assert context.metadata["done"] is False


def test_slurm_command_uses_longer_environment_start_multiplier() -> None:
    plan = build_harbor_command(
        HarborRunSpec(
            Path("candidate"),
            Path("out"),
            ["fix-git"],
            trials=1,
            concurrency=1,
            split="val",
            backend="slurm-pyxis",
        ),
        executable="harbor",
        help_text="--dataset --include-task-name --n-attempts --n-concurrent",
    )

    index = plan.command.index("--environment-build-timeout-multiplier")
    assert plan.command[index + 1] == "18"


class FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.timeouts: list[int | None] = []

    async def exec(self, command: str, timeout_sec: int | None = None):
        self.commands.append(command)
        self.timeouts.append(timeout_sec)
        return SimpleNamespace(stdout="ok\n", stderr="", return_code=0)


class FailingEnvironment:
    async def exec(self, command: str, timeout_sec: int | None = None):
        raise RuntimeError("exec failed")


class FakeOpenAI:
    def __init__(self, text: str) -> None:
        self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=text))
