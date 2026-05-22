from __future__ import annotations

import asyncio
import json
import subprocess
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


def test_load_harness_from_direct_harness_directory(tmp_path: Path) -> None:
    direct = tmp_path / "seed"
    direct.mkdir()
    (direct / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(command='echo direct')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )

    turn = load_harness(candidate_dir=direct).next_command(SimpleNamespace(), [])

    assert turn.command == "echo direct"


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


def test_harbor_agent_executes_candidate_tool_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(\n"
        "            HarnessToolCall('local_shell', {'command': 'echo one'}, 'call_1'),\n"
        "            HarnessToolCall('execute_commands', {'commands': [{'keystrokes': 'echo two', 'timeout_sec': 5}]}, 'call_2'),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == ["echo one", "echo two"]
    assert env.timeouts == [None, 5]
    payload = json.loads((tmp_path / "logs" / "harness-turn-02.json").read_text())
    assert payload["tool_name"] == "execute_commands"
    assert payload["tool_call_id"] == "call_2"


def test_harbor_agent_marks_codex_parallel_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "ITEMS = [\n"
        "    {'type': 'function_call', 'call_id': 'call_1', 'name': 'exec_command', 'arguments': '{\"cmd\":\"echo one\"}'},\n"
        "    {'type': 'function_call', 'call_id': 'call_2', 'name': 'exec_command', 'arguments': '{\"cmd\":\"echo two\"}'},\n"
        "]\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(\n"
        "            HarnessToolCall('exec_command', {'cmd': 'echo one'}, 'call_1'),\n"
        "            HarnessToolCall('exec_command', {'cmd': 'echo two'}, 'call_2'),\n"
        "        ), metadata={'codex_response_items': ITEMS})\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = UnifiedEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    first = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    second = json.loads((tmp_path / "logs" / "harness-turn-02.json").read_text())
    assert "codex_response_items" in first["metadata"]
    assert second["metadata"]["codex_output_only"] is True


def test_harbor_agent_executes_codex_exec_command_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(\n"
        "            HarnessToolCall('exec_command', {'cmd': 'pwd', 'workdir': 'src', 'timeout_ms': 1500}, 'call_1'),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == ["cd src && pwd"]
    assert env.timeouts == [2]


def test_harbor_agent_uses_unified_exec_command_when_available(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if len(history) == 0:\n"
        "            return HarnessTurn(tool_calls=(HarnessToolCall('exec_command', {\n"
        "                'cmd': 'python -i', 'workdir': '/app/src', 'shell': '/bin/sh',\n"
        "                'login': True, 'tty': True, 'yield_time_ms': 250,\n"
        "                'max_output_tokens': 50,\n"
        "            }, 'call_1'),))\n"
        "        if len(history) == 1:\n"
        "            return HarnessTurn(tool_calls=(HarnessToolCall('write_stdin', {\n"
        "                'session_id': 9, 'chars': 'exit()\\n', 'yield_time_ms': 100,\n"
        "            }, 'call_2'),))\n"
        "        return HarnessTurn(done=True)\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = UnifiedEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.exec_calls == [
        {
            "command": "python -i",
            "cwd": "/app/src",
            "timeout_sec": None,
            "shell": "/bin/sh",
            "login": True,
            "tty": True,
            "yield_time_ms": 250,
            "max_output_tokens": 50,
        }
    ]
    assert env.stdin_calls == [
        {
            "session_id": 9,
            "chars": "exit()\n",
            "yield_time_ms": 100,
            "max_output_tokens": None,
        }
    ]
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["metadata"]["unified_exec"]["session_id"] == 9
    assert payload["return_code"] is None


def test_harbor_agent_uses_unified_exec_for_native_local_shell(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(HarnessToolCall('local_shell', {\n"
        "            'action': {\n"
        "                'type': 'exec',\n"
        "                'command': ['/bin/bash', '-lc', 'pwd'],\n"
        "                'working_directory': 'src',\n"
        "                'timeout_ms': 1500,\n"
        "            }\n"
        "        }, 'call_1'),))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = UnifiedEnvironment()
    env._workdir = "/app"
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.exec_calls[0]["command"] == "pwd"
    assert env.exec_calls[0]["cwd"] == "/app/src"
    assert env.exec_calls[0]["timeout_sec"] == 2
    assert env.exec_calls[0]["shell"] == "/bin/bash"
    assert env.exec_calls[0]["login"] is True


def test_harbor_agent_resolves_relative_codex_workdir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(HarnessToolCall('exec_command', {\n"
        "            'cmd': 'pwd', 'workdir': 'src'\n"
        "        }, 'call_1'),))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = UnifiedEnvironment()
    env._workdir = "/app"
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.exec_calls[0]["cwd"] == "/app/src"
    assert env.exec_calls[0]["login"] is True


def test_harbor_agent_intercepts_shell_apply_patch_from_exec_command(tmp_path: Path) -> None:
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "hello.txt").write_text("old\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Update File: hello.txt\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
        "PATCH"
    )
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        f"COMMAND = {command!r}\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(HarnessToolCall('exec_command', {\n"
        "            'cmd': COMMAND\n"
        "        }, 'call_1'),))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = LocalUnifiedEnvironment(workdir)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert (workdir / "hello.txt").read_text(encoding="utf-8") == "new\n"
    assert env.exec_command_calls == 0
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["metadata"]["intercepted_apply_patch"] is True


def test_harbor_agent_executes_update_plan_without_environment(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(HarnessToolCall('update_plan', {\n"
        "            'plan': [{'step': 'inspect', 'status': 'in_progress'}]\n"
        "        }, 'call_1'),))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == []
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["tool_name"] == "update_plan"
    assert payload["stdout"] == "Plan updated."


def test_harbor_agent_collects_agents_context_for_codex_like_harness(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    wants_environment_context = True\n"
        "    wants_agents_context = True\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        text = task.metadata['agents_md'][0]['content']\n"
        "        return HarnessTurn(command=f'echo {text}')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = AgentsEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands[-1] == "echo use pytest"


def test_harbor_agent_passes_workdir_to_codex_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    wants_environment_context = True\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(command=f'echo {task.working_dir}')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = FakeEnvironment()
    env._workdir = "/task/workdir"
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == ["echo /task/workdir"]


def test_harbor_agent_executes_candidate_apply_patch_tool(tmp_path: Path) -> None:
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "hello.txt").write_text("old\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    patch = (
        "*** Begin Patch\n"
        "*** Update File: hello.txt\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End of File\n"
        "*** End Patch\n"
    )
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        f"PATCH = {patch!r}\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(HarnessToolCall('apply_patch', {'patch': PATCH}),))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", LocalEnvironment(workdir), SimpleNamespace(metadata=None)))

    assert (workdir / "hello.txt").read_text(encoding="utf-8") == "new\n"
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["tool_name"] == "apply_patch"
    assert payload["metadata"]["input"] == patch
    assert payload["return_code"] == 0


def test_harbor_agent_records_candidate_timeout_as_observation(tmp_path: Path) -> None:
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
    env = TimeoutEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)
    asyncio.run(agent.run("instruction", env, context))
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["return_code"] == 124
    assert payload["stderr"] == "Command timed out after 3 seconds"
    assert context.metadata["done"] is True


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


class UnifiedEnvironment:
    def __init__(self) -> None:
        self.exec_calls: list[dict[str, object]] = []
        self.stdin_calls: list[dict[str, object]] = []

    async def exec_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        shell: str | None = None,
        login: bool = False,
        tty: bool = False,
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ):
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
                "shell": shell,
                "login": login,
                "tty": tty,
                "yield_time_ms": yield_time_ms,
                "max_output_tokens": max_output_tokens,
            }
        )
        return SimpleNamespace(
            stdout="ready\n",
            stderr="",
            return_code=None,
            chunk_id="abc123",
            wall_time_seconds=0.25,
            session_id=9,
            original_token_count=2,
        )

    async def write_stdin(
        self,
        session_id: int,
        chars: str = "",
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ):
        self.stdin_calls.append(
            {
                "session_id": session_id,
                "chars": chars,
                "yield_time_ms": yield_time_ms,
                "max_output_tokens": max_output_tokens,
            }
        )
        return SimpleNamespace(
            stdout="done\n",
            stderr="",
            return_code=0,
            chunk_id="def456",
            wall_time_seconds=0.1,
            session_id=None,
            original_token_count=1,
        )


class LocalUnifiedEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.exec_command_calls = 0

    async def exec(self, command: str, timeout_sec: int | None = None):
        result = subprocess.run(
            command,
            cwd=self.workdir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return SimpleNamespace(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    async def exec_command(self, *args, **kwargs):
        self.exec_command_calls += 1
        raise AssertionError("shell apply_patch should be intercepted before exec_command")


class AgentsEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._workdir = "/app"

    async def exec(self, command: str, timeout_sec: int | None = None):
        self.commands.append(command)
        if "AGENTS.md" in command:
            return SimpleNamespace(
                stdout=json.dumps([{"path": "/app/AGENTS.md", "content": "use pytest"}]),
                stderr="",
                return_code=0,
            )
        return SimpleNamespace(stdout="ok\n", stderr="", return_code=0)


class LocalEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def exec(self, command: str, timeout_sec: int | None = None):
        result = subprocess.run(
            command,
            cwd=self.workdir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return SimpleNamespace(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )


class FailingEnvironment:
    async def exec(self, command: str, timeout_sec: int | None = None):
        raise RuntimeError("exec failed")


class TimeoutEnvironment:
    async def exec(self, command: str, timeout_sec: int | None = None):
        raise RuntimeError(f"Command timed out after {timeout_sec} seconds")


class FakeOpenAI:
    def __init__(self, text: str) -> None:
        self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=text))
