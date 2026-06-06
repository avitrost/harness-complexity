from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import plumbing.harbor_adapter as harbor_adapter
from plumbing.base_agent import load_harness
from plumbing.harbor_adapter import (
    HarborHarnessAgent,
    HarborRunSpec,
    build_harbor_command,
)
from plumbing.openai_client import set_client_factory
from plumbing.types import HarnessToolCall


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


def test_harbor_agent_can_execute_tool_calls_sequentially(tmp_path: Path) -> None:
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
        "            HarnessToolCall('local_shell', {'command': 'first'}, 'call_1'),\n"
        "            HarnessToolCall('local_shell', {'command': 'second'}, 'call_2'),\n"
        "        ), metadata={'sequential_tool_calls': True})\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = OrderedEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == ["first", "second"]
    second = json.loads((tmp_path / "logs" / "harness-turn-02.json").read_text())
    assert second["return_code"] == 0


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


def test_harbor_agent_provides_persistent_terminal_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    wants_persistent_terminal = True\n"
        "    def next_command(self, task, history):\n"
        "        term = task.metadata['persistent_terminal']\n"
        "        if not history:\n"
        "            assert term['available'] is True\n"
        "            assert term['session_id'] == 9\n"
        "            assert 'ready' in term['initial_output']\n"
        "            return HarnessTurn(tool_calls=(HarnessToolCall('write_stdin', {\n"
        "                'session_id': term['session_id'],\n"
        "                'commands': [\n"
        "                    {'chars': 'cd src\\n', 'yield_time_ms': 100},\n"
        "                    {'chars': 'pwd\\n', 'yield_time_ms': 200},\n"
        "                ],\n"
        "            }, 'call_1'),))\n"
        "        return HarnessTurn(done=True)\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = PersistentUnifiedEnvironment()
    env._workdir = "/app"
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.exec_calls[0]["command"] == harbor_adapter.PERSISTENT_TERMINAL_COMMAND
    assert env.exec_calls[0]["cwd"] == "/app"
    assert env.exec_calls[0]["tty"] is True
    assert [item["chars"] for item in env.stdin_calls[:2]] == ["cd src\n", "pwd\n"]
    assert env.stdin_calls[0]["yield_time_ms"] == 100
    assert env.stdin_calls[1]["yield_time_ms"] == 200
    assert env.stdin_calls[-1]["chars"] == "exit\n"
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["command"] == "write_stdin(session_id=9, commands=2)"
    assert payload["metadata"]["terminal_command_count"] == 2


def test_harbor_agent_provides_tmux_persistent_terminal_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    wants_persistent_terminal = 'tmux'\n"
        "    def next_command(self, task, history):\n"
        "        term = task.metadata['persistent_terminal']\n"
        "        if not history:\n"
        "            assert term['available'] is True\n"
        "            assert term['backend'] == 'tmux'\n"
        "            assert term['session_name'].startswith('harness-complexity-')\n"
        "            assert 'Current Terminal Screen:' in term['initial_output']\n"
        "            return HarnessTurn(tool_calls=(HarnessToolCall('write_stdin', {\n"
        "                'session_name': term['session_name'],\n"
        "                'commands': [\n"
        "                    {'chars': 'cd src\\n', 'yield_time_ms': 100},\n"
        "                    {'chars': 'pwd\\n', 'yield_time_ms': 200},\n"
        "                ],\n"
        "            }, 'call_1'),))\n"
        "        return HarnessTurn(done=True)\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TmuxEnvironment()
    env._workdir = "/app"
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert any("tmux new-session" in call["command"] for call in env.exec_calls)
    assert any("cd /app" in call["command"] for call in env.exec_calls)
    send_commands = [
        call["command"] for call in env.exec_calls if "tmux send-keys" in call["command"]
    ]
    assert len(send_commands) == 2
    assert "cd src" in send_commands[0]
    assert "pwd" in send_commands[1]
    assert "tmux kill-session" in env.exec_calls[-1]["command"]
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["command"].startswith("write_stdin(tmux_session=harness-complexity-")
    assert payload["metadata"]["backend"] == "tmux"
    assert payload["metadata"]["terminal_command_count"] == 2
    assert "New Terminal Output:" in payload["stdout"]


def test_tmux_persistent_terminal_fails_loudly_without_fallback() -> None:
    agent = SimpleNamespace(wants_persistent_terminal="tmux")

    with pytest.raises(RuntimeError, match="tmux missing"):
        asyncio.run(harbor_adapter._task_context("instruction", BrokenTmuxEnvironment(), agent))


def test_tmux_persistent_terminal_installs_tmux_when_missing() -> None:
    agent = SimpleNamespace(wants_persistent_terminal="tmux")
    env = InstallingTmuxEnvironment()

    task = asyncio.run(harbor_adapter._task_context("instruction", env, agent))

    terminal = task.metadata["persistent_terminal"]
    assert terminal["available"] is True
    assert any("apt-get install -y tmux" in call["command"] for call in env.exec_calls)
    assert any("tmux new-session" in call["command"] for call in env.exec_calls)


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


def test_harbor_agent_apply_patch_accepts_absolute_task_paths(tmp_path: Path) -> None:
    workdir = tmp_path / "task"
    workdir.mkdir()
    target = workdir / "hello.txt"
    target.write_text("old\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    patch = (
        "*** Begin Patch\n"
        f"*** Update File: {target}\n"
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

    assert target.read_text(encoding="utf-8") == "new\n"
    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["return_code"] == 0


def test_apply_patch_command_has_host_python_fallback() -> None:
    command = harbor_adapter._apply_patch_command("*** Begin Patch\n*** End Patch\n")

    assert "/opt/harbor-python/bin/python" in command
    assert "apply_patch failed: no Python runtime available" in command


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


def test_harbor_agent_records_stuck_tool_timeout_as_observation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(harbor_adapter, "TOOL_TIMEOUT_RESPONSE_GRACE_SEC", 0)
    monkeypatch.setattr(harbor_adapter, "EXEC_REQUEST_GRACE_SEC", 0)
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        f"PATCH = {patch!r}\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        if history:\n"
        "            return HarnessTurn(done=True)\n"
        "        return HarnessTurn(tool_calls=(\n"
        "            HarnessToolCall('apply_patch', {'patch': PATCH, 'timeout_sec': 1}),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = HangingEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert env.timeouts == [1]
    assert payload["return_code"] == 124
    assert payload["stderr"] == "Tool call timed out after 1 seconds"
    assert payload["metadata"]["adapter_timeout_sec"] == 1
    assert context.metadata["done"] is True


def test_harbor_agent_records_slurm_transport_reset_as_observation(tmp_path: Path) -> None:
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
        "            HarnessToolCall('exec_command', {'cmd': 'echo hello'}),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = SlurmResetEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["return_code"] == 1
    assert "Connection reset by peer" in payload["stderr"]
    assert context.metadata["done"] is True


def test_harbor_agent_records_slurm_srun_exit_as_observation(tmp_path: Path) -> None:
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
        "            HarnessToolCall('exec_command', {'cmd': 'echo hello'}),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = SlurmSrunExitedEnvironment()
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert payload["return_code"] == 1
    assert "srun exited before request" in payload["stderr"]
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
        context = SimpleNamespace(metadata=None)
        asyncio.run(agent.run("instruction", FakeEnvironment(), context))
    finally:
        set_client_factory(None)
    trace = (tmp_path / "logs" / "model-call-01.json").read_text(encoding="utf-8")
    assert '"content": "next?"' in trace
    assert '"response": "ok"' in trace
    assert context.metadata["model_accounting"]["model_calls"] == 1
    assert context.metadata["model_accounting"]["input_tokens"] > 0
    assert context.n_input_tokens == context.metadata["model_accounting"]["input_tokens"]


def test_harbor_agent_records_model_rate_limit_as_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        cause = RuntimeError('codex backend call failed: 429 Rate limit exceeded')\n"
        "        raise RuntimeError('terminal model tool call failed') from cause\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", FakeEnvironment(), context))

    result = json.loads((tmp_path / "logs" / "harness-result.json").read_text())
    turn = json.loads((tmp_path / "logs" / "harness-turn-01.json").read_text())
    assert result["done"] is False
    assert result["termination_reason"] == "model_call_error"
    assert turn["command"] == "<model call>"
    assert turn["return_code"] == 1
    assert "Rate limit exceeded" in turn["stderr"]
    assert context.metadata["termination_reason"] == "model_call_error"


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


def test_harbor_agent_soft_stops_before_harbor_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harbor_adapter, "MODEL_CALL_RUNWAY_SEC", 150)
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(command='echo should-not-run')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=1)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-result.json").read_text())
    assert env.commands == []
    assert payload["turns"] == 0
    assert payload["termination_reason"] == "soft_agent_timeout_before_model"
    assert payload["agent_timeout_sec"] == 1.0
    assert context.metadata["termination_reason"] == "soft_agent_timeout_before_model"


def test_harbor_agent_soft_stops_during_slow_model_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harbor_adapter, "MODEL_CALL_RUNWAY_SEC", 0.05)
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "import time\n"
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        time.sleep(0.2)\n"
        "        return HarnessTurn(command='echo too-late')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=0.1)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-result.json").read_text())
    assert env.commands == []
    assert payload["turns"] == 0
    assert payload["termination_reason"] == "soft_agent_timeout_during_model"
    assert context.metadata["termination_reason"] == "soft_agent_timeout_during_model"


def test_harbor_agent_caps_uncapped_tool_to_remaining_hard_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    ticks = iter([0.0, 10.0, 20.0, 21.0, 22.0])
    monkeypatch.setattr(harbor_adapter, "_monotonic", lambda: next(ticks, 22.0))
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
        "        return HarnessTurn(command='sleep maybe')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=300)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)

    asyncio.run(agent.run("instruction", env, SimpleNamespace(metadata=None)))

    assert env.commands == ["sleep maybe"]
    assert env.timeouts == [244]


def test_tool_wait_timeout_does_not_exceed_remaining_deadline() -> None:
    tool_call = HarnessToolCall("exec_command", {"cmd": "sleep maybe"})

    assert (
        harbor_adapter._tool_wait_timeout_sec(
            SimpleNamespace(exec_command=lambda: None),
            tool_call,
            default_timeout_sec=None,
            max_timeout_sec=190,
        )
        == 190
    )


def test_harbor_agent_soft_stops_before_tool_without_enough_runway(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(harbor_adapter, "EXEC_REQUEST_GRACE_SEC", 120)
    monkeypatch.setattr(harbor_adapter, "TOOL_TIMEOUT_RESPONSE_GRACE_SEC", 15)
    ticks = iter([0.0, 300.0, 581.0])
    monkeypatch.setattr(harbor_adapter, "_monotonic", lambda: next(ticks, 430.0))
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessToolCall, HarnessTurn\n"
        f"PATCH = {patch!r}\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(tool_calls=(\n"
        "            HarnessToolCall('apply_patch', {'patch': PATCH, 'timeout_sec': 30}),\n"
        "        ))\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=600)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-result.json").read_text())
    assert env.commands == []
    assert payload["turns"] == 0
    assert payload["termination_reason"] == "soft_agent_timeout_before_tools"
    assert context.metadata["termination_reason"] == "soft_agent_timeout_before_tools"


def test_harbor_agent_soft_stops_when_model_consumes_tool_runway(
    tmp_path: Path, monkeypatch
) -> None:
    ticks = iter([0.0, 100.0, 550.0, 551.0])
    monkeypatch.setattr(harbor_adapter, "_monotonic", lambda: next(ticks, 551.0))
    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "harness.py").write_text(
        "from plumbing.base_agent import BaseHarness\n"
        "from plumbing.types import HarnessTurn\n"
        "class H(BaseHarness):\n"
        "    def next_command(self, task, history):\n"
        "        return HarnessTurn(command='echo should-not-run')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=600)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    payload = json.loads((tmp_path / "logs" / "harness-result.json").read_text())
    assert env.commands == []
    assert payload["termination_reason"] == "soft_agent_timeout_before_tools"
    assert context.metadata["termination_reason"] == "soft_agent_timeout_before_tools"


def test_harbor_agent_runs_tool_when_capped_wait_fits_deadline(tmp_path: Path, monkeypatch) -> None:
    ticks = iter([0.0, 300.0, 409.0, 410.0, 411.0, 412.0])
    monkeypatch.setattr(harbor_adapter, "_monotonic", lambda: next(ticks, 412.0))
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
        "        return HarnessTurn(command='echo verify')\n"
        "def create_agent():\n"
        "    return H()\n",
        encoding="utf-8",
    )
    env = TaskTimeoutEnvironment(tmp_path / "task", timeout_sec=600)
    agent = HarborHarnessAgent(logs_dir=tmp_path / "logs", candidate_dir=workspace)
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("instruction", env, context))

    assert env.commands == ["echo verify"]
    assert env.timeouts == [155]
    assert context.metadata["termination_reason"] is None


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


class OrderedEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.first_done = False

    async def exec(self, command: str, timeout_sec: int | None = None):
        self.commands.append(command)
        self.timeouts.append(timeout_sec)
        if command == "first":
            await asyncio.sleep(0.01)
            self.first_done = True
            return SimpleNamespace(stdout="first\n", stderr="", return_code=0)
        if command == "second" and not self.first_done:
            return SimpleNamespace(stdout="", stderr="first not done\n", return_code=7)
        return SimpleNamespace(stdout="second\n", stderr="", return_code=0)


class TaskTimeoutEnvironment(FakeEnvironment):
    def __init__(self, task_dir: Path, timeout_sec: float) -> None:
        super().__init__()
        self.environment_dir = task_dir / "environment"
        self.environment_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            f"[agent]\ntimeout_sec = {timeout_sec}\n",
            encoding="utf-8",
        )


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


class PersistentUnifiedEnvironment(UnifiedEnvironment):
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
            stdout=f"{chars}ok\n",
            stderr="",
            return_code=None,
            chunk_id="def456",
            wall_time_seconds=0.1,
            session_id=session_id,
            original_token_count=1,
        )


class TmuxEnvironment:
    def __init__(self) -> None:
        self.exec_calls: list[dict[str, object]] = []
        self.sent = False

    async def exec(self, command: str, timeout_sec: int | None = None):
        self.exec_calls.append({"command": command, "timeout_sec": timeout_sec})
        if "tmux capture-pane" in command:
            stdout = "$ cd src\n$ pwd\n/app/src\n$ " if self.sent else "$ "
            return SimpleNamespace(stdout=stdout, stderr="", return_code=0)
        if "tmux send-keys" in command:
            self.sent = True
        return SimpleNamespace(stdout="", stderr="", return_code=0)


class BrokenTmuxEnvironment:
    async def exec(self, command: str, timeout_sec: int | None = None):
        return SimpleNamespace(stdout="", stderr="tmux missing", return_code=1)


class InstallingTmuxEnvironment(TmuxEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.installed = False

    async def exec(self, command: str, timeout_sec: int | None = None, user=None):
        self.exec_calls.append({"command": command, "timeout_sec": timeout_sec, "user": user})
        if command == "tmux -V" and not self.installed:
            return SimpleNamespace(stdout="", stderr="tmux missing", return_code=1)
        if "apt-get install -y tmux" in command:
            self.installed = True
            return SimpleNamespace(stdout="tmux 3.4\n", stderr="", return_code=0)
        if "tmux capture-pane" in command:
            stdout = "$ cd src\n$ pwd\n/app/src\n$ " if self.sent else "$ "
            return SimpleNamespace(stdout=stdout, stderr="", return_code=0)
        if "tmux send-keys" in command:
            self.sent = True
        return SimpleNamespace(stdout="tmux 3.4\n", stderr="", return_code=0)


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


class HangingEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.timeouts: list[int | None] = []

    async def exec(self, command: str, timeout_sec: int | None = None):
        self.commands.append(command)
        self.timeouts.append(timeout_sec)
        await asyncio.Event().wait()


class SlurmResetEnvironment:
    async def exec_command(self, *args, **kwargs):
        raise RuntimeError(
            "Slurm/Pyxis server request failed: [Errno 104] Connection reset by peer"
        )


class SlurmSrunExitedEnvironment:
    async def exec_command(self, *args, **kwargs):
        raise RuntimeError("Slurm/Pyxis srun exited before request: 0; output: cancelled")


class FakeOpenAI:
    def __init__(self, text: str) -> None:
        self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=text))
