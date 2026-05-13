from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plumbing.base_agent import load_harness
from plumbing.types import TaskContext

try:  # Harbor is installed as a CLI tool, not as a project test dependency.
    from harbor.agents.base import BaseAgent as HarborBaseAgent
except Exception:  # pragma: no cover - exercised when Harbor imports this file.
    HarborBaseAgent = object

TERMINAL_BENCH_DATASET = "terminal-bench@2.0"
HARBOR_AGENT_IMPORT_PATH = "plumbing.harbor_adapter:HarborHarnessAgent"


@dataclass(frozen=True)
class HarborRunSpec:
    candidate_dir: Path
    out_dir: Path
    tasks: list[str]
    trials: int
    concurrency: int
    split: str


@dataclass(frozen=True)
class HarborCommandPlan:
    command: list[str]
    runnable: bool
    task_flag: str | None
    note: str


class HarborHarnessAgent(HarborBaseAgent):
    SUPPORTS_WINDOWS: bool = True

    def __init__(
        self,
        logs_dir: Path | str,
        model_name: str | None = None,
        candidate_dir: str | Path = ".",
        **kwargs: Any,
    ):
        if HarborBaseAgent is object:
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.logger = None
        else:
            super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **kwargs)
        self.candidate_dir = Path(candidate_dir)

    @staticmethod
    def name() -> str:
        return "harness-complexity"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        agent = load_harness(candidate_dir=self.candidate_dir)
        command = agent.solve(TaskContext(instruction=instruction)).strip()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "harness-command.txt").write_text(command, encoding="utf-8")
        if not command:
            context.metadata = {"candidate_dir": str(self.candidate_dir), "empty_command": True}
            return
        result = await environment.exec(command=command)
        self._write_result_logs(command, result)
        context.metadata = {
            "candidate_dir": str(self.candidate_dir),
            "command": command,
            "return_code": getattr(result, "return_code", None),
            "stdout_chars": len(getattr(result, "stdout", "") or ""),
            "stderr_chars": len(getattr(result, "stderr", "") or ""),
        }

    def _write_result_logs(self, command: str, result: Any) -> None:
        payload = {
            "command": command,
            "return_code": getattr(result, "return_code", None),
            "stdout": getattr(result, "stdout", None),
            "stderr": getattr(result, "stderr", None),
        }
        (self.logs_dir / "harness-result.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def run_candidate(
    instruction: str,
    working_dir: str | None = None,
    candidate_dir: str | Path | None = None,
) -> str:
    agent = load_harness(candidate_dir=candidate_dir)
    return agent.solve(TaskContext(instruction=instruction, working_dir=working_dir))


def detect_harbor_executable() -> str | None:
    for name in ("harbor", "hb"):
        if shutil.which(name):
            return name
    local_harbor = Path.home() / ".local" / "bin" / "harbor.exe"
    if local_harbor.exists():
        return str(local_harbor)
    return None


def harbor_help(executable: str, *args: str) -> str:
    try:
        result = subprocess.run(
            [executable, *args, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except OSError:
        return ""
    return f"{result.stdout}\n{result.stderr}"


def has_harbor_run_flags(help_text: str) -> bool:
    required = ("--dataset", "--include-task-name", "--n-attempts", "--n-concurrent")
    return all(flag in help_text for flag in required)


def build_harbor_command(
    spec: HarborRunSpec,
    executable: str | None = None,
    help_text: str | None = None,
) -> HarborCommandPlan:
    exe = executable or detect_harbor_executable() or "harbor"
    help_blob = help_text if help_text is not None else harbor_help(exe, "run")
    task_flag = "--include-task-name"
    command = [
        exe,
        "run",
        "--dataset",
        TERMINAL_BENCH_DATASET,
        "--jobs-dir",
        str(spec.out_dir),
        "--n-attempts",
        str(spec.trials),
        "--n-concurrent",
        str(spec.concurrency),
        "--agent-import-path",
        HARBOR_AGENT_IMPORT_PATH,
        "--agent-kwarg",
        f"candidate_dir={spec.candidate_dir}",
        "--quiet",
        "--yes",
    ]
    for task in spec.tasks:
        command.extend([task_flag, task])
    runnable = bool(detect_harbor_executable() or executable) and has_harbor_run_flags(help_blob)
    note = (
        "Using Harbor terminal-bench@2.0 dataset filters."
        if runnable
        else "Harbor CLI was not found or did not expose expected run flags."
    )
    return HarborCommandPlan(command=command, runnable=runnable, task_flag=task_flag, note=note)
