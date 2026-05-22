from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

try:  # Harbor is installed as a CLI tool, not as a project test dependency.
    from harbor.agents.base import BaseAgent as HarborBaseAgent
except Exception:  # pragma: no cover - exercised when Harbor imports this file.
    HarborBaseAgent = object

CODEX_CLI_AGENT_IMPORT_PATH = "plumbing.codex_cli_agent:CodexCliAgent"
DEFAULT_CODEX_BIN = "/opt/harbor-python/bin/codex"
DEFAULT_CONTAINER_CODEX_HOME = "/root/.codex"
DEFAULT_PROMPT_PATH = "/tmp/codex-task-prompt.txt"
DEFAULT_LAST_MESSAGE_PATH = "/tmp/codex-last-message.txt"
DEFAULT_TIMEOUT_SEC = 7200


class CodexCliAgent(HarborBaseAgent):
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir: Path | str,
        model_name: str | None = None,
        codex_model: str | None = None,
        codex_reasoning_effort: str = "none",
        codex_bin: str = DEFAULT_CODEX_BIN,
        codex_home: str = DEFAULT_CONTAINER_CODEX_HOME,
        host_codex_auth_path: str | None = None,
        timeout_sec: int | str = DEFAULT_TIMEOUT_SEC,
        **kwargs: Any,
    ):
        if HarborBaseAgent is object:
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.logger = None
        else:
            super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **kwargs)
        self.codex_model = (
            codex_model or model_name or os.getenv("OPENAI_TERMINAL_MODEL", "gpt-5.4-mini")
        )
        self.codex_reasoning_effort = codex_reasoning_effort
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.host_codex_auth_path = Path(
            host_codex_auth_path
            or os.getenv("CODEX_AUTH_JSON_PATH")
            or Path.home() / ".codex" / "auth.json"
        )
        self.timeout_sec = int(timeout_sec)

    @staticmethod
    def name() -> str:
        return "codex-cli"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        await environment.exec(
            f"mkdir -p {shlex.quote(self.codex_home)} && chmod 700 {shlex.quote(self.codex_home)}",
            timeout_sec=30,
        )
        probe = await environment.exec(
            f"test -x {shlex.quote(self.codex_bin)} || command -v {shlex.quote(self.codex_bin)}",
            timeout_sec=30,
        )
        if probe.return_code:
            raise RuntimeError(
                f"Codex CLI is not available in the task environment: {self.codex_bin}"
            )
        if self.host_codex_auth_path.exists():
            await environment.upload_file(
                self.host_codex_auth_path,
                f"{self.codex_home}/auth.json",
            )
            await environment.exec(
                f"chmod 600 {shlex.quote(self.codex_home + '/auth.json')}",
                timeout_sec=30,
            )
        elif not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                f"Codex auth file not found: {self.host_codex_auth_path}; "
                "set OPENAI_API_KEY or provide host_codex_auth_path"
            )

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = self.logs_dir / "prompt.txt"
        prompt_file.write_text(instruction, encoding="utf-8")
        await environment.upload_file(prompt_file, DEFAULT_PROMPT_PATH)
        command = self._codex_command()
        (self.logs_dir / "codex_command.txt").write_text(command + "\n", encoding="utf-8")
        result = await environment.exec(
            command,
            env=self._codex_env(),
            timeout_sec=self.timeout_sec,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        (self.logs_dir / "codex_stdout.log").write_text(stdout, encoding="utf-8")
        (self.logs_dir / "codex_stderr.log").write_text(stderr, encoding="utf-8")
        last_message = await environment.exec(
            f"cat {shlex.quote(DEFAULT_LAST_MESSAGE_PATH)} 2>/dev/null || true",
            env=self._codex_env(),
            timeout_sec=30,
        )
        (self.logs_dir / "codex_last_message.txt").write_text(
            last_message.stdout or "",
            encoding="utf-8",
        )
        context.metadata = {
            "agent": self.name(),
            "codex_model": self.codex_model,
            "codex_reasoning_effort": self.codex_reasoning_effort,
            "return_code": result.return_code,
            "last_message_chars": len(last_message.stdout or ""),
        }

    def _codex_command(self) -> str:
        args = [
            self.codex_bin,
            "exec",
            "--model",
            self.codex_model,
            "-c",
            f'model_reasoning_effort="{self.codex_reasoning_effort}"',
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            DEFAULT_LAST_MESSAGE_PATH,
            "-",
        ]
        inner = " ".join(shlex.quote(str(item)) for item in args)
        inner += f" < {shlex.quote(DEFAULT_PROMPT_PATH)}"
        script = "\n".join(
            [
                "set -euo pipefail",
                f"setsid bash -c {shlex.quote(inner)} &",
                "pid=$!",
                "set +e",
                'wait "$pid"',
                "rc=$?",
                'kill -TERM -- -"$pid" 2>/dev/null || true',
                "sleep 0.2",
                'kill -KILL -- -"$pid" 2>/dev/null || true',
                'exit "$rc"',
            ]
        )
        return "bash -c " + shlex.quote(script)

    def _codex_env(self) -> dict[str, str]:
        env = {
            "CODEX_HOME": self.codex_home,
            "HOME": self.codex_home,
            "PATH": "/opt/harbor-python/bin:/usr/local/bin:/usr/bin:/bin",
        }
        if api_key := os.getenv("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = api_key
        return env
