from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import shlex
import shutil
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plumbing.base_agent import load_harness
from plumbing.openai_client import reset_trace_dir, set_trace_dir
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

try:  # Harbor is installed as a CLI tool, not as a project test dependency.
    from harbor.agents.base import BaseAgent as HarborBaseAgent
except Exception:  # pragma: no cover - exercised when Harbor imports this file.
    HarborBaseAgent = object

TERMINAL_BENCH_DATASET = "terminal-bench@2.0"
HARBOR_AGENT_IMPORT_PATH = "plumbing.harbor_adapter:HarborHarnessAgent"
SLURM_PYXIS_ENV_IMPORT_PATH = "plumbing.slurm_pyxis_environment:SlurmPyxisEnvironment"
SLURM_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER = "18"
MAX_OBSERVATION_CHARS = 6000
MODEL_CALL_RUNWAY_SEC = 60
TOOL_START_RUNWAY_SEC = 60
HARD_AGENT_TIMEOUT_GUARD_SEC = 20
TOOL_TIMEOUT_RESPONSE_GRACE_SEC = 15
EXEC_REQUEST_GRACE_SEC = 180
SHELL_TOOL_NAMES = {
    "local_shell",
    "shell",
    "exec",
    "execute",
    "execute_command",
    "execute_commands",
    "exec_command",
    "shell_command",
    "terminal",
    "bash",
}
WRITE_STDIN_TOOL_NAMES = {"write_stdin"}
PATCH_TOOL_NAMES = {"apply_patch", "patch"}
PLAN_TOOL_NAMES = {"update_plan", "plan"}
FUNCTION_HISTORY_TOOL_NAMES = {"exec_command", "write_stdin", "update_plan"}
_MODEL_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("HARBOR_MODEL_CALL_WORKERS", "512")),
    thread_name_prefix="harness-model",
)


@dataclass(frozen=True)
class HarborRunSpec:
    candidate_dir: Path
    out_dir: Path
    tasks: list[str]
    trials: int
    concurrency: int
    split: str
    backend: str = "docker"
    agent_import_path: str = HARBOR_AGENT_IMPORT_PATH
    agent_kwargs: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarborCommandPlan:
    command: list[str]
    runnable: bool
    task_flag: str | None
    note: str


class HarborHarnessAgent(HarborBaseAgent):
    SUPPORTS_WINDOWS: bool = False

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
        task = await _task_context(instruction, environment, agent)
        history: list[CommandResult] = []
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        done = False
        termination_reason: str | None = None
        started_at = _monotonic()
        agent_timeout_sec = _agent_timeout_seconds(self.logs_dir, environment)
        agent_deadline = _agent_deadline(started_at, agent_timeout_sec)
        turn_index = 0
        token = set_trace_dir(self.logs_dir)
        try:
            while True:
                model_timeout_sec = _model_call_timeout_sec(agent_deadline)
                if model_timeout_sec is not None and model_timeout_sec <= 0:
                    termination_reason = "soft_agent_timeout_before_model"
                    break
                turn_index += 1
                try:
                    turn = await _next_turn_with_timeout(
                        agent,
                        task,
                        history,
                        timeout_sec=model_timeout_sec,
                    )
                except TimeoutError:
                    termination_reason = "soft_agent_timeout_during_model"
                    break
                tool_calls = _turn_tool_calls(turn)
                if turn.done or not tool_calls:
                    done = turn.done
                    break
                if _insufficient_tool_runway(
                    environment,
                    tool_calls,
                    turn.timeout_sec,
                    agent_deadline,
                ):
                    termination_reason = "soft_agent_timeout_before_tools"
                    break
                max_timeout_sec = _remaining_timeout_sec(
                    _guarded_deadline(agent_deadline, HARD_AGENT_TIMEOUT_GUARD_SEC)
                )
                records = await asyncio.gather(
                    *(
                        _execute_tool_call_with_timeout(
                            environment,
                            tool_call,
                            turn.timeout_sec,
                            max_timeout_sec=max_timeout_sec,
                        )
                        for tool_call in tool_calls
                    )
                )
                turn_metadata = (
                    dict(turn.metadata) if isinstance(getattr(turn, "metadata", None), dict) else {}
                )
                has_codex_items = bool(turn_metadata.get("codex_response_items"))
                for record_index, record in enumerate(records):
                    extra: dict[str, Any] = {}
                    if has_codex_items:
                        if record_index == 0:
                            extra.update(turn_metadata)
                        else:
                            extra["codex_output_only"] = True
                    elif record_index == 0 and turn_metadata:
                        extra.update(turn_metadata)
                    if record_index == 0 and turn.assistant_content.strip():
                        extra["assistant_content"] = turn.assistant_content
                    if extra:
                        record = _with_metadata(record, extra)
                    history.append(record)
                    self._write_turn_log(turn_index, record)
                    turn_index += 1
                turn_index -= 1
        finally:
            reset_trace_dir(token)
            elapsed_sec = max(0.0, _monotonic() - started_at)
            self._write_result_logs(
                history,
                done,
                termination_reason=termination_reason,
                agent_timeout_sec=agent_timeout_sec,
                elapsed_sec=elapsed_sec,
            )
            context.metadata = {
                "candidate_dir": str(self.candidate_dir),
                "done": done,
                "turns": len(history),
                "last_return_code": history[-1].return_code if history else None,
                "termination_reason": termination_reason,
                "agent_timeout_sec": agent_timeout_sec,
                "elapsed_sec": elapsed_sec,
            }

    def _write_turn_log(self, turn_index: int, record: CommandResult) -> None:
        (self.logs_dir / f"harness-turn-{turn_index:02d}.json").write_text(
            json.dumps(record.__dict__, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_result_logs(
        self,
        history: list[CommandResult],
        done: bool,
        termination_reason: str | None = None,
        agent_timeout_sec: float | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        payload = {
            "done": done,
            "turns": len(history),
            "commands": [record.command for record in history],
            "last_return_code": history[-1].return_code if history else None,
        }
        if termination_reason:
            payload["termination_reason"] = termination_reason
        if agent_timeout_sec is not None:
            payload["agent_timeout_sec"] = agent_timeout_sec
        if elapsed_sec is not None:
            payload["elapsed_sec"] = elapsed_sec
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
    turn = agent.next_command(TaskContext(instruction=instruction, working_dir=working_dir), [])
    calls = _turn_tool_calls(turn)
    if calls:
        return _display_tool_call(calls[0])
    return turn.command


async def _task_context(instruction: str, environment: Any, agent: Any) -> TaskContext:
    if not getattr(agent, "wants_environment_context", False):
        return TaskContext(instruction=instruction)
    config = getattr(environment, "task_env_config", None)
    workdir = getattr(environment, "_workdir", None) or getattr(config, "workdir", None)
    metadata: dict[str, Any] = {}
    if getattr(agent, "wants_agents_context", False):
        metadata["agents_md"] = await _agents_context(environment, workdir)
    return TaskContext(instruction=instruction, working_dir=workdir, metadata=metadata)


def _agent_timeout_seconds(logs_dir: Path, environment: Any) -> float | None:
    trial_config = _trial_config(logs_dir)
    agent_config = trial_config.get("agent") if isinstance(trial_config.get("agent"), dict) else {}
    override = _float_or_none(agent_config.get("override_timeout_sec"))
    base = override if override is not None else _task_agent_timeout_seconds(environment)
    if base is None:
        return None
    cap = _float_or_none(agent_config.get("max_timeout_sec"))
    if cap is not None:
        base = min(base, cap)
    multiplier = _float_or_none(trial_config.get("agent_timeout_multiplier"))
    if multiplier is None:
        multiplier = _float_or_none(trial_config.get("timeout_multiplier")) or 1.0
    return max(0.0, base * multiplier)


def _trial_config(logs_dir: Path) -> dict[str, Any]:
    config_path = Path(logs_dir).parent / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _task_agent_timeout_seconds(environment: Any) -> float | None:
    environment_dir = getattr(environment, "environment_dir", None)
    if environment_dir is None:
        return None
    config_path = Path(environment_dir).parent / "task.toml"
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    agent_config = data.get("agent")
    if not isinstance(agent_config, dict):
        return None
    return _float_or_none(agent_config.get("timeout_sec"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _agent_deadline(started_at: float, agent_timeout_sec: float | None) -> float | None:
    if agent_timeout_sec is None:
        return None
    return started_at + max(0.0, agent_timeout_sec)


def _guarded_deadline(deadline: float | None, guard_sec: int) -> float | None:
    if deadline is None:
        return None
    return deadline - max(0, guard_sec)


def _remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return deadline - _monotonic()


def _insufficient_tool_runway(
    environment: Any,
    tool_calls: tuple[HarnessToolCall, ...],
    default_timeout_sec: int | None,
    agent_deadline: float | None,
) -> bool:
    remaining = _remaining_seconds(agent_deadline)
    if remaining is None:
        return False
    return remaining <= TOOL_START_RUNWAY_SEC


def _model_call_timeout_sec(agent_deadline: float | None) -> float | None:
    return _remaining_seconds(_guarded_deadline(agent_deadline, MODEL_CALL_RUNWAY_SEC))


def _remaining_timeout_sec(deadline: float | None) -> int | None:
    if deadline is None:
        return None
    remaining = deadline - _monotonic()
    if remaining <= 0:
        return 1
    return max(1, int(remaining))


def _monotonic() -> float:
    return time.monotonic()


async def _next_turn_with_timeout(
    agent: Any,
    task: TaskContext,
    history: list[CommandResult],
    timeout_sec: float | None,
) -> HarnessTurn:
    context = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        _MODEL_EXECUTOR,
        context.run,
        agent.next_command,
        task,
        history,
    )
    if timeout_sec is None:
        return await future
    try:
        return await asyncio.wait_for(future, timeout=max(0.001, timeout_sec))
    except TimeoutError:
        future.cancel()
        raise


def _cap_timeout(timeout_sec: int | None, max_timeout_sec: int | None) -> int | None:
    if max_timeout_sec is None:
        return timeout_sec
    if timeout_sec is None:
        return max_timeout_sec
    return max(1, min(timeout_sec, max_timeout_sec))


def _cap_yield_time_ms(yield_time_ms: int | None, max_timeout_sec: int | None) -> int | None:
    if max_timeout_sec is None:
        return yield_time_ms
    cap_ms = max(1, max_timeout_sec) * 1000
    if yield_time_ms is None:
        return None
    return max(1, min(yield_time_ms, cap_ms))


async def _execute_tool_call(
    environment: Any,
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
    max_timeout_sec: int | None = None,
) -> CommandResult:
    name = tool_call.name.strip() or "local_shell"
    lowered = name.lower()
    if lowered in WRITE_STDIN_TOOL_NAMES:
        return await _write_stdin_observed(
            environment,
            tool_call=tool_call,
            tool_name=name,
            max_timeout_sec=max_timeout_sec,
        )
    if lowered in PLAN_TOOL_NAMES:
        return _plan_observed(tool_call, name)
    if lowered in SHELL_TOOL_NAMES:
        if lowered in {"exec_command", "local_shell"} and hasattr(environment, "exec_command"):
            return await _exec_command_observed(
                environment,
                tool_call=tool_call,
                default_timeout_sec=default_timeout_sec,
                max_timeout_sec=max_timeout_sec,
                tool_name=name,
            )
        command, timeout_sec = _shell_command(tool_call, default_timeout_sec)
        timeout_sec = _cap_timeout(timeout_sec, max_timeout_sec)
        return await _exec_observed(
            environment,
            command=command,
            timeout_sec=timeout_sec,
            tool_name=name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": tool_call.arguments},
        )
    if lowered in PATCH_TOOL_NAMES:
        patch = _patch_text(tool_call.arguments)
        command = _apply_patch_command(patch)
        return await _exec_observed(
            environment,
            command=command,
            timeout_sec=_cap_timeout(
                _tool_timeout(tool_call.arguments, default_timeout_sec) or 30,
                max_timeout_sec,
            ),
            tool_name=name,
            tool_call_id=tool_call.call_id,
            display_command=_patch_display(patch),
            metadata={"input": patch, "patch_bytes": len(patch.encode("utf-8"))},
        )
    return CommandResult(
        command=f"<unsupported tool {name}>",
        return_code=2,
        stderr=f"Unsupported harness tool call: {name}",
        tool_name=name,
        tool_call_id=tool_call.call_id,
        metadata={"arguments": tool_call.arguments},
    )


async def _execute_tool_call_with_timeout(
    environment: Any,
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
    max_timeout_sec: int | None = None,
) -> CommandResult:
    wait_timeout_sec = _tool_wait_timeout_sec(
        environment,
        tool_call,
        default_timeout_sec,
        max_timeout_sec,
    )
    runtime_timeout_sec = _runtime_timeout_cap(max_timeout_sec)
    try:
        operation = _execute_tool_call(
            environment,
            tool_call,
            default_timeout_sec,
            max_timeout_sec=runtime_timeout_sec,
        )
        if wait_timeout_sec is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=wait_timeout_sec)
    except TimeoutError:
        return _tool_wait_timeout_result(tool_call, wait_timeout_sec)


def _tool_wait_timeout_sec(
    environment: Any,
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
    max_timeout_sec: int | None,
) -> int | None:
    name = tool_call.name.strip() or "local_shell"
    lowered = name.lower()
    if lowered in WRITE_STDIN_TOOL_NAMES:
        args = tool_call.arguments
        yield_time_ms = _cap_yield_time_ms(_tool_int(args.get("yield_time_ms")), max_timeout_sec)
        if yield_time_ms is None:
            yield_time_ms = 250 if args.get("chars") else 5000
        request_timeout_sec = max(
            EXEC_REQUEST_GRACE_SEC,
            int((yield_time_ms + 999) / 1000) + EXEC_REQUEST_GRACE_SEC,
        )
        return _cap_wait_timeout(request_timeout_sec, max_timeout_sec)
    if lowered in SHELL_TOOL_NAMES:
        args = _shell_args(tool_call.arguments)
        command = _shell_command_text(args)
        if lowered in {"exec_command", "local_shell"} and hasattr(environment, "exec_command"):
            if _extract_apply_patch(command):
                return _cap_exec_request_wait_timeout(
                    _cap_timeout(_tool_timeout(args, default_timeout_sec) or 30, max_timeout_sec),
                    max_timeout_sec,
                )
            yield_time_ms = _cap_yield_time_ms(
                _tool_int(args.get("yield_time_ms")), max_timeout_sec
            )
            wait_ms = yield_time_ms if yield_time_ms is not None else 10000
            request_timeout_sec = max(
                EXEC_REQUEST_GRACE_SEC,
                int(wait_ms / 1000) + EXEC_REQUEST_GRACE_SEC,
            )
            return _cap_wait_timeout(request_timeout_sec, max_timeout_sec)
        timeout_sec = _cap_timeout(_tool_timeout(args, default_timeout_sec), max_timeout_sec)
        return _cap_exec_request_wait_timeout(timeout_sec, max_timeout_sec)
    if lowered in PATCH_TOOL_NAMES:
        timeout_sec = _cap_timeout(
            _tool_timeout(tool_call.arguments, default_timeout_sec) or 30,
            max_timeout_sec,
        )
        return _cap_exec_request_wait_timeout(timeout_sec, max_timeout_sec)
    return None


def _cap_exec_request_wait_timeout(
    timeout_sec: int | None,
    max_timeout_sec: int | None,
) -> int | None:
    if timeout_sec is None:
        return _cap_wait_timeout(timeout_sec, max_timeout_sec)
    return _cap_wait_timeout(timeout_sec + EXEC_REQUEST_GRACE_SEC, max_timeout_sec)


def _cap_wait_timeout(timeout_sec: int | None, max_timeout_sec: int | None) -> int | None:
    if max_timeout_sec is None:
        if timeout_sec is None:
            return None
        return max(1, timeout_sec + TOOL_TIMEOUT_RESPONSE_GRACE_SEC)
    ceiling = max(1, max_timeout_sec)
    if timeout_sec is None:
        return ceiling
    return max(1, min(timeout_sec + TOOL_TIMEOUT_RESPONSE_GRACE_SEC, ceiling))


def _runtime_timeout_cap(max_timeout_sec: int | None) -> int | None:
    if max_timeout_sec is None:
        return None
    return max(1, max_timeout_sec - TOOL_TIMEOUT_RESPONSE_GRACE_SEC)


def _tool_wait_timeout_result(
    tool_call: HarnessToolCall,
    wait_timeout_sec: int | None,
) -> CommandResult:
    name = tool_call.name.strip() or "local_shell"
    return CommandResult(
        command=_display_tool_call(tool_call),
        return_code=124,
        stderr=f"Tool call timed out after {wait_timeout_sec} seconds",
        tool_name=name,
        tool_call_id=tool_call.call_id,
        metadata={
            "arguments": tool_call.arguments,
            "adapter_timeout_sec": wait_timeout_sec,
        },
    )


async def _exec_command_observed(
    environment: Any,
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
    max_timeout_sec: int | None,
    tool_name: str,
) -> CommandResult:
    args = _shell_args(tool_call.arguments)
    command = _shell_command_text(args)
    if not command.strip():
        return CommandResult(
            command="<invalid exec_command>",
            return_code=2,
            stderr="exec_command requires cmd",
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": tool_call.arguments},
        )
    cwd = _resolve_workdir(_tool_string(args.get("workdir")), environment)
    if patch := _extract_apply_patch(command):
        patch_command = _apply_patch_command(patch)
        if cwd:
            patch_command = f"cd {shlex.quote(cwd)} && {patch_command}"
        return await _exec_observed(
            environment,
            command=patch_command,
            timeout_sec=_cap_timeout(
                _tool_timeout(args, default_timeout_sec) or 30,
                max_timeout_sec,
            ),
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            display_command=_patch_display(patch),
            metadata={
                "arguments": tool_call.arguments,
                "input": patch,
                "intercepted_apply_patch": True,
            },
        )
    try:
        timeout_sec = _cap_timeout(_tool_timeout(args, default_timeout_sec), max_timeout_sec)
        result = await environment.exec_command(
            command=command,
            cwd=cwd,
            timeout_sec=timeout_sec,
            shell=_tool_string(args.get("shell")),
            login=bool(args.get("login")) if isinstance(args.get("login"), bool) else True,
            tty=bool(args.get("tty")) if isinstance(args.get("tty"), bool) else False,
            yield_time_ms=_cap_yield_time_ms(
                _tool_int(args.get("yield_time_ms")),
                max_timeout_sec,
            ),
            max_output_tokens=_tool_int(args.get("max_output_tokens")),
        )
    except RuntimeError as exc:
        if not _model_visible_tool_error(str(exc)):
            raise
        return CommandResult(
            command=command,
            return_code=1,
            stderr=str(exc),
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": tool_call.arguments},
        )
    metadata = {
        "arguments": tool_call.arguments,
        "unified_exec": _unified_exec_metadata(result),
    }
    return CommandResult(
        command=command,
        return_code=getattr(result, "return_code", None),
        stdout=_tail(getattr(result, "stdout", "") or ""),
        stderr=_tail(getattr(result, "stderr", "") or ""),
        tool_name=tool_name,
        tool_call_id=tool_call.call_id,
        metadata=metadata,
    )


async def _write_stdin_observed(
    environment: Any,
    tool_call: HarnessToolCall,
    tool_name: str,
    max_timeout_sec: int | None = None,
) -> CommandResult:
    if not hasattr(environment, "write_stdin"):
        return CommandResult(
            command="<unsupported tool write_stdin>",
            return_code=2,
            stderr="Environment does not support write_stdin",
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": tool_call.arguments},
        )
    args = tool_call.arguments
    session_id = _tool_int(args.get("session_id"))
    if session_id is None:
        return CommandResult(
            command="<invalid write_stdin session_id>",
            return_code=2,
            stderr="write_stdin requires session_id",
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": args},
        )
    chars = str(args.get("chars") or "")
    try:
        result = await environment.write_stdin(
            session_id=session_id,
            chars=chars,
            yield_time_ms=_cap_yield_time_ms(
                _tool_int(args.get("yield_time_ms")),
                max_timeout_sec,
            ),
            max_output_tokens=_tool_int(args.get("max_output_tokens")),
        )
    except RuntimeError as exc:
        if not _model_visible_tool_error(str(exc)):
            raise
        return CommandResult(
            command=f"write_stdin(session_id={session_id}, chars={len(chars)} chars)",
            return_code=1,
            stderr=str(exc),
            tool_name=tool_name,
            tool_call_id=tool_call.call_id,
            metadata={"arguments": args},
        )
    metadata = {
        "arguments": args,
        "unified_exec": _unified_exec_metadata(result),
    }
    return CommandResult(
        command=f"write_stdin(session_id={session_id}, chars={len(chars)} chars)",
        return_code=getattr(result, "return_code", None),
        stdout=_tail(getattr(result, "stdout", "") or ""),
        stderr=_tail(getattr(result, "stderr", "") or ""),
        tool_name=tool_name,
        tool_call_id=tool_call.call_id,
        metadata=metadata,
    )


def _plan_observed(tool_call: HarnessToolCall, tool_name: str) -> CommandResult:
    return CommandResult(
        command="update_plan",
        return_code=0,
        stdout="Plan updated.",
        tool_name="update_plan",
        tool_call_id=tool_call.call_id,
        metadata={"arguments": tool_call.arguments},
    )


async def _exec_observed(
    environment: Any,
    command: str,
    timeout_sec: int | None,
    tool_name: str,
    tool_call_id: str = "",
    display_command: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CommandResult:
    try:
        result = await environment.exec(command=command, timeout_sec=timeout_sec)
    except RuntimeError as exc:
        if not _model_visible_tool_error(str(exc)):
            raise
        return_code = 124 if "timed out" in str(exc).lower() and timeout_sec is not None else 1
        return CommandResult(
            command=display_command or command,
            return_code=return_code,
            stderr=str(exc),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
    return CommandResult(
        command=display_command or command,
        return_code=getattr(result, "return_code", None),
        stdout=_tail(getattr(result, "stdout", "") or ""),
        stderr=_tail(getattr(result, "stderr", "") or ""),
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        metadata=metadata or {},
    )


def _with_metadata(record: CommandResult, extra: dict[str, Any]) -> CommandResult:
    metadata = dict(record.metadata)
    metadata.update(extra)
    return CommandResult(
        command=record.command,
        return_code=record.return_code,
        stdout=record.stdout,
        stderr=record.stderr,
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        metadata=metadata,
    )


def _model_visible_tool_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "timed out",
            "exec_command failed",
            "write_stdin failed",
            "stdin is closed",
            "unknown session_id",
            "slurm/pyxis server request failed",
            "connection reset by peer",
            "connection refused",
            "remote end closed connection",
            "broken pipe",
        )
    )


def _turn_tool_calls(turn: HarnessTurn) -> list[HarnessToolCall]:
    calls: list[HarnessToolCall] = []
    for item in turn.tool_calls or ():
        calls.append(_coerce_tool_call(item))
    if calls:
        return calls
    command = turn.command.strip()
    if not command:
        return []
    return [
        HarnessToolCall(
            name="local_shell",
            arguments={"command": command, "timeout_sec": turn.timeout_sec},
        )
    ]


def _coerce_tool_call(value: Any) -> HarnessToolCall:
    if isinstance(value, HarnessToolCall):
        return value
    if isinstance(value, dict):
        name = str(value.get("name") or value.get("tool") or value.get("type") or "local_shell")
        arguments = value.get("arguments")
        if arguments is None:
            arguments = value.get("args")
        if arguments is None:
            arguments = {
                key: item
                for key, item in value.items()
                if key not in {"name", "tool", "type", "call_id", "id"}
            }
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        call_id = str(value.get("call_id") or value.get("id") or "")
        return HarnessToolCall(name=name, arguments=arguments, call_id=call_id)
    name = getattr(value, "name", None)
    if name:
        arguments = getattr(value, "arguments", {})
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        call_id = str(getattr(value, "call_id", "") or getattr(value, "id", "") or "")
        return HarnessToolCall(name=str(name), arguments=arguments, call_id=call_id)
    return HarnessToolCall(name="local_shell", arguments={"command": str(value)})


def _display_tool_call(tool_call: HarnessToolCall) -> str:
    lowered = tool_call.name.lower()
    if lowered in SHELL_TOOL_NAMES:
        return _shell_command(tool_call, None)[0]
    if lowered in WRITE_STDIN_TOOL_NAMES:
        session_id = _tool_int(tool_call.arguments.get("session_id"))
        chars = str(tool_call.arguments.get("chars") or "")
        return f"write_stdin(session_id={session_id}, chars={len(chars)} chars)"
    if lowered in PATCH_TOOL_NAMES:
        return _patch_display(_patch_text(tool_call.arguments))
    return f"<unsupported tool {tool_call.name}>"


def _shell_command(
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
) -> tuple[str, int | None]:
    args = _shell_args(tool_call.arguments)
    command = _shell_command_text(args)
    workdir = str(args.get("workdir") or "").strip()
    if workdir:
        command = f"cd {shlex.quote(workdir)} && {command}"
    return str(command), _tool_timeout(args, default_timeout_sec)


def _shell_command_text(args: dict[str, Any]) -> str:
    return str(
        args.get("command")
        or args.get("cmd")
        or args.get("shell")
        or args.get("keystrokes")
        or args.get("input")
        or ""
    )


def _shell_args(args: dict[str, Any]) -> dict[str, Any]:
    action = args.get("action")
    if isinstance(action, dict):
        merged = dict(action)
        merged.update({key: value for key, value in args.items() if key != "action"})
        args = merged
    args = _normalize_native_shell_args(args)
    commands = args.get("commands")
    if isinstance(commands, str):
        try:
            commands = json.loads(commands) if commands.startswith(("[", "{")) else [commands]
        except json.JSONDecodeError:
            commands = []
    if isinstance(commands, dict):
        commands = [commands]
    if isinstance(commands, list):
        for item in commands:
            if isinstance(item, dict):
                command = item.get("keystrokes") or item.get("command") or item.get("cmd")
                if command:
                    merged = dict(item)
                    merged.update({key: value for key, value in args.items() if key != "commands"})
                    merged["command"] = command
                    return merged
            elif item:
                merged = dict(args)
                merged["command"] = str(item)
                return merged
    return args


def _normalize_native_shell_args(args: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(args)
    if "working_directory" in normalized and "workdir" not in normalized:
        normalized["workdir"] = normalized["working_directory"]
    command = normalized.get("command")
    if isinstance(command, list):
        argv = [str(item) for item in command]
        if (
            len(argv) >= 3
            and Path(argv[0]).name in {"bash", "sh", "zsh"}
            and argv[1]
            in {
                "-c",
                "-lc",
            }
        ):
            normalized["shell"] = argv[0]
            normalized["login"] = argv[1] == "-lc"
            normalized["command"] = argv[2]
        else:
            normalized["command"] = shlex.join(argv)
    return normalized


def _tool_timeout(args: dict[str, Any], default: int | None = None) -> int | None:
    timeout_ms = args.get("timeout_ms")
    if isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool):
        return max(1, int((float(timeout_ms) + 999) // 1000))
    if isinstance(timeout_ms, str):
        try:
            return max(1, int((float(timeout_ms.strip()) + 999) // 1000))
        except ValueError:
            pass
    value = (
        args.get("timeout_sec")
        if args.get("timeout_sec") is not None
        else args.get("duration", args.get("timeout", default))
    )
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return max(1, int(value))
    if isinstance(value, str):
        try:
            return max(1, int(float(value.strip())))
        except ValueError:
            pass
    return default


def _tool_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _tool_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _resolve_workdir(workdir: str | None, environment: Any) -> str | None:
    if not workdir:
        return None
    if workdir.startswith("/"):
        return workdir
    config = getattr(environment, "task_env_config", None)
    base = getattr(environment, "_workdir", None) or getattr(config, "workdir", None)
    if not base:
        return workdir
    return os.path.normpath(f"{base.rstrip('/')}/{workdir}")


def _unified_exec_metadata(result: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(result, "chunk_id", ""),
        "wall_time_seconds": getattr(result, "wall_time_seconds", 0.0),
        "exit_code": getattr(result, "return_code", None),
        "session_id": getattr(result, "session_id", None),
        "original_token_count": getattr(result, "original_token_count", None),
    }


def _patch_text(args: dict[str, Any]) -> str:
    patch = args.get("patch") or args.get("input") or args.get("diff") or ""
    return str(patch)


def _extract_apply_patch(command: str) -> str | None:
    stripped = command.lstrip()
    if not stripped.startswith("apply_patch"):
        return None
    if "*** Begin Patch" not in command:
        return None
    lines = command.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("*** Begin Patch"):
            body = lines[index:]
            while body and body[-1].strip() in {"PATCH", "EOF", "'PATCH'", '"PATCH"'}:
                body.pop()
            return "\n".join(body)
    return None


def _patch_display(patch: str) -> str:
    if len(patch) <= MAX_OBSERVATION_CHARS:
        return f"apply_patch <<'PATCH'\n{patch}\nPATCH"
    omitted = len(patch) - MAX_OBSERVATION_CHARS
    return (
        f"apply_patch <<'PATCH'\n<omitted {omitted} chars>\n{patch[-MAX_OBSERVATION_CHARS:]}\nPATCH"
    )


def _apply_patch_command(patch: str) -> str:
    encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    return f"""PY=$(command -v python3 || command -v python); "$PY" - <<'PY'
import base64, os, pathlib, subprocess, sys, tempfile

PATCH = base64.b64decode({encoded!r}).decode("utf-8")

def safe_path(name):
    path = pathlib.PurePosixPath(name.strip())
    if path.is_absolute() or ".." in path.parts or not str(path):
        raise SystemExit(f"unsafe patch path: {{name}}")
    return pathlib.Path(str(path))

def read_lines(path):
    if not path.exists():
        raise SystemExit(f"patch target missing: {{path}}")
    return path.read_text(encoding="utf-8", errors="surrogateescape").splitlines()

def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(lines) + ("\\n" if lines else ""), encoding="utf-8", errors="surrogateescape")

def apply_hunk(lines, hunk, start):
    old = [text for tag, text in hunk if tag in " -"]
    new = [text for tag, text in hunk if tag in " +"]
    if not old:
        lines[start:start] = new
        return start + len(new)
    for index in range(start, len(lines) - len(old) + 1):
        if lines[index:index + len(old)] == old:
            lines[index:index + len(old)] = new
            return index + len(new)
    raise SystemExit("patch hunk did not match")

def parse_custom(text):
    rows = text.splitlines()
    if not rows or rows[0].strip() != "*** Begin Patch":
        return None
    ops, index = [], 1
    def next_op(row):
        return (
            row.strip() == "*** End Patch"
            or row.startswith("*** Add File: ")
            or row.startswith("*** Delete File: ")
            or row.startswith("*** Update File: ")
        )
    while index < len(rows):
        row = rows[index]
        if row.strip() == "*** End Patch":
            return ops
        if row.startswith("*** Add File: "):
            name, index, body = row.removeprefix("*** Add File: "), index + 1, []
            while index < len(rows) and not rows[index].startswith("*** "):
                if not rows[index].startswith("+"):
                    raise SystemExit("add-file patch line must start with +")
                body.append(rows[index][1:])
                index += 1
            ops.append(("add", name, None, [body]))
            continue
        if row.startswith("*** Delete File: "):
            ops.append(("delete", row.removeprefix("*** Delete File: "), None, []))
            index += 1
            continue
        if row.startswith("*** Update File: "):
            name, move_to, hunks = row.removeprefix("*** Update File: "), None, []
            index += 1
            current = []
            while index < len(rows) and not next_op(rows[index]):
                row = rows[index]
                if row.startswith("*** Move to: "):
                    move_to = row.removeprefix("*** Move to: ")
                elif row.startswith("@@"):
                    if current:
                        hunks.append(current)
                        current = []
                elif row == "*** End of File":
                    pass
                elif row[:1] in (" ", "+", "-"):
                    current.append((row[:1], row[1:]))
                else:
                    raise SystemExit(f"bad update patch line: {{row}}")
                index += 1
            if current:
                hunks.append(current)
            ops.append(("update", name, move_to, hunks))
            continue
        raise SystemExit(f"unknown patch operation: {{row}}")
    raise SystemExit("missing *** End Patch")

def apply_custom(text):
    for op, name, move_to, hunks in parse_custom(text):
        path = safe_path(name)
        if op == "add":
            if path.exists():
                raise SystemExit(f"patch add target exists: {{path}}")
            write_lines(path, hunks[0])
        elif op == "delete":
            if not path.exists():
                raise SystemExit(f"patch delete target missing: {{path}}")
            path.unlink()
        else:
            lines, cursor = read_lines(path), 0
            for hunk in hunks:
                cursor = apply_hunk(lines, hunk, cursor)
            target = safe_path(move_to) if move_to else path
            write_lines(target, lines)
            if target != path:
                path.unlink()

def apply_unified(text):
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        filename = handle.name
    try:
        commands = (
            ["git", "apply", "--whitespace=nowarn", filename],
            ["patch", "-p1", "-i", filename],
            ["patch", "-p0", "-i", filename],
        )
        errors = []
        for command in commands:
            try:
                result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            except FileNotFoundError as exc:
                errors.append(str(exc))
                continue
            if result.returncode == 0:
                return
            errors.append(result.stderr or result.stdout)
        raise SystemExit("\\n".join(errors)[-4000:])
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass

if not PATCH.strip():
    raise SystemExit("empty patch")
if PATCH.lstrip().startswith("*** Begin Patch"):
    apply_custom(PATCH.strip())
else:
    apply_unified(PATCH)
print("Patch applied.")
PY"""


async def _agents_context(environment: Any, workdir: str | None) -> list[dict[str, str]]:
    if not hasattr(environment, "exec"):
        return []
    command = _agents_context_command(workdir or ".")
    try:
        result = await environment.exec(command=command, timeout_sec=10)
    except Exception:
        return []
    if getattr(result, "return_code", 1):
        return []
    try:
        payload = json.loads(getattr(result, "stdout", "") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    kept = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            kept.append({"path": item["path"], "content": str(item.get("content") or "")})
    return kept


def _agents_context_command(workdir: str) -> str:
    root = json.dumps(workdir)
    return f"""PY=$(command -v python3 || command -v python); "$PY" - <<'PY'
import json
from pathlib import Path

root = Path({root})
try:
    root = root.resolve()
except OSError:
    pass
paths = []
for parent in (root, *root.parents):
    candidate = parent / "AGENTS.md"
    if candidate.is_file():
        paths.append(candidate)
try:
    paths.extend(sorted(root.rglob("AGENTS.md")))
except OSError:
    pass
seen, out, total = set(), [], 0
for path in paths:
    try:
        resolved = path.resolve()
    except OSError:
        continue
    if resolved in seen or not path.is_file():
        continue
    seen.add(resolved)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    text = text[:12000]
    total += len(text)
    out.append({{"path": str(path), "content": text}})
    if total >= 40000 or len(out) >= 20:
        break
print(json.dumps(out))
PY"""


def _tail(text: str) -> str:
    return text[-MAX_OBSERVATION_CHARS:]


def detect_harbor_executable() -> str | None:
    for name in ("harbor", "hb"):
        if shutil.which(name):
            return name
    local_harbor = Path.home() / ".local" / "bin" / "harbor"
    if local_harbor.exists():
        return str(local_harbor)
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
        spec.agent_import_path,
        "--agent-kwarg",
        f"candidate_dir={spec.candidate_dir}",
        "--quiet",
        "--yes",
    ]
    for item in spec.agent_kwargs:
        command.extend(["--agent-kwarg", item])
    if spec.backend == "slurm-pyxis":
        command.extend(
            [
                "--environment-build-timeout-multiplier",
                SLURM_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER,
                "--environment-import-path",
                SLURM_PYXIS_ENV_IMPORT_PATH,
                "--environment-kwarg",
                "sqsh_cache_dir=/wbl-fast/usrs/trost/tbench-sqsh-cache/images",
                "--environment-kwarg",
                "docker_tar_cache_dir=/wbl-fast/usrs/ee/agent-collab/docker-image-cache",
                "--environment-kwarg",
                "shared_dir=/wbl-fast/usrs/trost/harbor-slurm-pyxis",
            ]
        )
    for task in spec.tasks:
        command.extend([task_flag, task])
    runnable = bool(detect_harbor_executable() or executable) and has_harbor_run_flags(help_blob)
    note = (
        "Using Harbor terminal-bench@2.0 dataset filters."
        if runnable
        else "Harbor CLI was not found or did not expose expected run flags."
    )
    return HarborCommandPlan(command=command, runnable=runnable, task_flag=task_flag, note=note)
