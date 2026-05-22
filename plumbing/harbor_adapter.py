from __future__ import annotations

import base64
import json
import shlex
import shutil
import subprocess
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
PATCH_TOOL_NAMES = {"apply_patch", "patch"}


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
        task = TaskContext(instruction=instruction)
        history: list[CommandResult] = []
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        done = False
        turn_index = 0
        token = set_trace_dir(self.logs_dir)
        try:
            while True:
                turn_index += 1
                turn = agent.next_command(task, history)
                tool_calls = _turn_tool_calls(turn)
                if turn.done or not tool_calls:
                    done = turn.done
                    break
                for tool_call in tool_calls:
                    record = await _execute_tool_call(environment, tool_call, turn.timeout_sec)
                    history.append(record)
                    self._write_turn_log(turn_index, record)
                    turn_index += 1
                turn_index -= 1
        finally:
            reset_trace_dir(token)
            self._write_result_logs(history, done)
            context.metadata = {
                "candidate_dir": str(self.candidate_dir),
                "done": done,
                "turns": len(history),
                "last_return_code": history[-1].return_code if history else None,
            }

    def _write_turn_log(self, turn_index: int, record: CommandResult) -> None:
        (self.logs_dir / f"harness-turn-{turn_index:02d}.json").write_text(
            json.dumps(record.__dict__, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_result_logs(self, history: list[CommandResult], done: bool) -> None:
        payload = {
            "done": done,
            "turns": len(history),
            "commands": [record.command for record in history],
            "last_return_code": history[-1].return_code if history else None,
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
    turn = agent.next_command(TaskContext(instruction=instruction, working_dir=working_dir), [])
    calls = _turn_tool_calls(turn)
    if calls:
        return _display_tool_call(calls[0])
    return turn.command


async def _execute_tool_call(
    environment: Any,
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
) -> CommandResult:
    name = tool_call.name.strip() or "local_shell"
    lowered = name.lower()
    if lowered in SHELL_TOOL_NAMES:
        command, timeout_sec = _shell_command(tool_call, default_timeout_sec)
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
            timeout_sec=_tool_timeout(tool_call.arguments, default_timeout_sec) or 30,
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
        if timeout_sec is None or "timed out" not in str(exc).lower():
            raise
        return CommandResult(
            command=display_command or command,
            return_code=124,
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
    if lowered in PATCH_TOOL_NAMES:
        return _patch_display(_patch_text(tool_call.arguments))
    return f"<unsupported tool {tool_call.name}>"


def _shell_command(
    tool_call: HarnessToolCall,
    default_timeout_sec: int | None,
) -> tuple[str, int | None]:
    args = _shell_args(tool_call.arguments)
    command = (
        args.get("command")
        or args.get("cmd")
        or args.get("shell")
        or args.get("keystrokes")
        or args.get("input")
        or ""
    )
    workdir = str(args.get("workdir") or "").strip()
    if workdir:
        command = f"cd {shlex.quote(workdir)} && {command}"
    return str(command), _tool_timeout(args, default_timeout_sec)


def _shell_args(args: dict[str, Any]) -> dict[str, Any]:
    action = args.get("action")
    if isinstance(action, dict):
        merged = dict(action)
        merged.update({key: value for key, value in args.items() if key != "action"})
        args = merged
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


def _patch_text(args: dict[str, Any]) -> str:
    patch = args.get("patch") or args.get("input") or args.get("diff") or ""
    return str(patch)


def _patch_display(patch: str) -> str:
    if len(patch) <= MAX_OBSERVATION_CHARS:
        return f"apply_patch <<'PATCH'\n{patch}\nPATCH"
    omitted = len(patch) - MAX_OBSERVATION_CHARS
    return f"apply_patch <<'PATCH'\n<omitted {omitted} chars>\n{patch[-MAX_OBSERVATION_CHARS:]}\nPATCH"


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
