import json
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ToolModelResult, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessTurn, TaskContext

FIRST_COMMAND = "pwd && ls -la && find . -maxdepth 2 -type f | sort | sed -n '1,160p'"
MAX_PROMPT_HISTORY = 8
MAX_LONG_HISTORY = 3
MAX_COMMAND_CHARS = 14000
MAX_OUTPUT_CHARS = 30000
DEFAULT_TIMEOUT = 45
MIN_TIMEOUT = 2
MAX_TIMEOUT = 240
MAX_REPEAT = 2
EDIT_MARKERS = (
    "cat >",
    "cat <<",
    "tee ",
    "sed -i",
    "perl -pi",
    "python - <<",
    "python3 - <<",
    "write_text",
    "open(",
    "chmod ",
    "git apply",
    "patch ",
)
VERIFY_WORDS = (
    "test",
    "pytest",
    "unittest",
    "check",
    "verify",
    "cmp ",
    "diff ",
    "curl ",
    "make ",
    "cargo test",
    "npm test",
    "go test",
    "python",
    "python3",
)
RISKY_UNBOUNDED = (
    "tail -f",
    "watch ",
    "less ",
    "more ",
    "top",
    "htop",
    "vim ",
    "vi ",
    "nano ",
    "emacs ",
    "read ",
    "python -i",
    "python3 -i",
)
SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
JSON_ACTION_HINT = re.compile(r'"\s*(action|command|timeout_sec|cmd|tool|arguments)\s*"\s*:')

SYSTEM = (
    "You are a terminal coding agent. Use the provided tools instead of free-form text. "
    "Call execute_commands with analysis, plan, and command keystrokes, or call task_complete. "
    "Work in a loop: inspect/list/read, edit narrowly, verify with the smallest relevant check, then done. "
    "Prefer fast noninteractive commands. Bound output with sed/head/tail when reading large files. "
    "If a command fails, inspect the error or change approach; do not repeat it unchanged. "
    "If a tool is unavailable, fall back to POSIX shell, find, grep, sed, awk, or the available Python. "
    "Only mark done when recent terminal output gives evidence that the requested behavior now works."
)

TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "name": "execute_commands",
        "description": "Execute terminal commands with a short analysis and plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "plan": {"type": "string"},
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                        "required": ["keystrokes"],
                    },
                },
            },
            "required": ["analysis", "plan", "commands"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "task_complete",
        "description": "Signal that the task is complete.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class ActionKind(str, Enum):
    RUN = "run"
    DONE = "done"


class CommandRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Action:
    kind: ActionKind = ActionKind.RUN
    command: str = ""
    timeout_sec: int | None = None
    raw: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class CommandProfile:
    command: str
    words: list[str]
    timeout: int | None
    risk: CommandRisk
    writes: bool
    verifies: bool
    background: bool
    interactive: bool
    uses_network: bool
    long_running: bool
    bounded_output: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class HistoryStats:
    turn: int
    failures: int
    successes: int
    writes: int
    verifies: int
    repeats: int
    last_exit: int
    last_command: str
    last_stdout: str
    last_stderr: str
    recent_commands: list[str]
    recent_failures: list[str]
    recent_notes: list[str]

    @property
    def has_recent_write(self) -> bool:
        return self.writes > 0

    @property
    def has_recent_verify(self) -> bool:
        return self.verifies > 0

    @property
    def last_failed(self) -> bool:
        return self.last_exit != 0


@dataclass
class PromptBundle:
    system: str
    user: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


class CandidateHarness(BaseHarness):
    def __init__(self) -> None:
        self._parser_notes: list[str] = []
        self._guard_notes: list[str] = []
        self._done_block_count = 0
        self._last_model_text = ""

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        if not history:
            return HarnessTurn(done=False, command=FIRST_COMMAND, timeout_sec=20)
        stats = _history_stats(history)
        prompt = _build_prompt(task, history, stats, self._parser_notes, self._guard_notes)
        self._parser_notes = []
        self._guard_notes = []
        result = call_terminal_model_with_tools(prompt.messages(), TOOLS)
        self._last_model_text = result.content
        action = _parse_tool_result(result) or _parse_action(result.content)
        self._parser_notes.extend(action.notes)
        action = self._repair_action(action)
        if action.kind == ActionKind.DONE:
            return self._finish_or_verify(history, stats)
        profile = _profile_command(action.command, action.timeout_sec)
        command, guard_notes = _guard_command(profile, history, stats)
        self._guard_notes.extend(guard_notes)
        if not command.strip():
            command = _fallback_command(stats)
            self._guard_notes.append("empty command replaced by a compact state inspection")
        timeout = _select_timeout(command, profile.timeout, stats)
        return HarnessTurn(done=False, command=command, timeout_sec=timeout)

    def _repair_action(self, action: Action) -> Action:
        if action.kind == ActionKind.DONE:
            return action
        command = _clean_command(action.command)
        nested = _extract_action_from_command(command)
        if nested is not None and nested.command:
            nested.notes.append(
                "command contained an embedded action object; extracted its command"
            )
            if nested.timeout_sec is None:
                nested.timeout_sec = action.timeout_sec
            return nested
        if _looks_like_action_garbage(command):
            action.notes.append("malformed action-like shell text was rejected")
            action.command = ""
            return action
        action.command = command
        return action

    def _finish_or_verify(self, history: list[CommandResult], stats: HistoryStats) -> HarnessTurn:
        if _done_allowed(history, stats):
            return HarnessTurn(done=True, command="", timeout_sec=None)
        self._done_block_count += 1
        self._guard_notes.append("done was delayed because recent verification evidence was weak")
        command = _verification_probe(history, stats, self._done_block_count)
        return HarnessTurn(done=False, command=command, timeout_sec=45)


def _build_prompt(
    task: TaskContext,
    history: list[CommandResult],
    stats: HistoryStats,
    parser_notes: list[str],
    guard_notes: list[str],
) -> PromptBundle:
    sections = [
        f"Task:\n{task.instruction}",
        f"State:\n{_state_text(stats)}",
        f"Recent terminal history:\n{_history_text(history)}",
    ]
    notes = _format_notes(parser_notes, guard_notes)
    if notes:
        sections.append(f"Harness feedback:\n{notes}")
    sections.append(
        "Use execute_commands for exactly one next shell command. "
        "If the last command failed, fix the cause or inspect a different angle. "
        "If edits were made, run a relevant check before done. "
        "Use task_complete only after recent verification evidence."
    )
    return PromptBundle(system=SYSTEM, user="\n\n".join(sections))


def _format_notes(parser_notes: list[str], guard_notes: list[str]) -> str:
    merged = []
    for note in parser_notes + guard_notes:
        note = note.strip()
        if note and note not in merged:
            merged.append(note)
    return "\n".join(f"- {note}" for note in merged[-8:])


def _history_text(history: list[CommandResult]) -> str:
    start = max(0, len(history) - MAX_PROMPT_HISTORY)
    rows = []
    for index, result in enumerate(history[start:], start + 1):
        rows.append(_history_row(index, result, index > len(history) - MAX_LONG_HISTORY))
    return "\n\n".join(rows)


def _history_row(index: int, result: CommandResult, compact: bool) -> str:
    stdout_limit = 1200 if compact else 2600
    stderr_limit = 800 if compact else 1400
    command = _single_line_command(result.command, 1600 if not compact else 900)
    stdout = _clip(result.stdout, stdout_limit)
    stderr = _clip(result.stderr, stderr_limit)
    return (
        f"[{index}] $ {command}\n"
        f"exit={result.return_code}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _state_text(stats: HistoryStats) -> str:
    bits = [
        f"turn={stats.turn}",
        f"last_exit={stats.last_exit}",
        f"recent_writes={stats.writes}",
        f"recent_verifications={stats.verifies}",
        f"recent_repeats={stats.repeats}",
    ]
    if stats.last_failed:
        bits.append("last command failed; change approach or inspect the error")
    if stats.has_recent_write and not stats.has_recent_verify:
        bits.append("a recent write needs verification before done")
    if stats.repeats:
        bits.append("avoid repeating commands without changing inputs")
    if stats.recent_failures:
        bits.append("recent failing commands: " + "; ".join(stats.recent_failures[-3:]))
    if stats.recent_notes:
        bits.extend(stats.recent_notes[-4:])
    return "\n".join(f"- {bit}" for bit in bits)


def _history_stats(history: list[CommandResult]) -> HistoryStats:
    recent = history[-8:]
    failures = sum(1 for item in recent if item.return_code != 0)
    successes = sum(1 for item in recent if item.return_code == 0)
    writes = sum(1 for item in recent if _command_writes(item.command))
    verifies = sum(1 for item in recent if _command_verifies(item.command))
    repeats = _repeat_count(history)
    recent_failures = [
        _single_line_command(item.command, 140) for item in recent if item.return_code != 0
    ]
    recent_notes = _history_notes(history)
    last = history[-1]
    return HistoryStats(
        turn=len(history) + 1,
        failures=failures,
        successes=successes,
        writes=writes,
        verifies=verifies,
        repeats=repeats,
        last_exit=last.return_code,
        last_command=last.command,
        last_stdout=last.stdout or "",
        last_stderr=last.stderr or "",
        recent_commands=[item.command for item in recent],
        recent_failures=recent_failures,
        recent_notes=recent_notes,
    )


def _history_notes(history: list[CommandResult]) -> list[str]:
    notes = []
    last = history[-1]
    joined = "\n".join([last.stdout or "", last.stderr or ""])
    low = joined.lower()
    if "command not found" in low:
        notes.append("a required command was unavailable; use an installed fallback")
    if "no such file" in low or "not found" in low:
        notes.append("a path or command was missing; list nearby files before retrying")
    if "permission denied" in low:
        notes.append("permission failure observed; inspect ownership and executable bits")
    if "timed out" in low or "timeout" in low:
        notes.append("a command timed out; use a narrower command or explicit timeout")
    if _looks_like_action_garbage(last.command):
        notes.append("previous model response was malformed and ran as shell text")
    if len(history) >= 2 and history[-1].command.strip() == history[-2].command.strip():
        notes.append("previous command was repeated unchanged")
    return notes


def _parse_tool_result(result: ToolModelResult) -> Action | None:
    for call in result.tool_calls:
        if call.name == "task_complete":
            return Action(kind=ActionKind.DONE, raw=result.content)
        if call.name != "execute_commands":
            continue
        args = call.arguments
        commands = args.get("commands", [])
        if isinstance(commands, str):
            try:
                commands = json.loads(commands)
            except json.JSONDecodeError:
                commands = []
        if isinstance(commands, dict):
            commands = [commands]
        if not isinstance(commands, list):
            commands = []
        for item in commands:
            command = _tool_command(item)
            if command:
                return Action(
                    kind=ActionKind.RUN,
                    command=command,
                    timeout_sec=_tool_timeout(item),
                    raw=result.content,
                    notes=_tool_notes(args),
                )
        return Action(
            command="",
            raw=result.content,
            notes=["execute_commands contained no runnable command"],
        )
    return None


def _tool_command(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("keystrokes", "command", "cmd", "shell"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _tool_timeout(item: object) -> int | None:
    if not isinstance(item, dict):
        return None
    value = item.get("duration")
    if value is None:
        value = item.get("timeout_sec")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return max(MIN_TIMEOUT, min(MAX_TIMEOUT, int(value)))
    if isinstance(value, str) and value.strip().isdigit():
        return max(MIN_TIMEOUT, min(MAX_TIMEOUT, int(value.strip())))
    return None


def _tool_notes(args: dict[str, object]) -> list[str]:
    notes = []
    for key in ("analysis", "plan"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            notes.append(f"{key}: {value.strip()[:220]}")
    return notes


def _parse_action(text: str) -> Action:
    raw = text or ""
    stripped = _strip_fence(raw.strip())
    if not stripped:
        return Action(command="", raw=raw, notes=["empty model response"])
    if stripped.upper().startswith("DONE"):
        return Action(kind=ActionKind.DONE, raw=raw)
    objects = list(_iter_json_objects(stripped))
    if objects:
        action = _choose_action(objects, stripped, raw)
        if len(objects) > 1:
            action.notes.append(
                f"model returned {len(objects)} JSON objects; only the selected action was used"
            )
        return action
    value = _try_json_object(stripped)
    if value is not None:
        return _normalize_action(value, raw)
    possible = _extract_balanced_json(stripped)
    if possible is not None:
        action = _normalize_action(possible, raw)
        action.notes.append("extracted a JSON object from surrounding text")
        return action
    return Action(command=stripped, raw=raw, notes=["model response was not valid JSON"])


def _choose_action(objects: list[dict[str, object]], text: str, raw: str) -> Action:
    normalized = [_normalize_action(value, raw) for value in objects]
    for item in normalized:
        if item.kind == ActionKind.RUN and item.command.strip():
            return item
    for item in normalized:
        if item.kind == ActionKind.DONE:
            return item
    return Action(command=text, raw=raw, notes=["JSON objects did not contain a runnable command"])


def _iter_json_objects(text: str) -> Iterable[dict[str, object]]:
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, dict):
            yield value
        pos = start + max(end, 1)


def _try_json_object(text: str) -> dict[str, object] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_balanced_json(text: str) -> dict[str, object] | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = _try_json_object(text[start : index + 1])
                    if candidate is not None:
                        return candidate
                    break
        start = text.find("{", start + 1)
    return None


def _normalize_action(value: dict[str, object], raw: str) -> Action:
    action_name = str(
        value.get("action") or value.get("tool") or value.get("name") or value.get("type") or "run"
    ).lower()
    args = value.get("args") or value.get("arguments") or value.get("input") or {}
    if isinstance(args, str):
        args = _try_json_object(args) or _extract_balanced_json(args) or {}
    if action_name in {"done", "finish", "final", "complete", "task_complete"}:
        return Action(kind=ActionKind.DONE, raw=raw)
    command = value.get("command") or value.get("cmd") or value.get("shell")
    timeout = value.get("timeout_sec") or value.get("timeout") or value.get("duration")
    if command is None and isinstance(args, dict):
        command = args.get("command") or args.get("cmd") or args.get("shell")
    if timeout is None and isinstance(args, dict):
        timeout = args.get("timeout_sec") or args.get("timeout") or args.get("duration")
    notes = []
    if command is None:
        notes.append("JSON action lacked a command field")
    return Action(
        kind=ActionKind.RUN,
        command=str(command or ""),
        timeout_sec=_coerce_timeout(timeout),
        raw=raw,
        notes=notes,
    )


def _extract_action_from_command(command: str) -> Action | None:
    if not _looks_like_action_garbage(command):
        return None
    objects = list(_iter_json_objects(command))
    if not objects:
        return None
    return _choose_action(objects, command, command)


def _looks_like_action_garbage(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    if JSON_ACTION_HINT.search(stripped) and stripped.startswith("{"):
        return True
    if stripped.count('{"action"') > 1:
        return True
    if re.search(r"\}\s*\{", stripped) and JSON_ACTION_HINT.search(stripped):
        return True
    return False


def _profile_command(command: str, timeout: int | None) -> CommandProfile:
    cleaned = _clean_command(command)
    words = _shell_words(cleaned)
    writes = _command_writes(cleaned)
    verifies = _command_verifies(cleaned)
    background = _has_background(cleaned)
    interactive = _is_interactive(cleaned, words)
    uses_network = _uses_network(cleaned, words)
    long_running = _is_long_running(cleaned, words)
    bounded = _bounded_output(cleaned)
    risk = _risk_level(cleaned, writes, background, interactive, long_running)
    notes = []
    if interactive:
        notes.append("interactive command detected")
    if background:
        notes.append("background command detected")
    if not bounded and _could_emit_large_output(cleaned):
        notes.append("command may emit large output")
    return CommandProfile(
        command=cleaned,
        words=words,
        timeout=timeout,
        risk=risk,
        writes=writes,
        verifies=verifies,
        background=background,
        interactive=interactive,
        uses_network=uses_network,
        long_running=long_running,
        bounded_output=bounded,
        notes=notes,
    )


def _guard_command(
    profile: CommandProfile,
    history: list[CommandResult],
    stats: HistoryStats,
) -> tuple[str, list[str]]:
    command = profile.command
    notes = list(profile.notes)
    if not command:
        return "", notes
    if len(command) > MAX_COMMAND_CHARS:
        command = command[:MAX_COMMAND_CHARS]
        notes.append("overlong command was clipped")
    if _too_many_repeats(command, history):
        notes.append("repeated command replaced by a state inspection")
        return _fallback_command(stats), notes
    if profile.interactive:
        notes.append("interactive command replaced by a noninteractive inspection")
        return _fallback_command(stats), notes
    if profile.background and not _background_is_bounded(command):
        command = _bound_background(command)
        notes.append("background command was wrapped with a short status check")
    if _could_emit_large_output(command) and not profile.bounded_output and not profile.writes:
        command = _append_output_bound(command)
        notes.append("large-output command was bounded")
    if stats.last_failed and command.strip() == stats.last_command.strip():
        command = _fallback_command(stats)
        notes.append("unchanged retry after failure was replaced")
    return command, notes


def _fallback_command(stats: HistoryStats) -> str:
    probes = [
        "pwd",
        "ls -la | sed -n '1,120p'",
        "find . -maxdepth 2 -type f | sort | sed -n '1,160p'",
    ]
    if stats.last_failed:
        probes.append("printf '\\nLast command failed; inspect paths and available tools.\\n'")
    return " && ".join(probes)


def _verification_probe(
    history: list[CommandResult],
    stats: HistoryStats,
    block_count: int,
) -> str:
    candidates = []
    for item in reversed(history):
        cmd = item.command.strip()
        if _command_verifies(cmd) and item.return_code == 0:
            candidates.append(_repeatable_verify(cmd))
        if len(candidates) >= 2:
            break
    if candidates:
        return candidates[0]
    if block_count <= 1:
        return (
            "pwd && "
            "find . -maxdepth 3 -type f | sort | sed -n '1,200p' && "
            "printf '\\nRecent changes or generated files should be verified with the relevant local check.\\n'"
        )
    if stats.has_recent_write:
        return "find . -maxdepth 3 -type f -newer . 2>/dev/null | sort | sed -n '1,120p' || find . -maxdepth 2 -type f | sort | sed -n '1,120p'"
    return "pwd && ls -la && find . -maxdepth 2 -type f | sort | sed -n '1,120p'"


def _repeatable_verify(command: str) -> str:
    stripped = command.strip()
    if len(stripped) > 8000:
        return "printf 'Previous verification command was too large to repeat safely.\\n'"
    return stripped


def _done_allowed(history: list[CommandResult], stats: HistoryStats) -> bool:
    if not history:
        return False
    if stats.last_failed:
        return False
    if stats.has_recent_write and not stats.has_recent_verify:
        return False
    if len(history) < 2:
        return False
    if _looks_like_action_garbage(history[-1].command):
        return False
    if _command_verifies(history[-1].command):
        return history[-1].return_code == 0
    if stats.has_recent_verify and stats.successes >= 1:
        return True
    if not stats.has_recent_write and stats.successes >= 2:
        return True
    return False


def _select_timeout(command: str, requested: int | None, stats: HistoryStats) -> int:
    profile = _profile_command(command, requested)
    if requested is None:
        timeout = _default_timeout_for(profile, stats)
    else:
        timeout = requested
    if profile.interactive:
        timeout = min(timeout, 10)
    if profile.background:
        timeout = min(max(timeout, 5), 45)
    if _could_emit_large_output(command):
        timeout = min(timeout, 120)
    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))


def _default_timeout_for(profile: CommandProfile, stats: HistoryStats) -> int:
    command = profile.command
    low = command.lower()
    if profile.long_running:
        return 180
    if profile.uses_network:
        return 180
    if profile.verifies:
        return 120
    if profile.writes:
        return 90
    if "find " in low or "grep -r" in low or "grep -R" in command:
        return 60
    if stats.last_failed:
        return 35
    return DEFAULT_TIMEOUT


def _coerce_timeout(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if MIN_TIMEOUT <= value <= 600 else None
    if isinstance(value, float):
        rounded = int(value)
        return rounded if MIN_TIMEOUT <= rounded <= 600 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            number = int(stripped)
            return number if MIN_TIMEOUT <= number <= 600 else None
    return None


def _clean_command(text: str) -> str:
    text = _strip_fence(str(text)).strip()
    if text.upper().startswith("DONE"):
        return ""
    return _normalize_newlines(text)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _single_line_command(command: str, limit: int) -> str:
    command = _normalize_newlines(command).replace("\n", "\\n")
    return _clip(command, limit)


def _clip(text: str, limit: int) -> str:
    text = "" if text is None else str(text).strip()
    if len(text) <= limit:
        return text
    return f"<{len(text) - limit} chars omitted>\n{text[-limit:]}"


def _shell_words(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.replace("\n", " ").split()


def _command_writes(command: str) -> bool:
    low = command.lower()
    return any(marker in low for marker in EDIT_MARKERS) or any(
        token in low
        for token in (
            " > ",
            ">>",
            "mv ",
            "cp ",
            "rm ",
            "mkdir ",
            "touch ",
            "install ",
            "pip install",
            "npm install",
            "cargo add",
            "git commit",
        )
    )


def _command_verifies(command: str) -> bool:
    low = command.lower()
    return any(word in low for word in VERIFY_WORDS)


def _has_background(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    return bool(re.search(r"(^|[;&|]\s*)nohup\b", stripped)) or bool(
        re.search(r"(?<!&)&\s*($|[;])", stripped)
    )


def _background_is_bounded(command: str) -> bool:
    low = command.lower()
    return any(
        token in low for token in ("sleep ", "curl ", "timeout ", "pgrep ", "ss ", "netstat ")
    )


def _bound_background(command: str) -> str:
    return f"{command}\n" "sleep 2\n" "jobs || true\n" "ps -eo pid,cmd | sed -n '1,80p'"


def _is_interactive(command: str, words: list[str]) -> bool:
    low = command.lower()
    if any(token in low for token in RISKY_UNBOUNDED):
        return True
    if not words:
        return False
    first = _skip_assignments(words)
    return first in {"vim", "vi", "nano", "emacs", "less", "more", "top", "htop", "ssh"}


def _skip_assignments(words: list[str]) -> str:
    for word in words:
        if SHELL_ASSIGNMENT.match(word):
            continue
        if word in {"env", "time", "command", "xargs", "sudo"}:
            continue
        return word.rsplit("/", 1)[-1].lower()
    return ""


def _uses_network(command: str, words: list[str]) -> bool:
    low = command.lower()
    network_tokens = (
        "curl ",
        "wget ",
        "pip install",
        "npm install",
        "git clone",
        "apt-get",
        "apk add",
        "dnf install",
        "yum install",
        "ssh ",
        "scp ",
    )
    return any(token in low for token in network_tokens)


def _is_long_running(command: str, words: list[str]) -> bool:
    low = command.lower()
    long_tokens = (
        "make ",
        "cargo build",
        "cargo test",
        "npm test",
        "pytest",
        "go test",
        "docker ",
        "qemu",
        "snapshot_download",
        "pip install",
        "apt-get",
    )
    return any(token in low for token in long_tokens)


def _risk_level(
    command: str,
    writes: bool,
    background: bool,
    interactive: bool,
    long_running: bool,
) -> CommandRisk:
    if interactive or background:
        return CommandRisk.HIGH
    if writes or long_running:
        return CommandRisk.MEDIUM
    return CommandRisk.LOW


def _bounded_output(command: str) -> bool:
    low = command.lower()
    bounders = (
        "sed -n",
        "head",
        "tail",
        "wc ",
        "grep -n",
        "rg -n",
        "awk ",
        "python",
        "jq ",
        "cut ",
    )
    return any(token in low for token in bounders)


def _could_emit_large_output(command: str) -> bool:
    low = command.lower()
    if any(token in low for token in ("cat >", "cat <<", "tee ", "write_text", " > ")):
        return False
    if any(token in low for token in ("cat ", "find ", "ls -r", "grep -r", "grep -R", "rg ")):
        return True
    if "git diff" in low and "--stat" not in low and "sed -n" not in low:
        return True
    return False


def _append_output_bound(command: str) -> str:
    stripped = command.strip()
    if re.search(r"\|\s*(sed -n|head|tail|wc|jq|awk)\b", stripped):
        return stripped
    return f"( {stripped} ) | sed -n '1,240p'"


def _too_many_repeats(command: str, history: list[CommandResult]) -> bool:
    normalized = _canonical_command(command)
    count = 0
    for item in reversed(history[-6:]):
        if _canonical_command(item.command) == normalized:
            count += 1
    return count >= MAX_REPEAT


def _repeat_count(history: list[CommandResult]) -> int:
    if not history:
        return 0
    last = _canonical_command(history[-1].command)
    count = 0
    for item in reversed(history[:-1]):
        if _canonical_command(item.command) == last:
            count += 1
        else:
            break
    return count


def _canonical_command(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip())


def create_agent() -> CandidateHarness:
    return CandidateHarness()
