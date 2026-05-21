import json
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ToolModelResult, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessTurn, TaskContext

FIRST_COMMAND = "pwd && find . -maxdepth 2 -type f | sort | sed -n '1,160p'"
MAX_HISTORY_ITEMS = 18
MAX_OUTPUT_CHARS = 24000
MAX_PROMPT_HISTORY_CHARS = 36000
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 180
MIN_TIMEOUT = 3
RECOVERY_TIMEOUT = 10

SYSTEM_PROMPT = """
You are a terminal coding agent. Use the provided tools instead of free-form text.
Call execute_commands with analysis, plan, and command keystrokes, or call task_complete.
Think privately, but expose concise analysis and plan in the tool arguments.
Use a disciplined loop: inspect the workspace, read relevant files, make the smallest useful change, verify the changed behavior, then finish.
Prefer bounded commands. Use rg, find, sed -n, python snippets, and targeted tests. Avoid interactive programs, background servers, unbounded streams, and destructive cleanup.
When a command fails, inspect the failure or change approach. Do not repeat the same command without a reason.
Completion requires evidence from recent successful verification, especially after writes.
""".strip()

KIRA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "execute_commands",
        "description": "Execute terminal commands with a short analysis and plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Briefly describe the current state.",
                },
                "plan": {
                    "type": "string",
                    "description": "Briefly describe what the next command should accomplish.",
                },
                "commands": {
                    "type": "array",
                    "description": "One or more shell commands to run. The harness executes one per turn.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {
                                "type": "string",
                                "description": "Exact shell command text.",
                            },
                            "duration": {
                                "type": "number",
                                "description": "Expected command duration in seconds, capped by the harness.",
                            },
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
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


class ActionKind(str, Enum):
    RUN = "run"
    DONE = "done"
    RECOVER = "recover"


class RiskLevel(str, Enum):
    SAFE = "safe"
    WATCH = "watch"
    REWRITE = "rewrite"
    BLOCK = "block"


class CommandClass(str, Enum):
    INSPECT = "inspect"
    READ = "read"
    SEARCH = "search"
    EDIT = "edit"
    VERIFY = "verify"
    BUILD = "build"
    PACKAGE = "package"
    NETWORK = "network"
    PROCESS = "process"
    DESTRUCTIVE = "destructive"
    INTERACTIVE = "interactive"
    UNKNOWN = "unknown"


@dataclass
class ParsedAction:
    action: ActionKind
    command: str = ""
    timeout_sec: int | None = None
    raw: str = ""
    source: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class CommandDecision:
    command: str
    timeout_sec: int | None
    risk: RiskLevel
    classes: set[CommandClass]
    reason: str
    recovery: str = ""
    changed: bool = False
    refused: bool = False
    rewritten: bool = False


@dataclass
class HistorySignal:
    turn: int
    last_exit: int | None
    repeated_last: bool
    recent_edit: bool
    recent_verify: bool
    successful_verify_after_edit: bool
    recent_failure: bool
    recent_refusal: bool
    same_command_count: int
    last_command: str
    last_stdout_tail: str
    last_stderr_tail: str
    changed_files_hint: str
    summary: str
    facts: list[str]
    warnings: list[str]


@dataclass
class CommandFeatures:
    raw: str
    cleaned: str
    words: list[str]
    lowered: str
    first_word: str
    has_pipe: bool
    has_redirect: bool
    has_heredoc: bool
    has_subshell: bool
    has_background: bool
    has_glob: bool
    has_network: bool
    has_package: bool
    has_edit: bool
    has_verify: bool
    has_destructive: bool
    has_interactive: bool
    has_unbounded: bool
    has_timeout: bool
    requested_timeout: int | None
    line_count: int
    byte_count: int
    classes: set[CommandClass]


@dataclass
class PromptBundle:
    system: str
    user: str
    state_lines: list[str]
    policy_lines: list[str]


class TextClipper:
    def clip_middle(self, text: str, limit: int) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        head = max(0, limit // 3)
        tail = max(0, limit - head)
        omitted = len(text) - head - tail
        return text[:head] + f"\n<omitted {omitted} chars>\n" + text[-tail:]

    def clip_tail(self, text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return f"<omitted {len(text) - limit} chars>\n" + text[-limit:]

    def one_line(self, text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def compact_lines(self, lines: Iterable[str], limit: int) -> str:
        out: list[str] = []
        total = 0
        for line in lines:
            part = str(line).rstrip()
            total += len(part) + 1
            if total > limit:
                out.append(f"<truncated at {limit} chars>")
                break
            out.append(part)
        return "\n".join(out)


class JsonExtractor:
    def parse(self, text: str) -> list[Any]:
        text = self._strip_outer(text or "")
        values: list[Any] = []
        values.extend(self._decode_direct(text))
        values.extend(self._decode_fenced(text))
        values.extend(self._decode_concatenated(text))
        values.extend(self._decode_balanced(text))
        values.extend(self._decode_argument_strings(values))
        return self._dedupe(values)

    def _strip_outer(self, text: str) -> str:
        text = text.strip().strip("\ufeff")
        text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _decode_direct(self, text: str) -> list[Any]:
        if not text:
            return []
        try:
            return [json.loads(text)]
        except Exception:
            return []

    def _decode_fenced(self, text: str) -> list[Any]:
        values: list[Any] = []
        for block in re.findall(r"```(?:json|JSON|sh|bash)?\s*(.*?)```", text, flags=re.DOTALL):
            values.extend(self._decode_direct(block.strip()))
            values.extend(self._decode_concatenated(block.strip()))
        return values

    def _decode_concatenated(self, text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        values: list[Any] = []
        index = 0
        while index < len(text):
            match = re.search(r"[\[{]", text[index:])
            if match is None:
                break
            start = index + match.start()
            try:
                value, end = decoder.raw_decode(text[start:])
            except Exception:
                index = start + 1
                continue
            values.append(value)
            index = start + max(end, 1)
        return values

    def _decode_balanced(self, text: str) -> list[Any]:
        values: list[Any] = []
        for start, end in self._balanced_spans(text):
            fragment = text[start:end]
            values.extend(self._decode_direct(fragment))
        return values

    def _balanced_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        stack: list[str] = []
        start: int | None = None
        quote: str | None = None
        escape = False
        pairs = {"{": "}", "[": "]"}
        for index, char in enumerate(text):
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char in pairs:
                if not stack:
                    start = index
                stack.append(pairs[char])
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack and start is not None:
                    spans.append((start, index + 1))
                    start = None
        return spans

    def _decode_argument_strings(self, values: list[Any]) -> list[Any]:
        decoded: list[Any] = []
        for value in values:
            if isinstance(value, dict):
                for key in ("arguments", "args", "input"):
                    item = value.get(key)
                    if isinstance(item, str):
                        decoded.extend(self._decode_direct(item))
                        decoded.extend(self._decode_concatenated(item))
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, str):
                        decoded.extend(self._decode_direct(child))
        return decoded

    def _dedupe(self, values: list[Any]) -> list[Any]:
        seen: set[str] = set()
        out: list[Any] = []
        for value in values:
            try:
                key = json.dumps(value, sort_keys=True, default=str)
            except Exception:
                key = repr(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out


class ActionParser:
    def __init__(self) -> None:
        self.extractor = JsonExtractor()

    def parse(self, text: str) -> ParsedAction:
        raw = text or ""
        if self._looks_done(raw):
            return ParsedAction(ActionKind.DONE, raw=raw, source="text_done", confidence=0.45)
        candidates = self._json_candidates(raw)
        if candidates:
            return self._choose(candidates, raw)
        command = self._fallback_command(raw)
        if not command:
            return ParsedAction(ActionKind.RECOVER, raw=raw, source="empty", confidence=0.0)
        return ParsedAction(
            ActionKind.RUN, command=command, raw=raw, source="text", confidence=0.25
        )

    def _looks_done(self, text: str) -> bool:
        stripped = self._remove_fences(text).strip().lower()
        return stripped in {"done", "finished", "complete", "task_complete"}

    def _json_candidates(self, raw: str) -> list[ParsedAction]:
        values = self.extractor.parse(raw)
        actions: list[ParsedAction] = []
        for value in values:
            actions.extend(self._from_value(value, raw))
        return actions

    def _from_value(self, value: Any, raw: str) -> list[ParsedAction]:
        if isinstance(value, list):
            out: list[ParsedAction] = []
            for item in value:
                out.extend(self._from_value(item, raw))
            return out
        if not isinstance(value, dict):
            return []
        action = self._extract_action_name(value)
        args = self._extract_args(value)
        command = self._extract_command(value, args)
        timeout = self._extract_timeout(value, args)
        notes: list[str] = []
        if action in {"done", "finish", "finished", "complete", "task_complete", "final"}:
            return [ParsedAction(ActionKind.DONE, raw=raw, source="json", confidence=0.8)]
        if action in {"run", "command", "shell", "execute", "exec", "bash", "terminal"} or command:
            if not command and isinstance(args, str):
                command = args
            if not command:
                notes.append("json action requested run but command was empty")
                return [
                    ParsedAction(
                        ActionKind.RECOVER,
                        raw=raw,
                        source="json_empty",
                        confidence=0.2,
                        notes=notes,
                    )
                ]
            return [
                ParsedAction(
                    ActionKind.RUN,
                    command=str(command),
                    timeout_sec=timeout,
                    raw=raw,
                    source="json",
                    confidence=0.9,
                    notes=notes,
                )
            ]
        if "tool_calls" in value:
            return self._from_tool_calls(value.get("tool_calls"), raw)
        if "function" in value or "name" in value:
            return self._from_function_shape(value, raw)
        return []

    def _from_tool_calls(self, value: Any, raw: str) -> list[ParsedAction]:
        if not isinstance(value, list):
            return []
        out: list[ParsedAction] = []
        for item in value:
            out.extend(self._from_value(item, raw))
        return out

    def _from_function_shape(self, value: dict[str, Any], raw: str) -> list[ParsedAction]:
        function = value.get("function")
        if isinstance(function, dict):
            merged = dict(function)
            if "arguments" not in merged and "arguments" in value:
                merged["arguments"] = value["arguments"]
            return self._from_value(merged, raw)
        return []

    def _extract_action_name(self, value: dict[str, Any]) -> str:
        for key in ("action", "tool", "name", "type", "function"):
            item = value.get(key)
            if isinstance(item, str):
                return item.strip().lower()
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                return str(item.get("name")).strip().lower()
        return "run" if any(key in value for key in ("command", "cmd", "keystrokes")) else ""

    def _extract_args(self, value: dict[str, Any]) -> Any:
        args = value.get("args")
        if args is None:
            args = value.get("arguments")
        if args is None:
            args = value.get("input")
        if isinstance(args, str):
            decoded = self.extractor.parse(args)
            for item in decoded:
                if isinstance(item, dict):
                    return item
        return args

    def _extract_command(self, value: dict[str, Any], args: Any) -> Any:
        for key in ("command", "cmd", "shell", "keystrokes", "code"):
            item = value.get(key)
            if item is not None:
                return item
        if isinstance(args, dict):
            for key in ("command", "cmd", "shell", "keystrokes", "code"):
                item = args.get(key)
                if item is not None:
                    return item
            commands = args.get("commands")
            if isinstance(commands, list) and commands:
                first = commands[0]
                if isinstance(first, dict):
                    return first.get("command") or first.get("cmd") or first.get("keystrokes")
                if isinstance(first, str):
                    return first
        return None

    def _extract_timeout(self, value: dict[str, Any], args: Any) -> int | None:
        for key in ("timeout_sec", "timeout", "duration", "seconds"):
            parsed = self._to_int(value.get(key))
            if parsed is not None:
                return parsed
        if isinstance(args, dict):
            for key in ("timeout_sec", "timeout", "duration", "seconds"):
                parsed = self._to_int(args.get(key))
                if parsed is not None:
                    return parsed
        return None

    def _to_int(self, value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

    def _choose(self, candidates: list[ParsedAction], raw: str) -> ParsedAction:
        runnable = [item for item in candidates if item.action == ActionKind.RUN and item.command]
        done = [item for item in candidates if item.action == ActionKind.DONE]
        recover = [item for item in candidates if item.action == ActionKind.RECOVER]
        if runnable:
            best = runnable[0]
            best.raw = raw
            return best
        if done:
            best = done[0]
            best.raw = raw
            return best
        if recover:
            return recover[0]
        return ParsedAction(ActionKind.RECOVER, raw=raw, source="json_unusable", confidence=0.1)

    def _fallback_command(self, raw: str) -> str:
        text = self._remove_fences(raw).strip()
        text = re.sub(r"^\s*(command|cmd|shell)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _remove_fences(self, text: str) -> str:
        if "```" not in text:
            return text
        blocks = re.findall(r"```(?:json|JSON|sh|bash)?\s*(.*?)```", text, flags=re.DOTALL)
        if blocks:
            return "\n".join(blocks)
        return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))


class ShellCleaner:
    def clean(self, command: str) -> str:
        command = str(command or "")
        command = self._strip_json_echo(command)
        command = self._strip_fences(command)
        command = self._strip_prompt_prefix(command)
        command = self._normalize_newlines(command)
        command = self._strip_outer_shell_quotes(command)
        command = self._remove_trailing_done(command)
        command = self._trim_semicolon_noise(command)
        return command.strip()

    def _strip_json_echo(self, command: str) -> str:
        text = command.strip()
        if not text.startswith("{"):
            return command
        parsed = ActionParser().parse(text)
        if parsed.action == ActionKind.RUN and parsed.command and parsed.command != command:
            return parsed.command
        return command

    def _strip_fences(self, command: str) -> str:
        text = command.strip()
        blocks = re.findall(r"```(?:sh|bash|shell)?\s*(.*?)```", text, flags=re.DOTALL)
        if blocks:
            return "\n".join(block.strip() for block in blocks if block.strip())
        return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))

    def _strip_prompt_prefix(self, command: str) -> str:
        lines = command.splitlines()
        cleaned: list[str] = []
        for line in lines:
            line = re.sub(r"^\s*\$\s+", "", line)
            line = re.sub(r"^\s*>\s+", "", line)
            line = re.sub(r"^\s*(run|command|cmd)\s*:\s*", "", line, flags=re.IGNORECASE)
            cleaned.append(line)
        return "\n".join(cleaned)

    def _normalize_newlines(self, command: str) -> str:
        command = command.replace("\r\n", "\n").replace("\r", "\n")
        command = re.sub(r"\n{4,}", "\n\n", command)
        return command

    def _strip_outer_shell_quotes(self, command: str) -> str:
        text = command.strip()
        if len(text) < 2:
            return text
        if text[0] == text[-1] and text[0] in {"'", '"'} and "\n" not in text:
            try:
                parts = shlex.split(text)
            except Exception:
                return text
            if len(parts) == 1:
                return parts[0]
        return text

    def _remove_trailing_done(self, command: str) -> str:
        lines = command.splitlines()
        while lines and lines[-1].strip().lower() in {"done", "finish", "complete"}:
            lines.pop()
        return "\n".join(lines)

    def _trim_semicolon_noise(self, command: str) -> str:
        text = command.strip()
        text = re.sub(r";\s*$", "", text)
        return text


class CommandClassifier:
    def features(self, command: str, requested_timeout: int | None = None) -> CommandFeatures:
        cleaned = command.strip()
        words = self._words(cleaned)
        lowered = cleaned.lower()
        first = words[0] if words else ""
        classes = self._classes(cleaned, words, lowered)
        return CommandFeatures(
            raw=command,
            cleaned=cleaned,
            words=words,
            lowered=lowered,
            first_word=first,
            has_pipe="|" in cleaned,
            has_redirect=bool(re.search(r"(^|\s)(>|>>|<|2>|&>)", cleaned)),
            has_heredoc="<<" in cleaned,
            has_subshell="$(`" in cleaned or "$(`" in cleaned or "$(`" in cleaned,
            has_background=bool(re.search(r"(^|\s)(nohup\s|setsid\s)|&\s*$", cleaned)),
            has_glob=any(ch in cleaned for ch in ["*", "?", "["]),
            has_network=self._has_network(lowered),
            has_package=self._has_package(lowered),
            has_edit=self._has_edit(lowered),
            has_verify=self._has_verify(lowered),
            has_destructive=self._has_destructive(lowered),
            has_interactive=self._has_interactive(lowered, words),
            has_unbounded=self._has_unbounded(lowered, words),
            has_timeout=bool(re.search(r"(^|\s)(timeout|gtimeout)\s+", lowered)),
            requested_timeout=requested_timeout,
            line_count=len(cleaned.splitlines()),
            byte_count=len(cleaned.encode("utf-8", errors="ignore")),
            classes=classes,
        )

    def _words(self, command: str) -> list[str]:
        try:
            return shlex.split(command, comments=False, posix=True)
        except Exception:
            return re.findall(r"[^\s;|&]+", command)

    def _classes(self, command: str, words: list[str], lowered: str) -> set[CommandClass]:
        classes: set[CommandClass] = set()
        if self._has_edit(lowered):
            classes.add(CommandClass.EDIT)
        if self._has_verify(lowered):
            classes.add(CommandClass.VERIFY)
        if self._has_network(lowered):
            classes.add(CommandClass.NETWORK)
        if self._has_package(lowered):
            classes.add(CommandClass.PACKAGE)
        if self._has_destructive(lowered):
            classes.add(CommandClass.DESTRUCTIVE)
        if self._has_interactive(lowered, words):
            classes.add(CommandClass.INTERACTIVE)
        if self._has_process(lowered):
            classes.add(CommandClass.PROCESS)
        if self._has_search(lowered, words):
            classes.add(CommandClass.SEARCH)
        if self._has_read(lowered, words):
            classes.add(CommandClass.READ)
        if self._has_build(lowered, words):
            classes.add(CommandClass.BUILD)
        if not classes and self._has_inspect(lowered, words):
            classes.add(CommandClass.INSPECT)
        if not classes:
            classes.add(CommandClass.UNKNOWN)
        return classes

    def _has_search(self, lowered: str, words: list[str]) -> bool:
        return (
            bool(words and words[0] in {"rg", "grep", "find", "fd", "ag"})
            or " rg " in f" {lowered} "
        )

    def _has_read(self, lowered: str, words: list[str]) -> bool:
        if not words:
            return False
        return (
            words[0] in {"cat", "sed", "head", "tail", "less", "more", "nl", "awk", "jq"}
            or "sed -n" in lowered
        )

    def _has_inspect(self, lowered: str, words: list[str]) -> bool:
        if not words:
            return False
        return words[0] in {"pwd", "ls", "tree", "file", "stat", "wc", "du", "git"}

    def _has_edit(self, lowered: str) -> bool:
        markers = [
            "cat >",
            "cat >>",
            "tee ",
            "sed -i",
            "perl -pi",
            "python - <<",
            "python3 - <<",
            "apply_patch",
            "truncate ",
            "mv ",
            "cp ",
            "install ",
            "chmod ",
            "chown ",
        ]
        return any(marker in lowered for marker in markers) or bool(
            re.search(r"(^|\s)(touch|mkdir|rm|rmdir)\s", lowered)
        )

    def _has_verify(self, lowered: str) -> bool:
        markers = [
            "pytest",
            "npm test",
            "cargo test",
            "go test",
            "python -m pytest",
            "python3 -m pytest",
            "make test",
            "ctest",
            "bats",
            "ruff",
            "mypy",
            "py_compile",
            "shellcheck",
            "diff ",
            "cmp ",
            "sha256sum",
            "openssl verify",
        ]
        return any(marker in lowered for marker in markers) or bool(
            re.search(r"(^|\s)(test|check|verify|benchmark)\b", lowered)
        )

    def _has_build(self, lowered: str, words: list[str]) -> bool:
        markers = [
            "make",
            "cmake",
            "ninja",
            "cargo build",
            "npm run",
            "pip wheel",
            "python setup.py",
            "gcc",
            "g++",
            "clang",
            "javac",
            "go build",
        ]
        return any(marker in lowered for marker in markers) or bool(
            words and words[0] in {"make", "cmake", "ninja", "gcc", "g++", "clang", "javac"}
        )

    def _has_network(self, lowered: str) -> bool:
        markers = [
            "curl ",
            "wget ",
            "ssh ",
            "scp ",
            "rsync ",
            "git clone",
            "git fetch",
            "pip install",
            "npm install",
            "apt-get",
            "apk add",
            "dnf install",
            "yum install",
        ]
        return any(marker in lowered for marker in markers)

    def _has_package(self, lowered: str) -> bool:
        markers = [
            "pip install",
            "npm install",
            "apt-get",
            "apk add",
            "dnf install",
            "yum install",
            "conda install",
            "gem install",
        ]
        return any(marker in lowered for marker in markers)

    def _has_process(self, lowered: str) -> bool:
        return any(
            marker in lowered
            for marker in [
                "ps ",
                "pgrep",
                "pkill",
                "kill ",
                "jobs",
                "nohup",
                "setsid",
                "systemctl",
                "service ",
            ]
        )

    def _has_destructive(self, lowered: str) -> bool:
        markers = [
            "rm -rf /",
            "rm -fr /",
            "mkfs",
            "dd if=",
            ":(){",
            "chmod -r 777 /",
            "chown -r",
            "git reset --hard",
            "git clean -fd",
            "truncate -s 0",
        ]
        return any(marker in lowered for marker in markers) or bool(
            re.search(r"(^|\s)rm\s+(-[^\n]*[rf][^\n]*|[^\n]*\*)", lowered)
        )

    def _has_interactive(self, lowered: str, words: list[str]) -> bool:
        interactive = {
            "vim",
            "vi",
            "nano",
            "emacs",
            "less",
            "more",
            "top",
            "htop",
            "python",
            "python3",
            "node",
            "irb",
            "ghci",
            "mysql",
            "psql",
            "sqlite3",
        }
        if (
            words
            and words[0] in interactive
            and "-c" not in words
            and "<<" not in lowered
            and "-e" not in words
        ):
            return True
        return any(
            marker in lowered for marker in [" read -p", "select ", "prompt=", "--interactive"]
        )

    def _has_unbounded(self, lowered: str, words: list[str]) -> bool:
        if "yes " in lowered or lowered.strip() == "yes":
            return True
        if any(
            marker in lowered
            for marker in [
                "tail -f",
                "watch ",
                "while true",
                "for (;;)",
                "sleep infinity",
                "nc -l",
                "python -m http.server",
                "npm start",
                "npm run dev",
            ]
        ):
            return True
        if (
            words
            and words[0] in {"cat", "grep", "rg", "find"}
            and not any(
                bound in lowered
                for bound in ["head", "sed -n", "maxdepth", "-m ", "--max-count", "| head", "| sed"]
            )
        ):
            return True
        return False


class TimeoutPolicy:
    def bound(
        self, features: CommandFeatures, requested: int | None
    ) -> tuple[int | None, list[str]]:
        notes: list[str] = []
        base = requested if requested is not None else self._default_for(features)
        if base is None:
            return None, notes
        if base < MIN_TIMEOUT:
            notes.append(f"raised timeout from {base} to {MIN_TIMEOUT}")
            base = MIN_TIMEOUT
        cap = self._cap_for(features)
        if base > cap:
            notes.append(f"bounded timeout from {base} to {cap}")
            base = cap
        return base, notes

    def _default_for(self, features: CommandFeatures) -> int | None:
        if (
            CommandClass.INSPECT in features.classes
            or CommandClass.READ in features.classes
            or CommandClass.SEARCH in features.classes
        ):
            return 20
        if CommandClass.VERIFY in features.classes:
            return 120
        if CommandClass.BUILD in features.classes:
            return 150
        if CommandClass.PACKAGE in features.classes or CommandClass.NETWORK in features.classes:
            return 120
        if features.line_count > 6 or features.has_heredoc:
            return 90
        return DEFAULT_TIMEOUT

    def _cap_for(self, features: CommandFeatures) -> int:
        if features.has_unbounded or CommandClass.INTERACTIVE in features.classes:
            return 20
        if CommandClass.VERIFY in features.classes or CommandClass.BUILD in features.classes:
            return MAX_TIMEOUT
        if CommandClass.PACKAGE in features.classes or CommandClass.NETWORK in features.classes:
            return 150
        if CommandClass.READ in features.classes or CommandClass.SEARCH in features.classes:
            return 60
        return 120


class CommandRewriter:
    def rewrite(self, command: str, features: CommandFeatures) -> tuple[str, list[str]]:
        rewritten = command.strip()
        notes: list[str] = []
        rewritten, note = self._add_timeout(rewritten, features)
        if note:
            notes.append(note)
        rewritten, note = self._bound_read_output(rewritten, features)
        if note:
            notes.append(note)
        rewritten, note = self._disable_pagers(rewritten)
        if note:
            notes.append(note)
        rewritten, note = self._make_package_noninteractive(rewritten)
        if note:
            notes.append(note)
        rewritten, note = self._remove_background_tail(rewritten)
        if note:
            notes.append(note)
        return rewritten.strip(), notes

    def _add_timeout(self, command: str, features: CommandFeatures) -> tuple[str, str]:
        if features.has_timeout:
            return command, ""
        if (
            features.has_unbounded
            or CommandClass.INTERACTIVE in features.classes
            or CommandClass.NETWORK in features.classes
        ):
            return (
                f"timeout {RECOVERY_TIMEOUT if features.has_unbounded else 120}s bash -lc {shlex.quote(command)}",
                "wrapped command with timeout",
            )
        return command, ""

    def _bound_read_output(self, command: str, features: CommandFeatures) -> tuple[str, str]:
        if not features.has_unbounded:
            return command, ""
        lowered = command.lower()
        if any(marker in lowered for marker in ["| head", "| sed -n", "| tail"]):
            return command, ""
        if features.first_word in {"cat", "grep", "rg", "find"}:
            return f"{command} | sed -n '1,240p'", "bounded potentially large output"
        return command, ""

    def _disable_pagers(self, command: str) -> tuple[str, str]:
        if re.search(r"(^|\s)git\s+", command) and "--no-pager" not in command:
            return (
                re.sub(r"(^|\s)git\s+", r"\1git --no-pager ", command, count=1),
                "disabled git pager",
            )
        return command, ""

    def _make_package_noninteractive(self, command: str) -> tuple[str, str]:
        lower = command.lower()
        changed = False
        if "apt-get" in lower and "-y" not in lower:
            command = re.sub(
                r"apt-get\s+install",
                "DEBIAN_FRONTEND=noninteractive apt-get install -y",
                command,
                flags=re.IGNORECASE,
            )
            changed = True
        if "apk add" in lower and "--no-cache" not in lower:
            command = re.sub(r"apk\s+add", "apk add --no-cache", command, flags=re.IGNORECASE)
            changed = True
        return command, "made package command noninteractive" if changed else ""

    def _remove_background_tail(self, command: str) -> tuple[str, str]:
        if re.search(r"&\s*$", command):
            return re.sub(r"&\s*$", "", command).strip(), "removed background marker"
        return command, ""


class CompletionGate:
    def allow_done(self, history: list[CommandResult], signal: HistorySignal) -> tuple[bool, str]:
        if not history:
            return False, "no terminal evidence yet"
        if signal.recent_failure:
            return False, "last command failed; inspect or fix before done"
        if signal.recent_edit and not signal.successful_verify_after_edit:
            return False, "recent edit lacks successful verification"
        if not signal.recent_verify and len(history) > 2:
            return False, "no recent verification command"
        if signal.same_command_count >= 3:
            return False, "repeated command loop detected; need a different verification or summary"
        return True, "recent terminal evidence is sufficient"


class HistoryAnalyzer:
    def __init__(self) -> None:
        self.clipper = TextClipper()
        self.classifier = CommandClassifier()

    def analyze(self, history: list[CommandResult]) -> HistorySignal:
        turn = len(history) + 1
        if not history:
            return HistorySignal(
                turn,
                None,
                False,
                False,
                False,
                False,
                False,
                False,
                0,
                "",
                "",
                "",
                "",
                "No commands have run yet.",
                [],
                [],
            )
        last = history[-1]
        recent = history[-8:]
        repeated_last = len(history) > 1 and last.command.strip() == history[-2].command.strip()
        same_count = self._same_tail_count(history)
        recent_edit = any(self._is_edit(item.command) for item in recent)
        recent_verify = any(
            self._is_verify(item.command) and item.return_code == 0 for item in recent
        )
        successful_after_edit = self._successful_verify_after_last_edit(history)
        recent_failure = last.return_code not in (0, None)
        recent_refusal = "HARNESS_RECOVERY" in (last.command or "")
        facts = self._facts(history)
        warnings = self._warnings(
            history, same_count, recent_failure, recent_edit, successful_after_edit
        )
        summary = self._summary(history, facts, warnings)
        return HistorySignal(
            turn=turn,
            last_exit=last.return_code,
            repeated_last=repeated_last,
            recent_edit=recent_edit,
            recent_verify=recent_verify,
            successful_verify_after_edit=successful_after_edit,
            recent_failure=recent_failure,
            recent_refusal=recent_refusal,
            same_command_count=same_count,
            last_command=last.command,
            last_stdout_tail=self.clipper.clip_tail(last.stdout, 3000),
            last_stderr_tail=self.clipper.clip_tail(last.stderr, 2000),
            changed_files_hint=self._changed_files_hint(history),
            summary=summary,
            facts=facts,
            warnings=warnings,
        )

    def format_history(self, history: list[CommandResult]) -> str:
        start = max(0, len(history) - MAX_HISTORY_ITEMS)
        chunks: list[str] = []
        budget = MAX_PROMPT_HISTORY_CHARS
        for index, item in enumerate(history[start:], start + 1):
            stdout = self.clipper.clip_tail(item.stdout, 2600)
            stderr = self.clipper.clip_tail(item.stderr, 1600)
            chunk = f"[{index}] $ {item.command}\nexit={item.return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            chunks.append(chunk)
        text = "\n\n".join(chunks)
        return self.clipper.clip_middle(text, budget)

    def _same_tail_count(self, history: list[CommandResult]) -> int:
        if not history:
            return 0
        last = history[-1].command.strip()
        count = 0
        for item in reversed(history):
            if item.command.strip() == last:
                count += 1
            else:
                break
        return count

    def _is_edit(self, command: str) -> bool:
        return CommandClass.EDIT in self.classifier.features(command).classes

    def _is_verify(self, command: str) -> bool:
        return CommandClass.VERIFY in self.classifier.features(command).classes

    def _successful_verify_after_last_edit(self, history: list[CommandResult]) -> bool:
        last_edit_index = -1
        for index, item in enumerate(history):
            if self._is_edit(item.command):
                last_edit_index = index
        if last_edit_index < 0:
            return any(
                self._is_verify(item.command) and item.return_code == 0 for item in history[-6:]
            )
        for item in history[last_edit_index + 1 :]:
            if self._is_verify(item.command) and item.return_code == 0:
                return True
        return False

    def _facts(self, history: list[CommandResult]) -> list[str]:
        facts: list[str] = []
        for item in history[-10:]:
            features = self.classifier.features(item.command)
            status = "ok" if item.return_code == 0 else f"exit {item.return_code}"
            labels = ",".join(sorted(cls.value for cls in features.classes))
            facts.append(f"{status}: {labels}: {self.clipper.one_line(item.command, 180)}")
        return facts

    def _warnings(
        self,
        history: list[CommandResult],
        same_count: int,
        recent_failure: bool,
        recent_edit: bool,
        verified: bool,
    ) -> list[str]:
        warnings: list[str] = []
        if recent_failure:
            warnings.append("last command failed; do not repeat it blindly")
        if same_count >= 2:
            warnings.append("same command was repeated recently")
        if recent_edit and not verified:
            warnings.append("edits need verification before done")
        if len(history) > 18 and not any(self._is_verify(item.command) for item in history[-8:]):
            warnings.append("long run without recent verification")
        return warnings

    def _changed_files_hint(self, history: list[CommandResult]) -> str:
        names: list[str] = []
        for item in history[-12:]:
            cmd = item.command
            for match in re.findall(r"(?:>|>>|tee\s+-?a?\s+|Path\(['\"])([A-Za-z0-9_./-]+)", cmd):
                if match not in names:
                    names.append(match)
        return ", ".join(names[:12])

    def _summary(self, history: list[CommandResult], facts: list[str], warnings: list[str]) -> str:
        lines = [f"turns_completed={len(history)}"]
        lines.extend(f"fact: {fact}" for fact in facts[-8:])
        lines.extend(f"warning: {warning}" for warning in warnings)
        return "\n".join(lines)


class PromptBuilder:
    def __init__(self) -> None:
        self.analyzer = HistoryAnalyzer()

    def build(
        self, task: TaskContext, history: list[CommandResult], signal: HistorySignal, recovery: str
    ) -> PromptBundle:
        state_lines = self._state_lines(signal, recovery)
        policy_lines = self._policy_lines(signal)
        user = self._user(task, history, state_lines, policy_lines, recovery)
        return PromptBundle(SYSTEM_PROMPT, user, state_lines, policy_lines)

    def _user(
        self,
        task: TaskContext,
        history: list[CommandResult],
        state_lines: list[str],
        policy_lines: list[str],
        recovery: str,
    ) -> str:
        sections = [
            "Task instruction:\n" + task.instruction,
            "State summary:\n" + "\n".join(state_lines),
            "Recent terminal history:\n" + self.analyzer.format_history(history),
            "Command policy:\n" + "\n".join(policy_lines),
        ]
        if recovery:
            sections.append("Harness recovery note:\n" + recovery)
        sections.append(
            "Use execute_commands for the next shell action, or task_complete only when "
            "recent terminal evidence supports completion."
        )
        return "\n\n".join(sections)

    def _state_lines(self, signal: HistorySignal, recovery: str) -> list[str]:
        lines = [
            f"turn={signal.turn}",
            f"last_exit={signal.last_exit}",
            f"recent_edit={signal.recent_edit}",
            f"recent_verify={signal.recent_verify}",
            f"successful_verify_after_edit={signal.successful_verify_after_edit}",
            f"same_command_count={signal.same_command_count}",
        ]
        if signal.changed_files_hint:
            lines.append("possible_changed_files=" + signal.changed_files_hint)
        for warning in signal.warnings:
            lines.append("warning=" + warning)
        if recovery:
            lines.append("recovery_required=true")
        return lines

    def _policy_lines(self, signal: HistorySignal) -> list[str]:
        lines = [
            "Run one bounded shell command per turn.",
            "Prefer reading exact files and running exact checks over broad exploration.",
            "Use sed -n/head/tail limits for large files and command output.",
            "Avoid interactive editors, REPLs, background servers, endless loops, and broad deletes.",
            "If a command was blocked or rewritten, choose a safer bounded alternative.",
            "If the last command failed, inspect the error or adjust the approach.",
            "After a write-like command, run a relevant verification command before done.",
            "Only finish when recent terminal evidence supports completion.",
        ]
        if signal.recent_edit and not signal.successful_verify_after_edit:
            lines.append("Current gate: verification is required before done.")
        if signal.same_command_count >= 2:
            lines.append("Current gate: do not repeat the last command unchanged.")
        return lines


class RecoveryMemory:
    def __init__(self) -> None:
        self.last_history_len = -1
        self.message = ""

    def update_after_decision(self, history_len: int, decision: CommandDecision) -> None:
        if decision.refused or decision.rewritten or decision.recovery:
            self.last_history_len = history_len
            self.message = decision.recovery or decision.reason
            return
        if history_len > self.last_history_len:
            self.message = ""

    def current(self, history_len: int) -> str:
        if self.message and history_len <= self.last_history_len + 1:
            return self.message
        return ""


class ToolActionAdapter:
    def parse(self, result: ToolModelResult) -> ParsedAction | None:
        for call in result.tool_calls:
            if call.name == "task_complete":
                return ParsedAction(
                    ActionKind.DONE,
                    raw=result.content,
                    source="tool:task_complete",
                    confidence=0.95,
                )
            if call.name == "execute_commands":
                action = self._execute_commands_action(call.arguments, result.content)
                if action is not None:
                    return action
        return None

    def _execute_commands_action(self, arguments: dict[str, Any], raw: str) -> ParsedAction | None:
        commands = arguments.get("commands", [])
        if isinstance(commands, str):
            try:
                commands = json.loads(commands)
            except json.JSONDecodeError:
                commands = []
        if not isinstance(commands, list):
            return None
        for command in commands:
            keystrokes = self._keystrokes(command)
            if not keystrokes:
                continue
            return ParsedAction(
                ActionKind.RUN,
                command=keystrokes,
                timeout_sec=self._duration(command),
                raw=raw,
                source="tool:execute_commands",
                confidence=0.98,
                notes=self._notes(arguments),
            )
        return ParsedAction(
            ActionKind.RECOVER,
            raw=raw,
            source="tool:execute_commands_empty",
            confidence=0.2,
            notes=["execute_commands contained no runnable command"],
        )

    def _keystrokes(self, command: Any) -> str:
        if isinstance(command, str):
            return command.strip()
        if isinstance(command, dict):
            for key in ("keystrokes", "command", "cmd", "shell"):
                value = command.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _duration(self, command: Any) -> int | None:
        if not isinstance(command, dict):
            return None
        value = command.get("duration")
        if value is None:
            value = command.get("timeout_sec")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return max(MIN_TIMEOUT, min(int(value), MAX_TIMEOUT))

    def _notes(self, arguments: dict[str, Any]) -> list[str]:
        notes: list[str] = []
        for key in ("analysis", "plan"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                notes.append(f"{key}: {value.strip()[:240]}")
        return notes[:2]


class CommandMediator:
    def __init__(self) -> None:
        self.cleaner = ShellCleaner()
        self.classifier = CommandClassifier()
        self.timeout = TimeoutPolicy()
        self.rewriter = CommandRewriter()

    def mediate(
        self, action: ParsedAction, history: list[CommandResult], signal: HistorySignal
    ) -> CommandDecision:
        if action.action == ActionKind.DONE:
            return CommandDecision(
                "", None, RiskLevel.SAFE, {CommandClass.UNKNOWN}, "model requested done"
            )
        command = self.cleaner.clean(action.command)
        if not command:
            return self._recovery(
                "empty command after parsing",
                "Return a concrete bounded command that inspects, edits, or verifies.",
            )
        command = self._avoid_repetition(command, history)
        features = self.classifier.features(command, action.timeout_sec)
        risk, reason = self._risk(features, command, signal)
        if risk == RiskLevel.BLOCK:
            return self._recovery(reason, self._recovery_for(features, command), features.classes)
        rewritten = command
        notes: list[str] = []
        if risk == RiskLevel.REWRITE:
            rewritten, notes = self.rewriter.rewrite(command, features)
            features = self.classifier.features(rewritten, action.timeout_sec)
        bounded, timeout_notes = self.timeout.bound(features, action.timeout_sec)
        notes.extend(timeout_notes)
        changed = rewritten != action.command or bool(notes)
        recovery = "; ".join(notes)
        return CommandDecision(
            rewritten,
            bounded,
            risk,
            features.classes,
            reason,
            recovery,
            changed,
            False,
            rewritten != command,
        )

    def _avoid_repetition(self, command: str, history: list[CommandResult]) -> str:
        if not history:
            return command
        last = history[-1].command.strip()
        if command.strip() != last:
            return command
        if history[-1].return_code == 0:
            return command
        escaped = command.replace("'", "'\"'\"'")
        return f"printf '%s\n' 'Previous command failed; inspecting current directory and recent files before retrying.' && pwd && find . -maxdepth 2 -type f | sort | sed -n '1,120p' && printf '%s\n' 'Blocked repeat: {escaped}'"

    def _risk(
        self, features: CommandFeatures, command: str, signal: HistorySignal
    ) -> tuple[RiskLevel, str]:
        if not command.strip():
            return RiskLevel.BLOCK, "empty command"
        if features.byte_count > 50000:
            return RiskLevel.BLOCK, "command is too large for one mediated turn"
        if CommandClass.DESTRUCTIVE in features.classes:
            return RiskLevel.BLOCK, "destructive command refused"
        if self._writes_outside_workspace(command):
            return RiskLevel.BLOCK, "write target appears outside current workspace"
        if CommandClass.INTERACTIVE in features.classes:
            return RiskLevel.REWRITE, "interactive command needs a bounded noninteractive form"
        if features.has_unbounded:
            return RiskLevel.REWRITE, "command may produce unbounded output or run indefinitely"
        if features.has_background:
            return RiskLevel.REWRITE, "background execution is not useful in this turn loop"
        if features.requested_timeout and features.requested_timeout > MAX_TIMEOUT:
            return RiskLevel.REWRITE, "requested timeout is too large"
        if signal.same_command_count >= 2 and command.strip() == signal.last_command.strip():
            return RiskLevel.BLOCK, "same command repeated too many times"
        return RiskLevel.SAFE, "command accepted"

    def _writes_outside_workspace(self, command: str) -> bool:
        lowered = command.lower()
        if not any(marker in lowered for marker in ["> /", "tee /", "path('/", 'path("/', "rm /"]):
            return False
        allowed_prefixes = ["/app", "/workspace", "/tmp", "/root", "/home"]
        for match in re.findall(
            r"(?:>|>>|tee\s+-?a?\s+|Path\(['\"]|rm\s+(?:-[^\s]+\s+)?)(/[^\s'\"]+)", command
        ):
            if not any(
                match.startswith(prefix + "/") or match == prefix for prefix in allowed_prefixes
            ):
                return True
        return False

    def _recovery(
        self, reason: str, recovery: str, classes: set[CommandClass] | None = None
    ) -> CommandDecision:
        command = "printf '%s\n' 'HARNESS_RECOVERY: command blocked; choose a safer bounded command next.'"
        return CommandDecision(
            command,
            RECOVERY_TIMEOUT,
            RiskLevel.BLOCK,
            classes or {CommandClass.UNKNOWN},
            reason,
            recovery,
            True,
            True,
            False,
        )

    def _recovery_for(self, features: CommandFeatures, command: str) -> str:
        if CommandClass.DESTRUCTIVE in features.classes:
            return "The proposed command was destructive. Use a narrow inspection or backup-preserving edit instead."
        if CommandClass.INTERACTIVE in features.classes:
            return "The proposed command looked interactive. Use a noninteractive flag, here-doc, or script file."
        if features.has_unbounded:
            return "The proposed command looked unbounded. Add timeout and output limits such as head or sed -n."
        if features.byte_count > 50000:
            return "The proposed command was too large. Write a small script or inspect first."
        return "Choose a safer bounded command."


class TurnPlanner:
    def __init__(self) -> None:
        self.parser = ActionParser()
        self.tools = ToolActionAdapter()
        self.analyzer = HistoryAnalyzer()
        self.prompts = PromptBuilder()
        self.mediator = CommandMediator()
        self.gate = CompletionGate()
        self.memory = RecoveryMemory()

    def first_turn(self) -> HarnessTurn:
        return HarnessTurn(done=False, command=FIRST_COMMAND, timeout_sec=30)

    def next(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        if not history:
            return self.first_turn()
        signal = self.analyzer.analyze(history)
        recovery = self.memory.current(len(history))
        bundle = self.prompts.build(task, history, signal, recovery)
        response = call_terminal_model_with_tools(
            [
                {"role": "system", "content": bundle.system},
                {"role": "user", "content": bundle.user},
            ],
            KIRA_TOOLS,
        )
        parsed = self.tools.parse(response) or self.parser.parse(response.content)
        if parsed.action == ActionKind.DONE:
            allowed, reason = self.gate.allow_done(history, signal)
            if allowed:
                self.memory.update_after_decision(
                    len(history),
                    CommandDecision("", None, RiskLevel.SAFE, {CommandClass.UNKNOWN}, reason),
                )
                return HarnessTurn(command="", done=True, timeout_sec=None)
            decision = CommandDecision(
                command="printf '%s\n' 'HARNESS_RECOVERY: done blocked; run verification or inspect remaining failure.'",
                timeout_sec=RECOVERY_TIMEOUT,
                risk=RiskLevel.BLOCK,
                classes={CommandClass.VERIFY},
                reason=reason,
                recovery=reason,
                changed=True,
                refused=True,
            )
            self.memory.update_after_decision(len(history), decision)
            return HarnessTurn(
                command=decision.command, done=False, timeout_sec=decision.timeout_sec
            )
        decision = self.mediator.mediate(parsed, history, signal)
        self.memory.update_after_decision(len(history), decision)
        return HarnessTurn(
            command=decision.command or FIRST_COMMAND, done=False, timeout_sec=decision.timeout_sec
        )


class CandidateHarness(BaseHarness):
    def __init__(self) -> None:
        self.planner = TurnPlanner()

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        return self.planner.next(task, history)


class PolicyLibrary:
    def describe(self, signal: HistorySignal) -> list[str]:
        lines: list[str] = []
        lines.extend(self.workspace_orientation(signal))
        lines.extend(self.bounded_output(signal))
        lines.extend(self.failure_recovery(signal))
        lines.extend(self.edit_verification(signal))
        lines.extend(self.json_output(signal))
        lines.extend(self.shell_quoting(signal))
        lines.extend(self.here_document(signal))
        lines.extend(self.python_snippet(signal))
        lines.extend(self.search_strategy(signal))
        lines.extend(self.file_reading(signal))
        lines.extend(self.test_selection(signal))
        lines.extend(self.build_selection(signal))
        lines.extend(self.package_install(signal))
        lines.extend(self.network_access(signal))
        lines.extend(self.process_control(signal))
        lines.extend(self.background_jobs(signal))
        lines.extend(self.interactive_tools(signal))
        lines.extend(self.large_file_handling(signal))
        lines.extend(self.binary_file_handling(signal))
        lines.extend(self.archive_handling(signal))
        lines.extend(self.permission_change(signal))
        lines.extend(self.service_start(signal))
        lines.extend(self.database_checks(signal))
        lines.extend(self.config_changes(signal))
        lines.extend(self.dependency_pinning(signal))
        lines.extend(self.language_detection(signal))
        lines.extend(self.git_inspection(signal))
        lines.extend(self.git_mutation(signal))
        lines.extend(self.diff_review(signal))
        lines.extend(self.completion_evidence(signal))
        lines.extend(self.timeout_choice(signal))
        lines.extend(self.command_repetition(signal))
        lines.extend(self.stdout_budget(signal))
        lines.extend(self.stderr_triage(signal))
        lines.extend(self.path_safety(signal))
        lines.extend(self.absolute_paths(signal))
        lines.extend(self.temporary_files(signal))
        lines.extend(self.cleanup_policy(signal))
        lines.extend(self.script_generation(signal))
        lines.extend(self.macro_command(signal))
        lines.extend(self.compiler_errors(signal))
        lines.extend(self.unit_failures(signal))
        lines.extend(self.integration_failures(signal))
        lines.extend(self.benchmark_checks(signal))
        lines.extend(self.data_validation(signal))
        lines.extend(self.checksum_checks(signal))
        lines.extend(self.formatters(signal))
        lines.extend(self.linters(signal))
        lines.extend(self.static_analysis(signal))
        lines.extend(self.security_sensitive(signal))
        lines.extend(self.credential_avoidance(signal))
        lines.extend(self.environment_variables(signal))
        lines.extend(self.container_limits(signal))
        lines.extend(self.disk_usage(signal))
        lines.extend(self.memory_usage(signal))
        lines.extend(self.parallelism(signal))
        lines.extend(self.long_running_tasks(signal))
        lines.extend(self.polling(signal))
        lines.extend(self.server_probe(signal))
        lines.extend(self.port_checks(signal))
        lines.extend(self.text_transform(signal))
        lines.extend(self.structured_data(signal))
        lines.extend(self.json_processing(signal))
        lines.extend(self.csv_processing(signal))
        lines.extend(self.xml_processing(signal))
        lines.extend(self.yaml_processing(signal))
        lines.extend(self.sql_processing(signal))
        lines.extend(self.regex_usage(signal))
        lines.extend(self.numeric_validation(signal))
        lines.extend(self.image_metadata(signal))
        lines.extend(self.notebook_handling(signal))
        lines.extend(self.symlink_handling(signal))
        lines.extend(self.hidden_files(signal))
        lines.extend(self.generated_files(signal))
        lines.extend(self.patch_application(signal))
        lines.extend(self.backup_before_edit(signal))
        lines.extend(self.idempotency(signal))
        lines.extend(self.rollback_plan(signal))
        lines.extend(self.partial_progress(signal))
        lines.extend(self.final_sanity(signal))
        lines.extend(self.tool_contract(signal))
        lines.extend(self.model_response_repair(signal))
        lines.extend(self.concatenated_json(signal))
        lines.extend(self.fenced_json(signal))
        lines.extend(self.malformed_action(signal))
        lines.extend(self.empty_command(signal))
        lines.extend(self.unsafe_delete(signal))
        lines.extend(self.root_write(signal))
        lines.extend(self.pager_disable(signal))
        lines.extend(self.noninteractive_flags(signal))
        lines.extend(self.output_sampling(signal))
        lines.extend(self.tail_inspection(signal))
        lines.extend(self.head_inspection(signal))
        lines.extend(self.find_depth(signal))
        lines.extend(self.rg_preference(signal))
        lines.extend(self.sed_ranges(signal))
        lines.extend(self.awk_limits(signal))
        lines.extend(self.jq_filters(signal))
        lines.extend(self.python_compile(signal))
        lines.extend(self.node_checks(signal))
        lines.extend(self.go_checks(signal))
        lines.extend(self.rust_checks(signal))
        lines.extend(self.java_checks(signal))
        lines.extend(self.c_checks(signal))
        lines.extend(self.shell_checks(signal))
        lines.extend(self.make_checks(signal))
        lines.extend(self.cmake_checks(signal))
        lines.extend(self.docker_absence(signal))
        lines.extend(self.no_task_assumptions(signal))
        lines.extend(self.evidence_summary(signal))
        lines.extend(self.turn_budget(signal))
        lines.extend(self.history_compression(signal))
        lines.extend(self.state_tracking(signal))
        lines.extend(self.changed_file_tracking(signal))
        lines.extend(self.risk_explanation(signal))
        lines.extend(self.recovery_prompting(signal))
        lines.extend(self.safe_rewrite(signal))
        lines.extend(self.refusal_path(signal))
        lines.extend(self.done_gate(signal))
        lines.extend(self.verification_gate(signal))
        lines.extend(self.post_edit_gate(signal))
        lines.extend(self.pre_edit_inspect(signal))
        lines.extend(self.smallest_change(signal))
        lines.extend(self.broad_search(signal))
        lines.extend(self.narrow_search(signal))
        lines.extend(self.multi_command_batch(signal))
        lines.extend(self.single_command_turn(signal))
        lines.extend(self.command_cleanup(signal))
        lines.extend(self.quote_cleanup(signal))
        lines.extend(self.json_arguments(signal))
        lines.extend(self.tool_name_alias(signal))
        lines.extend(self.timeout_alias(signal))
        lines.extend(self.duration_alias(signal))
        lines.extend(self.keystroke_alias(signal))
        lines.extend(self.shell_alias(signal))
        lines.extend(self.finish_alias(signal))
        lines.extend(self.response_selection(signal))
        lines.extend(self.last_object_preference(signal))
        lines.extend(self.dedupe_actions(signal))
        lines.extend(self.balanced_braces(signal))
        lines.extend(self.escape_handling(signal))
        lines.extend(self.unicode_handling(signal))
        lines.extend(self.line_endings(signal))
        lines.extend(self.case_sensitivity(signal))
        lines.extend(self.permission_denied(signal))
        lines.extend(self.missing_command(signal))
        lines.extend(self.missing_file(signal))
        lines.extend(self.module_import(signal))
        lines.extend(self.syntax_error(signal))
        lines.extend(self.assertion_error(signal))
        lines.extend(self.segfault(signal))
        lines.extend(self.timeout_error(signal))
        lines.extend(self.oom_error(signal))
        lines.extend(self.network_error(signal))
        lines.extend(self.package_error(signal))
        lines.extend(self.compiler_warning(signal))
        lines.extend(self.test_flake(signal))
        lines.extend(self.version_probe(signal))
        lines.extend(self.help_probe(signal))
        lines.extend(self.readme_probe(signal))
        lines.extend(self.eval_probe(signal))
        lines.extend(self.harness_probe(signal))
        lines.extend(self.workspace_listing(signal))
        lines.extend(self.root_listing(signal))
        lines.extend(self.maxdepth_listing(signal))
        lines.extend(self.size_probe(signal))
        lines.extend(self.file_type_probe(signal))
        lines.extend(self.git_status(signal))
        lines.extend(self.git_log(signal))
        lines.extend(self.git_show(signal))
        lines.extend(self.diff_stat(signal))
        lines.extend(self.config_probe(signal))
        lines.extend(self.dependency_probe(signal))
        lines.extend(self.lockfile_probe(signal))
        lines.extend(self.entrypoint_probe(signal))
        lines.extend(self.main_probe(signal))
        lines.extend(self.cli_probe(signal))
        lines.extend(self.api_probe(signal))
        lines.extend(self.schema_probe(signal))
        lines.extend(self.sample_data_probe(signal))
        lines.extend(self.expected_output_probe(signal))
        lines.extend(self.actual_output_probe(signal))
        lines.extend(self.comparison_probe(signal))
        lines.extend(self.roundtrip_probe(signal))
        lines.extend(self.smoke_test(signal))
        lines.extend(self.targeted_test(signal))
        lines.extend(self.full_test(signal))
        lines.extend(self.performance_test(signal))
        lines.extend(self.artifact_check(signal))
        lines.extend(self.final_command(signal))
        lines.extend(self.blocked_done_message(signal))
        lines.extend(self.blocked_run_message(signal))
        lines.extend(self.safe_cat(signal))
        lines.extend(self.safe_grep(signal))
        lines.extend(self.safe_find(signal))
        lines.extend(self.safe_python(signal))
        lines.extend(self.safe_perl(signal))
        lines.extend(self.safe_sed(signal))
        lines.extend(self.safe_make(signal))
        lines.extend(self.safe_npm(signal))
        lines.extend(self.safe_pip(signal))
        lines.extend(self.safe_cargo(signal))
        lines.extend(self.safe_go(signal))
        lines.extend(self.safe_java(signal))
        lines.extend(self.safe_sqlite(signal))
        lines.extend(self.safe_openssl(signal))
        lines.extend(self.safe_tar(signal))
        lines.extend(self.safe_unzip(signal))
        lines.extend(self.safe_chmod(signal))
        lines.extend(self.safe_cp(signal))
        lines.extend(self.safe_mv(signal))
        lines.extend(self.safe_rm(signal))
        lines.extend(self.safe_mkdir(signal))
        lines.extend(self.safe_touch(signal))
        lines.extend(self.safe_ln(signal))
        lines.extend(self.safe_ps(signal))
        lines.extend(self.safe_kill(signal))
        lines.extend(self.safe_nc(signal))
        lines.extend(self.safe_curl(signal))
        lines.extend(self.safe_wget(signal))
        lines.extend(self.safe_ssh(signal))
        lines.extend(self.safe_timeout(signal))
        lines.extend(self.safe_yes(signal))
        lines.extend(self.safe_sleep(signal))
        lines.extend(self.safe_watch(signal))
        lines.extend(self.safe_tail(signal))
        lines.extend(self.safe_head(signal))
        lines.extend(self.safe_tree(signal))
        lines.extend(self.safe_du(signal))
        lines.extend(self.safe_wc(signal))
        lines.extend(self.safe_sort(signal))
        lines.extend(self.safe_uniq(signal))
        lines.extend(self.safe_xargs(signal))
        lines.extend(self.safe_parallel(signal))
        lines.extend(self.safe_env(signal))
        lines.extend(self.safe_export(signal))
        lines.extend(self.safe_source(signal))
        lines.extend(self.safe_bash(signal))
        lines.extend(self.safe_sh(signal))
        lines.extend(self.safe_zsh(signal))
        lines.extend(self.safe_fish(signal))
        lines.extend(self.safe_vim(signal))
        lines.extend(self.safe_editor(signal))
        lines.extend(self.safe_repl(signal))
        lines.extend(self.safe_server(signal))
        lines.extend(self.safe_daemon(signal))
        lines.extend(self.safe_service(signal))
        lines.extend(self.safe_systemctl(signal))
        lines.extend(self.safe_docker(signal))
        lines.extend(self.safe_mount(signal))
        lines.extend(self.safe_dd(signal))
        lines.extend(self.safe_mkfs(signal))
        lines.extend(self.safe_chown(signal))
        lines.extend(self.safe_sudo(signal))
        lines.extend(self.safe_root(signal))
        lines.extend(self.safe_home(signal))
        lines.extend(self.safe_tmp(signal))
        lines.extend(self.safe_app(signal))
        lines.extend(self.safe_workspace(signal))
        return self._trim(lines)

    def _trim(self, lines: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= 90:
                break
        return out

    def workspace_orientation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "workspace orientation: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "workspace orientation: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("workspace orientation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "workspace orientation: prefer bounded terminal evidence before acting."
            )
        return guidance

    def bounded_output(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("bounded output: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "bounded output: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("bounded output: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("bounded output: avoid bounded terminal evidence before acting.")
        return guidance

    def failure_recovery(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "failure recovery: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "failure recovery: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("failure recovery: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("failure recovery: require bounded terminal evidence before acting.")
        return guidance

    def edit_verification(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "edit verification: after failure, record a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "edit verification: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("edit verification: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("edit verification: record bounded terminal evidence before acting.")
        return guidance

    def json_output(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("json output: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("json output: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("json output: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("json output: bound bounded terminal evidence before acting.")
        return guidance

    def shell_quoting(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("shell quoting: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "shell quoting: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("shell quoting: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("shell quoting: inspect bounded terminal evidence before acting.")
        return guidance

    def here_document(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("here document: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "here document: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("here document: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("here document: verify bounded terminal evidence before acting.")
        return guidance

    def python_snippet(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("python snippet: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "python snippet: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("python snippet: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("python snippet: rewrite bounded terminal evidence before acting.")
        return guidance

    def search_strategy(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "search strategy: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "search strategy: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("search strategy: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("search strategy: summarize bounded terminal evidence before acting.")
        return guidance

    def file_reading(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("file reading: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("file reading: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("file reading: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("file reading: choose bounded terminal evidence before acting.")
        return guidance

    def test_selection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("test selection: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "test selection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("test selection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("test selection: prefer bounded terminal evidence before acting.")
        return guidance

    def build_selection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("build selection: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "build selection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("build selection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("build selection: avoid bounded terminal evidence before acting.")
        return guidance

    def package_install(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("package install: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "package install: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("package install: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("package install: require bounded terminal evidence before acting.")
        return guidance

    def network_access(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("network access: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "network access: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("network access: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("network access: record bounded terminal evidence before acting.")
        return guidance

    def process_control(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("process control: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "process control: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("process control: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("process control: bound bounded terminal evidence before acting.")
        return guidance

    def background_jobs(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("background jobs: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "background jobs: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("background jobs: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("background jobs: inspect bounded terminal evidence before acting.")
        return guidance

    def interactive_tools(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "interactive tools: after failure, verify a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "interactive tools: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("interactive tools: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("interactive tools: verify bounded terminal evidence before acting.")
        return guidance

    def large_file_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "large file handling: after failure, rewrite a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "large file handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("large file handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("large file handling: rewrite bounded terminal evidence before acting.")
        return guidance

    def binary_file_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "binary file handling: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "binary file handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("binary file handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "binary file handling: summarize bounded terminal evidence before acting."
            )
        return guidance

    def archive_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("archive handling: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "archive handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("archive handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("archive handling: choose bounded terminal evidence before acting.")
        return guidance

    def permission_change(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "permission change: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "permission change: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("permission change: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("permission change: prefer bounded terminal evidence before acting.")
        return guidance

    def service_start(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("service start: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "service start: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("service start: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("service start: avoid bounded terminal evidence before acting.")
        return guidance

    def database_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("database checks: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "database checks: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("database checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("database checks: require bounded terminal evidence before acting.")
        return guidance

    def config_changes(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("config changes: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "config changes: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("config changes: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("config changes: record bounded terminal evidence before acting.")
        return guidance

    def dependency_pinning(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "dependency pinning: after failure, bound a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "dependency pinning: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("dependency pinning: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("dependency pinning: bound bounded terminal evidence before acting.")
        return guidance

    def language_detection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "language detection: after failure, inspect a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "language detection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("language detection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("language detection: inspect bounded terminal evidence before acting.")
        return guidance

    def git_inspection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("git inspection: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "git inspection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("git inspection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("git inspection: verify bounded terminal evidence before acting.")
        return guidance

    def git_mutation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("git mutation: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("git mutation: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("git mutation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("git mutation: rewrite bounded terminal evidence before acting.")
        return guidance

    def diff_review(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("diff review: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("diff review: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("diff review: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("diff review: summarize bounded terminal evidence before acting.")
        return guidance

    def completion_evidence(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "completion evidence: after failure, choose a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "completion evidence: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("completion evidence: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("completion evidence: choose bounded terminal evidence before acting.")
        return guidance

    def timeout_choice(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("timeout choice: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "timeout choice: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("timeout choice: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("timeout choice: prefer bounded terminal evidence before acting.")
        return guidance

    def command_repetition(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "command repetition: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "command repetition: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("command repetition: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("command repetition: avoid bounded terminal evidence before acting.")
        return guidance

    def stdout_budget(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("stdout budget: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "stdout budget: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("stdout budget: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("stdout budget: require bounded terminal evidence before acting.")
        return guidance

    def stderr_triage(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("stderr triage: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "stderr triage: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("stderr triage: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("stderr triage: record bounded terminal evidence before acting.")
        return guidance

    def path_safety(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("path safety: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("path safety: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("path safety: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("path safety: bound bounded terminal evidence before acting.")
        return guidance

    def absolute_paths(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("absolute paths: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "absolute paths: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("absolute paths: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("absolute paths: inspect bounded terminal evidence before acting.")
        return guidance

    def temporary_files(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("temporary files: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "temporary files: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("temporary files: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("temporary files: verify bounded terminal evidence before acting.")
        return guidance

    def cleanup_policy(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("cleanup policy: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "cleanup policy: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("cleanup policy: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("cleanup policy: rewrite bounded terminal evidence before acting.")
        return guidance

    def script_generation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "script generation: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "script generation: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("script generation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("script generation: summarize bounded terminal evidence before acting.")
        return guidance

    def macro_command(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("macro command: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "macro command: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("macro command: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("macro command: choose bounded terminal evidence before acting.")
        return guidance

    def compiler_errors(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("compiler errors: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "compiler errors: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("compiler errors: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("compiler errors: prefer bounded terminal evidence before acting.")
        return guidance

    def unit_failures(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("unit failures: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "unit failures: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("unit failures: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("unit failures: avoid bounded terminal evidence before acting.")
        return guidance

    def integration_failures(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "integration failures: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "integration failures: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("integration failures: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "integration failures: require bounded terminal evidence before acting."
            )
        return guidance

    def benchmark_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("benchmark checks: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "benchmark checks: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("benchmark checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("benchmark checks: record bounded terminal evidence before acting.")
        return guidance

    def data_validation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("data validation: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "data validation: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("data validation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("data validation: bound bounded terminal evidence before acting.")
        return guidance

    def checksum_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("checksum checks: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "checksum checks: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("checksum checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("checksum checks: inspect bounded terminal evidence before acting.")
        return guidance

    def formatters(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("formatters: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("formatters: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("formatters: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("formatters: verify bounded terminal evidence before acting.")
        return guidance

    def linters(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("linters: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("linters: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("linters: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("linters: rewrite bounded terminal evidence before acting.")
        return guidance

    def static_analysis(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "static analysis: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "static analysis: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("static analysis: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("static analysis: summarize bounded terminal evidence before acting.")
        return guidance

    def security_sensitive(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "security sensitive: after failure, choose a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "security sensitive: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("security sensitive: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("security sensitive: choose bounded terminal evidence before acting.")
        return guidance

    def credential_avoidance(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "credential avoidance: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "credential avoidance: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("credential avoidance: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("credential avoidance: prefer bounded terminal evidence before acting.")
        return guidance

    def environment_variables(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "environment variables: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "environment variables: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("environment variables: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("environment variables: avoid bounded terminal evidence before acting.")
        return guidance

    def container_limits(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "container limits: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "container limits: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("container limits: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("container limits: require bounded terminal evidence before acting.")
        return guidance

    def disk_usage(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("disk usage: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("disk usage: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("disk usage: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("disk usage: record bounded terminal evidence before acting.")
        return guidance

    def memory_usage(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("memory usage: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("memory usage: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("memory usage: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("memory usage: bound bounded terminal evidence before acting.")
        return guidance

    def parallelism(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("parallelism: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("parallelism: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("parallelism: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("parallelism: inspect bounded terminal evidence before acting.")
        return guidance

    def long_running_tasks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "long running tasks: after failure, verify a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "long running tasks: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("long running tasks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("long running tasks: verify bounded terminal evidence before acting.")
        return guidance

    def polling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("polling: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("polling: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("polling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("polling: rewrite bounded terminal evidence before acting.")
        return guidance

    def server_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("server probe: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("server probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("server probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("server probe: summarize bounded terminal evidence before acting.")
        return guidance

    def port_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("port checks: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("port checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("port checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("port checks: choose bounded terminal evidence before acting.")
        return guidance

    def text_transform(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("text transform: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "text transform: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("text transform: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("text transform: prefer bounded terminal evidence before acting.")
        return guidance

    def structured_data(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("structured data: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "structured data: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("structured data: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("structured data: avoid bounded terminal evidence before acting.")
        return guidance

    def json_processing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("json processing: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "json processing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("json processing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("json processing: require bounded terminal evidence before acting.")
        return guidance

    def csv_processing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("csv processing: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "csv processing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("csv processing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("csv processing: record bounded terminal evidence before acting.")
        return guidance

    def xml_processing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("xml processing: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "xml processing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("xml processing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("xml processing: bound bounded terminal evidence before acting.")
        return guidance

    def yaml_processing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("yaml processing: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "yaml processing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("yaml processing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("yaml processing: inspect bounded terminal evidence before acting.")
        return guidance

    def sql_processing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("sql processing: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "sql processing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("sql processing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("sql processing: verify bounded terminal evidence before acting.")
        return guidance

    def regex_usage(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("regex usage: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("regex usage: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("regex usage: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("regex usage: rewrite bounded terminal evidence before acting.")
        return guidance

    def numeric_validation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "numeric validation: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "numeric validation: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("numeric validation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "numeric validation: summarize bounded terminal evidence before acting."
            )
        return guidance

    def image_metadata(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("image metadata: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "image metadata: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("image metadata: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("image metadata: choose bounded terminal evidence before acting.")
        return guidance

    def notebook_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "notebook handling: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "notebook handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("notebook handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("notebook handling: prefer bounded terminal evidence before acting.")
        return guidance

    def symlink_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("symlink handling: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "symlink handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("symlink handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("symlink handling: avoid bounded terminal evidence before acting.")
        return guidance

    def hidden_files(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("hidden files: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("hidden files: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("hidden files: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("hidden files: require bounded terminal evidence before acting.")
        return guidance

    def generated_files(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("generated files: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "generated files: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("generated files: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("generated files: record bounded terminal evidence before acting.")
        return guidance

    def patch_application(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("patch application: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "patch application: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("patch application: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("patch application: bound bounded terminal evidence before acting.")
        return guidance

    def backup_before_edit(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "backup before edit: after failure, inspect a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "backup before edit: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("backup before edit: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("backup before edit: inspect bounded terminal evidence before acting.")
        return guidance

    def idempotency(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("idempotency: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("idempotency: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("idempotency: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("idempotency: verify bounded terminal evidence before acting.")
        return guidance

    def rollback_plan(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("rollback plan: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "rollback plan: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("rollback plan: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("rollback plan: rewrite bounded terminal evidence before acting.")
        return guidance

    def partial_progress(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "partial progress: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "partial progress: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("partial progress: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("partial progress: summarize bounded terminal evidence before acting.")
        return guidance

    def final_sanity(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("final sanity: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("final sanity: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("final sanity: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("final sanity: choose bounded terminal evidence before acting.")
        return guidance

    def tool_contract(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("tool contract: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "tool contract: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("tool contract: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("tool contract: prefer bounded terminal evidence before acting.")
        return guidance

    def model_response_repair(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "model response repair: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "model response repair: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("model response repair: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("model response repair: avoid bounded terminal evidence before acting.")
        return guidance

    def concatenated_json(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "concatenated json: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "concatenated json: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("concatenated json: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("concatenated json: require bounded terminal evidence before acting.")
        return guidance

    def fenced_json(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("fenced json: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("fenced json: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("fenced json: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("fenced json: record bounded terminal evidence before acting.")
        return guidance

    def malformed_action(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("malformed action: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "malformed action: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("malformed action: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("malformed action: bound bounded terminal evidence before acting.")
        return guidance

    def empty_command(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("empty command: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "empty command: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("empty command: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("empty command: inspect bounded terminal evidence before acting.")
        return guidance

    def unsafe_delete(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("unsafe delete: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "unsafe delete: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("unsafe delete: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("unsafe delete: verify bounded terminal evidence before acting.")
        return guidance

    def root_write(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("root write: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("root write: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("root write: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("root write: rewrite bounded terminal evidence before acting.")
        return guidance

    def pager_disable(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("pager disable: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "pager disable: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("pager disable: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("pager disable: summarize bounded terminal evidence before acting.")
        return guidance

    def noninteractive_flags(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "noninteractive flags: after failure, choose a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "noninteractive flags: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("noninteractive flags: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("noninteractive flags: choose bounded terminal evidence before acting.")
        return guidance

    def output_sampling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("output sampling: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "output sampling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("output sampling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("output sampling: prefer bounded terminal evidence before acting.")
        return guidance

    def tail_inspection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("tail inspection: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "tail inspection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("tail inspection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("tail inspection: avoid bounded terminal evidence before acting.")
        return guidance

    def head_inspection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("head inspection: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "head inspection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("head inspection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("head inspection: require bounded terminal evidence before acting.")
        return guidance

    def find_depth(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("find depth: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("find depth: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("find depth: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("find depth: record bounded terminal evidence before acting.")
        return guidance

    def rg_preference(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("rg preference: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "rg preference: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("rg preference: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("rg preference: bound bounded terminal evidence before acting.")
        return guidance

    def sed_ranges(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("sed ranges: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("sed ranges: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("sed ranges: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("sed ranges: inspect bounded terminal evidence before acting.")
        return guidance

    def awk_limits(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("awk limits: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("awk limits: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("awk limits: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("awk limits: verify bounded terminal evidence before acting.")
        return guidance

    def jq_filters(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("jq filters: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("jq filters: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("jq filters: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("jq filters: rewrite bounded terminal evidence before acting.")
        return guidance

    def python_compile(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "python compile: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "python compile: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("python compile: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("python compile: summarize bounded terminal evidence before acting.")
        return guidance

    def node_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("node checks: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("node checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("node checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("node checks: choose bounded terminal evidence before acting.")
        return guidance

    def go_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("go checks: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("go checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("go checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("go checks: prefer bounded terminal evidence before acting.")
        return guidance

    def rust_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("rust checks: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("rust checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("rust checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("rust checks: avoid bounded terminal evidence before acting.")
        return guidance

    def java_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("java checks: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("java checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("java checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("java checks: require bounded terminal evidence before acting.")
        return guidance

    def c_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("c checks: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("c checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("c checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("c checks: record bounded terminal evidence before acting.")
        return guidance

    def shell_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("shell checks: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("shell checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("shell checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("shell checks: bound bounded terminal evidence before acting.")
        return guidance

    def make_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("make checks: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("make checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("make checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("make checks: inspect bounded terminal evidence before acting.")
        return guidance

    def cmake_checks(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("cmake checks: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("cmake checks: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("cmake checks: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("cmake checks: verify bounded terminal evidence before acting.")
        return guidance

    def docker_absence(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("docker absence: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "docker absence: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("docker absence: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("docker absence: rewrite bounded terminal evidence before acting.")
        return guidance

    def no_task_assumptions(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "no task assumptions: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "no task assumptions: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("no task assumptions: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "no task assumptions: summarize bounded terminal evidence before acting."
            )
        return guidance

    def evidence_summary(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("evidence summary: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "evidence summary: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("evidence summary: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("evidence summary: choose bounded terminal evidence before acting.")
        return guidance

    def turn_budget(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("turn budget: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("turn budget: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("turn budget: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("turn budget: prefer bounded terminal evidence before acting.")
        return guidance

    def history_compression(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "history compression: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "history compression: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("history compression: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("history compression: avoid bounded terminal evidence before acting.")
        return guidance

    def state_tracking(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("state tracking: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "state tracking: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("state tracking: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("state tracking: require bounded terminal evidence before acting.")
        return guidance

    def changed_file_tracking(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "changed file tracking: after failure, record a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "changed file tracking: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("changed file tracking: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "changed file tracking: record bounded terminal evidence before acting."
            )
        return guidance

    def risk_explanation(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("risk explanation: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "risk explanation: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("risk explanation: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("risk explanation: bound bounded terminal evidence before acting.")
        return guidance

    def recovery_prompting(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "recovery prompting: after failure, inspect a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "recovery prompting: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("recovery prompting: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("recovery prompting: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_rewrite(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe rewrite: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe rewrite: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe rewrite: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe rewrite: verify bounded terminal evidence before acting.")
        return guidance

    def refusal_path(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("refusal path: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("refusal path: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("refusal path: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("refusal path: rewrite bounded terminal evidence before acting.")
        return guidance

    def done_gate(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("done gate: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("done gate: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("done gate: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("done gate: summarize bounded terminal evidence before acting.")
        return guidance

    def verification_gate(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "verification gate: after failure, choose a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "verification gate: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("verification gate: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("verification gate: choose bounded terminal evidence before acting.")
        return guidance

    def post_edit_gate(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("post edit gate: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "post edit gate: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("post edit gate: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("post edit gate: prefer bounded terminal evidence before acting.")
        return guidance

    def pre_edit_inspect(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("pre edit inspect: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "pre edit inspect: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("pre edit inspect: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("pre edit inspect: avoid bounded terminal evidence before acting.")
        return guidance

    def smallest_change(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("smallest change: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "smallest change: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("smallest change: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("smallest change: require bounded terminal evidence before acting.")
        return guidance

    def broad_search(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("broad search: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("broad search: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("broad search: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("broad search: record bounded terminal evidence before acting.")
        return guidance

    def narrow_search(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("narrow search: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "narrow search: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("narrow search: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("narrow search: bound bounded terminal evidence before acting.")
        return guidance

    def multi_command_batch(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "multi command batch: after failure, inspect a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "multi command batch: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("multi command batch: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("multi command batch: inspect bounded terminal evidence before acting.")
        return guidance

    def single_command_turn(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "single command turn: after failure, verify a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "single command turn: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("single command turn: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("single command turn: verify bounded terminal evidence before acting.")
        return guidance

    def command_cleanup(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("command cleanup: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "command cleanup: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("command cleanup: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("command cleanup: rewrite bounded terminal evidence before acting.")
        return guidance

    def quote_cleanup(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("quote cleanup: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "quote cleanup: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("quote cleanup: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("quote cleanup: summarize bounded terminal evidence before acting.")
        return guidance

    def json_arguments(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("json arguments: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "json arguments: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("json arguments: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("json arguments: choose bounded terminal evidence before acting.")
        return guidance

    def tool_name_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("tool name alias: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "tool name alias: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("tool name alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("tool name alias: prefer bounded terminal evidence before acting.")
        return guidance

    def timeout_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("timeout alias: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "timeout alias: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("timeout alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("timeout alias: avoid bounded terminal evidence before acting.")
        return guidance

    def duration_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("duration alias: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "duration alias: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("duration alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("duration alias: require bounded terminal evidence before acting.")
        return guidance

    def keystroke_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("keystroke alias: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "keystroke alias: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("keystroke alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("keystroke alias: record bounded terminal evidence before acting.")
        return guidance

    def shell_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("shell alias: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("shell alias: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("shell alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("shell alias: bound bounded terminal evidence before acting.")
        return guidance

    def finish_alias(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("finish alias: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("finish alias: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("finish alias: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("finish alias: inspect bounded terminal evidence before acting.")
        return guidance

    def response_selection(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "response selection: after failure, verify a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "response selection: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("response selection: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("response selection: verify bounded terminal evidence before acting.")
        return guidance

    def last_object_preference(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "last object preference: after failure, rewrite a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "last object preference: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("last object preference: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "last object preference: rewrite bounded terminal evidence before acting."
            )
        return guidance

    def dedupe_actions(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "dedupe actions: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "dedupe actions: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("dedupe actions: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("dedupe actions: summarize bounded terminal evidence before acting.")
        return guidance

    def balanced_braces(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("balanced braces: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "balanced braces: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("balanced braces: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("balanced braces: choose bounded terminal evidence before acting.")
        return guidance

    def escape_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("escape handling: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "escape handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("escape handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("escape handling: prefer bounded terminal evidence before acting.")
        return guidance

    def unicode_handling(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("unicode handling: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "unicode handling: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("unicode handling: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("unicode handling: avoid bounded terminal evidence before acting.")
        return guidance

    def line_endings(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("line endings: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("line endings: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("line endings: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("line endings: require bounded terminal evidence before acting.")
        return guidance

    def case_sensitivity(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("case sensitivity: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "case sensitivity: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("case sensitivity: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("case sensitivity: record bounded terminal evidence before acting.")
        return guidance

    def permission_denied(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("permission denied: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "permission denied: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("permission denied: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("permission denied: bound bounded terminal evidence before acting.")
        return guidance

    def missing_command(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("missing command: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "missing command: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("missing command: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("missing command: inspect bounded terminal evidence before acting.")
        return guidance

    def missing_file(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("missing file: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("missing file: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("missing file: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("missing file: verify bounded terminal evidence before acting.")
        return guidance

    def module_import(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("module import: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "module import: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("module import: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("module import: rewrite bounded terminal evidence before acting.")
        return guidance

    def syntax_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("syntax error: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("syntax error: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("syntax error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("syntax error: summarize bounded terminal evidence before acting.")
        return guidance

    def assertion_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("assertion error: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "assertion error: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("assertion error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("assertion error: choose bounded terminal evidence before acting.")
        return guidance

    def segfault(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("segfault: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("segfault: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("segfault: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("segfault: prefer bounded terminal evidence before acting.")
        return guidance

    def timeout_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("timeout error: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "timeout error: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("timeout error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("timeout error: avoid bounded terminal evidence before acting.")
        return guidance

    def oom_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("oom error: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("oom error: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("oom error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("oom error: require bounded terminal evidence before acting.")
        return guidance

    def network_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("network error: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "network error: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("network error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("network error: record bounded terminal evidence before acting.")
        return guidance

    def package_error(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("package error: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "package error: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("package error: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("package error: bound bounded terminal evidence before acting.")
        return guidance

    def compiler_warning(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "compiler warning: after failure, inspect a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "compiler warning: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("compiler warning: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("compiler warning: inspect bounded terminal evidence before acting.")
        return guidance

    def test_flake(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("test flake: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("test flake: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("test flake: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("test flake: verify bounded terminal evidence before acting.")
        return guidance

    def version_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("version probe: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "version probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("version probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("version probe: rewrite bounded terminal evidence before acting.")
        return guidance

    def help_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("help probe: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("help probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("help probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("help probe: summarize bounded terminal evidence before acting.")
        return guidance

    def readme_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("readme probe: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("readme probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("readme probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("readme probe: choose bounded terminal evidence before acting.")
        return guidance

    def eval_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("eval probe: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("eval probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("eval probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("eval probe: prefer bounded terminal evidence before acting.")
        return guidance

    def harness_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("harness probe: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "harness probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("harness probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("harness probe: avoid bounded terminal evidence before acting.")
        return guidance

    def workspace_listing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "workspace listing: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "workspace listing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("workspace listing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("workspace listing: require bounded terminal evidence before acting.")
        return guidance

    def root_listing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("root listing: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("root listing: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("root listing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("root listing: record bounded terminal evidence before acting.")
        return guidance

    def maxdepth_listing(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("maxdepth listing: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "maxdepth listing: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("maxdepth listing: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("maxdepth listing: bound bounded terminal evidence before acting.")
        return guidance

    def size_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("size probe: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("size probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("size probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("size probe: inspect bounded terminal evidence before acting.")
        return guidance

    def file_type_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("file type probe: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "file type probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("file type probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("file type probe: verify bounded terminal evidence before acting.")
        return guidance

    def git_status(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("git status: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("git status: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("git status: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("git status: rewrite bounded terminal evidence before acting.")
        return guidance

    def git_log(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("git log: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("git log: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("git log: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("git log: summarize bounded terminal evidence before acting.")
        return guidance

    def git_show(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("git show: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("git show: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("git show: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("git show: choose bounded terminal evidence before acting.")
        return guidance

    def diff_stat(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("diff stat: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("diff stat: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("diff stat: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("diff stat: prefer bounded terminal evidence before acting.")
        return guidance

    def config_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("config probe: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("config probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("config probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("config probe: avoid bounded terminal evidence before acting.")
        return guidance

    def dependency_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "dependency probe: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "dependency probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("dependency probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("dependency probe: require bounded terminal evidence before acting.")
        return guidance

    def lockfile_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("lockfile probe: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "lockfile probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("lockfile probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("lockfile probe: record bounded terminal evidence before acting.")
        return guidance

    def entrypoint_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("entrypoint probe: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "entrypoint probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("entrypoint probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("entrypoint probe: bound bounded terminal evidence before acting.")
        return guidance

    def main_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("main probe: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("main probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("main probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("main probe: inspect bounded terminal evidence before acting.")
        return guidance

    def cli_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("cli probe: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("cli probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("cli probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("cli probe: verify bounded terminal evidence before acting.")
        return guidance

    def api_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("api probe: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("api probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("api probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("api probe: rewrite bounded terminal evidence before acting.")
        return guidance

    def schema_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("schema probe: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("schema probe: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("schema probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("schema probe: summarize bounded terminal evidence before acting.")
        return guidance

    def sample_data_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "sample data probe: after failure, choose a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "sample data probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("sample data probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("sample data probe: choose bounded terminal evidence before acting.")
        return guidance

    def expected_output_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "expected output probe: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "expected output probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("expected output probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append(
                "expected output probe: prefer bounded terminal evidence before acting."
            )
        return guidance

    def actual_output_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "actual output probe: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "actual output probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("actual output probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("actual output probe: avoid bounded terminal evidence before acting.")
        return guidance

    def comparison_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "comparison probe: after failure, require a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "comparison probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("comparison probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("comparison probe: require bounded terminal evidence before acting.")
        return guidance

    def roundtrip_probe(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("roundtrip probe: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "roundtrip probe: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("roundtrip probe: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("roundtrip probe: record bounded terminal evidence before acting.")
        return guidance

    def smoke_test(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("smoke test: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("smoke test: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("smoke test: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("smoke test: bound bounded terminal evidence before acting.")
        return guidance

    def targeted_test(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("targeted test: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "targeted test: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("targeted test: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("targeted test: inspect bounded terminal evidence before acting.")
        return guidance

    def full_test(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("full test: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("full test: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("full test: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("full test: verify bounded terminal evidence before acting.")
        return guidance

    def performance_test(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "performance test: after failure, rewrite a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "performance test: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("performance test: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("performance test: rewrite bounded terminal evidence before acting.")
        return guidance

    def artifact_check(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "artifact check: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "artifact check: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("artifact check: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("artifact check: summarize bounded terminal evidence before acting.")
        return guidance

    def final_command(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("final command: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "final command: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("final command: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("final command: choose bounded terminal evidence before acting.")
        return guidance

    def blocked_done_message(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "blocked done message: after failure, prefer a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "blocked done message: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("blocked done message: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("blocked done message: prefer bounded terminal evidence before acting.")
        return guidance

    def blocked_run_message(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "blocked run message: after failure, avoid a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "blocked run message: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("blocked run message: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("blocked run message: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_cat(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe cat: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe cat: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe cat: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe cat: require bounded terminal evidence before acting.")
        return guidance

    def safe_grep(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe grep: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe grep: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe grep: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe grep: record bounded terminal evidence before acting.")
        return guidance

    def safe_find(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe find: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe find: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe find: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe find: bound bounded terminal evidence before acting.")
        return guidance

    def safe_python(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe python: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe python: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe python: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe python: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_perl(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe perl: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe perl: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe perl: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe perl: verify bounded terminal evidence before acting.")
        return guidance

    def safe_sed(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sed: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sed: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sed: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sed: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_make(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe make: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe make: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe make: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe make: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_npm(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe npm: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe npm: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe npm: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe npm: choose bounded terminal evidence before acting.")
        return guidance

    def safe_pip(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe pip: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe pip: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe pip: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe pip: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_cargo(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe cargo: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe cargo: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe cargo: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe cargo: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_go(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe go: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe go: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe go: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe go: require bounded terminal evidence before acting.")
        return guidance

    def safe_java(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe java: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe java: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe java: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe java: record bounded terminal evidence before acting.")
        return guidance

    def safe_sqlite(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sqlite: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sqlite: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sqlite: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sqlite: bound bounded terminal evidence before acting.")
        return guidance

    def safe_openssl(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe openssl: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe openssl: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe openssl: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe openssl: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_tar(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe tar: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe tar: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe tar: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe tar: verify bounded terminal evidence before acting.")
        return guidance

    def safe_unzip(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe unzip: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe unzip: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe unzip: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe unzip: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_chmod(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe chmod: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe chmod: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe chmod: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe chmod: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_cp(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe cp: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe cp: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe cp: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe cp: choose bounded terminal evidence before acting.")
        return guidance

    def safe_mv(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe mv: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe mv: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe mv: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe mv: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_rm(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe rm: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe rm: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe rm: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe rm: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_mkdir(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe mkdir: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe mkdir: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe mkdir: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe mkdir: require bounded terminal evidence before acting.")
        return guidance

    def safe_touch(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe touch: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe touch: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe touch: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe touch: record bounded terminal evidence before acting.")
        return guidance

    def safe_ln(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe ln: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe ln: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe ln: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe ln: bound bounded terminal evidence before acting.")
        return guidance

    def safe_ps(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe ps: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe ps: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe ps: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe ps: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_kill(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe kill: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe kill: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe kill: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe kill: verify bounded terminal evidence before acting.")
        return guidance

    def safe_nc(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe nc: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe nc: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe nc: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe nc: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_curl(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe curl: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe curl: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe curl: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe curl: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_wget(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe wget: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe wget: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe wget: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe wget: choose bounded terminal evidence before acting.")
        return guidance

    def safe_ssh(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe ssh: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe ssh: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe ssh: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe ssh: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_timeout(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe timeout: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe timeout: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe timeout: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe timeout: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_yes(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe yes: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe yes: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe yes: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe yes: require bounded terminal evidence before acting.")
        return guidance

    def safe_sleep(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sleep: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sleep: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sleep: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sleep: record bounded terminal evidence before acting.")
        return guidance

    def safe_watch(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe watch: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe watch: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe watch: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe watch: bound bounded terminal evidence before acting.")
        return guidance

    def safe_tail(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe tail: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe tail: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe tail: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe tail: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_head(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe head: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe head: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe head: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe head: verify bounded terminal evidence before acting.")
        return guidance

    def safe_tree(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe tree: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe tree: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe tree: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe tree: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_du(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe du: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe du: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe du: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe du: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_wc(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe wc: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe wc: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe wc: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe wc: choose bounded terminal evidence before acting.")
        return guidance

    def safe_sort(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sort: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sort: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sort: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sort: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_uniq(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe uniq: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe uniq: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe uniq: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe uniq: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_xargs(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe xargs: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe xargs: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe xargs: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe xargs: require bounded terminal evidence before acting.")
        return guidance

    def safe_parallel(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe parallel: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "safe parallel: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("safe parallel: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe parallel: record bounded terminal evidence before acting.")
        return guidance

    def safe_env(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe env: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe env: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe env: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe env: bound bounded terminal evidence before acting.")
        return guidance

    def safe_export(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe export: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe export: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe export: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe export: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_source(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe source: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe source: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe source: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe source: verify bounded terminal evidence before acting.")
        return guidance

    def safe_bash(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe bash: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe bash: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe bash: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe bash: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_sh(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sh: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sh: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sh: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sh: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_zsh(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe zsh: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe zsh: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe zsh: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe zsh: choose bounded terminal evidence before acting.")
        return guidance

    def safe_fish(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe fish: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe fish: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe fish: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe fish: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_vim(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe vim: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe vim: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe vim: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe vim: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_editor(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe editor: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe editor: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe editor: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe editor: require bounded terminal evidence before acting.")
        return guidance

    def safe_repl(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe repl: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe repl: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe repl: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe repl: record bounded terminal evidence before acting.")
        return guidance

    def safe_server(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe server: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe server: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe server: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe server: bound bounded terminal evidence before acting.")
        return guidance

    def safe_daemon(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe daemon: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe daemon: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe daemon: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe daemon: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_service(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe service: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe service: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe service: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe service: verify bounded terminal evidence before acting.")
        return guidance

    def safe_systemctl(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe systemctl: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "safe systemctl: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("safe systemctl: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe systemctl: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_docker(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe docker: after failure, summarize a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe docker: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe docker: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe docker: summarize bounded terminal evidence before acting.")
        return guidance

    def safe_mount(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe mount: after failure, choose a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe mount: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe mount: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe mount: choose bounded terminal evidence before acting.")
        return guidance

    def safe_dd(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe dd: after failure, prefer a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe dd: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe dd: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe dd: prefer bounded terminal evidence before acting.")
        return guidance

    def safe_mkfs(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe mkfs: after failure, avoid a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe mkfs: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe mkfs: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe mkfs: avoid bounded terminal evidence before acting.")
        return guidance

    def safe_chown(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe chown: after failure, require a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe chown: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe chown: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe chown: require bounded terminal evidence before acting.")
        return guidance

    def safe_sudo(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe sudo: after failure, record a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe sudo: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe sudo: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe sudo: record bounded terminal evidence before acting.")
        return guidance

    def safe_root(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe root: after failure, bound a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe root: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe root: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe root: bound bounded terminal evidence before acting.")
        return guidance

    def safe_home(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe home: after failure, inspect a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe home: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe home: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe home: inspect bounded terminal evidence before acting.")
        return guidance

    def safe_tmp(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe tmp: after failure, verify a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe tmp: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe tmp: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe tmp: verify bounded terminal evidence before acting.")
        return guidance

    def safe_app(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append("safe app: after failure, rewrite a smaller diagnostic command.")
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append("safe app: edits are not complete until a relevant check succeeds.")
        if signal.same_command_count >= 2:
            guidance.append("safe app: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe app: rewrite bounded terminal evidence before acting.")
        return guidance

    def safe_workspace(self, signal: HistorySignal) -> list[str]:
        guidance: list[str] = []
        if signal.recent_failure:
            guidance.append(
                "safe workspace: after failure, summarize a smaller diagnostic command."
            )
        if signal.recent_edit and not signal.successful_verify_after_edit:
            guidance.append(
                "safe workspace: edits are not complete until a relevant check succeeds."
            )
        if signal.same_command_count >= 2:
            guidance.append("safe workspace: repeated commands need a changed hypothesis.")
        if not guidance:
            guidance.append("safe workspace: summarize bounded terminal evidence before acting.")
        return guidance


_ORIGINAL_POLICY_LINES = PromptBuilder._policy_lines


def _policy_lines_with_library(self: PromptBuilder, signal: HistorySignal) -> list[str]:
    lines = _ORIGINAL_POLICY_LINES(self, signal)
    lines.extend(PolicyLibrary().describe(signal))
    return lines[:120]


PromptBuilder._policy_lines = _policy_lines_with_library


def create_agent() -> CandidateHarness:
    return CandidateHarness()
