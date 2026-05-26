import base64
import hashlib
import json
import os
import time
import zlib
from collections import namedtuple
from dataclasses import dataclass, field, replace
from datetime import date
from functools import partial
from typing import Any
from plumbing.base_agent import BaseHarness
from plumbing.openai_client import (
    ToolModelResult,
    call_terminal_model,
    call_terminal_model_with_tools,
)
from plumbing.types import HarnessToolCall, HarnessTurn


def _z(*parts):
    return zlib.decompress(base64.b64decode("".join(parts))).decode()


CODEX_BASE_INSTRUCTIONS = _z(
    "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",
)
APPLY_PATCH_GRAMMAR = _z(
    "eNptUU1PwzAMvftXWJGQ+qGt9x7ohqCnIXHhXIXGWyvapBoZ4sCPx8lcWtAujf3y3vNr/OH12Zf4RqfeNpP2bYfdxb7nS",
    "NZce1jdlaiyLMOHgOBLQBQeavjlyv2TNcttBRAcS9TGNKHCbzQ0kKe5u0xGSwczSZz2xmDdD8QtHvm0eiT2jF5DbymHlZ",
    "VoHiNySwarSUJ+jcjNGW2n7Yma0X1SJQ3/y8wosUi2eVrAHIXluQpglhZBvrnHAAOsfGToM5fo3f90V2KJiShaZz19eX4",
    "hAYJfyqtxx1hW8JfISrXbKebzgUoCpou15ExCUCZt4hdVuqSG2Xu1SXeMzxM3DXf9OLmzx9aNo7PbQ/0DhYKyfA==",
)
SUMMARIZATION_PROMPT = _z(
    "eNpFkMFOwzAMhu99Ch9BKnsAbqOaRMW27jAJOJrkL42WJpWTaOvb4wwJDpGc6PPnP/6MhVhAC2SMMrvwTUzdcDzvPs7Uv",
    "e66t9PQH7UcDqdtd+6H44Y6AWcoN3GwcRwplXlmWUkNxCHmCUL7/YHyxJmuznsSKAN90MPpsmmaPhhfLJ6bJ+qKCEKmRe",
    "K3gkkdli5YycK45GJINLOFkv28RMmsrIkh45bbWqQs7EJOLen8knT4IhihToOkXe81hmBWKFGO9AWyMYAejAcLBfVQylj",
    "So8LbsJIRl51hT5Yzt4Qbz4vHr//frI2wsNVY07hQ0DQvqBeNjVadUkwuAtvevzRGo+ksxUAT/FJ3XRdyn1/XlcCzjkl+",
    "/RPegWuUy+YHFSeVIA==",
)
SUMMARY_PREFIX = _z(
    "eNptkDFuwzAMRa/yD1D4ANm69QIdMtIWYwuRxUCkYvj2Je20QIBuEkW+/6jPKrZwQ6E6d5oZqyQuUKNmnGAClfJk2JIVj",
    "yZj4RVUU5xTn7yFoH1dqe2QG7JptNZ7rnO0TKw64CodVFSwkKNoimqgPTmSjGM0LiZSAkCGjRujqweM+1l5VxzwrS8tJ4",
    "09lwSpB2WTdj9HFlIPbkxpx8hckaTyoU9PyQmpP0qeyMI2pgZ8RWwwQ+2119+qh4nL/vNjH+F6vOZ6k7Y6021yPQ1/SW5",
    "Kqll9vWwLdukNslU3orJ7/fID7yWSCg==",
)
CODEX_UPSTREAM_COMMIT = "9f42c89c0112771dc29100a6f3fc904049b2655f"
CODEX_UPSTREAM_DATE = "2026-05-24"
MAX_OBSERVATION_CHARS = 20000
MAX_FUNCTION_OUTPUT_CHARS = 24000
MAX_CONTEXT_HISTORY_ITEMS = 96
MAX_CONTEXT_HISTORY_CHARS = 90000
MAX_RAW_RESPONSE_ITEMS = 48
COMPACT_USER_MESSAGE_MAX_TOKENS = 20000
FUNCTION_HISTORY_TOOLS = {"exec_command", "write_stdin", "update_plan"}
FUNCTION_CALL_TYPES = {"function_call", "local_shell_call"}
FUNCTION_OUTPUT_TYPES = {"function_call_output"}
CUSTOM_CALL_TYPES = {"custom_tool_call"}
CUSTOM_OUTPUT_TYPES = {"custom_tool_call_output"}
TOOL_OUTPUT_TYPES = FUNCTION_OUTPUT_TYPES | CUSTOM_OUTPUT_TYPES | {"tool_search_output"}
TOOL_CALL_TYPES = FUNCTION_CALL_TYPES | CUSTOM_CALL_TYPES | {"tool_search_call"}
SHELL_TOOL_NAMES = {"exec_command", "shell_command", "local_shell", "local_shell_call"}
PERMISSIONS_SANDBOX_DANGER_FULL_ACCESS = (
    "Filesystem sandboxing defines which files can be read or written. "
    "`sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all "
    "commands are permitted. Network access is enabled."
)
PERMISSIONS_APPROVAL_NEVER = (
    "Approval policy is currently never. Do not provide the `sandbox_"
    "permissions` for any reason, commands will be rejected."
)
PORT_PARITY_MANIFEST = tuple(
    json.loads(
        _z(
            "eNqVlk9v4zYQxb8K4bMcAz0mJ6+b7AZYZw3HPSyKQqCpsUWEIlX+cewt+t37SEqOnN3CysWAJZKa37yZN/zzn",
            "0lonbfEm8ntRJiKjlPrZsJYmjkrZo6ck0bPfLD6xrrb222Qqipba5rWT4qJ89wHh61SCxUqqvCsPfnaaDx7t1",
            "YbT3HpmlxrNE5mUrfBF8wbo5hrSbiCtdxypUixneL7gnFdsS13hKWIMgiPYBzjllhryZE9UHUz+bf4KIUNunS",
            "8aZXU+9LS34HcBY2T8d1OXvIsEI2suKcv3GoceaPp6EthmgYvBoB4vTWWmVeE6mtiJniyTBnT3jHiombDfcyS",
            "MB1cXr6I8TNvuXbCytaPAIwZdDObPpQAN3iwTn97yeKSUiC311R723rz884e8SHopEUSSATnTZNljOuyQBXFO",
            "Cu2g/5soLmnxo1G6svhDLXA33XQXjZ0XbAByhnClTEgiO6CGlblqi+8MwYztiKLCmHSvZXbHevkpSMJHO0gYK",
            "wrkGZ2qCmCtaS9Oo3mrJFHRXjualKqjM0A5GtSlTGGvpCSTkmP8tUiy6XzldTp6QBzkUNjQadsJYquAUVNDU8",
            "HIGVozPMTiClMG1dL7Q1bpc+zSqJgPw7I21adypZ7UY/GHO55x/NgiXbGNuwrty9sbzlyYZNgUusUMZuvVl+/",
            "l6v5ZvGl/LyeL5fz9eiwIaZHs6byu0eqFjnXsbC+pSSN6aa88gFRch9rsct9Vi8eP+BZIhA1PUgnt4ouVepUq",
            "YlD173LSvUl2sTkZOsYAddhlWDhe7KzWqKD7emdFv/jgnnvMm+96CBpmY6USv7gZ3fYhmpPnu3QHFsuXlI5oW",
            "W4jPJweFZ0EIwILmKap86fAN5QQ7HWuhc4axRVf0p2eKnh71Ty4E3ZvSs9dy+jGRd5kxlS3h/xpI+L5eE2gzv",
            "s5JFlO28VFyn8aZdW5mreUgIPDtCvEp+JShrz4iAh9jQ4/aOwlxL2iacRDQW8A3oxafTU7xtCLiUGJqoql1z2",
            "c3fSmE4OK6uu9Fo09cWSyHHoAHNFxkNa1MUYowCG9q5sqlFlOI+rHwe3gjVp9MIFRufUwsDY43Sdf75/2jzfN",
            "DDxOGjrPMYRdtzpWC3JcougYeSsyy/GF99HMd14TWakD9IaHbeVbw5yTZVHLb3kqqu9T3H2wix+cdaA8BkJVj",
            "QdLGLitSpYGiMFi5eVfImKA/OH0XTmYTtkxby67rqxS+70AcaWbCPTtcqVw8vZCNDV29YrAv7ONWp7ugtKTbk",
            "QUaxWBYcLFOp3irFgzQFaVfirDCJ6o5O9rMk90VXqFBnZ5tNvIyD3gdtK8mxgDj9bc0Smr1flfboUgGZllBSn",
            "AQo+zGBKiN1ojFFUGqgR9umO9Ryz7kssDcvu4gE/iVcP7EjDLBtONv8GLgqF+U9Amz8eC7ZcrIqYrT3UKeI38",
            "OtJwZm8PRU4l6tYFAUTyoSKIXsOdkVaXFysTCMxsy7l+7beYJyuHzffy+X86fHh/nkzHAPWVFCUuWB3MMII7C",
            "EYxbvg9sR4lOBdEs6dGE0kPtcxhRgYWL6N7CD86z8Ijp/A",
        )
    )
)
ENABLE_PORT_PARITY_MANIFEST = True
ENABLE_HISTORY_REPLAY = True
ENABLE_CONTEXT_MANAGER = True
ENABLE_CONTEXT_NORMALIZATION = True
ENABLE_CONTEXT_BUDGETING = True
ENABLE_MODEL_CONTEXT_COMPACTION = True
ENABLE_PATCH_TOOL = True
ENABLE_PLAN_TOOL = True
ENABLE_WRITE_STDIN_TOOL = True
ENABLE_UNIFIED_EXEC_OUTPUT_FORMAT = True
ENABLE_MODEL_RESPONSE_ITEM_REPLAY = True
ENABLE_MODEL_CALL_RESILIENCE = True
ENABLE_RECOVERY_POLICY = True
ENABLE_COMMAND_CLASSIFICATION = True
ENABLE_COMPLETION_POLICY = True
ENABLE_INSTRUMENTATION = True


@dataclass(frozen=True)
class FeatureSet:
    port_parity_manifest: bool = True
    history_replay: bool = True
    context_manager: bool = True
    context_normalization: bool = True
    context_budgeting: bool = True
    model_context_compaction: bool = True
    patch_tool: bool = True
    plan_tool: bool = True
    write_stdin_tool: bool = True
    unified_exec_output_format: bool = True
    model_response_item_replay: bool = True
    model_call_resilience: bool = True
    recovery_policy: bool = True
    command_classification: bool = True
    completion_policy: bool = True
    instrumentation: bool = True

    @classmethod
    def from_globals(cls):
        return cls(
            port_parity_manifest=ENABLE_PORT_PARITY_MANIFEST,
            history_replay=ENABLE_HISTORY_REPLAY,
            context_manager=ENABLE_CONTEXT_MANAGER,
            context_normalization=ENABLE_CONTEXT_NORMALIZATION,
            context_budgeting=ENABLE_CONTEXT_BUDGETING,
            model_context_compaction=ENABLE_MODEL_CONTEXT_COMPACTION,
            patch_tool=ENABLE_PATCH_TOOL,
            plan_tool=ENABLE_PLAN_TOOL,
            write_stdin_tool=ENABLE_WRITE_STDIN_TOOL,
            unified_exec_output_format=ENABLE_UNIFIED_EXEC_OUTPUT_FORMAT,
            model_response_item_replay=ENABLE_MODEL_RESPONSE_ITEM_REPLAY,
            model_call_resilience=ENABLE_MODEL_CALL_RESILIENCE,
            recovery_policy=ENABLE_RECOVERY_POLICY,
            command_classification=ENABLE_COMMAND_CLASSIFICATION,
            completion_policy=ENABLE_COMPLETION_POLICY,
            instrumentation=ENABLE_INSTRUMENTATION,
        )

    def with_overrides(self, overrides):
        return replace(self, **overrides)


PROFILE_OVERRIDES = {
    "codex_full": {},
    "no_instrumentation": {"instrumentation": False, "port_parity_manifest": False},
    "no_classifier": {"command_classification": False},
    "no_recovery": {"recovery_policy": False},
    "no_compaction": {"model_context_compaction": False},
    "exec_only_tools": {"patch_tool": False, "plan_tool": False, "write_stdin_tool": False},
    "minimal_loop": {
        "history_replay": False,
        "context_manager": False,
        "context_normalization": False,
        "context_budgeting": False,
        "model_context_compaction": False,
        "patch_tool": False,
        "plan_tool": False,
        "write_stdin_tool": False,
        "unified_exec_output_format": False,
        "model_response_item_replay": False,
        "model_call_resilience": False,
        "recovery_policy": False,
        "command_classification": False,
        "port_parity_manifest": False,
        "instrumentation": False,
    },
}
DEFAULT_PROFILE_NAME = "codex_full"


def resolve_features(profile=None):
    if isinstance(profile, FeatureSet):
        return profile
    features = FeatureSet.from_globals()
    profile_name = profile or os.getenv("CODEX_HARNESS_PROFILE") or DEFAULT_PROFILE_NAME
    if profile_name not in PROFILE_OVERRIDES:
        expected = ", ".join(sorted(PROFILE_OVERRIDES))
        raise ValueError(
            f"unknown CODEX_HARNESS_PROFILE {profile_name!r}; expected one of: {expected}"
        )
    return features.with_overrides(PROFILE_OVERRIDES.get(profile_name, {}))


def _current_date():
    return os.getenv("CODEX_CURRENT_DATE") or date.today().isoformat()


def _construct(symbol_name, *args):
    return globals()[symbol_name](*args)


def _local_timezone_name():
    if timezone := os.getenv("TZ"):
        return timezone
    try:
        with open("/etc/timezone", encoding="utf-8") as timezone_file:
            timezone = timezone_file.read().strip()
        if timezone:
            return timezone
    except OSError:
        pass
    local_name = time.tzname[0] if time.tzname else ""
    if local_name in {"UTC", "GMT"}:
        return "Etc/UTC"
    return local_name or "Etc/UTC"


@dataclass(frozen=True)
class TurnEnvironment:
    cwd: str = "."
    shell: str = "bash"
    current_date: str = field(default_factory=_current_date)
    timezone: str = field(default_factory=_local_timezone_name)
    approval_policy: str = "never"
    sandbox_mode: str = "danger-full-access"
    network_access: str = "enabled"


@dataclass(frozen=True)
class TurnContext:
    cwd: str = "."
    supports_parallel_tool_calls: bool = True
    personality: str | None = None
    output_schema: dict[str, Any] | None = None
    environment: TurnEnvironment = field(default_factory=TurnEnvironment)


@dataclass(frozen=True)
class Prompt:
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool
    base_instructions: str
    developer_messages: list[dict[str, Any]] = field(default_factory=list)
    personality: str | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_strict: bool = True

    def messages(self):
        return [
            {"role": "system", "content": self.base_instructions},
            *self.developer_messages,
            *self.input,
        ]


ToolPayload = namedtuple(
    "ToolPayload",
    "name arguments input call_id payload_type",
    defaults=({}, "", "", "function"),
)
ToolCall = namedtuple("ToolCall", "tool_name call_id payload")
ContextStats = namedtuple(
    "ContextStats",
    "raw_items normalized_items pruned_items estimated_bytes estimated_tokens "
    "compacted compaction_summary_chars compaction_reused",
    defaults=(False, 0, False),
)
CodexPromptBundle = namedtuple("CodexPromptBundle", "messages input_items tools stats")


@dataclass(frozen=True)
class CommandAssessment:
    kind: str
    risky: bool = False
    long_running: bool = False
    needs_verification: bool = False
    notes: tuple[str, ...] = ()


CompactionCheckpoint = namedtuple(
    "CompactionCheckpoint",
    "prefix_digest prefix_len replacement summary_text",
)


_TOOL_SPECS = json.loads(
    _z(
        "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",
        "Yr0/CoIO0WzeanM1Th538A6iL0BA==",
    )
)


def _tool_spec(name):
    return json.loads(json.dumps(_TOOL_SPECS[name]))


_exec_command_tool = partial(_tool_spec, "exec_command")
_write_stdin_tool = partial(_tool_spec, "write_stdin")
_update_plan_tool = partial(_tool_spec, "update_plan")
_apply_patch_tool = partial(_tool_spec, "apply_patch")


class StableJson:
    @staticmethod
    def dumps(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def object_or_empty(value):
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}


class TextBudget:
    @staticmethod
    def approx_token_count(text):
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

    @staticmethod
    def clip_tail(text, limit, marker="omitted"):
        text = text or ""
        if limit < 0 or len(text) <= limit:
            return text
        return f"<{marker} {len(text) - limit} chars>\n{text[-limit:]}"

    @staticmethod
    def clip_middle(text, limit):
        text = text or ""
        if limit < 0 or len(text) <= limit:
            return text
        head = max(0, limit // 3)
        tail = max(0, limit - head)
        omitted = len(text) - head - tail
        return f"{text[:head]}\n<omitted {omitted} chars>\n{text[-tail:]}"

    @staticmethod
    def item_bytes(item):
        return len(StableJson.dumps(item).encode("utf-8", errors="replace"))


def _built_tools(features=None):
    features = features or FeatureSet.from_globals()
    tools = [_exec_command_tool()]
    if features.write_stdin_tool:
        tools.append(_construct("_write_stdin_tool"))
    if features.plan_tool:
        tools.append(_construct("_update_plan_tool"))
    if features.patch_tool:
        tools.append(_construct("_apply_patch_tool"))
    return tools


class ToolRouter:
    def __init__(self, tools):
        self._tools = tools
        self._tool_names = {str(tool.get("name", "")) for tool in tools}

    def model_visible_specs(self):
        return [dict(tool) for tool in self._tools]

    def build_tool_call(self, item):
        item_type = str(item.get("type") or "")
        if item_type == "function_call":
            name = str(item.get("name") or "")
            namespace = item.get("namespace")
            if namespace:
                name = f"{namespace}.{name}"
            arguments = self._arguments_from_item(item)
            payload = self._payload_from_model_call(
                name, arguments, str(item.get("arguments") or "")
            )
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        if item_type == "local_shell_call":
            arguments = self._local_shell_arguments(item)
            payload = self._payload_from_model_call(
                "local_shell", arguments, StableJson.dumps(arguments)
            )
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        if item_type == "custom_tool_call":
            name = str(item.get("name") or "")
            raw_input = item.get("input", item.get("arguments", ""))
            arguments = {"input": raw_input} if not isinstance(raw_input, dict) else dict(raw_input)
            payload = self._payload_from_model_call(name, arguments, str(raw_input or ""))
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        return None

    def tool_calls_from_result(self, result):
        calls: list[HarnessToolCall] = []
        seen: set[tuple[str, str]] = set()
        for call in result.tool_calls:
            payload = self._payload_from_model_call(call.name, call.arguments, call.arguments_text)
            if payload is None:
                continue
            key = (call.call_id, payload.name)
            if call.call_id and key in seen:
                continue
            seen.add(key)
            calls.append(HarnessToolCall(payload.name, payload.arguments, call.call_id))
        if calls:
            return calls
        for item in result.response_items:
            tool_call = self.build_tool_call(item)
            if tool_call is None:
                continue
            key = (tool_call.call_id, tool_call.tool_name)
            if tool_call.call_id and key in seen:
                continue
            seen.add(key)
            calls.append(
                HarnessToolCall(
                    tool_call.payload.name, tool_call.payload.arguments, tool_call.call_id
                )
            )
        return calls

    def _payload_from_model_call(self, name, arguments, arguments_text):
        plain_name = name.rsplit(".", 1)[-1]
        if plain_name == "apply_patch":
            patch = self._patch_input(arguments, arguments_text)
            return ToolPayload("apply_patch", {"patch": patch}, patch, payload_type="custom")
        if plain_name == "write_stdin":
            return ToolPayload("write_stdin", self._write_stdin_arguments(arguments))
        if plain_name in {"update_plan", "plan"}:
            return ToolPayload("update_plan", self._plan_arguments(arguments))
        if plain_name in SHELL_TOOL_NAMES:
            return ToolPayload("exec_command", self._exec_arguments(arguments))
        if plain_name in self._tool_names:
            return ToolPayload(plain_name, dict(arguments))
        return ToolPayload(plain_name or name, dict(arguments))

    def _arguments_from_item(self, item):
        raw = item.get("arguments", {})
        return StableJson.object_or_empty(raw)

    def _local_shell_arguments(self, item):
        args = StableJson.object_or_empty(item.get("action"))
        for key in ("command", "cmd", "timeout_sec", "timeout_ms", "duration", "workdir"):
            if key in item and key not in args:
                args[key] = item[key]
        return args

    def _exec_arguments(self, arguments):
        args = dict(arguments)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        if "working_directory" in args and "workdir" not in args:
            args["workdir"] = args.pop("working_directory")
        command = args.get("cmd")
        if isinstance(command, list):
            args["cmd"] = self._join_argv(command)
        if "cmd" not in args and "input" in args:
            args["cmd"] = str(args["input"])
        if isinstance(args.get("cmd"), str):
            args["cmd"] = args["cmd"].replace("find ..", "find .")
        return args

    def _write_stdin_arguments(self, arguments):
        args = dict(arguments)
        if "process_id" in args and "session_id" not in args:
            args["session_id"] = args.pop("process_id")
        args.setdefault("chars", "")
        return args

    def _plan_arguments(self, arguments):
        args = dict(arguments)
        if "plan" not in args:
            args["plan"] = []
        return args

    def _patch_input(self, arguments, arguments_text):
        for key in ("input", "patch", "diff", "command"):
            if key in arguments:
                return str(arguments[key])
        return arguments_text

    def _join_argv(self, argv):
        return " ".join((_shell_quote(str(item)) for item in argv))


def _shell_quote(value):
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-./:=,@%"
    if all((ch in safe for ch in value)):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


class ToolOutputFormatter:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def tool_output_item(self, record, call_id):
        output = self.tool_output_text(record)
        if self._is_custom_output(record):
            return {"type": "custom_tool_call_output", "call_id": call_id, "output": output}
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    def tool_output_text(self, record):
        if self.features.unified_exec_output_format and self._has_unified_exec(record):
            return self.unified_exec_text(record)
        return self.generic_function_text(record)

    def unified_exec_text(self, record):
        metadata = self._unified_exec(record)
        output = self._combined_output(record)
        sections = []
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            sections.append(f"Chunk ID: {chunk_id}")
        wall_time = _float_or_zero(metadata.get("wall_time_seconds"))
        sections.append(f"Wall time: {wall_time:.4f} seconds")
        exit_code = metadata.get("exit_code", record.return_code)
        if exit_code is not None:
            sections.append(f"Process exited with code {exit_code}")
        session_id = metadata.get("session_id")
        if session_id is not None:
            sections.append(f"Process running with session ID {session_id}")
        original_token_count = metadata.get("original_token_count")
        if original_token_count is not None:
            sections.append(f"Original token count: {original_token_count}")
        sections.append("Output:")
        sections.append(TextBudget.clip_tail(output, self._max_tokens_to_chars(record)))
        return "\n".join(sections)

    def generic_function_text(self, record):
        sections = ["Wall time: 0.0000 seconds"]
        if record.return_code is not None:
            sections.append(f"Process exited with code {record.return_code}")
        sections.append("Output:")
        sections.append(TextBudget.clip_tail(self._combined_output(record), MAX_OBSERVATION_CHARS))
        return "\n".join(sections)

    def failure_response_item(self, call_id, payload, message):
        item_type = (
            "custom_tool_call_output"
            if payload.payload_type == "custom"
            else "function_call_output"
        )
        return {"type": item_type, "call_id": call_id, "output": message}

    def _is_custom_output(self, record):
        return record.tool_name == "apply_patch" or record.tool_name in CUSTOM_CALL_TYPES

    def _has_unified_exec(self, record):
        return isinstance(record.metadata, dict) and isinstance(
            record.metadata.get("unified_exec"), dict
        )

    def _unified_exec(self, record):
        if isinstance(record.metadata, dict) and isinstance(
            record.metadata.get("unified_exec"), dict
        ):
            return dict(record.metadata["unified_exec"])
        return {}

    def _combined_output(self, record):
        stdout = record.stdout or ""
        stderr = record.stderr or ""
        if stderr:
            return f"{stdout}\nSTDERR:\n{stderr}".strip()
        return stdout

    def _max_tokens_to_chars(self, record):
        arguments = record.metadata.get("arguments") if isinstance(record.metadata, dict) else None
        if isinstance(arguments, dict):
            max_tokens = arguments.get("max_output_tokens")
            if isinstance(max_tokens, (int, float)) and max_tokens > 0:
                return max(MAX_OBSERVATION_CHARS // 4, int(max_tokens) * 4)
        return MAX_OBSERVATION_CHARS


def _float_or_zero(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class ResponseItemFactory:
    def function_call(self, call_id, name, arguments):
        args_text = (
            arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True)
        )
        return {"type": "function_call", "call_id": call_id, "name": name, "arguments": args_text}

    def custom_tool_call(self, call_id, name, input_text):
        return {"type": "custom_tool_call", "call_id": call_id, "name": name, "input": input_text}

    def assistant_message(self, content, phase=None):
        item: dict[str, Any] = {"role": "assistant", "content": content}
        if phase is not None:
            item["phase"] = phase
        return item

    def responses_message(self, role, text):
        item_type = "output_text" if role == "assistant" else "input_text"
        return {"type": "message", "role": role, "content": [{"type": item_type, "text": text}]}


class HistoryReplay:
    def __init__(self, formatter, features=None):
        self.formatter = formatter
        self.features = features or FeatureSet.from_globals()
        self.items = ResponseItemFactory()

    def input_items(self, task, history):
        items = [{"role": "user", "content": InitialContextBuilder().render(task)}]
        if not self.features.history_replay:
            return items
        for index, record in enumerate(history, start=1):
            items.extend(self.record_items(index, record))
        return items

    def record_items(self, index, record):
        call_id = record.tool_call_id or f"call_{index}"
        raw_items = self.raw_codex_response_items(record)
        if raw_items is not None:
            return [*raw_items, self.formatter.tool_output_item(record, call_id)]
        if self.is_output_only(record):
            return [self.formatter.tool_output_item(record, call_id)]
        items = self.assistant_history(record)
        items.extend(self.synthetic_tool_pair(record, call_id))
        return items

    def raw_codex_response_items(self, record):
        if not self.features.model_response_item_replay:
            return None
        if not isinstance(record.metadata, dict):
            return None
        raw_items = record.metadata.get("codex_response_items")
        if not isinstance(raw_items, list) or not raw_items:
            return None
        kept = []
        for item in raw_items[:MAX_RAW_RESPONSE_ITEMS]:
            if isinstance(item, dict):
                kept.append(self.sanitize_response_item(item))
        return kept or None

    def is_output_only(self, record):
        return isinstance(record.metadata, dict) and bool(record.metadata.get("codex_output_only"))

    def assistant_history(self, record):
        if not isinstance(record.metadata, dict):
            return []
        content = str(record.metadata.get("assistant_content") or "").strip()
        if not content:
            return []
        return [self.items.assistant_message(content)]

    def synthetic_tool_pair(self, record, call_id):
        if record.tool_name == "apply_patch":
            patch = self.patch_from_record(record)
            return [
                self.items.custom_tool_call(call_id, "apply_patch", patch),
                self.formatter.tool_output_item(record, call_id),
            ]
        tool_name = (
            record.tool_name if record.tool_name in FUNCTION_HISTORY_TOOLS else "exec_command"
        )
        arguments = self.arguments_from_record(record)
        return [
            self.items.function_call(call_id, tool_name, arguments),
            self.formatter.tool_output_item(record, call_id),
        ]

    def arguments_from_record(self, record):
        args = record.metadata.get("arguments") if isinstance(record.metadata, dict) else None
        if not isinstance(args, dict):
            args = {"cmd": record.command}
        args = dict(args)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        return args

    def patch_from_record(self, record):
        if isinstance(record.metadata, dict):
            for key in ("input", "patch", "diff"):
                if key in record.metadata:
                    return str(record.metadata[key])
        marker = "apply_patch <<'PATCH'\n"
        if record.command.startswith(marker) and record.command.endswith("\nPATCH"):
            return record.command[len(marker) : -len("\nPATCH")]
        return record.command

    def sanitize_response_item(self, item):
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role", "assistant"))
            return {
                "type": "message",
                "role": role,
                "content": self._content_items(item.get("content", []), role),
            }
        if item_type == "function_call":
            cleaned = {
                "type": "function_call",
                "name": str(item.get("name", "")),
                "arguments": str(item.get("arguments", "")),
                "call_id": str(item.get("call_id", "")),
            }
            if item.get("namespace") is not None:
                cleaned["namespace"] = item["namespace"]
            return cleaned
        if item_type == "custom_tool_call":
            return {
                "type": "custom_tool_call",
                "name": str(item.get("name", "")),
                "input": str(item.get("input", "")),
                "call_id": str(item.get("call_id", "")),
            }
        if item_type in TOOL_OUTPUT_TYPES:
            cleaned = {"type": item_type, "call_id": str(item.get("call_id", ""))}
            if "output" in item:
                cleaned["output"] = item["output"]
            return cleaned
        cleaned = dict(item)
        cleaned.pop("id", None)
        return cleaned

    def _content_items(self, content, role):
        item_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            return [{"type": item_type, "text": content}]
        if not isinstance(content, list):
            return [{"type": item_type, "text": str(content)}]
        items = []
        for content_item in content:
            if isinstance(content_item, dict):
                cleaned = dict(content_item)
                cleaned.pop("id", None)
                cleaned.setdefault("type", item_type)
                if cleaned.get("type") in {"input_text", "output_text"}:
                    cleaned["text"] = str(cleaned.get("text", ""))
                items.append(cleaned)
            else:
                items.append({"type": item_type, "text": str(content_item)})
        return items


class ConversationNormalizer:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def normalize(self, items):
        if not self.features.context_normalization:
            return list(items)
        normalized = [dict(item) for item in items]
        self.ensure_call_outputs_present(normalized)
        self.remove_orphan_outputs(normalized)
        return normalized

    def ensure_call_outputs_present(self, items):
        insertions: list[tuple[int, dict[str, Any]]] = []
        output_call_ids = self.output_call_ids(items)
        for index, item in enumerate(items):
            call_id = self.call_id_for_call(item)
            if not call_id or call_id in output_call_ids:
                continue
            insertions.append((index + 1, self.synthetic_aborted_output(item, call_id)))
        for index, output in reversed(insertions):
            items.insert(index, output)

    def remove_orphan_outputs(self, items):
        call_ids = self.call_ids(items)
        kept = []
        for item in items:
            if item.get("type") in TOOL_OUTPUT_TYPES:
                call_id = str(item.get("call_id") or "")
                if call_id and call_id not in call_ids:
                    continue
            kept.append(item)
        items[:] = kept

    def call_ids(self, items):
        ids = set()
        for item in items:
            call_id = self.call_id_for_call(item)
            if call_id:
                ids.add(call_id)
        return ids

    def output_call_ids(self, items):
        ids = set()
        for item in items:
            if item.get("type") in TOOL_OUTPUT_TYPES:
                call_id = str(item.get("call_id") or "")
                if call_id:
                    ids.add(call_id)
        return ids

    def call_id_for_call(self, item):
        if item.get("type") in TOOL_CALL_TYPES:
            return str(item.get("call_id") or "")
        return ""

    def synthetic_aborted_output(self, call_item, call_id):
        if call_item.get("type") in CUSTOM_CALL_TYPES:
            return {"type": "custom_tool_call_output", "call_id": call_id, "output": "aborted"}
        return {"type": "function_call_output", "call_id": call_id, "output": "aborted"}


class ContextCompactor:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()
        self.checkpoint: CompactionCheckpoint | None = None

    def maybe_compact(self, items, initial_context=None):
        raw_items = self.copy_items(items)
        reused = False
        working = raw_items
        if self.features.model_context_compaction:
            applied = self.apply_checkpoint(raw_items)
            if applied is not None:
                working = applied
                reused = True
        if not self.features.model_context_compaction or not self.should_compact(working):
            return (working, self.checkpoint if reused else None, reused)
        compacted, summary_text = self.compact(working, initial_context or [])
        if compacted is None:
            return (working, self.checkpoint if reused else None, reused)
        checkpoint = CompactionCheckpoint(
            prefix_digest=self.digest(raw_items),
            prefix_len=len(raw_items),
            replacement=tuple(self.copy_items(compacted)),
            summary_text=summary_text,
        )
        self.checkpoint = checkpoint
        return (self.copy_items(compacted), checkpoint, False)

    def should_compact(self, items):
        return (
            len(items) > MAX_CONTEXT_HISTORY_ITEMS
            or sum((TextBudget.item_bytes(item) for item in items)) > MAX_CONTEXT_HISTORY_CHARS
        )

    def apply_checkpoint(self, items):
        checkpoint = self.checkpoint
        if checkpoint is None or len(items) < checkpoint.prefix_len:
            return None
        prefix = items[: checkpoint.prefix_len]
        if self.digest(prefix) != checkpoint.prefix_digest:
            return None
        suffix = items[checkpoint.prefix_len :]
        return [*self.copy_items(list(checkpoint.replacement)), *self.copy_items(suffix)]

    def compact(self, items, initial_context):
        compact_input = [
            {"role": "system", "content": CODEX_BASE_INSTRUCTIONS},
            *self.copy_items(items),
            ResponseItemFactory().responses_message("user", SUMMARIZATION_PROMPT),
        ]
        try:
            summary_suffix = call_terminal_model(compact_input)
        except Exception:
            return (None, "")
        summary_text = f"{SUMMARY_PREFIX}\n{summary_suffix}"
        user_messages = self.collect_user_messages(items, initial_context)
        return (
            self.build_compacted_history(initial_context, user_messages, summary_text),
            summary_text,
        )

    def collect_user_messages(self, items, initial_context):
        initial_digests = {self.digest([item]) for item in initial_context}
        messages: list[str] = []
        for item in items:
            if self.digest([item]) in initial_digests:
                continue
            if self.message_role(item) != "user":
                continue
            text = self.message_text(item)
            if text and (not self.is_summary_message(text)):
                messages.append(text)
        return messages

    def build_compacted_history(self, initial_context, user_messages, summary_text):
        history = self.copy_items(initial_context)
        for message in self.selected_user_messages(user_messages):
            history.append(ResponseItemFactory().responses_message("user", message))
        history.append(
            ResponseItemFactory().responses_message(
                "user", summary_text if summary_text else "(no summary available)"
            )
        )
        return history

    def selected_user_messages(self, user_messages):
        selected: list[str] = []
        remaining = COMPACT_USER_MESSAGE_MAX_TOKENS
        for message in reversed(user_messages):
            if remaining <= 0:
                break
            tokens = TextBudget.approx_token_count(message)
            if tokens <= remaining:
                selected.append(message)
                remaining -= tokens
            else:
                selected.append(TextBudget.clip_middle(message, remaining * 4))
                break
        selected.reverse()
        return selected

    def is_summary_message(self, message):
        return message.startswith(f"{SUMMARY_PREFIX}\n")

    def message_role(self, item):
        if item.get("type") == "message" or "role" in item:
            return str(item.get("role") or "")
        return ""

    def message_text(self, item):
        content = item.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        pieces = []
        for content_item in content:
            if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                pieces.append(content_item["text"])
        return "\n".join((piece for piece in pieces if piece))

    def digest(self, items):
        return hashlib.sha256(StableJson.dumps(items).encode("utf-8")).hexdigest()

    def copy_items(self, items):
        return [json.loads(json.dumps(item)) for item in items]


class ContextManager:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()
        self.normalizer = (
            _construct("ConversationNormalizer", self.features)
            if self.features.context_normalization
            else None
        )
        self.compactor = (
            _construct("ContextCompactor", self.features)
            if self.features.model_context_compaction
            else None
        )

    def prepare(self, items, initial_context=None):
        raw_count = len(items)
        normalized = self.normalizer.normalize(items) if self.normalizer else list(items)
        normalized_count = len(normalized)
        if self.compactor:
            normalized, checkpoint, reused = self.compactor.maybe_compact(
                normalized, initial_context or []
            )
        else:
            checkpoint = None
            reused = False
        budgeted = self.apply_budget(normalized)
        estimated_bytes = sum((TextBudget.item_bytes(item) for item in budgeted))
        return (
            budgeted,
            ContextStats(
                raw_items=raw_count,
                normalized_items=normalized_count,
                pruned_items=max(0, normalized_count - len(budgeted)),
                estimated_bytes=estimated_bytes,
                estimated_tokens=max(1, estimated_bytes // 4),
                compacted=checkpoint is not None,
                compaction_summary_chars=len(checkpoint.summary_text) if checkpoint else 0,
                compaction_reused=reused,
            ),
        )

    def apply_budget(self, items):
        if not self.features.context_budgeting:
            return list(items)
        clipped = [self.clip_item(item) for item in items]
        if len(clipped) > MAX_CONTEXT_HISTORY_ITEMS:
            clipped = self.drop_oldest_pairs(clipped, len(clipped) - MAX_CONTEXT_HISTORY_ITEMS)
        while sum((TextBudget.item_bytes(item) for item in clipped)) > MAX_CONTEXT_HISTORY_CHARS:
            if len(clipped) <= 2:
                break
            clipped = self.drop_oldest_pairs(clipped, 1)
        return clipped

    def clip_item(self, item):
        item = dict(item)
        if item.get("type") in TOOL_OUTPUT_TYPES and isinstance(item.get("output"), str):
            item["output"] = TextBudget.clip_tail(str(item["output"]), MAX_FUNCTION_OUTPUT_CHARS)
        if item.get("type") == "message":
            item["content"] = self.clip_content_items(item.get("content", []))
        elif "content" in item and isinstance(item["content"], str):
            item["content"] = TextBudget.clip_middle(
                str(item["content"]), MAX_FUNCTION_OUTPUT_CHARS
            )
        return item

    def clip_content_items(self, content):
        if not isinstance(content, list):
            return content
        clipped = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                updated = dict(item)
                updated["text"] = TextBudget.clip_middle(
                    str(updated["text"]), MAX_FUNCTION_OUTPUT_CHARS
                )
                clipped.append(updated)
            else:
                clipped.append(item)
        return clipped

    def drop_oldest_pairs(self, items, target_drop):
        if target_drop <= 0:
            return items
        kept = list(items)
        dropped = 0
        index = 1 if kept and kept[0].get("role") == "user" else 0
        while dropped < target_drop and index < len(kept):
            removed = kept.pop(index)
            dropped += 1
            call_id = str(removed.get("call_id") or "")
            if removed.get("type") in TOOL_CALL_TYPES and call_id:
                match = self.find_output_index(kept, call_id)
                if match is not None:
                    kept.pop(match)
                    dropped += 1
            elif removed.get("type") in TOOL_OUTPUT_TYPES and call_id:
                match = self.find_call_index(kept, call_id)
                if match is not None and match != 0:
                    kept.pop(match)
                    dropped += 1
        return kept

    def find_output_index(self, items, call_id):
        for index, item in enumerate(items):
            if item.get("type") in TOOL_OUTPUT_TYPES and item.get("call_id") == call_id:
                return index
        return None

    def find_call_index(self, items, call_id):
        for index, item in enumerate(items):
            if item.get("type") in TOOL_CALL_TYPES and item.get("call_id") == call_id:
                return index
        return None


class NoopContextManager:
    def prepare(self, items, initial_context=None):
        kept = [dict(item) for item in items]
        estimated_bytes = sum((TextBudget.item_bytes(item) for item in kept))
        return (
            kept,
            ContextStats(
                raw_items=len(items),
                normalized_items=len(items),
                pruned_items=0,
                estimated_bytes=estimated_bytes,
                estimated_tokens=max(1, estimated_bytes // 4),
            ),
        )


class PermissionsInstructionsRenderer:
    def messages(self, environment):
        return [{"role": "developer", "content": self.render(environment)}]

    def render(self, environment):
        sandbox_text = self.sandbox_text(environment)
        approval_text = self.approval_text(environment)
        return (
            f"<permissions instructions>\n"
            f"{sandbox_text}\n"
            f"{approval_text}\n"
            f"</permissions instructions>"
        )

    def sandbox_text(self, environment):
        if environment.sandbox_mode == "danger-full-access":
            return PERMISSIONS_SANDBOX_DANGER_FULL_ACCESS
        return (
            "Filesystem sandboxing defines which files can be read or written. "
            f"`sandbox_mode` is `{environment.sandbox_mode}`. "
            f"Network access is {environment.network_access}."
        )

    def approval_text(self, environment):
        if environment.approval_policy == "never":
            return PERMISSIONS_APPROVAL_NEVER
        return (
            "Approvals are your mechanism to get user consent to run shell commands without the "
            f"sandbox. `approval_policy` is `{environment.approval_policy}`."
        )


class InitialContextBuilder:
    def render(self, task):
        cwd = task.working_dir or "."
        environment = TurnEnvironment(cwd=cwd)
        sections = [self.environment_context(environment)]
        agents = AgentInstructionsRenderer().render(task)
        if agents:
            sections.append(agents)
        sections.append(str(task.instruction))
        return "\n\n".join((section for section in sections if section))

    def environment_context(self, environment):
        return (
            f"<environment_context>\n"
            f"  <cwd>{environment.cwd}</cwd>\n"
            f"  <shell>{environment.shell}</shell>\n"
            f"  <current_date>{environment.current_date}</current_date>\n"
            f"  <timezone>{environment.timezone}</timezone>\n"
            f"</environment_context>"
        )


class AgentInstructionsRenderer:
    def render(self, task):
        agents = task.metadata.get("agents_md") if isinstance(task.metadata, dict) else None
        if not isinstance(agents, list):
            return ""
        sections = []
        for item in self.sorted_agents(agents):
            path = str(item.get("path") or "AGENTS.md")
            content = str(item.get("content") or "").strip()
            if content:
                sections.append(f"<agents_md path={json.dumps(path)}>\n{content}\n</agents_md>")
        return "\n".join(sections)

    def sorted_agents(self, agents):
        clean = [dict(item) for item in agents if isinstance(item, dict)]
        return sorted(
            clean,
            key=lambda item: (str(item.get("path") or "").count("/"), str(item.get("path") or "")),
        )


class PromptBuilder:
    def __init__(self, router, features=None):
        self.features = features or FeatureSet.from_globals()
        self.router = router
        self.history = (
            _construct(
                "HistoryReplay", _construct("ToolOutputFormatter", self.features), self.features
            )
            if self.features.history_replay
            else None
        )
        self.context_manager = (
            _construct("ContextManager", self.features)
            if self.features.context_manager
            else NoopContextManager()
        )

    def build(self, task, history, context):
        raw_items = (
            self.history.input_items(task, history)
            if self.history
            else [{"role": "user", "content": InitialContextBuilder().render(task)}]
        )
        input_items, stats = self.context_manager.prepare(raw_items, raw_items[:1])
        prompt = build_prompt(input_items, self.router, context, CODEX_BASE_INSTRUCTIONS)
        return CodexPromptBundle(
            messages=prompt.messages(), input_items=input_items, tools=prompt.tools, stats=stats
        )


def build_prompt(input, router, turn_context, base_instructions):
    return Prompt(
        input=input,
        tools=router.model_visible_specs(),
        parallel_tool_calls=turn_context.supports_parallel_tool_calls,
        base_instructions=base_instructions,
        developer_messages=PermissionsInstructionsRenderer().messages(turn_context.environment),
        personality=turn_context.personality,
        output_schema=turn_context.output_schema,
        output_schema_strict=True,
    )


class CommandClassifier:
    def classify(self, arguments):
        command = str(arguments.get("cmd") or arguments.get("command") or "")
        lowered = command.lower()
        notes: list[str] = []
        if not command.strip():
            return CommandAssessment("empty", risky=False, notes=("empty command",))
        if self.is_destructive(lowered):
            notes.append("destructive filesystem or git operation")
            return CommandAssessment(
                "destructive", risky=True, needs_verification=True, notes=tuple(notes)
            )
        if self.is_package_install(lowered):
            return CommandAssessment("package_install", long_running=True, needs_verification=True)
        if self.is_test(lowered):
            return CommandAssessment("test", long_running=True)
        if self.is_build(lowered):
            return CommandAssessment("build", long_running=True, needs_verification=True)
        if self.is_server(lowered):
            return CommandAssessment("server", long_running=True)
        if self.is_git(lowered):
            return CommandAssessment(
                "git", risky="reset --hard" in lowered, needs_verification=True
            )
        if self.is_edit(lowered):
            return CommandAssessment("edit", needs_verification=True)
        return CommandAssessment("inspection")

    def is_destructive(self, command):
        patterns = (
            "rm -rf",
            "git reset --hard",
            "git checkout --",
            "mkfs",
            "dd if=",
            "truncate -s 0",
        )
        return any((pattern in command for pattern in patterns))

    def is_package_install(self, command):
        return any(
            (
                token in command
                for token in (
                    "pip install",
                    "npm install",
                    "pnpm install",
                    "yarn install",
                    "apt-get install",
                    "uv sync",
                )
            )
        )

    def is_test(self, command):
        return any(
            (token in command for token in ("pytest", "npm test", "cargo test", "go test", "tox"))
        )

    def is_build(self, command):
        return any(
            (
                token in command
                for token in ("npm run build", "cargo build", "make", "cmake", "go build")
            )
        )

    def is_server(self, command):
        return any(
            (
                token in command
                for token in (
                    "uvicorn",
                    "flask run",
                    "npm run dev",
                    "vite",
                    "python -m http.server",
                )
            )
        )

    def is_git(self, command):
        return command.strip().startswith("git ")

    def is_edit(self, command):
        return any(
            (token in command for token in ("apply_patch", "sed -i", "perl -pi", "python - <<"))
        )


class ExecutionPolicy:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def annotate_tool_calls(self, calls):
        return list(calls)

    def assessments(self, calls):
        if not self.features.command_classification:
            return []
        classifier = _construct("CommandClassifier")
        assessments = []
        for call in calls:
            if call.name != "exec_command":
                continue
            assessment = classifier.classify(call.arguments)
            assessments.append(
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "command": str(call.arguments.get("cmd") or ""),
                    "assessment": assessment.__dict__,
                }
            )
        return assessments


class ModelCallResilience:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def call(self, messages, tools):
        if not self.features.model_call_resilience:
            return call_terminal_model_with_tools(
                messages, tools, tool_choice="auto", parallel_tool_calls=True
            )
        try:
            return call_terminal_model_with_tools(
                messages, tools, tool_choice="auto", parallel_tool_calls=True
            )
        except Exception as exc:
            return ToolModelResult(
                content="",
                tool_calls=[],
                request_metadata={"model_call_error": str(exc)},
                response_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"Model call failed before tool selection: {exc}",
                            }
                        ],
                    }
                ],
            )


class RecoveryPolicy:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def fallback_turn(self, result, history, metadata):
        if not self.features.recovery_policy:
            return None
        if result.tool_calls or result.content.strip():
            return None
        if not history:
            metadata["codex_recovery"] = "empty_response_initial_reconnaissance"
            return HarnessTurn(
                tool_calls=(
                    HarnessToolCall(
                        "exec_command",
                        {
                            "cmd": "pwd && find . -maxdepth 2 -type f | sort | sed -n '1,200p'",
                            "yield_time_ms": 1000,
                            "max_output_tokens": 12000,
                        },
                        "recovery_initial_recon",
                    ),
                ),
                metadata=metadata,
            )
        metadata["codex_recovery"] = "empty_response_recent_status"
        return HarnessTurn(
            tool_calls=(
                HarnessToolCall(
                    "exec_command",
                    {
                        "cmd": (
                            "pwd && git status --short 2>/dev/null || true && find . -maxde"
                            "pth 2 -type f | sort | sed -n '1,120p'"
                        ),
                        "yield_time_ms": 1000,
                        "max_output_tokens": 12000,
                    },
                    "recovery_status",
                ),
            ),
            metadata=metadata,
        )


class NullRecoveryPolicy:
    def fallback_turn(self, result, history, metadata):
        return None


class CompletionPolicy:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def is_complete(self, result, tool_calls):
        if tool_calls:
            return False
        return bool(self.visible_text(result).strip())

    def visible_text(self, result):
        if not self.features.completion_policy:
            return result.content
        if result.content.strip():
            return result.content
        chunks: list[str] = []
        for item in result.response_items:
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        chunks.append(content_item["text"])
            elif isinstance(content, str):
                chunks.append(content)
        return "\n".join(chunks)


class Instrumentation:
    def __init__(self, features=None):
        self.features = features or FeatureSet.from_globals()

    def turn_metadata(self, result, bundle, tool_calls, assessments=None):
        metadata: dict[str, Any] = {
            "codex_upstream_commit": CODEX_UPSTREAM_COMMIT,
            "codex_upstream_date": CODEX_UPSTREAM_DATE,
            "codex_port_stats": bundle.stats._asdict(),
            "codex_tool_count": len(bundle.tools),
            "codex_tool_names": [tool.get("name") for tool in bundle.tools],
            "codex_emitted_tool_calls": len(tool_calls),
        }
        if self.features.port_parity_manifest:
            metadata["codex_port_manifest"] = PORT_PARITY_MANIFEST
        if result.request_metadata:
            metadata["codex_request_metadata"] = result.request_metadata
        if result.response_items:
            metadata["codex_response_items"] = result.response_items
        if result.response_id:
            metadata["codex_response_id"] = result.response_id
        if assessments:
            metadata["codex_command_assessments"] = assessments
        return metadata


class NullInstrumentation:
    def turn_metadata(self, result, bundle, tool_calls, assessments=None):
        return {}


class CandidateHarness(BaseHarness):
    wants_environment_context = True
    wants_agents_context = True

    def __init__(self, profile=None):
        self.features = resolve_features(profile)
        self.router = ToolRouter(_built_tools(self.features))
        self.context = TurnContext()
        self.prompt_builder = PromptBuilder(self.router, self.features)
        self.model = ModelCallResilience(self.features)
        self.completion = CompletionPolicy(self.features)
        self.recovery = (
            _construct("RecoveryPolicy", self.features)
            if self.features.recovery_policy
            else NullRecoveryPolicy()
        )
        self.execution_policy = ExecutionPolicy(self.features)
        self.instrumentation = (
            _construct("Instrumentation", self.features)
            if self.features.instrumentation
            else NullInstrumentation()
        )

    def next_command(self, task, history):
        bundle = self.prompt_builder.build(task, history, self._turn_context_for_task(task))
        result = self.model.call(bundle.messages, bundle.tools)
        tool_calls = self.router.tool_calls_from_result(result)
        tool_calls = self.execution_policy.annotate_tool_calls(tool_calls)
        assessments = self.execution_policy.assessments(tool_calls)
        metadata = self.instrumentation.turn_metadata(result, bundle, tool_calls, assessments)
        recovery = self.recovery.fallback_turn(result, history, metadata)
        if recovery is not None:
            return recovery
        if tool_calls:
            return HarnessTurn(
                tool_calls=tuple(tool_calls),
                assistant_content=self.completion.visible_text(result),
                metadata=metadata,
            )
        return HarnessTurn(
            done=self.completion.is_complete(result, tool_calls),
            assistant_content=self.completion.visible_text(result),
            metadata=metadata,
        )

    def _turn_context_for_task(self, task):
        cwd = task.working_dir or "."
        environment = TurnEnvironment(cwd=cwd)
        return TurnContext(cwd=cwd, environment=environment)


create_agent = CandidateHarness
