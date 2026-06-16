from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CODEX_BASE_INSTRUCTIONS = """You are a coding agent working in a terminal.

Use the available tools to solve the user's task. Inspect the files, run commands, edit files, and verify your work as needed.

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. Do NOT guess or make up an answer.

You MUST adhere to the following criteria when solving queries:

- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the `apply_patch` tool to edit files (NEVER try `applypatch` or `apply-patch`, only `apply_patch`): {"command":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n- pass\\n+ return 123\\n*** End Patch"]}

When you are done, respond with the final answer and do not call more tools.
"""

SUMMARIZATION_PROMPT = """Summarize the current task for the next model.

Include progress, decisions, constraints, important files/data, and remaining steps.
Be concise."""
SUMMARY_PREFIX = "Continuation summary from a prior model:"

_ROOT = Path(__file__).resolve().parents[1]
_BASE_HARNESS_PATH = _ROOT / "seeds" / "codex_full" / "harness.py"
_BASE_MODULE_NAME = "_codex_full_minimal_surfaces_base"


def _load_base_module() -> Any:
    sys.modules.pop(_BASE_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load Codex full harness from {_BASE_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _install_minimal_surfaces(module)
    return module


def _install_minimal_surfaces(module: Any) -> None:
    module.CODEX_BASE_INSTRUCTIONS = CODEX_BASE_INSTRUCTIONS
    module.SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT
    module.SUMMARY_PREFIX = SUMMARY_PREFIX

    class MinimalPermissionsInstructionsRenderer:
        def messages(self, environment: Any) -> list[dict[str, Any]]:
            return []

        def render(self, environment: Any) -> str:
            return ""

    class MinimalInitialContextBuilder(module.InitialContextBuilder):
        def environment_context(self, environment: Any) -> str:
            return (
                f'<env cwd="{environment.cwd}" shell="{environment.shell}" '
                f'date="{environment.current_date}" tz="{environment.timezone}" />'
            )

    class MinimalToolOutputFormatter(module.ToolOutputFormatter):
        def unified_exec_text(self, record: Any) -> str:
            metadata = self._unified_exec(record)
            output = self._combined_output(record)
            chunk_id = metadata.get("chunk_id")
            wall_time = module._float_or_zero(metadata.get("wall_time_seconds"))
            exit_code = metadata.get("exit_code", record.return_code)
            session_id = metadata.get("session_id")
            original_token_count = metadata.get("original_token_count")
            status = (
                f"chunk={_compact_value(chunk_id)} "
                f"wall={wall_time:.4f}s "
                f"exit={_compact_value(exit_code)} "
                f"session={_compact_value(session_id)} "
                f"original_tokens={_compact_value(original_token_count)}"
            )
            return "\n".join(
                (
                    status,
                    "output:",
                    module.TextBudget.clip_tail(output, self._max_tokens_to_chars(record)),
                )
            )

    module.PermissionsInstructionsRenderer = MinimalPermissionsInstructionsRenderer
    module.InitialContextBuilder = MinimalInitialContextBuilder
    module.ToolOutputFormatter = MinimalToolOutputFormatter
    module._unified_exec_output_schema = _minimal_unified_exec_output_schema
    module._exec_command_tool = _minimal_exec_command_tool
    module._write_stdin_tool = _minimal_write_stdin_tool
    module._update_plan_tool = _minimal_update_plan_tool
    module._apply_patch_tool = lambda: _minimal_apply_patch_tool(module)


def _compact_value(value: Any) -> str:
    return "null" if value is None else str(value)


def _minimal_string(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if description:
        schema["description"] = description
    return schema


def _minimal_number(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if description:
        schema["description"] = description
    return schema


def _minimal_boolean(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean"}
    if description:
        schema["description"] = description
    return schema


def _minimal_object(
    properties: dict[str, Any],
    required: list[str],
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": additional_properties,
    }


def _minimal_array(items: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": items}
    if description:
        schema["description"] = description
    return schema


def _minimal_unified_exec_output_schema() -> dict[str, Any]:
    return _minimal_object(
        {
            "chunk_id": _minimal_string("chunk id"),
            "wall_time_seconds": _minimal_number("seconds"),
            "exit_code": _minimal_number("exit code"),
            "session_id": _minimal_number("running session id"),
            "original_token_count": _minimal_number("pre-truncation tokens"),
            "output": _minimal_string("stdout/stderr text"),
        },
        ["wall_time_seconds", "output"],
    )


def _minimal_exec_command_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "exec_command",
        "description": "Run shell command; return output or session_id.",
        "strict": False,
        "parameters": _minimal_object(
            {
                "cmd": _minimal_string("command"),
                "workdir": _minimal_string("working directory"),
                "tty": _minimal_boolean("allocate PTY"),
                "yield_time_ms": _minimal_number("wait milliseconds"),
                "max_output_tokens": _minimal_number("output token cap"),
            },
            ["cmd"],
        ),
        "output_schema": _minimal_unified_exec_output_schema(),
    }


def _minimal_write_stdin_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "write_stdin",
        "description": "Send stdin or poll a running session.",
        "strict": False,
        "parameters": _minimal_object(
            {
                "session_id": _minimal_number("session id"),
                "chars": _minimal_string("stdin bytes; empty polls"),
                "yield_time_ms": _minimal_number("wait milliseconds"),
                "max_output_tokens": _minimal_number("output token cap"),
            },
            ["session_id"],
        ),
        "output_schema": _minimal_unified_exec_output_schema(),
    }


def _minimal_update_plan_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_plan",
        "description": "Update task plan.",
        "strict": False,
        "parameters": _minimal_object(
            {
                "explanation": _minimal_string(),
                "plan": _minimal_array(
                    _minimal_object(
                        {
                            "step": _minimal_string(),
                            "status": _minimal_string("pending, in_progress, or completed"),
                        },
                        ["step", "status"],
                    )
                ),
            },
            ["plan"],
        ),
    }


def _minimal_apply_patch_tool(module: Any) -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a Begin/End Patch diff.",
        "format": {"type": "grammar", "syntax": "lark", "definition": module.APPLY_PATCH_GRAMMAR},
    }


_BASE = _load_base_module()
CandidateHarness = _BASE.CandidateHarness


def create_agent() -> Any:
    _install_minimal_surfaces(_BASE)
    return _BASE.create_agent()
