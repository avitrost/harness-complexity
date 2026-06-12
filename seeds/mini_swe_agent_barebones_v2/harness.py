from __future__ import annotations

import json
import platform
import shlex
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ModelToolCall, ToolModelResult, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

import os

UPSTREAM_COMMIT = "2afd0fb81bacbf0aacfac9ded6f093c5acd0bf7c"
FORMAT_RETRY_LIMIT = int(os.getenv("MINI_SWE_FORMAT_RETRY_LIMIT", "3"))
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
DEFAULT_COMMAND_TIMEOUT_SEC = 30

MINI_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.
"""

MINI_INSTANCE_TEMPLATE = """Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

**CRITICAL REQUIREMENTS:**

- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}
"""

MINI_OBSERVATION_TEMPLATE = """{%- if output.output | length < 10000 -%}
{
  "returncode": {{ output.returncode }},
  "output": {{ output.output | tojson }}
  {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
}
{%- else -%}
{
  "returncode": {{ output.returncode }},
  "output_head": {{ output.output[:5000] | tojson }},
  "output_tail": {{ output.output[-5000:] | tojson }},
  "elided_chars": {{ output.output | length - 10000 }},
  "warning": "Output too long."
  {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
}
{%- endif -%}
"""

MINI_FORMAT_ERROR_TEMPLATE = """Tool call error:

<error>
{{error}}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
"""

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}

ENVIRONMENT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


class MiniSweAgentV2Harness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        submission = _submission_from_history(history)
        if submission is not None:
            return HarnessTurn(
                done=True,
                assistant_content=submission,
                metadata=_base_metadata(exit_status="Submitted", submission=submission),
            )

        messages = _initial_messages(task)
        messages.extend(_history_messages(history))
        last_result: ToolModelResult | None = None
        turn_messages: list[dict[str, Any]] = []
        for retry_index in range(FORMAT_RETRY_LIMIT + 1):
            result = call_terminal_model_with_tools(messages, [BASH_TOOL])
            last_result = result
            actions, error = _parse_actions(result.tool_calls)
            if actions:
                return HarnessTurn(
                    tool_calls=tuple(actions),
                    assistant_content=result.content,
                    metadata=_turn_metadata(result, retry_index, turn_messages),
                )
            retry_messages = [*_assistant_items(result), _format_error_message(error)]
            messages.extend(retry_messages)
            turn_messages.extend(retry_messages)

        content = last_result.content if last_result is not None else ""
        return HarnessTurn(
            done=True,
            assistant_content=content,
            metadata=_base_metadata(exit_status="FormatError", unresolved_format_error=True),
        )


def create_agent() -> MiniSweAgentV2Harness:
    return MiniSweAgentV2Harness()


def _initial_messages(task: TaskContext) -> list[dict[str, Any]]:
    vars_ = _template_vars(task)
    return [
        {"role": "system", "content": _render(MINI_SYSTEM_TEMPLATE, vars_)},
        {"role": "user", "content": _render(MINI_INSTANCE_TEMPLATE, vars_)},
    ]


def _template_vars(task: TaskContext) -> dict[str, Any]:
    uname = platform.uname()
    return {
        **ENVIRONMENT_ENV,
        "task": task.instruction,
        "cwd": task.working_dir or "",
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        **{str(key): value for key, value in task.metadata.items()},
    }


def _history_messages(history: list[CommandResult]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history:
        metadata = item.metadata or {}
        turn_messages = metadata.get("mini_swe_agent_v2_messages")
        if isinstance(turn_messages, list):
            messages.extend(_sanitize_response_items(turn_messages))
        else:
            response_items = metadata.get("mini_swe_agent_v2_response_items")
            if isinstance(response_items, list):
                messages.extend(_sanitize_response_items(response_items))
        messages.append(_observation_message(item))
    return messages


def _observation_message(result: CommandResult) -> dict[str, Any]:
    call_id = result.tool_call_id or f"call_{abs(hash(result.command))}"
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": _render(MINI_OBSERVATION_TEMPLATE, {"output": _output_record(result)}),
    }


def _output_record(result: CommandResult) -> dict[str, Any]:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    output = stdout if not stderr else f"{stdout}{stderr}"
    exception_info = ""
    if result.return_code is None:
        exception_info = "command did not return an exit code"
    return {
        "output": output,
        "returncode": -1 if result.return_code is None else result.return_code,
        "exception_info": exception_info,
    }


def _submission_from_history(history: list[CommandResult]) -> str | None:
    if not history:
        return None
    last = history[-1]
    if last.return_code == 0 and last.command.strip() == SUBMIT_COMMAND:
        output = (last.stdout or "") + (last.stderr or "")
        return _submission_from_output(output)
    output = (last.stdout or "") + (last.stderr or "")
    lines = output.lstrip().splitlines(keepends=True)
    if last.return_code == 0 and lines and lines[0].strip() == SUBMIT_MARKER:
        return "".join(lines[1:])
    return None


def _submission_from_output(output: str) -> str:
    lines = output.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SUBMIT_MARKER:
        return "".join(lines[1:])
    return ""


def _parse_actions(calls: list[ModelToolCall]) -> tuple[list[HarnessToolCall], str]:
    if not calls:
        return (
            [],
            "No tool calls found in the response. Every response MUST include at least one tool call.",
        )
    actions: list[HarnessToolCall] = []
    errors: list[str] = []
    for index, call in enumerate(calls):
        error = ""
        if call.arguments_text:
            try:
                parsed = json.loads(call.arguments_text)
            except Exception as exc:
                parsed = call.arguments
                error = f"Error parsing tool call arguments: {exc}."
        else:
            parsed = call.arguments
        if call.name != "bash":
            error += f"Unknown tool '{call.name}'."
        if not isinstance(parsed, dict) or "command" not in parsed:
            error += "Missing 'command' argument in bash tool call."
        if error:
            errors.append(error.strip())
            continue
        actions.append(
            HarnessToolCall(
                "bash",
                {
                    "command": _runtime_command(str(parsed["command"])),
                    "timeout_sec": DEFAULT_COMMAND_TIMEOUT_SEC,
                },
                _call_id(call, index),
            )
        )
    if errors:
        return [], " ".join(errors)
    return actions, ""


def _assistant_items(result: ToolModelResult) -> list[dict[str, Any]]:
    items = _sanitize_response_items(result.response_items)
    if items:
        return _fill_missing_tool_call_ids(items, result.tool_calls)
    assistant: list[dict[str, Any]] = []
    if result.content:
        assistant.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.content}],
            }
        )
    for index, call in enumerate(result.tool_calls):
        assistant.append(
            {
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments_text or json.dumps(call.arguments, sort_keys=True),
                "call_id": _call_id(call, index),
            }
        )
    return assistant


def _fill_missing_tool_call_ids(
    items: list[dict[str, Any]],
    calls: list[ModelToolCall],
) -> list[dict[str, Any]]:
    fixed: list[dict[str, Any]] = []
    call_index = 0
    for item in items:
        current = dict(item)
        if current.get("type") == "function_call":
            call = calls[call_index] if call_index < len(calls) else None
            if not current.get("call_id") and call is not None:
                current["call_id"] = _call_id(call, call_index)
            call_index += 1
        fixed.append(current)
    return fixed


def _call_id(call: ModelToolCall, index: int) -> str:
    return call.call_id or f"call_mini_{index + 1}"


def _runtime_command(command: str) -> str:
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in ENVIRONMENT_ENV.items())
    return f"export {assignments};\n{command}"


def _format_error_message(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": _render(MINI_FORMAT_ERROR_TEMPLATE, {"error": error, "actions": []}),
    }


def _turn_metadata(
    result: ToolModelResult,
    retry_index: int,
    previous_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = _base_metadata(mini_swe_agent_v2_format_retries=retry_index)
    response_items = _assistant_items(result)
    metadata["mini_swe_agent_v2_response_items"] = response_items
    metadata["mini_swe_agent_v2_messages"] = [*previous_messages, *response_items]
    metadata["mini_swe_agent_v2_usage"] = result.usage
    metadata["mini_swe_agent_v2_request_metadata"] = result.request_metadata
    metadata["mini_swe_agent_v2_response_id"] = result.response_id
    return metadata


def _base_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "sequential_tool_calls": True,
        "mini_swe_agent_v2_upstream_commit": UPSTREAM_COMMIT,
        "mini_swe_agent_v2_config": {
            "agent": {"step_limit": 0, "cost_limit": 3.0, "mode": "confirm"},
            "environment": {"env": ENVIRONMENT_ENV},
            "model": {"model_kwargs": {"drop_params": True}},
        },
        "mini_swe_agent_v2_timestamp": time.time(),
    }
    metadata.update(extra)
    return metadata


def _sanitize_response_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [_sanitize_response_item(item) for item in items if isinstance(item, dict)]


def _sanitize_response_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "message":
        role = str(item.get("role", "assistant"))
        return {
            "type": "message",
            "role": role,
            "content": _content_items(item.get("content", []), role),
        }
    if item_type == "function_call":
        return {
            "type": "function_call",
            "name": str(item.get("name", "")),
            "arguments": str(item.get("arguments", "")),
            "call_id": str(item.get("call_id", "")),
        }
    if item_type in {"reasoning", "function_call_output", "custom_tool_call_output"}:
        return dict(item)
    return dict(item)


def _content_items(content: Any, role: str) -> list[dict[str, str]]:
    default_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": default_type, "text": content}]
    if isinstance(content, list):
        return [
            (
                {"type": str(item.get("type") or default_type), "text": str(item.get("text", ""))}
                if isinstance(item, dict)
                else {"type": default_type, "text": str(item)}
            )
            for item in content
        ]
    return [{"type": default_type, "text": str(content)}]


def _render(template: str, values: dict[str, Any]) -> str:
    return Template(template, undefined=StrictUndefined).render(**values)
