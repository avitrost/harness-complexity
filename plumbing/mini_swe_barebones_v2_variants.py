from __future__ import annotations

import json
import os
import platform
import shlex
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ModelToolCall, ToolModelResult, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

UPSTREAM_COMMIT = "2afd0fb81bacbf0aacfac9ded6f093c5acd0bf7c"
FORMAT_RETRY_LIMIT = int(os.getenv("MINI_SWE_FORMAT_RETRY_LIMIT", "3"))
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
DEFAULT_COMMAND_TIMEOUT_SEC = 30

MINI_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.
"""

MINI_INSTANCE_TEMPLATE = """Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

{{command_rules}}
"""

NONPERSISTENT_RULES = """## Command Execution Rules

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
- Arguments: {"command": "your_command_here"}"""

PERSISTENT_BASH_RULES = NONPERSISTENT_RULES.replace(
    "2. The system executes the command(s) in a subshell",
    "2. The system executes the command(s) in one persistent interactive shell",
).replace(
    "- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.",
    "- Directory, shell state, environment variable changes, and running processes persist across bash tool calls in one interactive shell.",
)

RICH_TERMINAL_RULES = """## Command Execution Rules

You are operating in an environment where

1. You issue at least one terminal tool call
2. The system executes the tool call(s)
3. You see the result(s)
4. You write your next tool call(s)

**CRITICAL REQUIREMENTS:**

- Your response MUST include AT LEAST ONE terminal tool call.
- Use `exec_command` to run shell commands.
- `exec_command` may return a `session_id` when a command is still running. Use `write_stdin` with that `session_id` to send input or poll for more output.
- A persistent tmux terminal is available as `{{tmux_session}}`. Use `write_stdin` with `tmux_session="{{tmux_session}}"` when you want directory changes, environment variables, running processes, or interactive terminal state to persist across turns.
- One-shot `exec_command` calls do not share shell state with later one-shot `exec_command` calls unless you explicitly use files or a persistent session.
- Submit your changes and finish your work by calling `exec_command` with exactly this command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>"""

RICH_TERMINAL_EXAMPLES = """Examples:

- Run a one-shot command with `exec_command`: `{"cmd": "pwd && ls -la"}`
- Run or poll a long-running command: first call `exec_command` with `{"cmd": "python3 script.py", "yield_time_ms": 1000}`; if the result includes a `session_id`, call `write_stdin` with `{"session_id": SESSION_ID, "chars": "", "yield_time_ms": 1000}` to poll for more output.
- Use the persistent tmux terminal: call `write_stdin` with `{"tmux_session": "{{tmux_session}}", "chars": "cd /app && export FOO=bar\\n", "yield_time_ms": 1000}`; later calls to the same `tmux_session` can rely on that terminal state."""

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

Every response needs to use the requested terminal tool at least once to execute commands.

{{tool_guidance}}

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

EXEC_COMMAND_TOOL = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a shell command, returning output and possibly a session_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to execute."},
            "command": {"type": "string", "description": "Alias for cmd."},
            "workdir": {"type": "string", "description": "Optional working directory."},
            "shell": {"type": "string", "description": "Optional shell binary."},
            "login": {"type": "boolean", "description": "Whether to use login shell semantics."},
            "tty": {"type": "boolean", "description": "Whether to allocate a TTY."},
            "yield_time_ms": {
                "type": "number",
                "description": "How long to wait for output before yielding.",
            },
            "max_output_tokens": {
                "type": "number",
                "description": "Maximum output tokens to return.",
            },
            "timeout_sec": {"type": "number", "description": "Command timeout in seconds."},
        },
    },
}

WRITE_STDIN_TOOL = {
    "type": "function",
    "name": "write_stdin",
    "description": "Write input to an existing session_id or tmux_session and return output.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "number", "description": "Unified exec session id."},
            "tmux_session": {"type": "string", "description": "Persistent tmux session name."},
            "chars": {"type": "string", "description": "Characters to send; empty string polls."},
            "yield_time_ms": {
                "type": "number",
                "description": "How long to wait for output before yielding.",
            },
            "max_output_tokens": {
                "type": "number",
                "description": "Maximum output tokens to return.",
            },
        },
    },
}

ENVIRONMENT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


class MiniSweBarebonesV2Variant(BaseHarness):
    def __init__(self, mode: str, include_examples: bool = False) -> None:
        self.mode = mode
        self.include_examples = include_examples
        if mode == "bash_persistent":
            self.wants_persistent_terminal = True
        elif mode == "rich_terminal":
            self.wants_persistent_terminal = "tmux"

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        submission = _submission_from_history(history)
        if submission is not None:
            return HarnessTurn(
                done=True,
                assistant_content=submission,
                metadata=_base_metadata(self.mode, exit_status="Submitted", submission=submission),
            )

        messages = _initial_messages(task, self.mode, self.include_examples)
        messages.extend(_history_messages(history))
        tools = _tools(self.mode)
        last_result: ToolModelResult | None = None
        turn_messages: list[dict[str, Any]] = []
        for retry_index in range(FORMAT_RETRY_LIMIT + 1):
            result = call_terminal_model_with_tools(messages, tools)
            last_result = result
            actions, error = _parse_actions(result.tool_calls, task, self.mode)
            if actions:
                return HarnessTurn(
                    tool_calls=tuple(actions),
                    assistant_content=result.content,
                    metadata=_turn_metadata(result, retry_index, turn_messages, self.mode),
                )
            retry_messages = [*_assistant_items(result), _format_error_message(error, self.mode)]
            messages.extend(retry_messages)
            turn_messages.extend(retry_messages)

        content = last_result.content if last_result is not None else ""
        return HarnessTurn(
            done=True,
            assistant_content=content,
            metadata=_base_metadata(
                self.mode,
                exit_status="FormatError",
                unresolved_format_error=True,
            ),
        )


def create_bash_persistent_agent() -> MiniSweBarebonesV2Variant:
    return MiniSweBarebonesV2Variant("bash_persistent")


def create_rich_terminal_agent() -> MiniSweBarebonesV2Variant:
    return MiniSweBarebonesV2Variant("rich_terminal", include_examples=True)


def create_rich_terminal_no_examples_agent() -> MiniSweBarebonesV2Variant:
    return MiniSweBarebonesV2Variant("rich_terminal", include_examples=False)


def _initial_messages(
    task: TaskContext,
    mode: str,
    include_examples: bool = False,
) -> list[dict[str, Any]]:
    vars_ = _template_vars(task)
    vars_["command_rules"] = _command_rules(mode, include_examples, vars_)
    return [
        {"role": "system", "content": _render(MINI_SYSTEM_TEMPLATE, vars_)},
        {"role": "user", "content": _render(MINI_INSTANCE_TEMPLATE, vars_)},
    ]


def _command_rules(mode: str, include_examples: bool, vars_: dict[str, Any]) -> str:
    if mode == "bash_persistent":
        return PERSISTENT_BASH_RULES
    rules = _render(RICH_TERMINAL_RULES, vars_)
    if include_examples:
        rules = f"{rules}\n\n{_render(RICH_TERMINAL_EXAMPLES, vars_)}"
    return rules


def _template_vars(task: TaskContext) -> dict[str, Any]:
    uname = platform.uname()
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    terminal = metadata.get("persistent_terminal") if isinstance(metadata, dict) else {}
    terminal = terminal if isinstance(terminal, dict) else {}
    return {
        **ENVIRONMENT_ENV,
        "task": task.instruction,
        "cwd": task.working_dir or "",
        "tmux_session": str(terminal.get("session_name") or ""),
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        **{str(key): value for key, value in metadata.items()},
    }


def _tools(mode: str) -> list[dict[str, Any]]:
    if mode == "bash_persistent":
        return [BASH_TOOL]
    return [EXEC_COMMAND_TOOL, WRITE_STDIN_TOOL]


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
    if last.return_code == 0 and _is_submit_command(last.command):
        output = (last.stdout or "") + (last.stderr or "")
        return "" if _output_contains_submit_marker(output) else None
    output = (last.stdout or "") + (last.stderr or "")
    lines = output.lstrip().splitlines(keepends=True)
    if last.return_code == 0 and lines and lines[0].strip() == SUBMIT_MARKER:
        return "".join(lines[1:])
    return None


def _is_submit_command(command: str) -> bool:
    return any(line.strip() == SUBMIT_COMMAND for line in command.splitlines())


def _submission_from_output(output: str) -> str:
    lines = output.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == SUBMIT_MARKER:
            return "".join(lines[index + 1 :])
    return ""


def _output_contains_submit_marker(output: str) -> bool:
    return any(line.strip() == SUBMIT_MARKER for line in output.splitlines())


def _parse_actions(
    calls: list[ModelToolCall],
    task: TaskContext,
    mode: str,
) -> tuple[list[HarnessToolCall], str]:
    if not calls:
        return [], "No tool calls found in the response. Every response MUST include a tool call."
    actions: list[HarnessToolCall] = []
    errors: list[str] = []
    for index, call in enumerate(calls):
        parsed, error = _parsed_call_arguments(call)
        if mode == "bash_persistent":
            action, action_error = _bash_persistent_action(task, call, parsed, error, index)
        else:
            action, action_error = _rich_terminal_action(call, parsed, error, index)
        if action_error:
            errors.append(action_error)
            continue
        if action is not None:
            actions.append(action)
    if errors:
        return [], " ".join(errors)
    return actions, ""


def _parsed_call_arguments(call: ModelToolCall) -> tuple[Any, str]:
    if call.arguments_text:
        try:
            return json.loads(call.arguments_text), ""
        except Exception as exc:
            return call.arguments, f"Error parsing tool call arguments: {exc}."
    return call.arguments, ""


def _bash_persistent_action(
    task: TaskContext,
    call: ModelToolCall,
    parsed: Any,
    error: str,
    index: int,
) -> tuple[HarnessToolCall | None, str]:
    if call.name != "bash":
        error += f"Unknown tool '{call.name}'."
    if not isinstance(parsed, dict) or "command" not in parsed:
        error += "Missing 'command' argument in bash tool call."
    session_id = _persistent_session_id(task)
    if session_id is None:
        error += "Persistent bash session is unavailable."
    if error:
        return None, error.strip()
    return (
        HarnessToolCall(
            "persistent_bash",
            {
                "command": _runtime_command(str(parsed["command"])),
                "timeout_sec": DEFAULT_COMMAND_TIMEOUT_SEC,
                "session_id": session_id,
            },
            _call_id(call, index),
        ),
        "",
    )


def _persistent_session_id(task: TaskContext) -> int | None:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    terminal = metadata.get("persistent_terminal") if isinstance(metadata, dict) else {}
    terminal = terminal if isinstance(terminal, dict) else {}
    session_id = terminal.get("session_id")
    return session_id if isinstance(session_id, int) else None


def _rich_terminal_action(
    call: ModelToolCall,
    parsed: Any,
    error: str,
    index: int,
) -> tuple[HarnessToolCall | None, str]:
    plain = call.name.rsplit(".", 1)[-1]
    if not isinstance(parsed, dict):
        return None, (error + "Tool arguments must be an object.").strip()
    if plain == "exec_command":
        args = dict(parsed)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        if not str(args.get("cmd") or "").strip():
            return None, (error + "Missing 'cmd' argument in exec_command tool call.").strip()
        args.setdefault("timeout_sec", DEFAULT_COMMAND_TIMEOUT_SEC)
        return HarnessToolCall("exec_command", args, _call_id(call, index)), error.strip()
    if plain == "write_stdin":
        args = dict(parsed)
        if "process_id" in args and "session_id" not in args:
            args["session_id"] = args.pop("process_id")
        args.setdefault("chars", "")
        if "session_id" not in args and "tmux_session" not in args and "session_name" not in args:
            return None, (error + "write_stdin requires session_id or tmux_session.").strip()
        return HarnessToolCall("write_stdin", args, _call_id(call, index)), error.strip()
    return None, (error + f"Unknown tool '{call.name}'.").strip()


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


def _format_error_message(error: str, mode: str) -> dict[str, str]:
    if mode == "bash_persistent":
        guidance = (
            "Call the bash tool with your command as the argument:\n"
            "- Tool: bash\n"
            '- Arguments: {"command": "your_command_here"}'
        )
    else:
        guidance = (
            "Use exec_command for shell commands and write_stdin for existing session_id "
            "or tmux_session interaction."
        )
    return {
        "role": "user",
        "content": _render(
            MINI_FORMAT_ERROR_TEMPLATE,
            {"error": error, "tool_guidance": guidance},
        ),
    }


def _turn_metadata(
    result: ToolModelResult,
    retry_index: int,
    previous_messages: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    metadata = _base_metadata(mode, mini_swe_agent_v2_format_retries=retry_index)
    response_items = _assistant_items(result)
    metadata["mini_swe_agent_v2_response_items"] = response_items
    metadata["mini_swe_agent_v2_messages"] = [*previous_messages, *response_items]
    metadata["mini_swe_agent_v2_usage"] = result.usage
    metadata["mini_swe_agent_v2_request_metadata"] = result.request_metadata
    metadata["mini_swe_agent_v2_response_id"] = result.response_id
    return metadata


def _base_metadata(mode: str, **extra: Any) -> dict[str, Any]:
    metadata = {
        "sequential_tool_calls": True,
        "mini_swe_agent_v2_variant": mode,
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
