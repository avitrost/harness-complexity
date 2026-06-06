from __future__ import annotations

import json

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ModelToolCall, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer."
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
DEFAULT_COMMAND_TIMEOUT_SEC = 30
MAX_OBSERVATION_CHARS = 10000

INSTANCE_PROMPT = """Task:
{task}

Use the bash tool to execute commands.

Each bash command runs in a new, non-persistent subshell. Directory changes,
environment variables, aliases, shell functions, and shell options do not persist
between bash calls. Filesystem changes do persist.

Every response must include at least one bash tool call.
Tool format:
- Tool: bash
- Arguments: {{"command": "your_command_here"}}

When the task is complete, finish by issuing exactly this bash command and no
other command:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

After that command, you cannot continue working on the task.

Previous command results:
{history}
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


class BarebonesMiniSweAgentHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        submission = _submission_from_history(history)
        if submission is not None:
            return HarnessTurn(done=True, assistant_content=submission)

        result = call_terminal_model_with_tools(_messages(task, history), [BASH_TOOL])
        tool_calls = tuple(
            tool_call
            for index, call in enumerate(result.tool_calls)
            if (tool_call := _bash_tool_call(call, index)) is not None
        )
        if tool_calls:
            return HarnessTurn(
                tool_calls=tool_calls,
                assistant_content=result.content,
                metadata={"sequential_tool_calls": True},
            )
        return HarnessTurn(done=True, assistant_content=result.content)


def create_agent() -> BarebonesMiniSweAgentHarness:
    return BarebonesMiniSweAgentHarness()


def _messages(task: TaskContext, history: list[CommandResult]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": INSTANCE_PROMPT.format(
                task=task.instruction,
                history=_history_text(history),
            ),
        },
    ]


def _history_text(history: list[CommandResult]) -> str:
    if not history:
        return "(none)"
    return "\n\n".join(_history_item(item) for item in history)


def _history_item(item: CommandResult) -> str:
    output = (item.stdout or "") + (item.stderr or "")
    return (
        f"$ {item.command}\n"
        f"returncode: {-1 if item.return_code is None else item.return_code}\n"
        f"output:\n{_clip(output, MAX_OBSERVATION_CHARS)}"
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"<{omitted} chars omitted>\n{text[-limit:]}"


def _bash_tool_call(call: ModelToolCall, index: int) -> HarnessToolCall | None:
    if call.name != "bash":
        return None
    args = call.arguments
    if call.arguments_text:
        try:
            parsed = json.loads(call.arguments_text)
        except json.JSONDecodeError:
            parsed = args
    else:
        parsed = args
    if not isinstance(parsed, dict):
        return None
    command = str(parsed.get("command") or "").strip()
    if not command:
        return None
    return HarnessToolCall(
        "bash",
        {"command": command, "timeout_sec": DEFAULT_COMMAND_TIMEOUT_SEC},
        call.call_id or f"call_barebones_{index + 1}",
    )


def _submission_from_history(history: list[CommandResult]) -> str | None:
    if not history:
        return None
    last = history[-1]
    output = (last.stdout or "") + (last.stderr or "")
    lines = output.lstrip().splitlines(keepends=True)
    if last.return_code == 0 and lines and lines[0].strip() == SUBMIT_MARKER:
        return "".join(lines[1:])
    if last.return_code == 0 and last.command.strip() == SUBMIT_COMMAND:
        return ""
    return None
