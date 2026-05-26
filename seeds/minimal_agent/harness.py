from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model_with_tools
from plumbing.types import HarnessToolCall, HarnessTurn

SYSTEM = (
    "You are a terminal coding agent. Use exec_command for shell work. "
    "When the task is complete, reply with final text and no tool call."
)
TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "description": "Run one shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "yield_time_ms": {"type": "number"},
                "max_output_tokens": {"type": "number"},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    }
]


class CandidateHarness(BaseHarness):
    def next_command(self, task, history):
        result = call_terminal_model_with_tools(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _prompt(task, history)},
            ],
            TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        calls = tuple(tool for call in result.tool_calls if (tool := _tool_call(call)))
        if calls:
            return HarnessTurn(tool_calls=calls, assistant_content=result.content)
        text = _visible_text(result)
        return HarnessTurn(done=bool(text.strip()), assistant_content=text)


def _prompt(task, history):
    cwd = task.working_dir or "."
    return f"<cwd>{cwd}</cwd>\n\nTask:\n{task.instruction}\n\nRecent terminal history:\n{_history(history)}"


def _history(history):
    if not history:
        return "(none)"
    rows = []
    for item in history[-4:]:
        rows.append(
            f"$ {item.command}\n"
            f"exit={item.return_code}\n"
            f"stdout:\n{_clip(item.stdout, 3000)}\n"
            f"stderr:\n{_clip(item.stderr, 1200)}"
        )
    return "\n\n".join(rows)


def _clip(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else f"<{len(text) - limit} chars omitted>\n{text[-limit:]}"


def _tool_call(call):
    if call.name.rsplit(".", 1)[-1] != "exec_command":
        return None
    args = dict(call.arguments or {})
    cmd = str(args.get("cmd") or args.get("command") or "").strip()
    if not cmd:
        return None
    args["cmd"] = cmd
    return HarnessToolCall("exec_command", args, call.call_id)


def _visible_text(result):
    if result.content.strip():
        return result.content
    chunks = []
    for item in result.response_items:
        if item.get("type") == "message" and item.get("role") == "assistant":
            chunks.extend(_content_text(item.get("content")))
    return "\n".join(chunks)


def _content_text(content):
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [item["text"] for item in content if isinstance(item, dict) and "text" in item]
    return []


def create_agent():
    return CandidateHarness()
