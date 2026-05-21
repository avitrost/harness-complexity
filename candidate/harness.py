import json

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessTurn, TaskContext

FIRST_COMMAND = "pwd && ls -la"
EDIT_MARKERS = ("cat >", "tee ", "sed -i", "perl -pi", "python - <<", "cat <<")
SYSTEM = (
    "You are a terminal coding agent. Use the provided tools instead of free-form text. "
    "Call execute_commands for shell work or task_complete when done. "
    "Use the loop: inspect/list/read, edit narrowly, verify, then done. "
    "Prefer rg/find/sed/python for precise work. Avoid interactive commands, "
    "background servers, and unbounded output. Done requires evidence."
)
TOOLS = json.loads(
    """[{"type":"function","name":"execute_commands","description":"Execute terminal commands.","parameters":{"type":"object","properties":{"analysis":{"type":"string"},"plan":{"type":"string"},"commands":{"type":"array","items":{"type":"object","properties":{"keystrokes":{"type":"string"},"duration":{"type":"number"}},"required":["keystrokes"]}}},"required":["analysis","plan","commands"],"additionalProperties":false}},{"type":"function","name":"task_complete","description":"Signal completion.","parameters":{"type":"object","properties":{},"additionalProperties":false}}]"""
)


class CandidateHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        if not history:
            return HarnessTurn(done=False, command=FIRST_COMMAND)
        user = (
            f"Task:\n{task.instruction}\n\n"
            f"Recent terminal history:\n{_history(history)}\n\n"
            f"State hints: {_state(history)}\n\n"
            "Use execute_commands for exactly one next shell command. If the last command failed, "
            "inspect the error or change approach. After edits, run the smallest "
            "relevant check before task_complete."
        )
        result = call_terminal_model_with_tools(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], TOOLS
        )
        action = _parse_tool(result) or _parse_action(result.content)
        done = action["action"] == "done"
        command = _clean_command(action.get("command", ""))
        timeout = _timeout(action.get("timeout_sec"))
        return HarnessTurn("" if done else command or FIRST_COMMAND, done, timeout)


def _history(history: list[CommandResult]) -> str:
    start = max(0, len(history) - 6)
    rows = (
        f"[{i}] $ {x.command}\nexit={x.return_code}\n"
        f"stdout:\n{_clip(x.stdout, 2200)}\n"
        f"stderr:\n{_clip(x.stderr, 1200)}"
        for i, x in enumerate(history[start:], start + 1)
    )
    return "\n\n".join(rows)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else f"<{len(text) - limit} chars omitted>\n{text[-limit:]}"


def _state(history: list[CommandResult]) -> str:
    last = history[-1]
    hints = [f"turn={len(history) + 1}", f"last_exit={last.return_code}"]
    if last.return_code:
        hints.append("last command failed; do not repeat it unchanged")
    if len(history) > 1 and last.command.strip() == history[-2].command.strip():
        hints.append("last command repeated")
    if any(any(marker in item.command for marker in EDIT_MARKERS) for item in history[-4:]):
        hints.append("recent write-like command seen; verify before done")
    return "; ".join(hints)


def _parse_tool(result) -> dict[str, object] | None:
    for call in result.tool_calls:
        if call.name == "task_complete":
            return {"action": "done", "command": ""}
        if call.name != "execute_commands":
            continue
        commands = call.arguments.get("commands") or []
        if isinstance(commands, str):
            try:
                commands = json.loads(commands) if commands.startswith(("[", "{")) else [commands]
            except json.JSONDecodeError:
                commands = []
        if isinstance(commands, dict):
            commands = [commands]
        for item in commands if isinstance(commands, list) else []:
            if isinstance(item, dict):
                command = item.get("keystrokes") or item.get("command") or item.get("cmd")
                timeout = item.get("duration") or item.get("timeout_sec")
            else:
                command, timeout = item, None
            if command:
                return {"action": "run", "command": str(command), "timeout_sec": timeout}
        return {"action": "run", "command": ""}
    return None


def _parse_action(text: str) -> dict[str, object]:
    text = _strip_fence(text.strip())
    if text.upper().startswith("DONE"):
        return {"action": "done", "command": ""}
    value = _json_object(text)
    if value is not None:
        return _normalize(value)
    return {"action": "run", "command": text}


def _json_object(text: str) -> dict[str, object] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalize(value: dict[str, object]) -> dict[str, object]:
    action = str(value.get("action") or value.get("tool") or value.get("name") or "run").lower()
    args = value.get("args") or value.get("arguments") or {}
    if isinstance(args, str):
        args = _json_object(args) or {}
    if action in {"done", "finish", "final"}:
        return {"action": "done", "command": ""}
    command = value.get("command") or value.get("cmd")
    if command is None and isinstance(args, dict):
        command = args.get("command") or args.get("cmd")
    timeout = value.get("timeout_sec")
    if timeout is None and isinstance(args, dict):
        timeout = args.get("timeout_sec")
    return {"action": "run", "command": str(command or ""), "timeout_sec": timeout}


def _timeout(value: object) -> int | None:
    if type(value) in (int, float):
        return max(1, min(600, int(value)))
    if isinstance(value, str) and value.strip().isdigit():
        return max(1, min(600, int(value)))
    return None


def _clean_command(text: str) -> str:
    text = _strip_fence(text).strip()
    return "" if text.upper().startswith("DONE") else text


def _strip_fence(text: str) -> str:
    if text.startswith("```"):
        return "\n".join(line for line in text.splitlines() if not line.startswith("```")).strip()
    return text


def create_agent() -> CandidateHarness:
    return CandidateHarness()
