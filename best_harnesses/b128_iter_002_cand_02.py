import json

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessTurn, TaskContext

FIRST_COMMAND = "pwd && ls -la"
SYSTEM = (
    "You are a terminal coding agent. Use the provided tools instead of free-form text. "
    "Call execute_commands for shell work or task_complete when done. Use this loop: "
    "inspect/list/read, edit narrowly, verify, then finish. Prefer rg/find/sed/python. "
    "Avoid interactive commands, unbounded output, and background servers without logs. "
    "Once verification evidence is visible, finish instead of repeating evidence commands."
)
TOOLS = json.loads(
    """[{"type":"function","name":"execute_commands","description":"Execute one terminal command.","parameters":{"type":"object","properties":{"analysis":{"type":"string"},"plan":{"type":"string"},"commands":{"type":"array","items":{"type":"object","properties":{"keystrokes":{"type":"string"},"duration":{"type":"number"},"timeout_sec":{"type":"number"}},"required":["keystrokes"]}}},"required":["analysis","plan","commands"],"additionalProperties":false}},{"type":"function","name":"task_complete","description":"Signal completion after verification evidence.","parameters":{"type":"object","properties":{},"additionalProperties":false}}]"""
)


class CandidateHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        if not history:
            return HarnessTurn(done=False, command=FIRST_COMMAND)
        if _verified_finish(history):
            return HarnessTurn(done=True, command="")
        user = (
            f"Task:\n{task.instruction}\n\nRecent terminal history:\n{_history(history)}\n\n"
            f"State hints: {_state(history)}\n\n"
            "Use execute_commands for exactly one next shell command. If the last command "
            "failed, inspect the error or change approach. After edits, run the smallest "
            "relevant check before task_complete. Do not keep reprinting the same passing "
            "evidence; call task_complete once the requested outcome is verified."
        )
        result = call_terminal_model_with_tools(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], TOOLS
        )
        action = _parse_tool(result) or _parse_action(result.content)
        done = action["action"] == "done"
        command = _strip(str(action.get("command", "")))
        timeout = _timeout(action.get("timeout_sec"))
        return HarnessTurn("" if done else command or FIRST_COMMAND, done, timeout)


def _history(history: list[CommandResult]) -> str:
    start = max(0, len(history) - 6)
    return "\n\n".join(
        f"[{i}] $ {x.command}\nexit={x.return_code}\nstdout:\n{_clip(x.stdout, 2200)}\nstderr:\n{_clip(x.stderr, 1200)}"
        for i, x in enumerate(history[start:], start + 1)
    )


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
    return "; ".join(hints)


def _verified_finish(history: list[CommandResult]) -> bool:
    if len(history) < 4 or history[-1].return_code or not history[-1].stdout.strip():
        return False
    prior = "\n".join(x.command.lower() for x in history[:-1])
    last = history[-1].command.lower()
    changed = "cat >|tee |write_text|sed -i|chmod |chown |mkdir |install |git commit|git push|make |openssl |sqlite3 ".split(
        "|"
    )
    checked = "test|check|verify|pytest|curl |openssl |stat |grep |sed -n|cat |ls |find |python|rscript|sqlite3 |git status|wc ".split(
        "|"
    )
    return any(x in prior for x in changed) and any(x in last for x in checked)


def _parse_tool(result) -> dict[str, object] | None:
    for call in result.tool_calls:
        if call.name == "task_complete":
            return {"action": "done", "command": ""}
        if call.name != "execute_commands":
            continue
        commands = call.arguments.get("commands") or []
        if isinstance(commands, (str, dict)):
            commands = [commands]
        for item in commands:
            item = item if isinstance(item, dict) else {"keystrokes": item}
            command = item.get("keystrokes") or item.get("command") or item.get("cmd")
            timeout = item.get("timeout_sec") or item.get("duration")
            if command:
                return {"action": "run", "command": str(command), "timeout_sec": timeout}
        return {"action": "run", "command": ""}
    return None


def _parse_action(text: str) -> dict[str, object]:
    text = _strip(text.strip())
    if text.upper().startswith("DONE"):
        return {"action": "done", "command": ""}
    value = _json_object(text)
    if not value:
        return {"action": "run", "command": text}
    action = str(value.get("action") or value.get("tool") or "run").lower()
    if action in {"done", "finish", "final"}:
        return {"action": "done", "command": ""}
    command = value.get("command") or value.get("cmd")
    commands = value.get("commands") if isinstance(value.get("commands"), list) else []
    first = commands[0] if commands and isinstance(commands[0], dict) else {}
    command = command or first.get("keystrokes") or first.get("command") or first.get("cmd")
    timeout = value.get("timeout_sec") or first.get("timeout_sec") or first.get("duration")
    return {"action": "run", "command": str(command or ""), "timeout_sec": timeout}


def _json_object(text: object) -> dict[str, object] | None:
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _timeout(value: object) -> int | None:
    if type(value) in (int, float) or (isinstance(value, str) and value.strip().isdigit()):
        return max(1, min(600, int(value)))
    return None


def _strip(text: str) -> str:
    if text.startswith("```"):
        return "\n".join(line for line in text.splitlines() if not line.startswith("```")).strip()
    return text


def create_agent() -> CandidateHarness:
    return CandidateHarness()
