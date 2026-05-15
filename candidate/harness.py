import json

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model
from plumbing.types import CommandResult, HarnessTurn, TaskContext

FIRST_COMMAND = "pwd && ls -la"
EDIT_MARKERS = ("cat >", "tee ", "sed -i", "perl -pi", "python - <<", "cat <<")
SYSTEM = (
    "You are a terminal coding agent with one tool: run shell commands. "
    'Return only JSON: {"action":"run","command":"..."} or {"action":"done"}. '
    "Use the loop: inspect/list/read, edit narrowly, verify, then done. "
    "Prefer rg/find/sed/python for precise work. Avoid interactive commands, "
    "background servers, and unbounded output. Done requires evidence."
)


class CandidateHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        if not history:
            return HarnessTurn(done=False, command=FIRST_COMMAND)
        user = (
            f"Task:\n{task.instruction}\n\n"
            f"Recent terminal history:\n{_history(history)}\n\n"
            f"State hints: {_state(history)}\n\n"
            "Choose exactly one next shell command. If the last command failed, "
            "inspect the error or change approach. After edits, run the smallest "
            "relevant check before done."
        )
        action = _parse_action(
            call_terminal_model(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        )
        done = action["action"] == "done"
        command = _clean_command(action.get("command", ""))
        return HarnessTurn(done=done, command="" if done else command or FIRST_COMMAND)


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


def _parse_action(text: str) -> dict[str, str]:
    text = _strip_fence(text.strip())
    if text.upper().startswith("DONE"):
        return {"action": "done", "command": ""}
    value = _json_object(text)
    if value is not None:
        return _normalize(value)
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("tool:") or not line.endswith(")"):
            continue
        name, _, rest = line[5:].strip().partition("(")
        if name.strip().lower() in {"done", "finish"}:
            return {"action": "done", "command": ""}
        args = _json_object(rest[:-1])
        if args is not None:
            return _normalize({"action": name.strip(), "args": args})
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


def _normalize(value: dict[str, object]) -> dict[str, str]:
    action = str(value.get("action") or value.get("tool") or value.get("name") or "run").lower()
    args = value.get("args") or value.get("arguments") or {}
    if isinstance(args, str):
        args = _json_object(args) or {}
    if action in {"done", "finish", "final"}:
        return {"action": "done", "command": ""}
    command = value.get("command") or value.get("cmd")
    if command is None and isinstance(args, dict):
        command = args.get("command") or args.get("cmd")
    return {"action": "run", "command": str(command or "")}


def _clean_command(text: str) -> str:
    text = _strip_fence(text).strip()
    return "" if text.upper().startswith("DONE") else text


def _strip_fence(text: str) -> str:
    if text.startswith("```"):
        return "\n".join(line for line in text.splitlines() if not line.startswith("```")).strip()
    return text


def create_agent() -> CandidateHarness:
    return CandidateHarness()
