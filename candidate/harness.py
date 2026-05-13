import json

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model
from plumbing.types import CommandResult, HarnessTurn, TaskContext


class CandidateHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        system = (
            'You are a terminal agent. Reply only as JSON: {"action":"run",'
            '"command":"..."} or {"action":"done"}. Use run to inspect, edit, and '
            "verify. Use done only after the task is solved."
        )
        recent = "\n\n".join(
            f"$ {item.command}\nexit={item.return_code}\n"
            f"stdout:\n{item.stdout[-2500:]}\nstderr:\n{item.stderr[-1200:]}"
            for item in history[-4:]
        )
        user = (
            f"Task:\n{task.instruction}\n\n"
            f"Recent terminal history:\n{recent or '(none)'}\n\n"
            "Choose one next action. Prefer targeted inspection first, focused edits, "
            "then a relevant check. Do not include markdown or commentary."
        )
        action = _parse_action(
            call_terminal_model(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        )
        done = action.get("action") == "done"
        return HarnessTurn(
            done=done, command="" if done else str(action.get("command", "")).strip()
        )


def _parse_action(text: str) -> dict[str, str]:
    text = text.strip()
    if text.upper().startswith("DONE"):
        return {"action": "done"}
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            value = {}
        if isinstance(value, dict):
            action = str(value.get("action", "run")).lower()
            command = str(value.get("command", ""))
            return {"action": action, "command": command}
    return {"action": "run", "command": text}


def create_agent() -> CandidateHarness:
    return CandidateHarness()
