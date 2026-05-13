from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model
from plumbing.types import CommandResult, HarnessTurn, TaskContext


class CandidateHarness(BaseHarness):
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        system_prompt = (
            "You are a terminal agent. Return exactly one shell command to run next, "
            "or return DONE if the task is complete. No markdown, no commentary."
        )
        recent = "\n\n".join(_format_result(item) for item in history[-6:])
        user_prompt = (
            f"Task:\n{task.instruction}\n\n"
            f"Recent terminal history:\n{recent or '(none)'}\n\n"
            "Choose the next safest useful command. Inspect before editing, make focused "
            "changes, and verify before DONE."
        )
        text = call_terminal_model(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        ).strip()
        return HarnessTurn(done=text.upper().startswith("DONE"), command=_clean_command(text))


def _format_result(result: CommandResult) -> str:
    return (
        f"$ {result.command}\n"
        f"exit={result.return_code}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )


def _clean_command(text: str) -> str:
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        return "\n".join(lines).strip()
    return "" if text.upper().startswith("DONE") else text


def create_agent() -> CandidateHarness:
    return CandidateHarness()
