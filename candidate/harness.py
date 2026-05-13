from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model
from plumbing.types import TaskContext


class CandidateHarness(BaseHarness):
    def solve(self, task: TaskContext) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful terminal agent. Solve the task generally, "
                    "use concise commands, inspect before editing, and verify results."
                ),
            },
            {"role": "user", "content": task.instruction},
        ]
        return call_terminal_model(messages)


def create_agent() -> CandidateHarness:
    return CandidateHarness()
