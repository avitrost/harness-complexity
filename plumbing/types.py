from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskContext:
    instruction: str
    working_dir: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    command: str
    return_code: int | None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class HarnessTurn:
    command: str = ""
    done: bool = False
    timeout_sec: int | None = None
