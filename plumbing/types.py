from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskContext:
    instruction: str
    working_dir: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
