from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalResult:
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    runtime_sec: float | None = None
