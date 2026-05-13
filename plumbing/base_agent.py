from __future__ import annotations

import importlib
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from plumbing.types import CommandResult, HarnessTurn, TaskContext


class BaseHarness(ABC):
    @abstractmethod
    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        """Return the next terminal command, or mark the task done."""


def load_harness(
    module_name: str = "candidate.harness",
    factory_name: str = "create_agent",
    candidate_dir: Path | str | None = None,
) -> BaseHarness:
    with _candidate_import_path(candidate_dir, module_name):
        module = importlib.import_module(module_name)
    factory: Any = getattr(module, factory_name)
    agent = factory()
    if not isinstance(agent, BaseHarness):
        raise TypeError(f"{module_name}.{factory_name}() did not return BaseHarness")
    return agent


@contextmanager
def _candidate_import_path(candidate_dir: Path | str | None, module_name: str) -> Iterator[None]:
    if candidate_dir is None:
        yield
        return
    root = Path(candidate_dir).resolve()
    if root.name == "candidate":
        root = root.parent
    sys.path.insert(0, str(root))
    for name in (module_name, module_name.rsplit(".", 1)[0]):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
