from __future__ import annotations

import hashlib
import importlib
import importlib.util
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
    direct_path = _direct_harness_path(candidate_dir, module_name)
    if direct_path is not None:
        module = _import_harness_file(direct_path)
    else:
        with _candidate_import_path(candidate_dir, module_name):
            module = importlib.import_module(module_name)
    factory: Any = getattr(module, factory_name)
    agent = factory()
    if not isinstance(agent, BaseHarness):
        raise TypeError(f"{module_name}.{factory_name}() did not return BaseHarness")
    return agent


def _direct_harness_path(candidate_dir: Path | str | None, module_name: str) -> Path | None:
    if candidate_dir is None or module_name != "candidate.harness":
        return None
    path = Path(candidate_dir).resolve() / "harness.py"
    return path if path.is_file() else None


def _import_harness_file(path: Path) -> Any:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_candidate_harness_{digest}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load harness from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    importlib.invalidate_caches()
    spec.loader.exec_module(module)
    return module


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
