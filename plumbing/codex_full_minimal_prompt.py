from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CODEX_BASE_INSTRUCTIONS = """You are a coding agent working in a terminal.

Use the available tools to solve the user's task. Inspect the files, run commands, edit files, and verify your work as needed.

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. Do NOT guess or make up an answer.

You MUST adhere to the following criteria when solving queries:

- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the `apply_patch` tool to edit files (NEVER try `applypatch` or `apply-patch`, only `apply_patch`): {"command":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n- pass\\n+ return 123\\n*** End Patch"]}

When you are done, respond with the final answer and do not call more tools.
"""

_ROOT = Path(__file__).resolve().parents[1]
_BASE_HARNESS_PATH = _ROOT / "seeds" / "codex_full" / "harness.py"
_BASE_MODULE_NAME = "_codex_full_minimal_prompt_base"


def _load_base_module() -> Any:
    sys.modules.pop(_BASE_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load Codex full harness from {_BASE_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    module.CODEX_BASE_INSTRUCTIONS = CODEX_BASE_INSTRUCTIONS
    return module


_BASE = _load_base_module()
CandidateHarness = _BASE.CandidateHarness


def create_agent() -> Any:
    _BASE.CODEX_BASE_INSTRUCTIONS = CODEX_BASE_INSTRUCTIONS
    return _BASE.create_agent()
