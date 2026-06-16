from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_HARNESS_PATH = ROOT / "seeds" / "mini_swe_agent_barebones_v2" / "harness.py"

CODEX_SYSTEM_TEMPLATE = """You are Codex, a coding agent running in a terminal-based coding assistant. You are expected to be precise, safe, and helpful.

Your capabilities:

- Receive user prompts and other context provided by the harness, such as files in the workspace.
- Use the available bash tool to inspect files, run commands, edit files, and verify behavior.
- Work iteratively: inspect the environment, make the smallest useful change, check results, and continue until the task is complete.

Within this context, Codex refers to the agentic coding interface, not an older code model.

# How You Work

Be concise and direct. Prefer concrete terminal actions over narration. Keep going until the task is solved, or until the next action is genuinely blocked.

# AGENTS.md

AGENTS.md files can provide repository-specific instructions. When AGENTS.md content is provided in the prompt, obey instructions that apply to files you inspect or modify. More-specific AGENTS.md files take precedence over broader ones.

# Tooling

This harness exposes exactly one tool:

- `bash(command: string)`: run a shell command in a fresh non-persistent subshell.

You do not have `apply_patch`, `exec_command`, `write_stdin`, planning tools, or any other tool in this harness.
"""

CODEX_DEVELOPER_TEMPLATE = """<permissions instructions>
Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all commands are permitted. Network access is enabled. Approval policy is currently never.
</permissions instructions>
"""

CODEX_INSTANCE_TEMPLATE = """<environment_context>
  <cwd>{{cwd}}</cwd>
  <shell>bash</shell>
</environment_context>
{{agents_context}}
Task:
{{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a fresh subshell
3. You see the result(s)
4. You write your next command(s)

**CRITICAL REQUIREMENTS:**

- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files.
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}
"""


def create_agent() -> Any:
    agent = _base().create_agent()
    agent.wants_environment_context = True
    agent.wants_agents_context = True
    return agent


def _initial_messages(task: Any) -> list[dict[str, str]]:
    base = _base()
    values = base._template_vars(task)
    values["cwd"] = values.get("cwd") or "."
    values["agents_context"] = _agents_context(values.get("agents_md"))
    return [
        {"role": "system", "content": base._render(CODEX_SYSTEM_TEMPLATE, values)},
        {"role": "developer", "content": base._render(CODEX_DEVELOPER_TEMPLATE, values)},
        {"role": "user", "content": base._render(CODEX_INSTANCE_TEMPLATE, values)},
    ]


def _agents_context(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    rendered = []
    for item in sorted((entry for entry in items if isinstance(entry, dict)), key=_agents_path):
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        path = str(item.get("path") or "AGENTS.md")
        rendered.append(f"<agents_md path='{path}'>\n{content}\n</agents_md>")
    return "\n".join(rendered) + ("\n" if rendered else "")


def _agents_path(item: dict[str, Any]) -> str:
    return str(item.get("path") or "")


def _base() -> Any:
    global _BASE
    try:
        return _BASE
    except NameError:
        module_name = "_mini_swe_agent_barebones_v2_codex_prompt_base"
        spec = importlib.util.spec_from_file_location(module_name, BASE_HARNESS_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load base harness from {BASE_HARNESS_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._initial_messages = _initial_messages
        _BASE = module
        return module
