from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_HARNESS_PATH = ROOT / "seeds" / "mini_swe_agent_barebones_v2" / "harness.py"


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
    user = base._render(CONTEXT_PREFIX_TEMPLATE, values) + base._render(
        base.MINI_INSTANCE_TEMPLATE, values
    )
    return [
        {"role": "system", "content": base._render(base.MINI_SYSTEM_TEMPLATE, values)},
        {"role": "user", "content": user},
    ]


CONTEXT_PREFIX_TEMPLATE = """<environment_context>
  <cwd>{{cwd}}</cwd>
  <shell>bash</shell>
</environment_context>
{{agents_context}}"""


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
        module_name = "_mini_swe_agent_barebones_v2_context_only_base"
        spec = importlib.util.spec_from_file_location(module_name, BASE_HARNESS_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load base harness from {BASE_HARNESS_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._initial_messages = _initial_messages
        _BASE = module
        return module
