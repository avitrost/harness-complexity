from __future__ import annotations

import ast
from dataclasses import dataclass
import html
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "seeds/codex_full/harness.py"
MINIMAL_SOURCE = ROOT / "plumbing/codex_full_minimal_prompt.py"
OUT = ROOT / "viz/codex_prompt_minimal_side_by_side.html"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEW_SYSTEM_PROMPT = """You are a coding agent working in a terminal.

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

NEW_SUMMARIZATION_PROMPT = """Summarize the current task for the next model.

Include progress, decisions, constraints, important files/data, and remaining steps.
Be concise."""

NEW_SUMMARY_PREFIX = "Continuation summary from a prior model:"

NEW_PERMISSIONS = ""

NEW_ENVIRONMENT = (
    '<env cwd="{environment.cwd}" shell="{environment.shell}" '
    'date="{environment.current_date}" tz="{environment.timezone}" />'
)

AGENTS_WRAPPER = '<agents_md path="AGENTS.md">\n{content}\n</agents_md>'

NEW_TOOL_SPECS = """[
  {"name":"exec_command","desc":"Run shell command; return output or session_id.","args":["cmd","workdir?","yield_time_ms?","max_output_tokens?","tty?"]},
  {"name":"write_stdin","desc":"Send stdin or poll a running session.","args":["session_id","chars?","yield_time_ms?","max_output_tokens?"]},
  {"name":"update_plan","desc":"Update task plan.","args":["plan","explanation?"]},
  {"name":"apply_patch","desc":"Apply a Begin/End Patch diff.","input":"patch text"}
]"""

NEW_TOOL_OUTPUT_FORMAT = """chunk=eb4a89 wall=0.0120s exit=0 session=1 original_tokens=100
output:
..."""

HISTORY_REPLAY_SHAPE = """[
  {"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\\"cmd\\":\\"ls\\"}"},
  {"type":"function_call_output","call_id":"call_1","output":"Chunk ID: ...\\nWall time: ...\\nOutput:\\n..."}
]"""


@dataclass(frozen=True)
class PromptComponent:
    title: str
    source: str
    original: str
    proposed: str
    note: str


def main() -> None:
    module = _load_full_harness()
    components = _components(module)
    html_text = _render_page(components)
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)


def _components(module: Any) -> list[PromptComponent]:
    return [
        PromptComponent(
            title="Main System Prompt",
            source="seeds/codex_full/harness.py:19; active proposal in plumbing/codex_full_minimal_prompt.py:8",
            original=_literal_constant(SOURCE, "CODEX_BASE_INSTRUCTIONS"),
            proposed=NEW_SYSTEM_PROMPT,
            note=(
                "This is the large prompt already cut in the minimal-prompt experiment. "
                "It remains the primary prose prior."
            ),
        ),
        PromptComponent(
            title="Context Compaction Request",
            source="seeds/codex_full/harness.py:315 and ContextCompactor.compact at line 1505",
            original=_literal_constant(SOURCE, "SUMMARIZATION_PROMPT"),
            proposed=NEW_SUMMARIZATION_PROMPT,
            note=(
                "Sent to the model only when context compaction triggers. It creates the "
                "handoff summary that later turns see."
            ),
        ),
        PromptComponent(
            title="Compaction Handoff Prefix",
            source="seeds/codex_full/harness.py:325 and ContextCompactor.compact at line 1511",
            original=_literal_constant(SOURCE, "SUMMARY_PREFIX"),
            proposed=NEW_SUMMARY_PREFIX,
            note="Prepended to summaries that are replayed into later turns.",
        ),
        PromptComponent(
            title="Permissions Developer Message",
            source=(
                "seeds/codex_full/harness.py:329, :334, and "
                "PermissionsInstructionsRenderer at line 1732"
            ),
            original=module.PermissionsInstructionsRenderer().render(
                module.TurnEnvironment(cwd="/app")
            ),
            proposed=NEW_PERMISSIONS,
            note=(
                "Currently sent as a developer message on every model call. Proposed "
                "replacement is intentionally empty: no text and no tags."
            ),
        ),
        PromptComponent(
            title="Environment Context Wrapper",
            source="seeds/codex_full/harness.py:1764 and InitialContextBuilder.environment_context at line 1775",
            original=module.InitialContextBuilder().environment_context(
                module.TurnEnvironment(cwd="/app")
            ),
            proposed=NEW_ENVIRONMENT,
            note=(
                "Prepended to the user task before the task text. The proposed replacement "
                "keeps the date dynamic via the same TurnEnvironment.current_date field."
            ),
        ),
        PromptComponent(
            title="AGENTS.md Wrapper",
            source="seeds/codex_full/harness.py:1786 in AgentInstructionsRenderer.render",
            original=AGENTS_WRAPPER,
            proposed=AGENTS_WRAPPER,
            note=(
                "Only appears when Harbor supplies AGENTS.md metadata for the task. "
                "Kept unchanged by request."
            ),
        ),
        PromptComponent(
            title="Tool Schemas And Descriptions",
            source="seeds/codex_full/harness.py:787-940 in _built_tools and tool spec helpers",
            original=json.dumps(module._built_tools(), indent=2),
            proposed=NEW_TOOL_SPECS,
            note=(
                "This is not a system prompt, but it is model-visible instruction text. "
                "The proposed compression is conceptual; real implementation must stay valid "
                "for the tool-calling API."
            ),
        ),
        PromptComponent(
            title="Unified Exec Output Format",
            source="seeds/codex_full/harness.py:1098 in ToolOutputFormatter.unified_exec_text",
            original="""Chunk ID: eb4a89
Wall time: 0.0120 seconds
Process exited with code 0
Process running with session ID 1
Original token count: 100
Output:
...""",
            proposed=NEW_TOOL_OUTPUT_FORMAT,
            note=(
                "This is replayed after shell calls and affects how the model reads prior "
                "state. The proposed compact line preserves chunk id, wall time, exit code, "
                "session id, original token count, and output."
            ),
        ),
        PromptComponent(
            title="History Replay Shape",
            source="seeds/codex_full/harness.py:1244-1310 in HistoryReplay and ResponseItemFactory",
            original=HISTORY_REPLAY_SHAPE,
            proposed=HISTORY_REPLAY_SHAPE,
            note="Kept unchanged by request.",
        ),
    ]


def _render_page(components: list[PromptComponent]) -> str:
    total_original = sum(len(component.original) for component in components)
    total_proposed = sum(len(component.proposed) for component in components)
    component_cards = "\n".join(_component_html(component) for component in components)
    toc = "\n".join(
        f'<a href="#{_slug(component.title)}">{html.escape(component.title)}</a>'
        for component in components
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codex Prompt Minimal Replacement</title>
  <style>
    :root {{
      --paper: #fbfaf7;
      --ink: #20201d;
      --muted: #6d6a62;
      --line: #d9d4ca;
      --panel: #fffefa;
      --accent: #9b4b2f;
      --soft: #efe8dc;
      --green: #496f5d;
    }}
    html {{ background: var(--paper); color: var(--ink); }}
    body {{
      margin: 0;
      font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1540px; margin: 0 auto; padding: 36px 32px 56px; }}
    h1 {{
      margin: 0 0 10px;
      font: 600 30px/1.1 ui-serif, Georgia, "Times New Roman", serif;
      letter-spacing: 0;
    }}
    .deck {{ max-width: 1040px; margin: 0 0 22px; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin: 22px 0 20px;
    }}
    .stat {{
      border-top: 2px solid var(--line);
      padding-top: 8px;
      min-height: 54px;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .value {{ display: block; margin-top: 2px; font-size: 20px; font-weight: 650; }}
    .note {{
      border-left: 4px solid var(--accent);
      background: var(--soft);
      padding: 12px 14px;
      margin: 0 0 20px;
      max-width: 1040px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 0 0 28px;
      max-width: 1100px;
    }}
    nav a {{ color: var(--green); text-decoration: none; border-bottom: 1px solid color-mix(in srgb, var(--green), transparent 55%); }}
    article {{ margin: 0 0 30px; }}
    .component-title {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
      margin: 0 0 8px;
    }}
    h2 {{ margin: 0; font-size: 18px; line-height: 1.25; }}
    .source {{ color: var(--muted); font-size: 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(340px, .66fr);
      gap: 18px;
      align-items: start;
    }}
    section {{
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: baseline;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 94%, white);
      padding: 12px 14px;
    }}
    h3 {{ margin: 0; font-size: 14px; line-height: 1.2; }}
    .meta {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
    pre {{
      margin: 0;
      padding: 16px 18px 22px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12.5px/1.48 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      tab-size: 2;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    .component-note {{ color: var(--muted); max-width: 1040px; margin: 0 0 10px; }}
    @media (max-width: 960px) {{
      main {{ padding: 24px 16px 42px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      header {{ position: static; }}
      .component-title {{ display: block; }}
      .source {{ margin-top: 4px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Codex Prompt Surfaces vs Minimal Replacements</h1>
  <p class="deck">Original model-visible prompt surfaces from <code>{html.escape(str(SOURCE.relative_to(ROOT)))}</code>, paired with proposed compressed replacements. This artifact is a design map only; it does not change harness behavior by itself.</p>

  <div class="stats">
    <div class="stat"><span class="label">Components</span><span class="value">{len(components)}</span></div>
    <div class="stat"><span class="label">Original Chars</span><span class="value">{total_original:,}</span></div>
    <div class="stat"><span class="label">Proposed Chars</span><span class="value">{total_proposed:,}</span></div>
    <div class="stat"><span class="label">Reduction</span><span class="value">{_pct_reduction(total_original, total_proposed)}</span></div>
  </div>

  <p class="note">The main system prompt is already reduced in <code>{html.escape(str(MINIMAL_SOURCE.relative_to(ROOT)))}</code>. The other rows show additional places where Codex-shaped prompt text still reaches the model.</p>
  <nav>{toc}</nav>

  {component_cards}
</main>
</body>
</html>
"""


def _component_html(component: PromptComponent) -> str:
    original_lines = len(component.original.splitlines())
    proposed_lines = len(component.proposed.splitlines())
    return f"""<article id="{_slug(component.title)}">
  <div class="component-title">
    <h2>{html.escape(component.title)}</h2>
    <div class="source">{html.escape(component.source)}</div>
  </div>
  <p class="component-note">{html.escape(component.note)}</p>
  <div class="grid">
    <section>
      <header>
        <h3>Original</h3>
        <span class="meta">{original_lines} lines · {len(component.original):,} chars</span>
      </header>
      <pre>{html.escape(component.original)}</pre>
    </section>
    <section>
      <header>
        <h3>Proposed Minimal</h3>
        <span class="meta">{proposed_lines} lines · {len(component.proposed):,} chars</span>
      </header>
      <pre>{html.escape(component.proposed)}</pre>
    </section>
  </div>
</article>"""


def _literal_constant(path: Path, name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise TypeError(f"{name} is not a string")
        return value
    raise ValueError(f"{name} not found")


def _load_full_harness() -> Any:
    module_name = "_codex_prompt_compare_full_harness"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _pct_reduction(original: int, proposed: int) -> str:
    if original <= 0:
        return "n/a"
    return f"{(1 - proposed / original) * 100:.1f}%"


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


if __name__ == "__main__":
    main()
