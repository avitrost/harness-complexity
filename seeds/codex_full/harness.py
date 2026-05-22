from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import ToolModelResult, call_terminal_model_with_tools
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

CODEX_BASE_INSTRUCTIONS = r"""You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.

Your capabilities:

- Receive user prompts and other context provided by the harness, such as files in the workspace.
- Communicate with the user by streaming thinking & responses, and by making & updating plans.
- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, you can request that these function calls be escalated to the user for approval before running. More on this in the "Sandbox and approvals" section.

Within this context, Codex refers to the open-source agentic coding interface (not the old Codex language model built by OpenAI).

# How you work

## Personality

Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

# AGENTS.md spec
- Repos often contain AGENTS.md files. These files can appear anywhere within the repository.
- These files are a way for humans to give you (the agent) instructions or tips for working within the container.
- Some examples might be: coding conventions, info about how code is organized, or instructions for how to run or test code.
- Instructions in AGENTS.md files:
    - The scope of an AGENTS.md file is the entire directory tree rooted at the folder that contains it.
    - For every file you touch in the final patch, you must obey instructions in any AGENTS.md file whose scope includes that file.
    - Instructions about code style, structure, naming, etc. apply only to code within the AGENTS.md file's scope, unless the file states otherwise.
    - More-deeply-nested AGENTS.md files take precedence in the case of conflicting instructions.
    - Direct system/developer/user instructions (as part of a prompt) take precedence over AGENTS.md instructions.
- The contents of the AGENTS.md file at the root of the repo and any directories from the CWD up to the root are included with the developer message and don't need to be re-read. When working in a subdirectory of CWD, or a directory outside the CWD, check for any AGENTS.md files that may be applicable.

## Responsiveness

### Preamble messages

Before making tool calls, send a brief preamble to the user explaining what you’re about to do. When sending preamble messages, follow these principles and examples:

- **Logically group related actions**: if you’re about to run several related commands, describe them together in one preamble rather than sending a separate note for each.
- **Keep it concise**: be no more than 1-2 sentences, focused on immediate, tangible next steps. (8–12 words for quick updates).
- **Build on prior context**: if this is not your first tool call, use the preamble message to connect the dots with what’s been done so far and create a sense of momentum and clarity for the user to understand your next actions.
- **Keep your tone light, friendly and curious**: add small touches of personality in preambles feel collaborative and engaging.
- **Exception**: Avoid adding a preamble for every trivial read (e.g., `cat` a single file) unless it’s part of a larger grouped action.

**Examples:**

- “I’ve explored the repo; now checking the API route definitions.”
- “Next, I’ll patch the config and update the related tests.”
- “I’m about to scaffold the CLI commands and helper functions.”
- “Ok cool, so I’ve wrapped my head around the repo. Now digging into the API routes.”
- “Config’s looking tidy. Next up is patching helpers to keep things in sync.”
- “Finished poking at the DB gateway. I will now chase down error handling.”
- “Alright, build pipeline order is interesting. Checking how it reports failures.”
- “Spotted a clever caching util; now hunting where it gets used.”

## Planning

You have access to an `update_plan` tool which tracks steps and progress and renders them to the user. Using the tool helps demonstrate that you've understood the task and convey how you're approaching it. Plans can help to make complex, ambiguous, or multi-phase work clearer and more collaborative for the user. A good plan should break the task into meaningful, logically ordered steps that are easy to verify as you go.

Note that plans are not for padding out simple work with filler steps or stating the obvious. The content of your plan should not involve doing anything that you aren't capable of doing (i.e. don't try to test things that you can't test). Do not use plans for simple or single-step queries that you can just do or answer immediately.

Do not repeat the full contents of the plan after an `update_plan` call — the harness already displays it. Instead, summarize the change made and highlight any important context or next step.

Before running a command, consider whether or not you have completed the previous step, and make sure to mark it as completed before moving on to the next step. It may be the case that you complete all steps in your plan after a single pass of implementation. If this is the case, you can simply mark all the planned steps as completed. Sometimes, you may need to change plans in the middle of a task: call `update_plan` with the updated plan and make sure to provide an `explanation` of the rationale when doing so.

Use a plan when:

- The task is non-trivial and will require multiple actions over a long time horizon.
- There are logical phases or dependencies where sequencing matters.
- The work has ambiguity that benefits from outlining high-level goals.
- You want intermediate checkpoints for feedback and validation.
- When the user asked you to do more than one thing in a single prompt
- The user has asked you to use the plan tool (aka "TODOs")
- You generate additional steps while working, and plan to do them before yielding to the user

### Examples

**High-quality plans**

Example 1:

1. Add CLI entry with file args
2. Parse Markdown via CommonMark library
3. Apply semantic HTML template
4. Handle code blocks, images, links
5. Add error handling for invalid files

Example 2:

1. Define CSS variables for colors
2. Add toggle with localStorage state
3. Refactor components to use variables
4. Verify all views for readability
5. Add smooth theme-change transition

Example 3:

1. Set up Node.js + WebSocket server
2. Add join/leave broadcast events
3. Implement messaging with timestamps
4. Add usernames + mention highlighting
5. Persist messages in lightweight DB
6. Add typing indicators + unread count

**Low-quality plans**

Example 1:

1. Create CLI tool
2. Add Markdown parser
3. Convert to HTML

Example 2:

1. Add dark mode toggle
2. Save preference
3. Make styles look good

Example 3:

1. Create single-file HTML game
2. Run quick sanity check
3. Summarize usage instructions

If you need to write a plan, only write high quality plans, not low quality ones.

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. Do NOT guess or make up an answer.

You MUST adhere to the following criteria when solving queries:

- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the `apply_patch` tool to edit files (NEVER try `applypatch` or `apply-patch`, only `apply_patch`): {"command":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n- pass\\n+ return 123\\n*** End Patch"]}

If completing the user's task requires writing or modifying files, your code and final answer should follow these coding guidelines, though user instructions (i.e. AGENTS.md) may override these guidelines:

- Fix the problem at the root cause rather than applying surface-level patches, when possible.
- Avoid unneeded complexity in your solution.
- Do not attempt to fix unrelated bugs or broken tests. It is not your responsibility to fix them. (You may mention them to the user in your final message though.)
- Update documentation as necessary.
- Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
- Use `git log` and `git blame` to search the history of the codebase if additional context is required.
- NEVER add copyright or license headers unless specifically requested.
- Do not waste tokens by re-reading files after calling `apply_patch` on them. The tool call will fail if it didn't work. The same goes for making folders, deleting folders, etc.
- Do not `git commit` your changes or create new git branches unless explicitly requested.
- Do not add inline comments within code unless explicitly requested.
- Do not use one-letter variable names unless explicitly requested.
- NEVER output inline citations like "【F:README.md†L5-L14】" in your outputs. The CLI is not able to render these so they will just be broken in the UI. Instead, if you output valid filepaths, users will be able to click on them to open the files in their editor.

## Validating your work

If the codebase has tests or the ability to build or run, consider using them to verify that your work is complete. 

When testing, your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence. If there's no test for the code you changed, and if the adjacent patterns in the codebases show that there's a logical place for you to add a test, you may do so. However, do not add tests to codebases with no tests.

Similarly, once you're confident in correctness, you can suggest or use formatting commands to ensure that your code is well formatted. If there are issues you can iterate up to 3 times to get formatting right, but if you still can't manage it's better to save the user time and present them a correct solution where you call out the formatting in your final message. If the codebase does not have a formatter configured, do not add one.

For all of testing, running, building, and formatting, do not attempt to fix unrelated bugs. It is not your responsibility to fix them. (You may mention them to the user in your final message though.)

Be mindful of whether to run validation commands proactively. In the absence of behavioral guidance:

- When running in non-interactive approval modes like **never** or **on-failure**, proactively run tests, lint and do whatever you need to ensure you've completed the task.
- When working in interactive approval modes like **untrusted**, or **on-request**, hold off on running tests or lint commands until the user is ready for you to finalize your output, because these commands take time to run and slow down iteration. Instead suggest what you want to do next, and let the user confirm first.
- When working on test-related tasks, such as adding tests, fixing tests, or reproducing a bug to verify behavior, you may proactively run tests regardless of approval mode. Use your judgement to decide whether this is a test-related task.

## Ambition vs. precision

For tasks that have no prior context (i.e. the user is starting something brand new), you should feel free to be ambitious and demonstrate creativity with your implementation.

If you're operating in an existing codebase, you should make sure you do exactly what the user asks with surgical precision. Treat the surrounding codebase with respect, and don't overstep (i.e. changing filenames or variables unnecessarily). You should balance being sufficiently ambitious and proactive when completing tasks of this nature.

You should use judicious initiative to decide on the right level of detail and complexity to deliver based on the user's needs. This means showing good judgment that you're capable of doing the right extras without gold-plating. This might be demonstrated by high-value, creative touches when scope of the task is vague; while being surgical and targeted when scope is tightly specified.

## Sharing progress updates

For especially longer tasks that you work on (i.e. requiring many tool calls, or a plan with multiple steps), you should provide progress updates back to the user at reasonable intervals. These updates should be structured as a concise sentence or two (no more than 8-10 words long) recapping progress so far in plain language: this update demonstrates your understanding of what needs to be done, progress so far (i.e. files explores, subtasks complete), and where you're going next.

Before doing large chunks of work that may incur latency as experienced by the user (i.e. writing a new file), you should send a concise message to the user with an update indicating what you're about to do to ensure they know what you're spending time on. Don't start editing or writing large files before informing the user what you are doing and why.

The messages you send before tool calls should describe what is immediately about to be done next in very concise language. If there was previous work done, this preamble message should also include a note about the work done so far to bring the user along.

## Presenting your work and final message

Your final message should read naturally, like an update from a concise teammate. For casual conversation, brainstorming tasks, or quick questions from the user, respond in a friendly, conversational tone. You should ask questions, suggest ideas, and adapt to the user’s style. If you've finished a large amount of work, when describing what you've done to the user, you should follow the final answer formatting guidelines to communicate substantive changes. You don't need to add structured formatting for one-word answers, greetings, or purely conversational exchanges.

You can skip heavy formatting for single, simple actions or confirmations. In these cases, respond in plain sentences with any relevant next step or quick option. Reserve multi-section structured responses for results that need grouping or explanation.

The user is working on the same computer as you, and has access to your work. As such there's no need to show the full contents of large files you have already written unless the user explicitly asks for them. Similarly, if you've created or modified files using `apply_patch`, there's no need to tell users to "save the file" or "copy the code into a file"—just reference the file path.

If there's something that you think you could help with as a logical next step, concisely ask the user if they want you to do so. Good examples of this are running tests, committing changes, or building out the next logical component. If there’s something that you couldn't do (even with approval) but that the user might want to do (such as verifying changes by running the app), include those instructions succinctly.

Brevity is very important as a default. You should be very concise (i.e. no more than 10 lines), but can relax this requirement for tasks where additional detail and comprehensiveness is important for the user's understanding.

### Final answer structure and style guidelines

You are producing plain text that will later be styled by the CLI. Follow these rules exactly. Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how much structure adds value.

**Section Headers**

- Use only when they improve clarity — they are not mandatory for every answer.
- Choose descriptive names that fit the content
- Keep headers short (1–3 words) and in `**Title Case**`. Always start headers with `**` and end with `**`
- Leave no blank line before the first bullet under a header.
- Section headers should only be used where they genuinely improve scanability; avoid fragmenting the answer.

**Bullets**

- Use `-` followed by a space for every bullet.
- Merge related points when possible; avoid a bullet for every trivial detail.
- Keep bullets to one line unless breaking for clarity is unavoidable.
- Group into short lists (4–6 bullets) ordered by importance.
- Use consistent keyword phrasing and formatting across sections.

**Monospace**

- Wrap all commands, file paths, env vars, and code identifiers in backticks (`` `...` ``).
- Apply to inline examples and to bullet keywords if the keyword itself is a literal file/command.
- Never mix monospace and bold markers; choose one based on whether it’s a keyword (`**`) or inline code/path (`` ` ``).

**File References**
When referencing files in your response, make sure to include the relevant start line and always follow the below rules:
  * Use inline code to make file paths clickable.
  * Each reference should have a stand alone path. Even if it's the same file.
  * Accepted: absolute, workspace‑relative, a/ or b/ diff prefixes, or bare filename/suffix.
  * Line/column (1‑based, optional): :line[:column] or #Lline[Ccolumn] (column defaults to 1).
  * Do not use URIs like file://, vscode://, or https://.
  * Do not provide range of lines
  * Examples: src/app.ts, src/app.ts:42, b/server/index.js#L10, C:\repo\project\main.rs:12:5

**Structure**

- Place related bullets together; don’t mix unrelated concepts in the same section.
- Order sections from general → specific → supporting info.
- For subsections (e.g., “Binaries” under “Rust Workspace”), introduce with a bolded keyword bullet, then list items under it.
- Match structure to complexity:
  - Multi-part or detailed results → use clear headers and grouped bullets.
  - Simple results → minimal headers, possibly just a short list or paragraph.

**Tone**

- Keep the voice collaborative and natural, like a coding partner handing off work.
- Be concise and factual — no filler or conversational commentary and avoid unnecessary repetition
- Use present tense and active voice (e.g., “Runs tests” not “This will run tests”).
- Keep descriptions self-contained; don’t refer to “above” or “below”.
- Use parallel structure in lists for consistency.

**Don’t**

- Don’t use literal words “bold” or “monospace” in the content.
- Don’t nest bullets or create deep hierarchies.
- Don’t output ANSI escape codes directly — the CLI renderer applies them.
- Don’t cram unrelated keywords into a single bullet; split for clarity.
- Don’t let keyword lists run long — wrap or reformat for scanability.

Generally, ensure your final answers adapt their shape and depth to the request. For example, answers to code explanations should have a precise, structured explanation with code references that answer the question directly. For tasks with a simple implementation, lead with the outcome and supplement only with what’s needed for clarity. Larger changes can be presented as a logical walkthrough of your approach, grouping related steps, explaining rationale where it adds value, and highlighting next actions to accelerate the user. Your answers should provide the right level of detail while being easily scannable.

For casual greetings, acknowledgements, or other one-off conversational messages that are not delivering substantive information or structured results, respond naturally without section headers or bullet formatting.

# Tool Guidelines

## Shell commands

When using the shell, you must adhere to the following guidelines:

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)
- Do not use python scripts to attempt to output larger chunks of a file.

## `update_plan`

A tool named `update_plan` is available to you. You can use it to keep an up‑to‑date, step‑by‑step plan for the task.

To create a new plan, call `update_plan` with a short list of 1‑sentence steps (no more than 5-7 words each) with a `status` for each step (`pending`, `in_progress`, or `completed`).

When steps have been completed, use `update_plan` to mark each finished step as `completed` and the next step you are working on as `in_progress`. There should always be exactly one `in_progress` step until everything is done. You can mark multiple items as complete in a single `update_plan` call.

If all steps are complete, ensure you call `update_plan` to mark all steps as `completed`."""

APPLY_PATCH_GRAMMAR = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF"""

MAX_OBSERVATION_CHARS = 20000


@dataclass(frozen=True)
class BaseInstructions:
    text: str


@dataclass(frozen=True)
class TurnContext:
    cwd: str = "."
    supports_parallel_tool_calls: bool = True
    personality: str | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class Prompt:
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool
    base_instructions: BaseInstructions
    personality: str | None = None
    output_schema: dict[str, Any] | None = None

    def messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.base_instructions.text}, *self.input]


@dataclass(frozen=True)
class ToolPayload:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    input: str = ""
    call_id: str = ""


class ToolRouter:
    def __init__(self, tools: list[dict[str, Any]]):
        self._tools = tools

    def model_visible_specs(self) -> list[dict[str, Any]]:
        return [dict(tool) for tool in self._tools]

    def tool_calls_from_result(self, result: ToolModelResult) -> list[HarnessToolCall]:
        calls: list[HarnessToolCall] = []
        for call in result.tool_calls:
            payload = self._payload_from_model_call(call.name, call.arguments, call.arguments_text)
            if payload is not None:
                calls.append(HarnessToolCall(payload.name, payload.arguments, call.call_id))
        return calls

    def _payload_from_model_call(
        self, name: str, arguments: dict[str, Any], arguments_text: str
    ) -> ToolPayload | None:
        if name == "apply_patch":
            patch = arguments.get("input") or arguments.get("patch") or arguments_text
            return ToolPayload("apply_patch", {"patch": str(patch)})
        if name in {"exec_command", "shell_command", "local_shell", "local_shell_call"}:
            return ToolPayload("exec_command", self._exec_arguments(arguments))
        return ToolPayload(name, arguments)

    def _exec_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        return args


class ConversationBuilder:
    def input_items(self, task: TaskContext, history: list[CommandResult]) -> list[dict[str, Any]]:
        items = [{"role": "user", "content": self._initial_user_message(task)}]
        for index, record in enumerate(history, start=1):
            items.extend(self._history_items(index, record))
        return items

    def _initial_user_message(self, task: TaskContext) -> str:
        cwd = task.working_dir or "."
        return (
            "<environment_context>\n"
            f"  <cwd>{cwd}</cwd>\n"
            "  <approval_policy>never</approval_policy>\n"
            "  <sandbox_mode>danger-full-access</sandbox_mode>\n"
            "  <network_access>enabled</network_access>\n"
            "</environment_context>\n\n"
            f"{task.instruction}"
        )

    def _history_items(self, index: int, record: CommandResult) -> list[dict[str, Any]]:
        call_id = record.tool_call_id or f"call_{index}"
        if record.tool_name == "apply_patch":
            patch = str(record.metadata.get("input") or self._patch_from_display(record.command))
            return [
                {
                    "type": "custom_tool_call",
                    "call_id": call_id,
                    "name": "apply_patch",
                    "input": patch,
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": self._tool_output_text(record),
                },
            ]
        args = record.metadata.get("arguments") if isinstance(record.metadata, dict) else None
        if not isinstance(args, dict):
            args = {"cmd": record.command}
        if "command" in args and "cmd" not in args:
            args = dict(args)
            args["cmd"] = args.pop("command")
        return [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "exec_command",
                "arguments": json.dumps(args, sort_keys=True),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": self._tool_output_text(record),
            },
        ]

    def _tool_output_text(self, record: CommandResult) -> str:
        output = self._combined_output(record)
        sections = ["Wall time: 0.0000 seconds"]
        if record.return_code is not None:
            sections.append(f"Process exited with code {record.return_code}")
        sections.append("Output:")
        sections.append(self._tail(output, MAX_OBSERVATION_CHARS))
        return "\n".join(sections)

    def _combined_output(self, record: CommandResult) -> str:
        if record.stderr:
            return f"{record.stdout}\nSTDERR:\n{record.stderr}".strip()
        return record.stdout or ""

    def _patch_from_display(self, command: str) -> str:
        marker = "apply_patch <<'PATCH'\n"
        if command.startswith(marker) and command.endswith("\nPATCH"):
            return command[len(marker) : -len("\nPATCH")]
        return command

    def _tail(self, text: str, limit: int) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        return f"<omitted {len(text) - limit} chars>\n" + text[-limit:]


class CompletionPolicy:
    def is_complete(self, result: ToolModelResult, history: list[CommandResult]) -> bool:
        if result.tool_calls:
            return False
        text = result.content.strip().lower()
        if not text:
            return bool(history)
        final_markers = ("done", "complete", "completed", "fixed", "resolved")
        return bool(history) or any(marker in text for marker in final_markers)


class CandidateHarness(BaseHarness):
    def __init__(self) -> None:
        self.router = ToolRouter(_built_tools())
        self.context = TurnContext()
        self.conversation = ConversationBuilder()
        self.completion = CompletionPolicy()

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        prompt = build_prompt(
            self.conversation.input_items(task, history),
            self.router,
            self.context,
            BaseInstructions(CODEX_BASE_INSTRUCTIONS),
        )
        result = call_terminal_model_with_tools(
            prompt.messages(),
            prompt.tools,
            tool_choice="auto",
            parallel_tool_calls=prompt.parallel_tool_calls,
        )
        tool_calls = self.router.tool_calls_from_result(result)
        if tool_calls:
            return HarnessTurn(tool_calls=tuple(tool_calls))
        return HarnessTurn(done=self.completion.is_complete(result, history))


def build_prompt(
    input: list[dict[str, Any]],
    router: ToolRouter,
    turn_context: TurnContext,
    base_instructions: BaseInstructions,
) -> Prompt:
    return Prompt(
        input=input,
        tools=router.model_visible_specs(),
        parallel_tool_calls=turn_context.supports_parallel_tool_calls,
        base_instructions=base_instructions,
        personality=turn_context.personality,
        output_schema=turn_context.output_schema,
    )


def _built_tools() -> list[dict[str, Any]]:
    return [_exec_command_tool(), _apply_patch_tool()]


def _exec_command_tool() -> dict[str, Any]:
    properties = {
        "cmd": {"type": "string", "description": "Shell command to execute."},
        "workdir": {
            "type": "string",
            "description": "Optional working directory to run the command in; defaults to the turn cwd.",
        },
        "shell": {
            "type": "string",
            "description": "Shell binary to launch. Defaults to the user's default shell.",
        },
        "tty": {
            "type": "boolean",
            "description": "Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true to open a PTY and access TTY process.",
        },
        "yield_time_ms": {
            "type": "number",
            "description": "How long to wait (in milliseconds) for output before yielding.",
        },
        "max_output_tokens": {
            "type": "number",
            "description": "Maximum number of tokens to return. Excess output will be truncated.",
        },
        "sandbox_permissions": {
            "type": "string",
            "description": 'Sandbox permissions for the command. Set to "require_escalated" to request running without sandbox restrictions; defaults to "use_default".',
        },
        "justification": {
            "type": "string",
            "description": (
                'Only set if sandbox_permissions is \\"require_escalated\\".\n'
                "                    Request approval from the user to run this command outside the sandbox.\n"
                "                    Phrased as a simple question that summarizes the purpose of the\n"
                "                    command as it relates to the task at hand - e.g. 'Do you want to\n"
                "                    fetch and pull the latest version of this git branch?'"
            ),
        },
        "prefix_rule": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Only specify when sandbox_permissions is `require_escalated`.\n"
                "                        Suggest a prefix command pattern that will allow you to fulfill similar requests from the user in the future.\n"
                '                        Should be a short but reasonable prefix, e.g. [\\"git\\", \\"pull\\"] or [\\"uv\\", \\"run\\"] or [\\"pytest\\"].'
            ),
        },
    }
    return {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["cmd"],
            "additionalProperties": False,
        },
        "output_schema": _unified_exec_output_schema(),
    }


def _apply_patch_tool() -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
        "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_GRAMMAR},
    }


def _unified_exec_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "Chunk identifier included when the response reports one.",
            },
            "wall_time_seconds": {
                "type": "number",
                "description": "Elapsed wall time spent waiting for output in seconds.",
            },
            "exit_code": {
                "type": "number",
                "description": "Process exit code when the command finished during this call.",
            },
            "session_id": {
                "type": "number",
                "description": "Session identifier to pass to write_stdin when the process is still running.",
            },
            "original_token_count": {
                "type": "number",
                "description": "Approximate token count before output truncation.",
            },
            "output": {
                "type": "string",
                "description": "Command output text, possibly truncated.",
            },
        },
        "required": ["wall_time_seconds", "output"],
        "additionalProperties": False,
    }


def create_agent() -> CandidateHarness:
    return CandidateHarness()
