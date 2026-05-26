from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import (
    ToolModelResult,
    call_terminal_model,
    call_terminal_model_with_tools,
)
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

SUMMARIZATION_PROMPT = r"""You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

SUMMARY_PREFIX = r"""Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"""

CODEX_UPSTREAM_COMMIT = "9f42c89c0112771dc29100a6f3fc904049b2655f"
CODEX_UPSTREAM_DATE = "2026-05-24"
MAX_OBSERVATION_CHARS = 20000
MAX_FUNCTION_OUTPUT_CHARS = 24000
MAX_CONTEXT_HISTORY_ITEMS = 96
MAX_CONTEXT_HISTORY_CHARS = 90000
MAX_RAW_RESPONSE_ITEMS = 48
COMPACT_USER_MESSAGE_MAX_TOKENS = 20000
FUNCTION_HISTORY_TOOLS = {"exec_command", "write_stdin", "update_plan"}
FUNCTION_CALL_TYPES = {"function_call", "local_shell_call"}
FUNCTION_OUTPUT_TYPES = {"function_call_output"}
CUSTOM_CALL_TYPES = {"custom_tool_call"}
CUSTOM_OUTPUT_TYPES = {"custom_tool_call_output"}
TOOL_OUTPUT_TYPES = FUNCTION_OUTPUT_TYPES | CUSTOM_OUTPUT_TYPES | {"tool_search_output"}
TOOL_CALL_TYPES = FUNCTION_CALL_TYPES | CUSTOM_CALL_TYPES | {"tool_search_call"}
SHELL_TOOL_NAMES = {"exec_command", "shell_command", "local_shell", "local_shell_call"}
PERMISSIONS_SANDBOX_DANGER_FULL_ACCESS = (
    "Filesystem sandboxing defines which files can be read or written. "
    "`sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all commands "
    "are permitted. Network access is enabled."
)
PERMISSIONS_APPROVAL_NEVER = (
    "Approval policy is currently never. Do not provide the `sandbox_permissions` "
    "for any reason, commands will be rejected."
)


# === 01. Port-Parity Manifest ===

PORT_PARITY_MANIFEST: tuple[dict[str, str], ...] = (
    {
        "upstream": "codex-rs/core/src/session/turn.rs::build_prompt",
        "status": "included",
        "python": "build_prompt",
        "notes": "Responses input, tool specs, parallel flag, and base instructions are preserved.",
    },
    {
        "upstream": "codex-rs/core/src/session/turn.rs::run_sampling_request",
        "status": "simplified",
        "python": "CandidateHarness.next_command",
        "notes": "Harbor owns the outer loop; each next_command reconstructs the Codex transcript.",
    },
    {
        "upstream": "codex-rs/core/src/tools/router.rs::ToolRouter::build_tool_call",
        "status": "included",
        "python": "ToolRouter.build_tool_call",
        "notes": "Function and custom tool calls are decoded from Responses items.",
    },
    {
        "upstream": "codex-rs/core/src/tools/parallel.rs::ToolCallRuntime",
        "status": "simplified",
        "python": "ToolRouter.tool_calls_from_result",
        "notes": "Parallel tool call ordering is preserved; Harbor executes returned calls concurrently.",
    },
    {
        "upstream": "codex-rs/core/src/tools/handlers/shell_spec.rs",
        "status": "included",
        "python": "_exec_command_tool and _write_stdin_tool",
        "notes": "Current unified exec tool schema and output schema are copied into Python dicts.",
    },
    {
        "upstream": "codex-rs/core/src/tools/handlers/apply_patch_spec.rs",
        "status": "included",
        "python": "_apply_patch_tool",
        "notes": "Freeform Lark grammar is pinned in APPLY_PATCH_GRAMMAR.",
    },
    {
        "upstream": "codex-rs/core/src/tools/context.rs::ExecCommandToolOutput",
        "status": "included",
        "python": "ToolOutputFormatter.unified_exec_text",
        "notes": "Model-visible unified exec output headings and ordering match Codex.",
    },
    {
        "upstream": "codex-rs/core/src/context_manager/history.rs",
        "status": "simplified",
        "python": "ContextManager",
        "notes": "Pair normalization and budget fallback are retained around compact.rs-style memento compaction.",
    },
    {
        "upstream": "codex-rs/core/src/compact.rs::run_inline_auto_compact_task",
        "status": "simplified",
        "python": "ContextCompactor",
        "notes": "Exact compact prompt/prefix and replacement-history shape are used without hooks or remote compaction.",
    },
    {
        "upstream": "codex-rs/core/src/context_manager/normalize.rs",
        "status": "included",
        "python": "ConversationNormalizer",
        "notes": "Missing outputs are synthesized and orphan outputs are removed with matching pairs.",
    },
    {
        "upstream": "codex-rs/core/src/agents_md.rs",
        "status": "simplified",
        "python": "AgentInstructionsRenderer",
        "notes": "Harbor collects AGENTS.md; the harness renders hierarchical context fragments.",
    },
    {
        "upstream": "codex-rs/core/src/context/environment_context.rs",
        "status": "included",
        "python": "InitialContextBuilder.environment_context",
        "notes": "Single-environment cwd, shell, date, and timezone fragment follows Codex format.",
    },
    {
        "upstream": "codex-rs/core/src/context/permissions_instructions.rs",
        "status": "included",
        "python": "PermissionsInstructionsRenderer",
        "notes": "Danger-full-access plus never-approval developer fragment is rendered exactly for TB2.",
    },
    {
        "upstream": "codex-rs/core/src/guardian and sandboxing",
        "status": "simplified",
        "python": "ExecutionPolicy",
        "notes": "TB2 runs noninteractively; approval/sandbox handling is represented in prompts and metadata.",
    },
    {
        "upstream": "TUI, MCP, plugins, apps, telemetry, realtime, cloud persistence",
        "status": "omitted",
        "python": "PORT_PARITY_MANIFEST",
        "notes": "Product surfaces not needed by a TB2 noninteractive harness are intentionally absent.",
    },
)


# === 02. Feature Switches ===

ENABLE_PORT_PARITY_MANIFEST = True
ENABLE_HISTORY_REPLAY = True
ENABLE_CONTEXT_MANAGER = True
ENABLE_CONTEXT_NORMALIZATION = True
ENABLE_CONTEXT_BUDGETING = True
ENABLE_MODEL_CONTEXT_COMPACTION = True
ENABLE_PATCH_TOOL = True
ENABLE_PLAN_TOOL = True
ENABLE_WRITE_STDIN_TOOL = True
ENABLE_UNIFIED_EXEC_OUTPUT_FORMAT = True
ENABLE_MODEL_RESPONSE_ITEM_REPLAY = True
ENABLE_MODEL_CALL_RESILIENCE = True
ENABLE_RECOVERY_POLICY = True
ENABLE_COMMAND_CLASSIFICATION = True
ENABLE_COMPLETION_POLICY = True
ENABLE_INSTRUMENTATION = True


@dataclass(frozen=True)
class FeatureSet:
    port_parity_manifest: bool = True
    history_replay: bool = True
    context_manager: bool = True
    context_normalization: bool = True
    context_budgeting: bool = True
    model_context_compaction: bool = True
    patch_tool: bool = True
    plan_tool: bool = True
    write_stdin_tool: bool = True
    unified_exec_output_format: bool = True
    model_response_item_replay: bool = True
    model_call_resilience: bool = True
    recovery_policy: bool = True
    command_classification: bool = True
    completion_policy: bool = True
    instrumentation: bool = True

    @classmethod
    def from_globals(cls) -> "FeatureSet":
        return cls(
            port_parity_manifest=ENABLE_PORT_PARITY_MANIFEST,
            history_replay=ENABLE_HISTORY_REPLAY,
            context_manager=ENABLE_CONTEXT_MANAGER,
            context_normalization=ENABLE_CONTEXT_NORMALIZATION,
            context_budgeting=ENABLE_CONTEXT_BUDGETING,
            model_context_compaction=ENABLE_MODEL_CONTEXT_COMPACTION,
            patch_tool=ENABLE_PATCH_TOOL,
            plan_tool=ENABLE_PLAN_TOOL,
            write_stdin_tool=ENABLE_WRITE_STDIN_TOOL,
            unified_exec_output_format=ENABLE_UNIFIED_EXEC_OUTPUT_FORMAT,
            model_response_item_replay=ENABLE_MODEL_RESPONSE_ITEM_REPLAY,
            model_call_resilience=ENABLE_MODEL_CALL_RESILIENCE,
            recovery_policy=ENABLE_RECOVERY_POLICY,
            command_classification=ENABLE_COMMAND_CLASSIFICATION,
            completion_policy=ENABLE_COMPLETION_POLICY,
            instrumentation=ENABLE_INSTRUMENTATION,
        )

    def with_overrides(self, overrides: dict[str, bool]) -> "FeatureSet":
        return replace(self, **overrides)


PROFILE_OVERRIDES: dict[str, dict[str, bool]] = {
    "codex_full": {},
    "no_instrumentation": {
        "instrumentation": False,
        "port_parity_manifest": False,
    },
    "no_classifier": {"command_classification": False},
    "no_recovery": {"recovery_policy": False},
    "no_compaction": {"model_context_compaction": False},
    "exec_only_tools": {
        "patch_tool": False,
        "plan_tool": False,
        "write_stdin_tool": False,
    },
    "minimal_loop": {
        "history_replay": False,
        "context_manager": False,
        "context_normalization": False,
        "context_budgeting": False,
        "model_context_compaction": False,
        "patch_tool": False,
        "plan_tool": False,
        "write_stdin_tool": False,
        "unified_exec_output_format": False,
        "model_response_item_replay": False,
        "model_call_resilience": False,
        "recovery_policy": False,
        "command_classification": False,
        "port_parity_manifest": False,
        "instrumentation": False,
    },
}
DEFAULT_PROFILE_NAME = "codex_full"


def resolve_features(profile: str | FeatureSet | None = None) -> FeatureSet:
    if isinstance(profile, FeatureSet):
        return profile
    features = FeatureSet.from_globals()
    profile_name = profile or os.getenv("CODEX_HARNESS_PROFILE") or DEFAULT_PROFILE_NAME
    if profile_name not in PROFILE_OVERRIDES:
        expected = ", ".join(sorted(PROFILE_OVERRIDES))
        raise ValueError(
            f"unknown CODEX_HARNESS_PROFILE {profile_name!r}; expected one of: {expected}"
        )
    return features.with_overrides(PROFILE_OVERRIDES.get(profile_name, {}))


def _current_date() -> str:
    return os.getenv("CODEX_CURRENT_DATE") or date.today().isoformat()


def _construct(symbol_name: str, *args: Any) -> Any:
    return globals()[symbol_name](*args)


def _local_timezone_name() -> str:
    if timezone := os.getenv("TZ"):
        return timezone
    try:
        with open("/etc/timezone", encoding="utf-8") as timezone_file:
            timezone = timezone_file.read().strip()
        if timezone:
            return timezone
    except OSError:
        pass
    local_name = time.tzname[0] if time.tzname else ""
    if local_name in {"UTC", "GMT"}:
        return "Etc/UTC"
    return local_name or "Etc/UTC"


# === 03. Data Models ===


@dataclass(frozen=True)
class BaseInstructions:
    text: str


@dataclass(frozen=True)
class TurnEnvironment:
    cwd: str = "."
    shell: str = "bash"
    current_date: str = field(default_factory=_current_date)
    timezone: str = field(default_factory=_local_timezone_name)
    approval_policy: str = "never"
    sandbox_mode: str = "danger-full-access"
    network_access: str = "enabled"


@dataclass(frozen=True)
class TurnContext:
    cwd: str = "."
    supports_parallel_tool_calls: bool = True
    personality: str | None = None
    output_schema: dict[str, Any] | None = None
    environment: TurnEnvironment = field(default_factory=TurnEnvironment)


@dataclass(frozen=True)
class Prompt:
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool
    base_instructions: BaseInstructions
    developer_messages: list[dict[str, Any]] = field(default_factory=list)
    personality: str | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_strict: bool = True

    def messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.base_instructions.text},
            *self.developer_messages,
            *self.input,
        ]


@dataclass(frozen=True)
class ToolPayload:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    input: str = ""
    call_id: str = ""
    payload_type: str = "function"


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    call_id: str
    payload: ToolPayload


@dataclass(frozen=True)
class ContextStats:
    raw_items: int
    normalized_items: int
    pruned_items: int
    estimated_bytes: int
    estimated_tokens: int
    compacted: bool = False
    compaction_summary_chars: int = 0
    compaction_reused: bool = False


@dataclass(frozen=True)
class CodexPromptBundle:
    messages: list[dict[str, Any]]
    input_items: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stats: ContextStats


@dataclass(frozen=True)
class CommandAssessment:
    kind: str
    risky: bool = False
    long_running: bool = False
    needs_verification: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactionCheckpoint:
    prefix_digest: str
    prefix_len: int
    replacement: tuple[dict[str, Any], ...]
    summary_text: str


# === 04. Small Structured Helpers ===


class JsonSchema:
    @staticmethod
    def string(description: str | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "string"}
        if description is not None:
            schema["description"] = description
        return schema

    @staticmethod
    def number(description: str | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "number"}
        if description is not None:
            schema["description"] = description
        return schema

    @staticmethod
    def boolean(description: str | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "boolean"}
        if description is not None:
            schema["description"] = description
        return schema

    @staticmethod
    def array(items: dict[str, Any], description: str | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "array", "items": items}
        if description is not None:
            schema["description"] = description
        return schema

    @staticmethod
    def object(
        properties: dict[str, Any],
        required: list[str] | None = None,
        additional_properties: bool = False,
        description: str | None = None,
    ) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": additional_properties,
        }
        if required is not None:
            schema["required"] = required
        if description is not None:
            schema["description"] = description
        return schema


class StableJson:
    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def object_or_empty(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}


class TextBudget:
    @staticmethod
    def approx_token_count(text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

    @staticmethod
    def clip_tail(text: str, limit: int, marker: str = "omitted") -> str:
        text = text or ""
        if limit < 0 or len(text) <= limit:
            return text
        return f"<{marker} {len(text) - limit} chars>\n{text[-limit:]}"

    @staticmethod
    def clip_middle(text: str, limit: int) -> str:
        text = text or ""
        if limit < 0 or len(text) <= limit:
            return text
        head = max(0, limit // 3)
        tail = max(0, limit - head)
        omitted = len(text) - head - tail
        return f"{text[:head]}\n<omitted {omitted} chars>\n{text[-tail:]}"

    @staticmethod
    def item_bytes(item: dict[str, Any]) -> int:
        return len(StableJson.dumps(item).encode("utf-8", errors="replace"))


# === 05. Upstream-Like Tool Schemas ===


def _built_tools(features: FeatureSet | None = None) -> list[dict[str, Any]]:
    features = features or FeatureSet.from_globals()
    tools = [_exec_command_tool()]
    if features.write_stdin_tool:
        tools.append(_construct("_write_stdin_tool"))
    if features.plan_tool:
        tools.append(_construct("_update_plan_tool"))
    if features.patch_tool:
        tools.append(_construct("_apply_patch_tool"))
    return tools


def _exec_command_tool() -> dict[str, Any]:
    properties = {
        "cmd": JsonSchema.string("Shell command to execute."),
        "workdir": JsonSchema.string(
            "Optional working directory to run the command in; defaults to the turn cwd."
        ),
        "shell": JsonSchema.string("Shell binary to launch. Defaults to the user's default shell."),
        "login": JsonSchema.boolean(
            "Whether to run the shell with -l/-i semantics. Defaults to true."
        ),
        "tty": JsonSchema.boolean(
            "Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true to open a PTY and access TTY process."
        ),
        "yield_time_ms": JsonSchema.number(
            "How long to wait (in milliseconds) for output before yielding."
        ),
        "max_output_tokens": JsonSchema.number(
            "Maximum number of tokens to return. Excess output will be truncated."
        ),
        "sandbox_permissions": JsonSchema.string(
            'Sandbox permissions for the command. Set to "require_escalated" to request running without sandbox restrictions; defaults to "use_default".'
        ),
        "justification": JsonSchema.string(
            'Only set if sandbox_permissions is \\"require_escalated\\".\n'
            "                    Request approval from the user to run this command outside the sandbox.\n"
            "                    Phrased as a simple question that summarizes the purpose of the\n"
            "                    command as it relates to the task at hand - e.g. 'Do you want to\n"
            "                    fetch and pull the latest version of this git branch?'"
        ),
        "prefix_rule": JsonSchema.array(
            JsonSchema.string(),
            "Only specify when sandbox_permissions is `require_escalated`.\n"
            "                        Suggest a prefix command pattern that will allow you to fulfill similar requests from the user in the future.\n"
            '                        Should be a short but reasonable prefix, e.g. [\\"git\\", \\"pull\\"] or [\\"uv\\", \\"run\\"] or [\\"pytest\\"].',
        ),
    }
    return {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
        "strict": False,
        "parameters": JsonSchema.object(properties, ["cmd"], False),
        "output_schema": _unified_exec_output_schema(),
    }


def _write_stdin_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "write_stdin",
        "description": "Writes characters to an existing unified exec session and returns recent output.",
        "strict": False,
        "parameters": JsonSchema.object(
            {
                "session_id": JsonSchema.number("Identifier of the running unified exec session."),
                "chars": JsonSchema.string("Bytes to write to stdin (may be empty to poll)."),
                "yield_time_ms": JsonSchema.number(
                    "How long to wait (in milliseconds) for output before yielding."
                ),
                "max_output_tokens": JsonSchema.number(
                    "Maximum number of tokens to return. Excess output will be truncated."
                ),
            },
            ["session_id"],
            False,
        ),
        "output_schema": _unified_exec_output_schema(),
    }


def _update_plan_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_plan",
        "description": (
            "Updates the task plan.\n"
            "Provide an optional explanation and a list of plan items, each with a step and status.\n"
            "At most one step can be in_progress at a time.\n"
        ),
        "strict": False,
        "parameters": JsonSchema.object(
            {
                "explanation": JsonSchema.string(),
                "plan": JsonSchema.array(
                    JsonSchema.object(
                        {
                            "step": JsonSchema.string(),
                            "status": JsonSchema.string("One of: pending, in_progress, completed"),
                        },
                        ["step", "status"],
                        False,
                    ),
                    "The list of steps",
                ),
            },
            ["plan"],
            False,
        ),
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


# === 06. Tool Router and Response Parsing ===


class ToolRouter:
    def __init__(self, tools: list[dict[str, Any]]):
        self._tools = tools
        self._tool_names = {str(tool.get("name", "")) for tool in tools}

    def model_visible_specs(self) -> list[dict[str, Any]]:
        return [dict(tool) for tool in self._tools]

    def build_tool_call(self, item: dict[str, Any]) -> ToolCall | None:
        item_type = str(item.get("type") or "")
        if item_type == "function_call":
            name = str(item.get("name") or "")
            namespace = item.get("namespace")
            if namespace:
                name = f"{namespace}.{name}"
            arguments = self._arguments_from_item(item)
            payload = self._payload_from_model_call(
                name, arguments, str(item.get("arguments") or "")
            )
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        if item_type == "local_shell_call":
            arguments = self._local_shell_arguments(item)
            payload = self._payload_from_model_call(
                "local_shell", arguments, StableJson.dumps(arguments)
            )
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        if item_type == "custom_tool_call":
            name = str(item.get("name") or "")
            raw_input = item.get("input", item.get("arguments", ""))
            arguments = {"input": raw_input} if not isinstance(raw_input, dict) else dict(raw_input)
            payload = self._payload_from_model_call(name, arguments, str(raw_input or ""))
            if payload is None:
                return None
            return ToolCall(payload.name, str(item.get("call_id") or ""), payload)
        return None

    def tool_calls_from_result(self, result: ToolModelResult) -> list[HarnessToolCall]:
        calls: list[HarnessToolCall] = []
        seen: set[tuple[str, str]] = set()
        for call in result.tool_calls:
            payload = self._payload_from_model_call(call.name, call.arguments, call.arguments_text)
            if payload is None:
                continue
            key = (call.call_id, payload.name)
            if call.call_id and key in seen:
                continue
            seen.add(key)
            calls.append(HarnessToolCall(payload.name, payload.arguments, call.call_id))
        if calls:
            return calls
        for item in result.response_items:
            tool_call = self.build_tool_call(item)
            if tool_call is None:
                continue
            key = (tool_call.call_id, tool_call.tool_name)
            if tool_call.call_id and key in seen:
                continue
            seen.add(key)
            calls.append(
                HarnessToolCall(
                    tool_call.payload.name,
                    tool_call.payload.arguments,
                    tool_call.call_id,
                )
            )
        return calls

    def _payload_from_model_call(
        self, name: str, arguments: dict[str, Any], arguments_text: str
    ) -> ToolPayload | None:
        plain_name = name.rsplit(".", 1)[-1]
        if plain_name == "apply_patch":
            patch = self._patch_input(arguments, arguments_text)
            return ToolPayload("apply_patch", {"patch": patch}, patch, payload_type="custom")
        if plain_name == "write_stdin":
            return ToolPayload("write_stdin", self._write_stdin_arguments(arguments))
        if plain_name in {"update_plan", "plan"}:
            return ToolPayload("update_plan", self._plan_arguments(arguments))
        if plain_name in SHELL_TOOL_NAMES:
            return ToolPayload("exec_command", self._exec_arguments(arguments))
        if plain_name in self._tool_names:
            return ToolPayload(plain_name, dict(arguments))
        return ToolPayload(plain_name or name, dict(arguments))

    def _arguments_from_item(self, item: dict[str, Any]) -> dict[str, Any]:
        raw = item.get("arguments", {})
        return StableJson.object_or_empty(raw)

    def _local_shell_arguments(self, item: dict[str, Any]) -> dict[str, Any]:
        args = StableJson.object_or_empty(item.get("action"))
        for key in ("command", "cmd", "timeout_sec", "timeout_ms", "duration", "workdir"):
            if key in item and key not in args:
                args[key] = item[key]
        return args

    def _exec_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        if "working_directory" in args and "workdir" not in args:
            args["workdir"] = args.pop("working_directory")
        command = args.get("cmd")
        if isinstance(command, list):
            args["cmd"] = self._join_argv(command)
        if "cmd" not in args and "input" in args:
            args["cmd"] = str(args["input"])
        if isinstance(args.get("cmd"), str):
            args["cmd"] = self._sanitize_command(args["cmd"])
        return args

    def _sanitize_command(self, command: str) -> str:
        return command.replace("find ..", "find .")

    def _write_stdin_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if "process_id" in args and "session_id" not in args:
            args["session_id"] = args.pop("process_id")
        args.setdefault("chars", "")
        return args

    def _plan_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if "plan" not in args:
            args["plan"] = []
        return args

    def _patch_input(self, arguments: dict[str, Any], arguments_text: str) -> str:
        for key in ("input", "patch", "diff", "command"):
            if key in arguments:
                return str(arguments[key])
        return arguments_text

    def _join_argv(self, argv: list[Any]) -> str:
        return " ".join(_shell_quote(str(item)) for item in argv)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-./:=,@%"
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


# === 07. Codex Tool Output Formatting ===


class ToolOutputFormatter:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def tool_output_item(self, record: CommandResult, call_id: str) -> dict[str, Any]:
        output = self.tool_output_text(record)
        if self._is_custom_output(record):
            return {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": output,
            }
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    def tool_output_text(self, record: CommandResult) -> str:
        if self.features.unified_exec_output_format and self._has_unified_exec(record):
            return self.unified_exec_text(record)
        return self.generic_function_text(record)

    def unified_exec_text(self, record: CommandResult) -> str:
        metadata = self._unified_exec(record)
        output = self._combined_output(record)
        sections = []
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            sections.append(f"Chunk ID: {chunk_id}")
        wall_time = _float_or_zero(metadata.get("wall_time_seconds"))
        sections.append(f"Wall time: {wall_time:.4f} seconds")
        exit_code = metadata.get("exit_code", record.return_code)
        if exit_code is not None:
            sections.append(f"Process exited with code {exit_code}")
        session_id = metadata.get("session_id")
        if session_id is not None:
            sections.append(f"Process running with session ID {session_id}")
        original_token_count = metadata.get("original_token_count")
        if original_token_count is not None:
            sections.append(f"Original token count: {original_token_count}")
        sections.append("Output:")
        sections.append(TextBudget.clip_tail(output, self._max_tokens_to_chars(record)))
        return "\n".join(sections)

    def generic_function_text(self, record: CommandResult) -> str:
        sections = ["Wall time: 0.0000 seconds"]
        if record.return_code is not None:
            sections.append(f"Process exited with code {record.return_code}")
        sections.append("Output:")
        sections.append(TextBudget.clip_tail(self._combined_output(record), MAX_OBSERVATION_CHARS))
        return "\n".join(sections)

    def failure_response_item(
        self, call_id: str, payload: ToolPayload, message: str
    ) -> dict[str, Any]:
        item_type = (
            "custom_tool_call_output"
            if payload.payload_type == "custom"
            else "function_call_output"
        )
        return {
            "type": item_type,
            "call_id": call_id,
            "output": message,
        }

    def _is_custom_output(self, record: CommandResult) -> bool:
        return record.tool_name == "apply_patch" or record.tool_name in CUSTOM_CALL_TYPES

    def _has_unified_exec(self, record: CommandResult) -> bool:
        return isinstance(record.metadata, dict) and isinstance(
            record.metadata.get("unified_exec"), dict
        )

    def _unified_exec(self, record: CommandResult) -> dict[str, Any]:
        if isinstance(record.metadata, dict) and isinstance(
            record.metadata.get("unified_exec"), dict
        ):
            return dict(record.metadata["unified_exec"])
        return {}

    def _combined_output(self, record: CommandResult) -> str:
        stdout = record.stdout or ""
        stderr = record.stderr or ""
        if stderr:
            return f"{stdout}\nSTDERR:\n{stderr}".strip()
        return stdout

    def _max_tokens_to_chars(self, record: CommandResult) -> int:
        arguments = record.metadata.get("arguments") if isinstance(record.metadata, dict) else None
        if isinstance(arguments, dict):
            max_tokens = arguments.get("max_output_tokens")
            if isinstance(max_tokens, (int, float)) and max_tokens > 0:
                return max(MAX_OBSERVATION_CHARS // 4, int(max_tokens) * 4)
        return MAX_OBSERVATION_CHARS


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# === 08. History Replay and Context Normalization ===


class ResponseItemFactory:
    def function_call(
        self, call_id: str, name: str, arguments: dict[str, Any] | str
    ) -> dict[str, Any]:
        args_text = (
            arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True)
        )
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": args_text,
        }

    def custom_tool_call(self, call_id: str, name: str, input_text: str) -> dict[str, Any]:
        return {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name,
            "input": input_text,
        }

    def assistant_message(self, content: str, phase: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"role": "assistant", "content": content}
        if phase is not None:
            item["phase"] = phase
        return item

    def responses_message(self, role: str, text: str) -> dict[str, Any]:
        item_type = "output_text" if role == "assistant" else "input_text"
        return {
            "type": "message",
            "role": role,
            "content": [{"type": item_type, "text": text}],
        }


class HistoryReplay:
    def __init__(self, formatter: ToolOutputFormatter, features: FeatureSet | None = None):
        self.formatter = formatter
        self.features = features or FeatureSet.from_globals()
        self.items = ResponseItemFactory()

    def input_items(self, task: TaskContext, history: list[CommandResult]) -> list[dict[str, Any]]:
        items = [{"role": "user", "content": InitialContextBuilder().render(task)}]
        if not self.features.history_replay:
            return items
        for index, record in enumerate(history, start=1):
            items.extend(self.record_items(index, record))
        return items

    def record_items(self, index: int, record: CommandResult) -> list[dict[str, Any]]:
        call_id = record.tool_call_id or f"call_{index}"
        raw_items = self.raw_codex_response_items(record)
        if raw_items is not None:
            return [*raw_items, self.formatter.tool_output_item(record, call_id)]
        if self.is_output_only(record):
            return [self.formatter.tool_output_item(record, call_id)]
        items = self.assistant_history(record)
        items.extend(self.synthetic_tool_pair(record, call_id))
        return items

    def raw_codex_response_items(self, record: CommandResult) -> list[dict[str, Any]] | None:
        if not self.features.model_response_item_replay:
            return None
        if not isinstance(record.metadata, dict):
            return None
        raw_items = record.metadata.get("codex_response_items")
        if not isinstance(raw_items, list) or not raw_items:
            return None
        kept = []
        for item in raw_items[:MAX_RAW_RESPONSE_ITEMS]:
            if isinstance(item, dict):
                kept.append(self.sanitize_response_item(item))
        return kept or None

    def is_output_only(self, record: CommandResult) -> bool:
        return isinstance(record.metadata, dict) and bool(record.metadata.get("codex_output_only"))

    def assistant_history(self, record: CommandResult) -> list[dict[str, Any]]:
        if not isinstance(record.metadata, dict):
            return []
        content = str(record.metadata.get("assistant_content") or "").strip()
        if not content:
            return []
        return [self.items.assistant_message(content)]

    def synthetic_tool_pair(self, record: CommandResult, call_id: str) -> list[dict[str, Any]]:
        if record.tool_name == "apply_patch":
            patch = self.patch_from_record(record)
            return [
                self.items.custom_tool_call(call_id, "apply_patch", patch),
                self.formatter.tool_output_item(record, call_id),
            ]
        tool_name = (
            record.tool_name if record.tool_name in FUNCTION_HISTORY_TOOLS else "exec_command"
        )
        arguments = self.arguments_from_record(record)
        return [
            self.items.function_call(call_id, tool_name, arguments),
            self.formatter.tool_output_item(record, call_id),
        ]

    def arguments_from_record(self, record: CommandResult) -> dict[str, Any]:
        args = record.metadata.get("arguments") if isinstance(record.metadata, dict) else None
        if not isinstance(args, dict):
            args = {"cmd": record.command}
        args = dict(args)
        if "command" in args and "cmd" not in args:
            args["cmd"] = args.pop("command")
        return args

    def patch_from_record(self, record: CommandResult) -> str:
        if isinstance(record.metadata, dict):
            for key in ("input", "patch", "diff"):
                if key in record.metadata:
                    return str(record.metadata[key])
        marker = "apply_patch <<'PATCH'\n"
        if record.command.startswith(marker) and record.command.endswith("\nPATCH"):
            return record.command[len(marker) : -len("\nPATCH")]
        return record.command

    def sanitize_response_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role", "assistant"))
            return {
                "type": "message",
                "role": role,
                "content": self._content_items(item.get("content", []), role),
            }
        if item_type == "function_call":
            cleaned = {
                "type": "function_call",
                "name": str(item.get("name", "")),
                "arguments": str(item.get("arguments", "")),
                "call_id": str(item.get("call_id", "")),
            }
            if item.get("namespace") is not None:
                cleaned["namespace"] = item["namespace"]
            return cleaned
        if item_type == "custom_tool_call":
            return {
                "type": "custom_tool_call",
                "name": str(item.get("name", "")),
                "input": str(item.get("input", "")),
                "call_id": str(item.get("call_id", "")),
            }
        if item_type in TOOL_OUTPUT_TYPES:
            cleaned = {"type": item_type, "call_id": str(item.get("call_id", ""))}
            if "output" in item:
                cleaned["output"] = item["output"]
            return cleaned
        cleaned = dict(item)
        cleaned.pop("id", None)
        return cleaned

    def _content_items(self, content: Any, role: str) -> list[dict[str, Any]]:
        item_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            return [{"type": item_type, "text": content}]
        if not isinstance(content, list):
            return [{"type": item_type, "text": str(content)}]
        items = []
        for content_item in content:
            if isinstance(content_item, dict):
                cleaned = dict(content_item)
                cleaned.pop("id", None)
                cleaned.setdefault("type", item_type)
                if cleaned.get("type") in {"input_text", "output_text"}:
                    cleaned["text"] = str(cleaned.get("text", ""))
                items.append(cleaned)
            else:
                items.append({"type": item_type, "text": str(content_item)})
        return items


class ConversationNormalizer:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def normalize(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.features.context_normalization:
            return list(items)
        normalized = [dict(item) for item in items]
        self.ensure_call_outputs_present(normalized)
        self.remove_orphan_outputs(normalized)
        return normalized

    def ensure_call_outputs_present(self, items: list[dict[str, Any]]) -> None:
        insertions: list[tuple[int, dict[str, Any]]] = []
        output_call_ids = self.output_call_ids(items)
        for index, item in enumerate(items):
            call_id = self.call_id_for_call(item)
            if not call_id or call_id in output_call_ids:
                continue
            insertions.append((index + 1, self.synthetic_aborted_output(item, call_id)))
        for index, output in reversed(insertions):
            items.insert(index, output)

    def remove_orphan_outputs(self, items: list[dict[str, Any]]) -> None:
        call_ids = self.call_ids(items)
        kept = []
        for item in items:
            if item.get("type") in TOOL_OUTPUT_TYPES:
                call_id = str(item.get("call_id") or "")
                if call_id and call_id not in call_ids:
                    continue
            kept.append(item)
        items[:] = kept

    def call_ids(self, items: list[dict[str, Any]]) -> set[str]:
        ids = set()
        for item in items:
            call_id = self.call_id_for_call(item)
            if call_id:
                ids.add(call_id)
        return ids

    def output_call_ids(self, items: list[dict[str, Any]]) -> set[str]:
        ids = set()
        for item in items:
            if item.get("type") in TOOL_OUTPUT_TYPES:
                call_id = str(item.get("call_id") or "")
                if call_id:
                    ids.add(call_id)
        return ids

    def call_id_for_call(self, item: dict[str, Any]) -> str:
        if item.get("type") in TOOL_CALL_TYPES:
            return str(item.get("call_id") or "")
        return ""

    def synthetic_aborted_output(self, call_item: dict[str, Any], call_id: str) -> dict[str, Any]:
        if call_item.get("type") in CUSTOM_CALL_TYPES:
            return {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": "aborted",
            }
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": "aborted",
        }


class ContextCompactor:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()
        self.checkpoint: CompactionCheckpoint | None = None

    def maybe_compact(
        self, items: list[dict[str, Any]], initial_context: list[dict[str, Any]] | None = None
    ) -> tuple[list[dict[str, Any]], CompactionCheckpoint | None, bool]:
        raw_items = self.copy_items(items)
        reused = False
        working = raw_items
        if self.features.model_context_compaction:
            applied = self.apply_checkpoint(raw_items)
            if applied is not None:
                working = applied
                reused = True
        if not self.features.model_context_compaction or not self.should_compact(working):
            return working, self.checkpoint if reused else None, reused
        compacted, summary_text = self.compact(working, initial_context or [])
        if compacted is None:
            return working, self.checkpoint if reused else None, reused
        checkpoint = CompactionCheckpoint(
            prefix_digest=self.digest(raw_items),
            prefix_len=len(raw_items),
            replacement=tuple(self.copy_items(compacted)),
            summary_text=summary_text,
        )
        self.checkpoint = checkpoint
        return self.copy_items(compacted), checkpoint, False

    def should_compact(self, items: list[dict[str, Any]]) -> bool:
        return (
            len(items) > MAX_CONTEXT_HISTORY_ITEMS
            or sum(TextBudget.item_bytes(item) for item in items) > MAX_CONTEXT_HISTORY_CHARS
        )

    def apply_checkpoint(self, items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        checkpoint = self.checkpoint
        if checkpoint is None or len(items) < checkpoint.prefix_len:
            return None
        prefix = items[: checkpoint.prefix_len]
        if self.digest(prefix) != checkpoint.prefix_digest:
            return None
        suffix = items[checkpoint.prefix_len :]
        return [*self.copy_items(list(checkpoint.replacement)), *self.copy_items(suffix)]

    def compact(
        self, items: list[dict[str, Any]], initial_context: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str] | tuple[None, str]:
        compact_input = [
            {"role": "system", "content": CODEX_BASE_INSTRUCTIONS},
            *self.copy_items(items),
            ResponseItemFactory().responses_message("user", SUMMARIZATION_PROMPT),
        ]
        try:
            summary_suffix = call_terminal_model(compact_input)
        except Exception:
            return None, ""
        summary_text = f"{SUMMARY_PREFIX}\n{summary_suffix}"
        user_messages = self.collect_user_messages(items, initial_context)
        return (
            self.build_compacted_history(initial_context, user_messages, summary_text),
            summary_text,
        )

    def collect_user_messages(
        self, items: list[dict[str, Any]], initial_context: list[dict[str, Any]]
    ) -> list[str]:
        initial_digests = {self.digest([item]) for item in initial_context}
        messages: list[str] = []
        for item in items:
            if self.digest([item]) in initial_digests:
                continue
            if self.message_role(item) != "user":
                continue
            text = self.message_text(item)
            if text and not self.is_summary_message(text):
                messages.append(text)
        return messages

    def build_compacted_history(
        self,
        initial_context: list[dict[str, Any]],
        user_messages: list[str],
        summary_text: str,
    ) -> list[dict[str, Any]]:
        history = self.copy_items(initial_context)
        for message in self.selected_user_messages(user_messages):
            history.append(ResponseItemFactory().responses_message("user", message))
        history.append(
            ResponseItemFactory().responses_message(
                "user", summary_text if summary_text else "(no summary available)"
            )
        )
        return history

    def selected_user_messages(self, user_messages: list[str]) -> list[str]:
        selected: list[str] = []
        remaining = COMPACT_USER_MESSAGE_MAX_TOKENS
        for message in reversed(user_messages):
            if remaining <= 0:
                break
            tokens = TextBudget.approx_token_count(message)
            if tokens <= remaining:
                selected.append(message)
                remaining -= tokens
            else:
                selected.append(TextBudget.clip_middle(message, remaining * 4))
                break
        selected.reverse()
        return selected

    def is_summary_message(self, message: str) -> bool:
        return message.startswith(f"{SUMMARY_PREFIX}\n")

    def message_role(self, item: dict[str, Any]) -> str:
        if item.get("type") == "message" or "role" in item:
            return str(item.get("role") or "")
        return ""

    def message_text(self, item: dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        pieces = []
        for content_item in content:
            if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                pieces.append(content_item["text"])
        return "\n".join(piece for piece in pieces if piece)

    def digest(self, items: list[dict[str, Any]]) -> str:
        return hashlib.sha256(StableJson.dumps(items).encode("utf-8")).hexdigest()

    def copy_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [json.loads(json.dumps(item)) for item in items]


class ContextManager:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()
        self.normalizer = (
            _construct("ConversationNormalizer", self.features)
            if self.features.context_normalization
            else None
        )
        self.compactor = (
            _construct("ContextCompactor", self.features)
            if self.features.model_context_compaction
            else None
        )

    def prepare(
        self, items: list[dict[str, Any]], initial_context: list[dict[str, Any]] | None = None
    ) -> tuple[list[dict[str, Any]], ContextStats]:
        raw_count = len(items)
        normalized = self.normalizer.normalize(items) if self.normalizer else list(items)
        normalized_count = len(normalized)
        if self.compactor:
            normalized, checkpoint, reused = self.compactor.maybe_compact(
                normalized, initial_context or []
            )
        else:
            checkpoint = None
            reused = False
        budgeted = self.apply_budget(normalized)
        estimated_bytes = sum(TextBudget.item_bytes(item) for item in budgeted)
        return (
            budgeted,
            ContextStats(
                raw_items=raw_count,
                normalized_items=normalized_count,
                pruned_items=max(0, normalized_count - len(budgeted)),
                estimated_bytes=estimated_bytes,
                estimated_tokens=max(1, estimated_bytes // 4),
                compacted=checkpoint is not None,
                compaction_summary_chars=len(checkpoint.summary_text) if checkpoint else 0,
                compaction_reused=reused,
            ),
        )

    def apply_budget(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.features.context_budgeting:
            return list(items)
        clipped = [self.clip_item(item) for item in items]
        if len(clipped) > MAX_CONTEXT_HISTORY_ITEMS:
            clipped = self.drop_oldest_pairs(clipped, len(clipped) - MAX_CONTEXT_HISTORY_ITEMS)
        while sum(TextBudget.item_bytes(item) for item in clipped) > MAX_CONTEXT_HISTORY_CHARS:
            if len(clipped) <= 2:
                break
            clipped = self.drop_oldest_pairs(clipped, 1)
        return clipped

    def clip_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        if item.get("type") in TOOL_OUTPUT_TYPES and isinstance(item.get("output"), str):
            item["output"] = TextBudget.clip_tail(str(item["output"]), MAX_FUNCTION_OUTPUT_CHARS)
        if item.get("type") == "message":
            item["content"] = self.clip_content_items(item.get("content", []))
        elif "content" in item and isinstance(item["content"], str):
            item["content"] = TextBudget.clip_middle(
                str(item["content"]), MAX_FUNCTION_OUTPUT_CHARS
            )
        return item

    def clip_content_items(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content
        clipped = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                updated = dict(item)
                updated["text"] = TextBudget.clip_middle(
                    str(updated["text"]), MAX_FUNCTION_OUTPUT_CHARS
                )
                clipped.append(updated)
            else:
                clipped.append(item)
        return clipped

    def drop_oldest_pairs(
        self, items: list[dict[str, Any]], target_drop: int
    ) -> list[dict[str, Any]]:
        if target_drop <= 0:
            return items
        kept = list(items)
        dropped = 0
        index = 1 if kept and kept[0].get("role") == "user" else 0
        while dropped < target_drop and index < len(kept):
            removed = kept.pop(index)
            dropped += 1
            call_id = str(removed.get("call_id") or "")
            if removed.get("type") in TOOL_CALL_TYPES and call_id:
                match = self.find_output_index(kept, call_id)
                if match is not None:
                    kept.pop(match)
                    dropped += 1
            elif removed.get("type") in TOOL_OUTPUT_TYPES and call_id:
                match = self.find_call_index(kept, call_id)
                if match is not None and match != 0:
                    kept.pop(match)
                    dropped += 1
        return kept

    def find_output_index(self, items: list[dict[str, Any]], call_id: str) -> int | None:
        for index, item in enumerate(items):
            if item.get("type") in TOOL_OUTPUT_TYPES and item.get("call_id") == call_id:
                return index
        return None

    def find_call_index(self, items: list[dict[str, Any]], call_id: str) -> int | None:
        for index, item in enumerate(items):
            if item.get("type") in TOOL_CALL_TYPES and item.get("call_id") == call_id:
                return index
        return None


class NoopContextManager:
    def prepare(
        self, items: list[dict[str, Any]], initial_context: list[dict[str, Any]] | None = None
    ) -> tuple[list[dict[str, Any]], ContextStats]:
        kept = [dict(item) for item in items]
        estimated_bytes = sum(TextBudget.item_bytes(item) for item in kept)
        return (
            kept,
            ContextStats(
                raw_items=len(items),
                normalized_items=len(items),
                pruned_items=0,
                estimated_bytes=estimated_bytes,
                estimated_tokens=max(1, estimated_bytes // 4),
            ),
        )


# === 09. Prompt and Context Rendering ===


class PermissionsInstructionsRenderer:
    def messages(self, environment: TurnEnvironment) -> list[dict[str, Any]]:
        return [{"role": "developer", "content": self.render(environment)}]

    def render(self, environment: TurnEnvironment) -> str:
        sandbox_text = self.sandbox_text(environment)
        approval_text = self.approval_text(environment)
        return (
            "<permissions instructions>\n"
            f"{sandbox_text}\n"
            f"{approval_text}\n"
            "</permissions instructions>"
        )

    def sandbox_text(self, environment: TurnEnvironment) -> str:
        if environment.sandbox_mode == "danger-full-access":
            return PERMISSIONS_SANDBOX_DANGER_FULL_ACCESS
        return (
            "Filesystem sandboxing defines which files can be read or written. "
            f"`sandbox_mode` is `{environment.sandbox_mode}`. "
            f"Network access is {environment.network_access}."
        )

    def approval_text(self, environment: TurnEnvironment) -> str:
        if environment.approval_policy == "never":
            return PERMISSIONS_APPROVAL_NEVER
        return (
            "Approvals are your mechanism to get user consent to run shell commands "
            f"without the sandbox. `approval_policy` is `{environment.approval_policy}`."
        )


class InitialContextBuilder:
    def render(self, task: TaskContext) -> str:
        cwd = task.working_dir or "."
        environment = TurnEnvironment(cwd=cwd)
        sections = [self.environment_context(environment)]
        agents = AgentInstructionsRenderer().render(task)
        if agents:
            sections.append(agents)
        sections.append(str(task.instruction))
        return "\n\n".join(section for section in sections if section)

    def environment_context(self, environment: TurnEnvironment) -> str:
        return (
            "<environment_context>\n"
            f"  <cwd>{environment.cwd}</cwd>\n"
            f"  <shell>{environment.shell}</shell>\n"
            f"  <current_date>{environment.current_date}</current_date>\n"
            f"  <timezone>{environment.timezone}</timezone>\n"
            "</environment_context>"
        )


class AgentInstructionsRenderer:
    def render(self, task: TaskContext) -> str:
        agents = task.metadata.get("agents_md") if isinstance(task.metadata, dict) else None
        if not isinstance(agents, list):
            return ""
        sections = []
        for item in self.sorted_agents(agents):
            path = str(item.get("path") or "AGENTS.md")
            content = str(item.get("content") or "").strip()
            if content:
                sections.append(f"<agents_md path={json.dumps(path)}>\n{content}\n</agents_md>")
        return "\n".join(sections)

    def sorted_agents(self, agents: list[Any]) -> list[dict[str, Any]]:
        clean = [dict(item) for item in agents if isinstance(item, dict)]
        return sorted(
            clean,
            key=lambda item: (str(item.get("path") or "").count("/"), str(item.get("path") or "")),
        )


class PromptBuilder:
    def __init__(self, router: ToolRouter, features: FeatureSet | None = None):
        self.features = features or FeatureSet.from_globals()
        self.router = router
        self.history = (
            _construct(
                "HistoryReplay",
                _construct("ToolOutputFormatter", self.features),
                self.features,
            )
            if self.features.history_replay
            else None
        )
        self.context_manager = (
            _construct("ContextManager", self.features)
            if self.features.context_manager
            else NoopContextManager()
        )

    def build(
        self, task: TaskContext, history: list[CommandResult], context: TurnContext
    ) -> CodexPromptBundle:
        raw_items = (
            self.history.input_items(task, history)
            if self.history
            else [{"role": "user", "content": InitialContextBuilder().render(task)}]
        )
        input_items, stats = self.context_manager.prepare(raw_items, raw_items[:1])
        prompt = build_prompt(
            input_items,
            self.router,
            context,
            BaseInstructions(CODEX_BASE_INSTRUCTIONS),
        )
        return CodexPromptBundle(
            messages=prompt.messages(),
            input_items=input_items,
            tools=prompt.tools,
            stats=stats,
        )


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
        developer_messages=PermissionsInstructionsRenderer().messages(turn_context.environment),
        personality=turn_context.personality,
        output_schema=turn_context.output_schema,
        output_schema_strict=True,
    )


# === 10. Command Classification and Execution Policy ===


class CommandClassifier:
    def classify(self, arguments: dict[str, Any]) -> CommandAssessment:
        command = str(arguments.get("cmd") or arguments.get("command") or "")
        lowered = command.lower()
        notes: list[str] = []
        if not command.strip():
            return CommandAssessment("empty", risky=False, notes=("empty command",))
        if self.is_destructive(lowered):
            notes.append("destructive filesystem or git operation")
            return CommandAssessment(
                "destructive", risky=True, needs_verification=True, notes=tuple(notes)
            )
        if self.is_package_install(lowered):
            return CommandAssessment("package_install", long_running=True, needs_verification=True)
        if self.is_test(lowered):
            return CommandAssessment("test", long_running=True)
        if self.is_build(lowered):
            return CommandAssessment("build", long_running=True, needs_verification=True)
        if self.is_server(lowered):
            return CommandAssessment("server", long_running=True)
        if self.is_git(lowered):
            return CommandAssessment(
                "git", risky="reset --hard" in lowered, needs_verification=True
            )
        if self.is_edit(lowered):
            return CommandAssessment("edit", needs_verification=True)
        return CommandAssessment("inspection")

    def is_destructive(self, command: str) -> bool:
        patterns = (
            "rm -rf",
            "git reset --hard",
            "git checkout --",
            "mkfs",
            "dd if=",
            "truncate -s 0",
        )
        return any(pattern in command for pattern in patterns)

    def is_package_install(self, command: str) -> bool:
        return any(
            token in command
            for token in (
                "pip install",
                "npm install",
                "pnpm install",
                "yarn install",
                "apt-get install",
                "uv sync",
            )
        )

    def is_test(self, command: str) -> bool:
        return any(
            token in command for token in ("pytest", "npm test", "cargo test", "go test", "tox")
        )

    def is_build(self, command: str) -> bool:
        return any(
            token in command
            for token in ("npm run build", "cargo build", "make", "cmake", "go build")
        )

    def is_server(self, command: str) -> bool:
        return any(
            token in command
            for token in ("uvicorn", "flask run", "npm run dev", "vite", "python -m http.server")
        )

    def is_git(self, command: str) -> bool:
        return command.strip().startswith("git ")

    def is_edit(self, command: str) -> bool:
        return any(
            token in command for token in ("apply_patch", "sed -i", "perl -pi", "python - <<")
        )


class ExecutionPolicy:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def annotate_tool_calls(self, calls: list[HarnessToolCall]) -> list[HarnessToolCall]:
        return list(calls)

    def assessments(self, calls: list[HarnessToolCall]) -> list[dict[str, Any]]:
        if not self.features.command_classification:
            return []
        classifier = _construct("CommandClassifier")
        assessments = []
        for call in calls:
            if call.name != "exec_command":
                continue
            assessment = classifier.classify(call.arguments)
            assessments.append(
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "command": str(call.arguments.get("cmd") or ""),
                    "assessment": assessment.__dict__,
                }
            )
        return assessments


# === 11. Model Call Resilience ===


class ModelCallResilience:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolModelResult:
        if not self.features.model_call_resilience:
            return call_terminal_model_with_tools(
                messages,
                tools,
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        try:
            return call_terminal_model_with_tools(
                messages,
                tools,
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        except Exception as exc:
            return ToolModelResult(
                content="",
                tool_calls=[],
                request_metadata={"model_call_error": str(exc)},
                response_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"Model call failed before tool selection: {exc}",
                            }
                        ],
                    }
                ],
            )


class RecoveryPolicy:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def fallback_turn(
        self, result: ToolModelResult, history: list[CommandResult], metadata: dict[str, Any]
    ) -> HarnessTurn | None:
        if not self.features.recovery_policy:
            return None
        text = result.content.strip()
        lower = text.lower().replace("\u2019", "'")
        if result.tool_calls or (
            text
            and not lower.startswith(
                ("i'll ", "i will ", "i'm going to ", "i am going to ", "let me ")
            )
        ):
            return None
        if not history:
            metadata["codex_recovery"] = "empty_response_initial_reconnaissance"
            return HarnessTurn(
                tool_calls=(
                    HarnessToolCall(
                        "exec_command",
                        {
                            "cmd": "pwd && find . -maxdepth 2 -type f | sort | sed -n '1,200p'",
                            "yield_time_ms": 1000,
                            "max_output_tokens": 12000,
                        },
                        "recovery_initial_recon",
                    ),
                ),
                metadata=metadata,
            )
        metadata["codex_recovery"] = "empty_response_recent_status"
        return HarnessTurn(
            tool_calls=(
                HarnessToolCall(
                    "exec_command",
                    {
                        "cmd": "pwd && git status --short 2>/dev/null || true && find . -maxdepth 2 -type f | sort | sed -n '1,120p'",
                        "yield_time_ms": 1000,
                        "max_output_tokens": 12000,
                    },
                    "recovery_status",
                ),
            ),
            metadata=metadata,
        )


class NullRecoveryPolicy:
    def fallback_turn(
        self, result: ToolModelResult, history: list[CommandResult], metadata: dict[str, Any]
    ) -> HarnessTurn | None:
        return None


# === 12. Completion Policy ===


class CompletionPolicy:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def is_complete(self, result: ToolModelResult, tool_calls: list[HarnessToolCall]) -> bool:
        if tool_calls:
            return False
        return bool(self.visible_text(result).strip())

    def visible_text(self, result: ToolModelResult) -> str:
        if not self.features.completion_policy:
            return result.content
        if result.content.strip():
            return result.content
        chunks: list[str] = []
        for item in result.response_items:
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        chunks.append(content_item["text"])
            elif isinstance(content, str):
                chunks.append(content)
        return "\n".join(chunks)


# === 13. Instrumentation ===


class Instrumentation:
    def __init__(self, features: FeatureSet | None = None) -> None:
        self.features = features or FeatureSet.from_globals()

    def turn_metadata(
        self,
        result: ToolModelResult,
        bundle: CodexPromptBundle,
        tool_calls: list[HarnessToolCall],
        assessments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "codex_upstream_commit": CODEX_UPSTREAM_COMMIT,
            "codex_upstream_date": CODEX_UPSTREAM_DATE,
            "codex_port_stats": bundle.stats.__dict__,
            "codex_tool_count": len(bundle.tools),
            "codex_tool_names": [tool.get("name") for tool in bundle.tools],
            "codex_emitted_tool_calls": len(tool_calls),
        }
        if self.features.port_parity_manifest:
            metadata["codex_port_manifest"] = PORT_PARITY_MANIFEST
        if result.request_metadata:
            metadata["codex_request_metadata"] = result.request_metadata
        if result.response_items:
            metadata["codex_response_items"] = result.response_items
        if result.response_id:
            metadata["codex_response_id"] = result.response_id
        if assessments:
            metadata["codex_command_assessments"] = assessments
        return metadata


class NullInstrumentation:
    def turn_metadata(
        self,
        result: ToolModelResult,
        bundle: CodexPromptBundle,
        tool_calls: list[HarnessToolCall],
        assessments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {}


# === 14. Candidate Harness ===


class CandidateHarness(BaseHarness):
    wants_environment_context = True
    wants_agents_context = True

    def __init__(self, profile: str | FeatureSet | None = None) -> None:
        self.features = resolve_features(profile)
        self.router = ToolRouter(_built_tools(self.features))
        self.context = TurnContext()
        self.prompt_builder = PromptBuilder(self.router, self.features)
        self.model = ModelCallResilience(self.features)
        self.completion = CompletionPolicy(self.features)
        self.recovery = (
            _construct("RecoveryPolicy", self.features)
            if self.features.recovery_policy
            else NullRecoveryPolicy()
        )
        self.execution_policy = ExecutionPolicy(self.features)
        self.instrumentation = (
            _construct("Instrumentation", self.features)
            if self.features.instrumentation
            else NullInstrumentation()
        )

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        bundle = self.prompt_builder.build(task, history, self._turn_context_for_task(task))
        result = self.model.call(bundle.messages, bundle.tools)
        tool_calls = self.router.tool_calls_from_result(result)
        tool_calls = self.execution_policy.annotate_tool_calls(tool_calls)
        assessments = self.execution_policy.assessments(tool_calls)
        metadata = self.instrumentation.turn_metadata(result, bundle, tool_calls, assessments)
        recovery = self.recovery.fallback_turn(result, history, metadata)
        if recovery is not None:
            return recovery
        if tool_calls:
            return HarnessTurn(
                tool_calls=tuple(tool_calls),
                assistant_content=self.completion.visible_text(result),
                metadata=metadata,
            )
        return HarnessTurn(
            done=self.completion.is_complete(result, tool_calls),
            assistant_content=self.completion.visible_text(result),
            metadata=metadata,
        )

    def _turn_context_for_task(self, task: TaskContext) -> TurnContext:
        cwd = task.working_dir or "."
        environment = TurnEnvironment(cwd=cwd)
        return TurnContext(cwd=cwd, environment=environment)


# === 15. Factory ===


def create_agent() -> CandidateHarness:
    return CandidateHarness()
