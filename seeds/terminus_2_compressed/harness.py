from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import zlib
from dataclasses import dataclass
from typing import Any

from plumbing.base_agent import BaseHarness
from plumbing.openai_client import call_terminal_model
from plumbing.types import CommandResult, HarnessToolCall, HarnessTurn, TaskContext

MAX_OUTPUT_BYTES = 10000
MAX_REPAIR_ATTEMPTS = 2
MAX_YIELD_MS = 60000
DEFAULT_YIELD_MS = 1000
DEFAULT_CONTEXT_LIMIT_TOKENS = 128000
DEFAULT_SUMMARIZATION_THRESHOLD_TOKENS = 8000

TERMINUS_JSON_PLAIN_TEMPLATE_B64 = """
eNqdVsFuGzcQve9XDHypjUqCgwI9qAfDsVvARWsXiYEiSIKA2h1JrLnkhuRKVg3/e9/M7korJ0iL5hK
LSz6+eTPzhu9CSyYyGU+XN2RSsikbnymb9MAVbW1eUwpuY/2KylDXxldTZz3rhkTWk6HfrG8fif3Gxu
Br9nlG7wC7tc7RgmllNyzb5ARVnMpom2wDlnxFec0U2ty0mZYx1NRE3tjQJrcjfuSyzSDR35sUNtIq
GEc2UQ7KjBVDwRc7nA8bWwnbhcnlmhOFJaU1g8oepih+CbE2mXYCFzk1wSdIkOjXt3e3XcyCuQzOh
a1gpRzbMreR50Xx9FQQnRhv3A5inczp5FL+/rvjUbYxQgGcMJnBIYE/QlWKHGuLrUO8HVWuZvTnGm
SqIHwoMV90C2sQWrBIV4J64yzCqPpvKYu4nrlSHaByFTxfnEyEW+OMF17XqjW+aZyyipCicvH8KCD
cpP72QZwua0Iktl2GtuvdBb1mSg2XdmlLMgsEgOVOQKQJHzKxKdcDilA6kJ51rIYbwOw9fhOpkPLv
5IF3UDg8sMrpEk2d+eD1mH6v2mikYvD1fPZKV5+fJ98EKSvR9y8w+1cg/PdRGUoNfVLWnBlbkHQu
8L14w59bG5HJpWVXpXkxPSoArcrht9TbUSHY3OqdcqjPzLtv5EO2jaS6jNHsBHOQNiwkKM163yBF
caf9hMoa8XsZzOsQHONG6ytbghCq2i4PrQPiw2Y6rXhpWtddsjQOvYGtPkjJckJQZ0VxdURn3CDT
F6l4m2NnHj4b6+VPuZQfDY4dNmo3c1c6R81yGnvxzwR5lMDbtl5w1PZmgHeNsDU270UdVeM+tgXj
Kx80H/YMZrU3nSMRXs3Ov5Tg5vc/7t7cX97ez+leGQPO+oSWPlJgD92mzsuUCPxtwxEeZWsxnsN+
9GO0mQ8NqUphO3ZJu23Vnddci1z5pVwiv3hv3Sa0JOJi3LIbRQlrM4h8qx5++sGfEcSCZkpS4usF
mAEILtl1PdIAgrjwc8u+5DSRWCjX7eM05Z2Daqk0DR82zNFOU7qalpqLqxzd91f9UnVYui4KEe6
QVDIZxbLA9YPbSGFIqv5XrocUziV/Z/859TO6Q5vUNVdWPLwbdKc8W80mVFYTcoify3XAL5PPwCh
D0iEG4QhrGXgq1j6VPciqLCdoVQ+siDyVX8OQihtjJEyiL4Bq88ATanZ5HfwP9N4FNBdsW5usG7M
fJ7RdAfz90jr+2F8EX29gjk3U+Pa39gVWcVdMohJSmUzcYWLeZLGIBWd87HpVGKfaOIeFPUZeCzo
Jk9HyjLrjxm3NLlET8MxYoGyGJJoVnGGwI9BCTvsZKTNQqhJq6fibyIwfQnx6emH6J5Mjh3h1Pj
t/fpbxm9pFV5uZxE445UMFdVxndCutMl7qgvlxn4efpPeXXfgNngadDPBGL4L11QJ3QMXp+G/lp
XFTNyHqiwphsFrzz6NJ+d249fUhpl096vkt3CDjEfCi0YFzHVQa60vXVlK+ORqMZZhHaky5tzqI
aZaaNRwfXdZ6h+yi98GhMTEPg0vC8dXh2SWMFVoN7mug/asJiqzYcxQZtiZKilAxrZwBe6cfBO1
+OKImhY8b42wFcxBTkboErBqK5Fj6+3Mr0uljZLCjcm0iJOKY1NNQPUnHjIzPq6FLjM7OEkmU7q
6bvJMaO5hoXwACIC+abB7kRsDKtC7uZSxeH56r8+IJ7q5zDr/wKrjqJ/x+VOmbD9uGhU+68Fz8A
xn2+Lw=
"""
TERMINUS_TIMEOUT_TEMPLATE_B64 = """
eNpdjjFuAzEMBHu9Yh8Q+AF+QdylSG8wJ14kWCcaImXEONzfQ50TF26EFbmc3Y/GtyxdMcmyUI3H
sP6pLYTPxLi+GGB54QjpBpqNG9Yx8O9Zedrgj9SoIZwMWXEV1fxVGJbI/OEnxpdVDHc2zLlmTQ7
lH5665fp9wGl+nLhtvyLlt6EqoozD9HDtBioqL0l36Uh0Y3D1jo6mijwkTZZ97HGlYPTQJL1Er1
U9uPuGa3Q2LnxXa3Jh54+qbaFyCOHdac9SvTXnQ42MIfM+9IwlVyrHENZ/fd4dG34BVEeFHg==
"""
TERMINUS_JSON_PLAIN_TEMPLATE = zlib.decompress(
    base64.b64decode("".join(TERMINUS_JSON_PLAIN_TEMPLATE_B64.split()))
).decode()
TERMINUS_TIMEOUT_TEMPLATE = zlib.decompress(
    base64.b64decode("".join(TERMINUS_TIMEOUT_TEMPLATE_B64.split()))
).decode()


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str
    analysis: str = ""
    plan: str = ""


class TerminusJSONPlainParser:
    required_fields = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        result = self._try_parse_response(response)
        if not result.error:
            return result
        for name, fix in self._get_auto_fixes():
            corrected, was_fixed = fix(response, result.error)
            if not was_fixed:
                continue
            corrected_result = self._try_parse_response(corrected)
            if not corrected_result.error:
                corrected_result.warning = self._combine_warnings(
                    f"AUTO-CORRECTED: {name} - please fix this in future responses",
                    corrected_result.warning,
                )
                return corrected_result
        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        warnings: list[str] = []
        json_content, extra = self._extract_json_content(response)
        warnings.extend(extra)
        warning = "- " + "\n- ".join(warnings) if warnings else ""
        if not json_content:
            return ParseResult([], False, "No valid JSON found in response", warning)
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as exc:
            suffix = (
                f" | Content: {json_content!r}"
                if len(json_content) < 200
                else f" | Content preview: {json_content[:100]!r}..."
            )
            return ParseResult([], False, f"Invalid JSON: {exc}{suffix}", warning)
        error = self._validate_json_structure(data, json_content, warnings)
        warning = "- " + "\n- ".join(warnings) if warnings else ""
        if error:
            return ParseResult([], False, error, warning)
        is_complete = data.get("task_complete", False)
        if isinstance(is_complete, str):
            is_complete = is_complete.lower() in ("true", "1", "yes")
        analysis = data.get("analysis", "")
        plan = data.get("plan", "")
        commands, error = self._parse_commands(data.get("commands", []), warnings)
        warning = "- " + "\n- ".join(warnings) if warnings else ""
        if error:
            if is_complete:
                warnings.append(error)
                return ParseResult([], True, "", "- " + "\n- ".join(warnings), analysis, plan)
            return ParseResult([], False, error, warning, analysis, plan)
        return ParseResult(commands, bool(is_complete), "", warning, analysis, plan)

    def _extract_json_content(self, response: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        start = end = -1
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(response or ""):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    end = index + 1
                    break
        if start == -1 or end == -1:
            return "", ["No valid JSON object found"]
        if response[:start].strip():
            warnings.append("Extra text detected before JSON object")
        if response[end:].strip():
            warnings.append("Extra text detected after JSON object")
        return response[start:end], warnings

    def _validate_json_structure(self, data: Any, response: str, warnings: list[str]) -> str:
        if not isinstance(data, dict):
            return "Response must be a JSON object"
        missing = [field for field in self.required_fields if field not in data]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"
        if not isinstance(data.get("analysis", ""), str):
            warnings.append("Field 'analysis' should be a string")
        if not isinstance(data.get("plan", ""), str):
            warnings.append("Field 'plan' should be a string")
        if not isinstance(data.get("commands", []), list):
            return "Field 'commands' must be an array"
        self._check_field_order(response, warnings)
        complete = data.get("task_complete")
        if complete is not None and not isinstance(complete, (bool, str)):
            warnings.append("Field 'task_complete' should be a boolean or string")
        return ""

    def _parse_commands(
        self, commands_data: list[Any], warnings: list[str]
    ) -> tuple[list[ParsedCommand], str]:
        commands: list[ParsedCommand] = []
        for index, data in enumerate(commands_data):
            label = f"Command {index + 1}"
            if not isinstance(data, dict):
                return [], f"{label} must be an object"
            if "keystrokes" not in data:
                return [], f"{label} missing required 'keystrokes' field"
            keystrokes = data["keystrokes"]
            if not isinstance(keystrokes, str):
                return [], f"{label} 'keystrokes' must be a string"
            duration = data.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                warnings.append(f"{label}: Invalid duration value, using default 1.0")
                duration = 1.0
            if "duration" not in data:
                warnings.append(f"{label}: Missing duration field, using default 1.0")
            unknown = set(data) - {"keystrokes", "duration"}
            if unknown:
                warnings.append(f"{label}: Unknown fields: {', '.join(unknown)}")
            if index < len(commands_data) - 1 and not keystrokes.endswith("\n"):
                warnings.append(
                    f"{label} should end with newline when followed by another command. "
                    "Otherwise the two commands will be concatenated together on the same line."
                )
            commands.append(ParsedCommand(keystrokes, float(duration)))
        return commands, ""

    def _get_auto_fixes(self):
        return [
            ("Fixed incomplete JSON by adding missing closing brace", self._fix_incomplete_json),
            ("Extracted JSON from mixed content", self._fix_mixed_content),
        ]

    def _fix_incomplete_json(self, response: str, error: str) -> tuple[str, bool]:
        if any(
            text in error
            for text in ("Invalid JSON", "Expecting", "Unterminated", "No valid JSON found")
        ):
            missing = response.count("{") - response.count("}")
            if missing > 0:
                return response + "}" * missing, True
        return response, False

    def _fix_mixed_content(self, response: str, error: str) -> tuple[str, bool]:
        if not error:
            return response, False
        for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL):
            try:
                json.loads(match)
            except json.JSONDecodeError:
                continue
            return match, True
        return response, False

    def _combine_warnings(self, first: str, rest: str) -> str:
        return f"- {first}\n{rest}" if rest else f"- {first}"

    def _check_field_order(self, response: str, warnings: list[str]) -> None:
        expected = ["analysis", "plan", "commands"]
        positions = {}
        for field in expected:
            match = re.search(f'"({field})"\\s*:', response)
            if match:
                positions[field] = match.start()
        if len(positions) < 2:
            return
        actual = [field for field, _ in sorted(positions.items(), key=lambda item: item[1])]
        wanted = [field for field in expected if field in positions]
        if actual != wanted:
            warnings.append(
                f"Fields appear in wrong order. Found: {' -> '.join(actual)}, "
                f"expected: {' -> '.join(wanted)}"
            )


class CandidateHarness(BaseHarness):
    wants_environment_context = True
    wants_persistent_terminal = "tmux"

    def __init__(self) -> None:
        self.parser = TerminusJSONPlainParser()
        self.compaction: dict[str, Any] | None = None
        self.summarization_count = 0
        self._last_compacted = False
        self._last_compaction_reused = False
        self._last_summary_chars = 0

    def next_command(self, task: TaskContext, history: list[CommandResult]) -> HarnessTurn:
        messages = self._messages(task, history)
        response = ""
        confirmed_once = False
        for _ in range(MAX_REPAIR_ATTEMPTS + 1):
            try:
                response = call_terminal_model(messages)
            except RuntimeError as exc:
                if not self._context_error(exc):
                    raise
                messages = self._summarize(task, history, messages, force=True)
                response = call_terminal_model(messages)
            parsed = self.parser.parse_response(response)
            if parsed.error:
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": self._parse_error(parsed)},
                    ]
                )
                continue
            if parsed.is_task_complete and (confirmed_once or self._pending_completion(history)):
                return HarnessTurn(
                    done=True,
                    assistant_content=response,
                    metadata=self._turn_metadata(parsed),
                )
            if parsed.commands:
                call = self._tool_call(task, parsed.commands)
                return HarnessTurn(
                    tool_calls=(call,),
                    assistant_content=response,
                    metadata=self._turn_metadata(parsed),
                )
            if parsed.is_task_complete:
                confirmed_once = True
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": self._completion_prompt(history)},
                    ]
                )
                continue
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": self._empty_command_prompt(history)},
                ]
            )
        return HarnessTurn(
            tool_calls=(HarnessToolCall("exec_command", {"cmd": "true", "yield_time_ms": 100}),),
            assistant_content=response,
        )

    def _turn_metadata(self, parsed: ParseResult) -> dict[str, Any]:
        return {
            "terminus_analysis": parsed.analysis,
            "terminus_plan": parsed.plan,
            "terminus_warning": parsed.warning,
            "terminus_task_complete_requested": parsed.is_task_complete,
            "terminus_summarization_count": self.summarization_count,
            "terminus_compacted": self._last_compacted,
            "terminus_compaction_reused": self._last_compaction_reused,
            "terminus_compaction_summary_chars": self._last_summary_chars,
        }

    def _messages(self, task: TaskContext, history: list[CommandResult]) -> list[dict[str, str]]:
        self._last_compacted = False
        self._last_compaction_reused = False
        self._last_summary_chars = 0
        messages = [
            {
                "role": "user",
                "content": TERMINUS_JSON_PLAIN_TEMPLATE.format(
                    instruction=task.instruction,
                    terminal_state=self._initial_terminal_state(task),
                ),
            }
        ]
        for record in history:
            assistant = record.metadata.get("assistant_content")
            if isinstance(assistant, str) and assistant:
                messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": self._observation(record)})
        return self._summarize(task, history, messages)

    def _summarize(
        self,
        task: TaskContext,
        history: list[CommandResult],
        messages: list[dict[str, str]],
        force: bool = False,
    ) -> list[dict[str, str]]:
        if not force and not self._should_summarize(messages, history):
            return messages
        digest = self._digest(messages)
        if self.compaction and self.compaction.get("digest") == digest:
            self._last_compacted = True
            self._last_compaction_reused = True
            self._last_summary_chars = len(str(self.compaction.get("summary") or ""))
            return list(self.compaction["messages"])
        self.summarization_count += 1
        summary_prompt = self._summary_prompt(task.instruction)
        summary = call_terminal_model([*messages, {"role": "user", "content": summary_prompt}])
        screen = (
            self._terminal_output(history[-1]) if history else self._initial_terminal_state(task)
        )
        question_prompt = self._question_prompt(task.instruction, summary, screen)
        questions = call_terminal_model([{"role": "user", "content": question_prompt}])
        answer_prompt = (
            "The next agent has a few questions for you, please answer each of them one "
            "by one in detail:\n\n" + questions
        )
        answers = call_terminal_model(
            [
                *messages,
                {"role": "user", "content": summary_prompt},
                {"role": "assistant", "content": summary},
                {"role": "user", "content": answer_prompt},
            ]
        )
        handoff = (
            "Here are the answers the other agent provided.\n\n"
            + answers
            + "\n\n"
            + "Continue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact with "
            "the terminal."
        )
        compacted = [
            messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": questions},
            {"role": "user", "content": handoff},
        ]
        self.compaction = {"digest": digest, "messages": compacted, "summary": summary}
        self._last_compacted = True
        self._last_summary_chars = len(summary)
        return compacted

    def _should_summarize(
        self,
        messages: list[dict[str, str]],
        history: list[CommandResult],
    ) -> bool:
        if not history:
            return False
        limit = self._env_int("TERMINUS_CONTEXT_LIMIT_TOKENS", DEFAULT_CONTEXT_LIMIT_TOKENS)
        threshold = self._env_int(
            "TERMINUS_PROACTIVE_SUMMARIZATION_THRESHOLD",
            DEFAULT_SUMMARIZATION_THRESHOLD_TOKENS,
        )
        return limit - self._token_count(messages) < threshold

    def _summary_prompt(self, instruction: str) -> str:
        return f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""

    def _question_prompt(self, instruction: str, summary: str, current_screen: str) -> str:
        return f"""You are picking up work from a previous AI agent on this task:

**Original Task:** {instruction}

**Summary from Previous Agent:**
{summary}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

    def _context_error(self, exc: BaseException) -> bool:
        text = str(exc).lower()
        cause = exc.__cause__
        while cause is not None:
            text += "\n" + str(cause).lower()
            cause = cause.__cause__
        return "context" in text and any(word in text for word in ("length", "limit", "tokens"))

    def _digest(self, messages: list[dict[str, str]]) -> str:
        data = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _token_count(self, messages: list[dict[str, str]]) -> int:
        return max(1, len(json.dumps(messages, ensure_ascii=False)) // 4)

    def _env_int(self, name: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(name, str(default)) or default))
        except ValueError:
            return default

    def _initial_terminal_state(self, task: TaskContext) -> str:
        terminal = self._persistent_terminal(task)
        output = terminal.get("initial_output") if terminal else None
        if isinstance(output, str) and output:
            return self._limit_output_length(output)
        cwd = task.working_dir or "."
        return self._limit_output_length(f"$ pwd\n{cwd}\n")

    def _observation(self, record: CommandResult) -> str:
        output = self._terminal_output(record)
        if record.return_code == 124:
            args = record.metadata.get("arguments")
            timeout = args.get("timeout_sec") if isinstance(args, dict) else "unknown"
            return TERMINUS_TIMEOUT_TEMPLATE.format(
                command=record.command,
                timeout_sec=timeout,
                terminal_state=output,
            )
        if record.metadata.get("terminus_task_complete_requested"):
            return self._completion_prompt_from_output(output)
        if record.metadata.get("terminus_warning"):
            warning = record.metadata["terminus_warning"]
            return f"Previous response had warnings:\n{warning}\n\n{output}"
        return output

    def _terminal_output(self, record: CommandResult) -> str:
        if record.tool_name == "write_stdin":
            return self._limit_output_length(
                ((record.stdout or "") + (record.stderr or "")) or "\n"
            )
        parts = [f"$ {record.command}"]
        if record.stdout:
            parts.append(record.stdout.rstrip())
        if record.stderr:
            parts.append(record.stderr.rstrip())
        if record.return_code is not None:
            parts.append(f"[exit code {record.return_code}]")
        return self._limit_output_length("\n".join(parts).strip() + "\n")

    def _tool_call(self, task: TaskContext, commands: list[ParsedCommand]) -> HarnessToolCall:
        text = "".join(command.keystrokes for command in commands)
        yield_ms = self._yield_ms(commands)
        terminal = self._persistent_terminal(task)
        session_name = terminal.get("session_name") if terminal else None
        if isinstance(session_name, str) and terminal.get("backend") == "tmux":
            return HarnessToolCall(
                "write_stdin",
                {
                    "session_name": session_name,
                    "commands": [
                        {"chars": command.keystrokes, "yield_time_ms": self._yield_ms([command])}
                        for command in commands
                    ],
                    "yield_time_ms": yield_ms,
                    "max_output_tokens": 6000,
                },
            )
        session_id = terminal.get("session_id") if terminal else None
        if isinstance(session_id, int):
            return HarnessToolCall(
                "write_stdin",
                {
                    "session_id": session_id,
                    "commands": [
                        {
                            "chars": self._tmux_keys(command.keystrokes),
                            "yield_time_ms": self._yield_ms([command]),
                        }
                        for command in commands
                    ],
                    "yield_time_ms": yield_ms,
                    "max_output_tokens": 6000,
                },
            )
        shell_text = self._tmux_keys(text).rstrip("\n") or self._wait_command(yield_ms)
        args: dict[str, Any] = {
            "cmd": shell_text,
            "yield_time_ms": yield_ms,
            "max_output_tokens": 6000,
            "tty": True,
        }
        if task.working_dir:
            args["workdir"] = task.working_dir
        return HarnessToolCall("exec_command", args)

    def _yield_ms(self, commands: list[ParsedCommand]) -> int:
        total = sum(max(0.1, min(command.duration, 60.0)) for command in commands)
        return max(100, min(MAX_YIELD_MS, int(total * 1000) or DEFAULT_YIELD_MS))

    def _tmux_keys(self, text: str) -> str:
        return text.replace("C-c", "\x03").replace("C-d", "\x04")

    def _wait_command(self, yield_ms: int) -> str:
        seconds = max(1, min(60, int((yield_ms + 999) / 1000)))
        return f"sleep {seconds}"

    def _persistent_terminal(self, task: TaskContext) -> dict[str, Any]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        terminal = metadata.get("persistent_terminal")
        if isinstance(terminal, dict) and terminal.get("available"):
            return terminal
        return {}

    def _pending_completion(self, history: list[CommandResult]) -> bool:
        return bool(history and history[-1].metadata.get("terminus_task_complete_requested"))

    def _completion_prompt(self, history: list[CommandResult]) -> str:
        output = self._terminal_output(history[-1]) if history else ""
        return self._completion_prompt_from_output(output)

    def _completion_prompt_from_output(self, output: str) -> str:
        return (
            f"Current terminal state:\n{output}\n\n"
            "Are you sure you want to mark the task as complete? "
            "This will trigger your solution to be graded and you won't be able to "
            'make any further corrections. If so, include "task_complete": true '
            "in your JSON response again."
        )

    def _parse_error(self, parsed: ParseResult) -> str:
        feedback = f"ERROR: {parsed.error}"
        if parsed.warning:
            feedback += f"\nWARNINGS: {parsed.warning}"
        return (
            f"Previous response had parsing errors:\n{feedback}\n\n"
            "Please fix these issues and provide a proper JSON response."
        )

    def _empty_command_prompt(self, history: list[CommandResult]) -> str:
        output = self._terminal_output(history[-1]) if history else ""
        return (
            "Your previous JSON response did not request any commands and did not confirm "
            f"completion.\n\nCurrent terminal state:\n{output}"
        )

    def _limit_output_length(self, output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        if len(output.encode("utf-8")) <= max_bytes:
            return output
        size = max_bytes // 2
        data = output.encode("utf-8")
        first = data[:size].decode("utf-8", errors="ignore")
        last = data[-size:].decode("utf-8", errors="ignore")
        omitted = len(data) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
        return f"{first}\n... [Output truncated: {omitted} bytes omitted] ...\n{last}"


def create_agent() -> CandidateHarness:
    return CandidateHarness()
