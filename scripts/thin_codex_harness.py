from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "seeds" / "codex_full" / "harness.py"


@dataclass(frozen=True)
class ThinProfile:
    flags: dict[str, bool] = field(default_factory=dict)
    drop_symbols: tuple[str, ...] = ()
    drop_import_lines: tuple[str, ...] = ()


THIN_PROFILES: dict[str, ThinProfile] = {
    "codex_full": ThinProfile(),
    "no_instrumentation": ThinProfile(
        flags={
            "ENABLE_INSTRUMENTATION": False,
            "ENABLE_PORT_PARITY_MANIFEST": False,
        },
        drop_symbols=(
            "Instrumentation",
            "PORT_PARITY_MANIFEST",
            "CODEX_UPSTREAM_COMMIT",
            "CODEX_UPSTREAM_DATE",
        ),
    ),
    "no_classifier": ThinProfile(
        flags={"ENABLE_COMMAND_CLASSIFICATION": False},
        drop_symbols=("CommandAssessment", "CommandClassifier"),
    ),
    "no_recovery": ThinProfile(
        flags={"ENABLE_RECOVERY_POLICY": False},
        drop_symbols=("RecoveryPolicy",),
    ),
    "no_compaction": ThinProfile(
        flags={"ENABLE_MODEL_CONTEXT_COMPACTION": False},
        drop_symbols=(
            "CompactionCheckpoint",
            "ContextCompactor",
            "SUMMARIZATION_PROMPT",
            "SUMMARY_PREFIX",
        ),
        drop_import_lines=("import hashlib", "call_terminal_model,"),
    ),
    "exec_only_tools": ThinProfile(
        flags={
            "ENABLE_PATCH_TOOL": False,
            "ENABLE_PLAN_TOOL": False,
            "ENABLE_WRITE_STDIN_TOOL": False,
        },
        drop_symbols=(
            "APPLY_PATCH_GRAMMAR",
            "_write_stdin_tool",
            "_update_plan_tool",
            "_apply_patch_tool",
        ),
    ),
    "minimal_loop": ThinProfile(
        flags={
            "ENABLE_PORT_PARITY_MANIFEST": False,
            "ENABLE_HISTORY_REPLAY": False,
            "ENABLE_CONTEXT_MANAGER": False,
            "ENABLE_CONTEXT_NORMALIZATION": False,
            "ENABLE_CONTEXT_BUDGETING": False,
            "ENABLE_MODEL_CONTEXT_COMPACTION": False,
            "ENABLE_PATCH_TOOL": False,
            "ENABLE_PLAN_TOOL": False,
            "ENABLE_WRITE_STDIN_TOOL": False,
            "ENABLE_UNIFIED_EXEC_OUTPUT_FORMAT": False,
            "ENABLE_MODEL_RESPONSE_ITEM_REPLAY": False,
            "ENABLE_MODEL_CALL_RESILIENCE": False,
            "ENABLE_RECOVERY_POLICY": False,
            "ENABLE_COMMAND_CLASSIFICATION": False,
            "ENABLE_INSTRUMENTATION": False,
        },
        drop_symbols=(
            "APPLY_PATCH_GRAMMAR",
            "SUMMARIZATION_PROMPT",
            "SUMMARY_PREFIX",
            "CODEX_UPSTREAM_COMMIT",
            "CODEX_UPSTREAM_DATE",
            "PORT_PARITY_MANIFEST",
            "CommandAssessment",
            "CompactionCheckpoint",
            "ResponseItemFactory",
            "HistoryReplay",
            "ConversationNormalizer",
            "ContextCompactor",
            "ContextManager",
            "CommandClassifier",
            "RecoveryPolicy",
            "Instrumentation",
            "_write_stdin_tool",
            "_update_plan_tool",
            "_apply_patch_tool",
            "MAX_CONTEXT_HISTORY_ITEMS",
            "MAX_CONTEXT_HISTORY_CHARS",
            "MAX_RAW_RESPONSE_ITEMS",
            "COMPACT_USER_MESSAGE_MAX_TOKENS",
        ),
        drop_import_lines=("import hashlib", "call_terminal_model,"),
    ),
}


def thin_source(source: str, profile_name: str) -> str:
    if profile_name not in THIN_PROFILES:
        raise KeyError(f"unknown profile {profile_name!r}")
    profile = THIN_PROFILES[profile_name]
    tree = ast.parse(source)
    replacements = _replacement_spans(tree, profile, profile_name)
    lines = source.splitlines()
    for start, end, replacement in sorted(replacements, reverse=True):
        replacement_lines = replacement.splitlines() if replacement else []
        lines[start - 1 : end] = replacement_lines
    if profile.drop_import_lines:
        lines = [
            line
            for line in lines
            if not any(fragment in line for fragment in profile.drop_import_lines)
        ]
    return "\n".join(lines).rstrip() + "\n"


def _replacement_spans(
    tree: ast.AST, profile: ThinProfile, profile_name: str
) -> list[tuple[int, int, str]]:
    replacements: list[tuple[int, int, str]] = []
    drop_symbols = set(profile.drop_symbols)
    for node in getattr(tree, "body", []):
        name = _top_level_name(node)
        if not name:
            continue
        if name in drop_symbols:
            replacements.append((_start_lineno(node), _end_lineno(node), ""))
        elif name in profile.flags:
            replacements.append(
                (_start_lineno(node), _end_lineno(node), f"{name} = {profile.flags[name]}")
            )
        elif name == "DEFAULT_PROFILE_NAME":
            replacements.append(
                (_start_lineno(node), _end_lineno(node), f'{name} = "{profile_name}"')
            )
    return replacements


def _top_level_name(node: ast.AST) -> str:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return ""


def _start_lineno(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return int(min(decorator.lineno for decorator in decorators))
    return int(node.lineno)


def _end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is None:
        raise ValueError(f"missing end_lineno for {type(node).__name__}")
    return int(end)


def write_profile(source_path: Path, output_path: Path, profile_name: str) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    thinned = thin_source(source, profile_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(thinned, encoding="utf-8")
    return {
        "profile": profile_name,
        "source": str(source_path),
        "output": str(output_path),
        "source_lines": len(source.splitlines()),
        "output_lines": len(thinned.splitlines()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(THIN_PROFILES), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    result = write_profile(args.source, args.output, args.profile)
    if args.validate:
        validation = subprocess.run(
            [sys.executable, "-m", "evaluator.validate_candidate", str(args.output)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result["validation_returncode"] = validation.returncode
        result["validation_stdout"] = validation.stdout
        result["validation_stderr"] = validation.stderr
        print(json.dumps(result, indent=2, sort_keys=True))
        return validation.returncode
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
