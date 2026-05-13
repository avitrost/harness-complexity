from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.splits import TEST_TASKS, VAL_TASKS  # noqa: E402

ALLOWED_LOCAL_IMPORTS = {
    "plumbing.base_agent",
    "plumbing.openai_client",
    "plumbing.text",
    "plumbing.types",
}
LOCAL_ROOTS = {
    "candidate",
    "evaluator",
    "experience",
    "final_test",
    "plumbing",
    "results",
    "scripts",
}
FORBIDDEN_PATTERNS = {
    "eval(": re.compile(r"\beval\s*\("),
    "exec(": re.compile(r"\bexec\s*\("),
    "compile(": re.compile(r"\bcompile\s*\("),
    "importlib": re.compile(r"\bimportlib\b"),
    "__import__": re.compile(r"\b__import__\b"),
    "subprocess": re.compile(r"\bsubprocess\b"),
    "os.system": re.compile(r"\bos\s*\.\s*system\b"),
    "shell=True": re.compile(r"\bshell\s*=\s*True\b"),
    "gpt-5.4-nano": re.compile(r"gpt-5\.4-nano"),
    "OPENAI_API_KEY": re.compile(r"OPENAI_API_KEY"),
}
FORBIDDEN_DIRS = (
    "experience",
    "final_test",
    "results",
    "splits",
    "recommendation_records",
    "tests",
    "solutions",
)


def audit_candidate(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(text):
            errors.append(f"forbidden pattern: {label}")
    for slug in [*VAL_TASKS, *TEST_TASKS]:
        if slug in text:
            errors.append(f"forbidden task slug: {slug}")
    _audit_forbidden_paths(text, errors)
    _audit_imports(text, errors)
    return {"path": str(path), "ok": not errors, "errors": errors}


def _audit_forbidden_paths(text: str, errors: list[str]) -> None:
    for dirname in FORBIDDEN_DIRS:
        pattern = re.compile(
            rf"(open|Path|read_text|read_bytes)\s*\([^)]*[\"']([^\"']*[\\/])?{dirname}([\\/\"'])"
        )
        if pattern.search(text):
            errors.append(f"forbidden path read: {dirname}")


def _audit_imports(text: str, errors: list[str]) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        errors.append(f"syntax error: {exc}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name, errors)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _check_import(module, errors)


def _check_import(module: str, errors: list[str]) -> None:
    root = module.split(".", 1)[0]
    if root in LOCAL_ROOTS and module not in ALLOWED_LOCAL_IMPORTS:
        errors.append(f"forbidden local import: {module}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = audit_candidate(args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
