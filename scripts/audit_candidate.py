from __future__ import annotations

import argparse
import ast
from collections import Counter
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
    "plumbing.mini_swe_barebones_v2_variants",
    "plumbing.openai_client",
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
    "compile(": re.compile(r"(?<![\w.])compile\s*\("),
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
    "history",
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
    tree = _parse_tree(text, errors)
    if tree is not None:
        _audit_imports(tree, errors)
        _audit_padding_shapes(tree, errors)
    return {"path": str(path), "ok": not errors, "errors": errors}


def _audit_forbidden_paths(text: str, errors: list[str]) -> None:
    for dirname in FORBIDDEN_DIRS:
        pattern = re.compile(
            rf"(open|Path|read_text|read_bytes)\s*\([^)]*[\"']([^\"']*[\\/])?{dirname}([\\/\"'])"
        )
        if pattern.search(text):
            errors.append(f"forbidden path read: {dirname}")


def _parse_tree(text: str, errors: list[str]) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        errors.append(f"syntax error: {exc}")
        return


def _audit_imports(tree: ast.AST, errors: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name, errors)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _check_import(module, errors)


def _audit_padding_shapes(tree: ast.AST, errors: list[str]) -> None:
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    for node in functions:
        lines = (getattr(node, "end_lineno", None) or node.lineno) - node.lineno + 1
        if lines > 500:
            errors.append(f"oversized function body: {node.name} has {lines} lines")
    numbered = [node.name for node in functions if re.search(r"_\d{1,4}$", node.name)]
    if len(numbered) > 24:
        errors.append(f"mechanical numbered functions: {len(numbered)}")
    numbered_classes = [node.name for node in classes if re.search(r"\d{1,4}$", node.name)]
    if len(numbered_classes) > 12:
        errors.append(f"mechanical numbered classes: {len(numbered_classes)}")
    numbered_assignments = [
        name
        for node in ast.walk(tree)
        for name in _assignment_names(node)
        if re.search(r"_\d{1,4}$", name)
    ]
    if len(numbered_assignments) > 24:
        errors.append(f"mechanical numbered assignments: {len(numbered_assignments)}")
    prefixes = Counter(_function_prefix(node.name) for node in functions)
    prefixes.pop("__", None)
    prefix, count = prefixes.most_common(1)[0] if prefixes else ("", 0)
    if count > 120 and count > len(functions) // 2:
        errors.append(f"mechanical function family: {prefix} has {count} functions")
    bodies = Counter(
        ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
        for node in functions
    )
    repeated = max(bodies.values(), default=0)
    if repeated > 8:
        errors.append(f"near-duplicate function bodies: {repeated}")
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            lines = (getattr(node, "end_lineno", None) or node.lineno) - node.lineno + 1
            if lines > 500:
                errors.append(f"large top-level data block: {lines} lines")


def _function_prefix(name: str) -> str:
    if name.startswith("__"):
        return "__"
    return name.split("_", 1)[0]


def _assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    return []


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
