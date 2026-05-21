from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tokenize
from typing import Any

import black


def black_format_text(text: str) -> str:
    try:
        return black.format_file_contents(
            text,
            fast=False,
            mode=black.FileMode(line_length=100),
        )
    except black.NothingChanged:
        return text


def count_loc(
    path: Path,
    max_lines: int | None = None,
    min_lines: int | None = None,
    min_sloc: int | None = None,
    max_sloc: int | None = None,
) -> dict[str, Any]:
    formatted = black_format_text(path.read_text(encoding="utf-8"))
    lines = formatted.splitlines()
    physical_loc = len(lines)
    ignored_lines = _multiline_string_body_lines(formatted)
    nonblank_noncomment_sloc = sum(
        1 for line_no, line in enumerate(lines, start=1) if _is_sloc(line_no, line, ignored_lines)
    )
    ok = (
        (min_lines is None or physical_loc >= min_lines)
        and (max_lines is None or physical_loc <= max_lines)
        and (min_sloc is None or nonblank_noncomment_sloc >= min_sloc)
        and (max_sloc is None or nonblank_noncomment_sloc <= max_sloc)
    )
    return {
        "path": str(path),
        "physical_loc": physical_loc,
        "nonblank_noncomment_sloc": nonblank_noncomment_sloc,
        "char_count": len(formatted),
        "min_lines": min_lines,
        "max_lines": max_lines,
        "min_sloc": min_sloc,
        "max_sloc": max_sloc,
        "ok": ok,
    }


def _is_sloc(line_no: int, line: str, ignored_lines: set[int]) -> bool:
    if line_no in ignored_lines:
        return False
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _multiline_string_body_lines(text: str) -> set[int]:
    ignored: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.STRING and token.start[0] < token.end[0]:
                ignored.update(range(token.start[0] + 1, token.end[0]))
    except tokenize.TokenError:
        return set()
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-lines", type=int)
    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--min-sloc", type=int)
    parser.add_argument("--max-sloc", type=int)
    args = parser.parse_args()
    result = count_loc(
        args.path,
        max_lines=args.max_lines,
        min_lines=args.min_lines,
        min_sloc=args.min_sloc,
        max_sloc=args.max_sloc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
