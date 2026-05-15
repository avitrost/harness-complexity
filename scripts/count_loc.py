from __future__ import annotations

import argparse
import json
from pathlib import Path
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
) -> dict[str, Any]:
    formatted = black_format_text(path.read_text(encoding="utf-8"))
    lines = formatted.splitlines()
    physical_loc = len(lines)
    nonblank_noncomment_sloc = sum(1 for line in lines if _is_sloc(line))
    ok = (min_lines is None or physical_loc >= min_lines) and (
        max_lines is None or physical_loc <= max_lines
    )
    return {
        "path": str(path),
        "physical_loc": physical_loc,
        "nonblank_noncomment_sloc": nonblank_noncomment_sloc,
        "char_count": len(formatted),
        "min_lines": min_lines,
        "max_lines": max_lines,
        "ok": ok,
    }


def _is_sloc(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-lines", type=int)
    parser.add_argument("--max-lines", type=int)
    args = parser.parse_args()
    result = count_loc(args.path, max_lines=args.max_lines, min_lines=args.min_lines)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
