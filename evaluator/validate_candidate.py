from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def validate_candidate(path: Path, max_lines: int, min_lines: int | None = None) -> dict[str, Any]:
    count_command = [
        sys.executable,
        "scripts/count_loc.py",
        str(path),
        "--max-lines",
        str(max_lines),
    ]
    if min_lines is not None:
        count_command.extend(["--min-lines", str(min_lines)])
    checks = [
        [sys.executable, "-m", "black", "--line-length", "100", str(path)],
        [sys.executable, "-m", "ruff", "check", str(path)],
        [sys.executable, "-m", "py_compile", str(path)],
        count_command,
        [sys.executable, "scripts/audit_candidate.py", str(path)],
    ]
    results = [_run(command) for command in checks]
    return {
        "path": str(path),
        "min_lines": min_lines,
        "max_lines": max_lines,
        "ok": all(r["ok"] for r in results),
        "checks": results,
    }


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    parsed: Any = None
    if result.stdout.strip().startswith("{"):
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "command": command,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("candidate/harness.py"))
    parser.add_argument("--min-lines", type=int)
    parser.add_argument("--max-lines", type=int, required=True)
    args = parser.parse_args()
    result = validate_candidate(args.path, args.max_lines, args.min_lines)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
