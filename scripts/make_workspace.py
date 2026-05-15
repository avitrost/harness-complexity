from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WORKSPACE_AGENTS = """# Candidate Workspace

Edit only candidate/harness.py and proposal.md.
You may inspect history/ for prior candidates from this budget, including source,
proposals, validation, scores, and terminal traces.
Do not inspect parent directories or experience/ directly; use the history/ snapshot.
Do not read final_test/, results/, tests/, benchmark solutions, or hidden files.

proposal.md should describe current workspace files inspected, observed failure modes,
hypothesis, changes made, expected benefit, and risks.
"""


def make_workspace(
    destination: Path,
    source: Path,
    history_dirs: list[Path] | None = None,
) -> Path:
    candidate_dir = destination / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    source_file = source / "harness.py" if source.is_dir() else source
    shutil.copy2(source_file, candidate_dir / "harness.py")
    proposal = destination / "proposal.md"
    if not proposal.exists():
        proposal.write_text("# Proposal\n", encoding="utf-8")
    history = destination / "history"
    history.mkdir(exist_ok=True)
    for path in history_dirs or []:
        shutil.copytree(
            path,
            history / path.name,
            ignore=shutil.ignore_patterns("__pycache__", "history"),
        )
    (destination / "AGENTS.md").write_text(WORKSPACE_AGENTS, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("candidate"))
    args = parser.parse_args()
    print(make_workspace(args.destination, args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
