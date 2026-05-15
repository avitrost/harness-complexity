from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WORKSPACE_AGENTS = """# Candidate Workspace

Edit only candidate/harness.py and proposal.md.
Do not inspect parent directories or prior experiment artifacts.
Do not read experience/, final_test/, results/, tests/, benchmark solutions, or hidden files.

proposal.md should describe current workspace files inspected, observed failure modes,
hypothesis, changes made, expected benefit, and risks.
"""


def make_workspace(destination: Path, source: Path) -> Path:
    candidate_dir = destination / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    source_file = source / "harness.py" if source.is_dir() else source
    shutil.copy2(source_file, candidate_dir / "harness.py")
    proposal = destination / "proposal.md"
    if not proposal.exists():
        proposal.write_text("# Proposal\n", encoding="utf-8")
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
