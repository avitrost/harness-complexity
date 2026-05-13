from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def make_workspace(destination: Path, source: Path) -> Path:
    candidate_dir = destination / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, candidate_dir / "harness.py")
    proposal = destination / "proposal.md"
    if not proposal.exists():
        proposal.write_text("# Proposal\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("seeds/seed_minimal.py"))
    args = parser.parse_args()
    print(make_workspace(args.destination, args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
