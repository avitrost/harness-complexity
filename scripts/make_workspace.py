from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
OFFICIAL_SKILL = REFERENCES / "meta_harness_terminal_bench_2_skill.md"

LOCAL_ADAPTATION = """

## LOCAL ADAPTATION

These notes override the reference instructions above wherever they conflict with
this repository:

- Edit `agents/baseline_kira.py` and `proposal.md` only. The outer loop copies
  `agents/baseline_kira.py` to `candidate/harness.py` before validation.
- The final `candidate/harness.py` must expose `create_agent()` returning a
  `plumbing.base_agent.BaseHarness`; it does not subclass Terminus2.
- The task prompt gives the exact Black-formatted physical line budget, and the
  final `candidate/harness.py` must satisfy it.
- Use `logs/frontier_val.json`, `logs/evolution_summary.jsonl`,
  `logs/trace_index.json`, `logs/failures.md`, `logs/jobs/`, and `history/` as
  the local run-history filesystem.
- You may inspect `references/terminus_kira.md` for Terminus-KIRA design
  patterns, but do not import from references or prior runs at runtime.
- Do not inspect parent directories; use the workspace-local snapshots.
- If Agent subagents are unavailable, perform the Analyze and Implement steps
  yourself in the main session.
- Write `proposal.md` instead of `pending_eval.json`; include inspected files,
  failure modes, hypothesis, changes, expected benefit, risks, and exact trace
  paths used as evidence.
"""

WORKSPACE_AGENTS = OFFICIAL_SKILL.read_text(encoding="utf-8") + LOCAL_ADAPTATION


def make_workspace(
    destination: Path,
    source: Path,
    history_dirs: list[Path] | None = None,
) -> Path:
    candidate_dir = destination / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    agents_dir = destination / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    source_file = source / "harness.py" if source.is_dir() else source
    shutil.copy2(source_file, candidate_dir / "harness.py")
    shutil.copy2(source_file, agents_dir / "baseline_kira.py")
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
    if REFERENCES.exists():
        shutil.copytree(
            REFERENCES,
            destination / "references",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    (destination / "AGENTS.md").write_text(WORKSPACE_AGENTS, encoding="utf-8")
    skill_dir = destination / ".claude" / "skills" / "meta-harness-terminal-bench-2"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(WORKSPACE_AGENTS, encoding="utf-8")
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
