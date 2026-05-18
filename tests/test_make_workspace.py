from pathlib import Path

from scripts.make_workspace import make_workspace


def test_make_workspace_writes_isolation_instructions(tmp_path: Path) -> None:
    source = tmp_path / "seed.py"
    source.write_text("x = 1\n", encoding="utf-8")
    workspace = make_workspace(tmp_path / "workspace", source)
    agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "Do not inspect parent directories" in agents
    assert "history/" in agents
    assert "logs/failures.md" in agents
    assert "logs/trace_index.json" in agents
    assert "Terminus-KIRA" in agents
    assert (workspace / "history").is_dir()
    assert (workspace / "agents" / "baseline_kira.py").is_file()
    assert (workspace / "references" / "terminus_kira.md").is_file()
    assert (
        workspace / ".claude" / "skills" / "meta-harness-terminal-bench-2" / "SKILL.md"
    ).is_file()
