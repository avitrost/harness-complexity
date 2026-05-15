from pathlib import Path

from scripts.make_workspace import make_workspace


def test_make_workspace_writes_isolation_instructions(tmp_path: Path) -> None:
    source = tmp_path / "seed.py"
    source.write_text("x = 1\n", encoding="utf-8")
    workspace = make_workspace(tmp_path / "workspace", source)
    agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "Do not inspect parent directories" in agents
    assert "history/" in agents
    assert (workspace / "history").is_dir()
