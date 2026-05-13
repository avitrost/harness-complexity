from pathlib import Path

from evaluator.optimize_budget import build_codex_command


def test_build_codex_command_uses_resolved_exec_binary(tmp_path: Path) -> None:
    codex = tmp_path / "codex.cmd"
    codex.write_text("", encoding="utf-8")
    command = build_codex_command(
        workspace=tmp_path,
        budget=128,
        codex_model="gpt-5.5-medium",
        repair=False,
        codex_bin=str(codex),
    )
    assert command[:4] == [str(codex), "exec", "--model", "gpt-5.5-medium"]
    assert "--skip-git-repo-check" in command
    assert "--sandbox" in command
