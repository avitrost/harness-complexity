from pathlib import Path

from evaluator.optimize_budget import build_codex_command


def test_build_codex_command_uses_resolved_exec_binary(tmp_path: Path) -> None:
    codex = tmp_path / "codex.cmd"
    codex.write_text("", encoding="utf-8")
    command = build_codex_command(
        budget=128,
        codex_model="gpt-5.5",
        codex_reasoning_effort="medium",
        repair=False,
        codex_bin=str(codex),
    )
    assert command[:4] == [str(codex), "exec", "--model", "gpt-5.5"]
    assert command[4:6] == ["-c", 'model_reasoning_effort="medium"']
    assert command[6:8] == ["--sandbox", "danger-full-access"]
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert "Do not inspect parent directories" in command[-1]
    assert "Read history/ first" in command[-1]
