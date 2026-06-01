import json
import sys
from types import SimpleNamespace

from scripts import run_tb2_model_sweep


def test_tb2_model_sweep_defaults_to_supported_codex_backend_models(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    monkeypatch.setattr(run_tb2_model_sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tb2_model_sweep",
            "--run-id",
            "sweep",
            "--out-root",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert run_tb2_model_sweep.main() == 0
    manifest = json.loads((tmp_path / "sweep" / "manifest.json").read_text())
    assert manifest["models"] == ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]
    assert "gpt-5.4-nano" not in manifest["models"]
    assert manifest["reasoning_effort"] == "medium"
    assert manifest["trials"] == 10
    assert manifest["concurrency_per_candidate"] == 45
    assert manifest["max_candidate_workers"] == 2
    assert manifest["effective_max_in_flight"] == 90
    assert manifest["include_codex_cli"] is False
    assert manifest["include_terminus_2"] is False

    models = [call[0][call[0].index("--codex-model") + 1] for call in calls]
    assert models == ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]
    for command, kwargs in calls:
        assert "--no-include-codex-cli" in command
        assert "--no-include-terminus-2" in command
        assert command[command.index("--trials") + 1] == "10"
        assert command[command.index("--concurrency") + 1] == "45"
        assert command[command.index("--max-candidate-workers") + 1] == "2"
        assert command[command.index("--codex-reasoning-effort") + 1] == "medium"
        assert kwargs["env"]["OPENAI_AUTH_MODE"] == "codex"
        assert kwargs["env"]["HARBOR_SLURM_PYXIS_PARTITION"] == "m7i-cpu2"


def test_tb2_model_sweep_can_select_models_and_candidates(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(run_tb2_model_sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tb2_model_sweep",
            "--run-id",
            "custom",
            "--out-root",
            str(tmp_path),
            "--model",
            "gpt-5.5",
            "--candidate",
            "seed_codex_full",
            "--candidate",
            "seed_terminus_2_compressed",
        ],
    )

    assert run_tb2_model_sweep.main() == 0
    manifest = json.loads((tmp_path / "custom" / "manifest.json").read_text())
    assert manifest["models"] == ["gpt-5.5"]
    assert manifest["candidates"] == ["seed_codex_full", "seed_terminus_2_compressed"]
    command = calls[0]
    assert command.count("--candidate") == 2
    assert "seed_codex_full" in command
    assert "seed_terminus_2_compressed" in command
