import json
import sys

from scripts import run_tb2_model_sweep


def test_tb2_model_sweep_defaults_to_supported_codex_backend_models(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.delenv("TERMINAL_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_TERMINAL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    monkeypatch.delenv("HARBOR_SLURM_PYXIS_PARTITION", raising=False)

    def fake_run_attempt(root, spec, args):
        calls.append((root, spec, args))
        return {
            "model": spec.model,
            "candidate": spec.candidate.name,
            "task": spec.task,
            "attempt": spec.attempt,
            "returncode": 0,
            "ran": False,
            "dry_run": True,
        }

    monkeypatch.setattr(run_tb2_model_sweep, "_run_attempt", fake_run_attempt)
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
    assert manifest["provider"] == "openai"
    assert manifest["models"] == ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]
    assert "gpt-5.4-nano" not in manifest["models"]
    assert manifest["reasoning_effort"] == "medium"
    assert manifest["trials"] == 10
    assert manifest["scheduler"] == "global_attempt_pool"
    assert manifest["global_concurrency"] == 45
    assert manifest["attempt_concurrency"] == 1
    assert manifest["effective_max_in_flight"] == 45
    assert manifest["attempt_cells"] == 3 * 9 * 9 * 10
    assert manifest["include_codex_cli"] is False
    assert manifest["include_terminus_2"] is False

    assert len(calls) == 3 * 9 * 9 * 10
    models = {call[1].model for call in calls}
    assert models == {"gpt-5.4-mini", "gpt-5.4", "gpt-5.5"}
    assert {call[2].concurrency for call in calls} == {45}
    assert run_tb2_model_sweep.os.environ["TERMINAL_MODEL_PROVIDER"] == "openai"
    assert run_tb2_model_sweep.os.environ["OPENAI_AUTH_MODE"] == "codex"
    assert run_tb2_model_sweep.os.environ["HARBOR_SLURM_PYXIS_PARTITION"] == "m7i-cpu2"


def test_tb2_model_sweep_can_select_models_and_candidates(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.delenv("TERMINAL_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_TERMINAL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)
    monkeypatch.delenv("HARBOR_SLURM_PYXIS_PARTITION", raising=False)

    def fake_run_attempt(root, spec, args):
        calls.append(spec)
        return {
            "model": spec.model,
            "candidate": spec.candidate.name,
            "task": spec.task,
            "attempt": spec.attempt,
            "returncode": 0,
        }

    monkeypatch.setattr(run_tb2_model_sweep, "_run_attempt", fake_run_attempt)
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
            "--task",
            "bn-fit-modify",
            "--trials",
            "2",
            "--concurrency",
            "1",
        ],
    )

    assert run_tb2_model_sweep.main() == 0
    manifest = json.loads((tmp_path / "custom" / "manifest.json").read_text())
    assert manifest["provider"] == "openai"
    assert manifest["models"] == ["gpt-5.5"]
    assert [item["name"] for item in manifest["candidates"]] == [
        "seed_codex_full",
        "seed_terminus_2_compressed",
    ]
    assert manifest["tasks"] == ["bn-fit-modify"]
    assert manifest["attempt_cells"] == 4
    assert [(spec.candidate.name, spec.task, spec.attempt) for spec in calls] == [
        ("seed_codex_full", "bn-fit-modify", 1),
        ("seed_codex_full", "bn-fit-modify", 2),
        ("seed_terminus_2_compressed", "bn-fit-modify", 1),
        ("seed_terminus_2_compressed", "bn-fit-modify", 2),
    ]


def test_tb2_model_sweep_can_select_anthropic_provider(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.delenv("TERMINAL_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_TERMINAL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_AUTH_MODE", raising=False)

    def fake_run_attempt(root, spec, args):
        calls.append((spec, args))
        return {
            "model": spec.model,
            "candidate": spec.candidate.name,
            "task": spec.task,
            "attempt": spec.attempt,
            "returncode": 0,
        }

    monkeypatch.setattr(run_tb2_model_sweep, "_run_attempt", fake_run_attempt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tb2_model_sweep",
            "--run-id",
            "claude",
            "--out-root",
            str(tmp_path),
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--candidate",
            "seed_minimal_agent",
            "--task",
            "bn-fit-modify",
            "--trials",
            "1",
            "--concurrency",
            "1",
        ],
    )

    assert run_tb2_model_sweep.main() == 0
    manifest = json.loads((tmp_path / "claude" / "manifest.json").read_text())
    assert manifest["provider"] == "anthropic"
    assert manifest["models"] == ["claude-sonnet-4-6"]
    assert calls[0][1].provider == "anthropic"
    assert run_tb2_model_sweep.os.environ["TERMINAL_MODEL_PROVIDER"] == "anthropic"
    assert "OPENAI_AUTH_MODE" not in run_tb2_model_sweep.os.environ
