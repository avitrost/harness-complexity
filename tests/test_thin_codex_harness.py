from __future__ import annotations

import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.thin_codex_harness import THIN_PROFILES, write_profile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "seeds" / "codex_full" / "harness.py"


def test_thin_profiles_generate_valid_python(tmp_path: Path) -> None:
    source_lines = SOURCE.read_text(encoding="utf-8").splitlines()

    for profile in THIN_PROFILES:
        output = tmp_path / profile / "harness.py"
        result = write_profile(SOURCE, output, profile)
        py_compile.compile(str(output), doraise=True)
        lint = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert lint.returncode == 0, lint.stdout + lint.stderr
        if profile != "codex_full":
            assert result["output_lines"] < len(source_lines)


def test_minimal_loop_profile_removes_codex_components(tmp_path: Path) -> None:
    output = tmp_path / "minimal_loop.py"
    write_profile(SOURCE, output, "minimal_loop")
    text = output.read_text(encoding="utf-8")

    assert "class HistoryReplay" not in text
    assert "class ContextManager" not in text
    assert "class ContextCompactor" not in text
    assert "class RecoveryPolicy" not in text
    assert "class Instrumentation" not in text

    module = _load_module(output)
    agent = module.create_agent()
    assert [tool["name"] for tool in agent.router.model_visible_specs()] == ["exec_command"]
    assert not agent.features.history_replay
    assert not agent.features.context_manager
    assert not agent.features.instrumentation


def test_runtime_profiles_disable_components(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HARNESS_PROFILE", "exec_only_tools")
    module = _load_module(SOURCE)
    agent = module.create_agent()

    assert [tool["name"] for tool in agent.router.model_visible_specs()] == ["exec_command"]
    assert not agent.features.patch_tool
    assert not agent.features.plan_tool
    assert not agent.features.write_stdin_tool


def test_unknown_runtime_profile_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HARNESS_PROFILE", "typo_profile")
    module = _load_module(SOURCE)

    with pytest.raises(ValueError, match="unknown CODEX_HARNESS_PROFILE"):
        module.create_agent()


def _load_module(path: Path):
    name = f"thin_profile_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
