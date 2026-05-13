from pathlib import Path

from scripts.audit_candidate import audit_candidate


def test_audit_rejects_forbidden_import(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("import plumbing.harbor_adapter\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("forbidden local import" in error for error in result["errors"])


def test_audit_rejects_task_slug(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text('PROMPT = "fix-git"\n', encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("forbidden task slug" in error for error in result["errors"])


def test_audit_accepts_seed_candidate() -> None:
    result = audit_candidate(Path("candidate/harness.py"))
    assert result["ok"], result["errors"]
