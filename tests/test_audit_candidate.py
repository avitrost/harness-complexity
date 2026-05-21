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


def test_audit_rejects_history_reads(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("open('history/index.json').read()\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("forbidden path read: history" in error for error in result["errors"])


def test_audit_allows_regex_compile(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("import re\nPATTERN = re.compile('x')\n", encoding="utf-8")
    result = audit_candidate(path)
    assert result["ok"], result["errors"]


def test_audit_rejects_builtin_compile(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("compile('x = 1', '<x>', 'exec')\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("forbidden pattern: compile(" in error for error in result["errors"])


def test_audit_rejects_mechanical_duplicate_helpers(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    helpers = "\n".join(
        f"def helper_{index}(x):\n    return str(x).strip()\n" for index in range(30)
    )
    path.write_text(helpers, encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("mechanical numbered functions" in error for error in result["errors"])
    assert any("near-duplicate function bodies" in error for error in result["errors"])


def test_audit_rejects_mechanical_numbered_classes(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    classes = "\n".join(f"class Policy{index}:\n    pass\n" for index in range(20))
    path.write_text(classes, encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("mechanical numbered classes" in error for error in result["errors"])


def test_audit_rejects_mechanical_function_family(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    helpers = "\n".join(
        f"def review_case_{index}(x):\n    return x + {index}\n" for index in range(130)
    )
    path.write_text(f"{helpers}\ndef choose(x):\n    return x\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("mechanical function family" in error for error in result["errors"])


def test_audit_rejects_mechanical_numbered_assignments(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    fields = "\n".join(f"metric_{index:04d}: int = 0" for index in range(30))
    indented_fields = fields.replace("\n", "\n    ")
    path.write_text(f"class Metrics:\n    {indented_fields}\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("mechanical numbered assignments" in error for error in result["errors"])


def test_audit_rejects_large_top_level_data_block(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    rows = "\n".join(f"    ({index}, 'x')," for index in range(510))
    path.write_text(f"RULES = (\n{rows}\n)\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("large top-level data block" in error for error in result["errors"])


def test_audit_rejects_oversized_function_body(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    branches = "\n".join(f"    if x == {index}:\n        return {index}" for index in range(260))
    path.write_text(f"def dispatch(x):\n{branches}\n    return -1\n", encoding="utf-8")
    result = audit_candidate(path)
    assert not result["ok"]
    assert any("oversized function body" in error for error in result["errors"])


def test_audit_accepts_seed_candidate() -> None:
    result = audit_candidate(Path("candidate/harness.py"))
    assert result["ok"], result["errors"]
