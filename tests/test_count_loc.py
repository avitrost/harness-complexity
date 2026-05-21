from pathlib import Path

from scripts.count_loc import count_loc


def test_count_loc_counts_black_formatted_physical_lines(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("x=1\n\n# comment\ny = 2\n", encoding="utf-8")
    result = count_loc(path, max_lines=4)
    assert result["physical_loc"] == 4
    assert result["nonblank_noncomment_sloc"] == 2
    assert result["ok"] is True


def test_count_loc_fails_over_budget(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("a = 1\nb = 2\n", encoding="utf-8")
    assert count_loc(path, max_lines=1)["ok"] is False


def test_count_loc_fails_under_bucket_floor(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("a = 1\nb = 2\n", encoding="utf-8")
    assert count_loc(path, min_lines=3, max_lines=4)["ok"] is False


def test_count_loc_sloc_budget_ignores_padding_comments(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text("a = 1\n# padding\n# padding\n", encoding="utf-8")
    result = count_loc(path, min_sloc=2, max_sloc=4)
    assert result["physical_loc"] == 3
    assert result["nonblank_noncomment_sloc"] == 1
    assert result["ok"] is False


def test_count_loc_sloc_budget_ignores_multiline_string_bodies(tmp_path: Path) -> None:
    path = tmp_path / "harness.py"
    path.write_text('DATA = """\\\nsynthetic\npadding\n""".splitlines()\nx = 1\n', encoding="utf-8")
    result = count_loc(path, min_sloc=4)
    assert result["physical_loc"] == 5
    assert result["nonblank_noncomment_sloc"] == 3
    assert result["ok"] is False
