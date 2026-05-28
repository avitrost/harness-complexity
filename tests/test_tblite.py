from __future__ import annotations

from pathlib import Path

import pytest

from evaluator import tblite


def test_discover_tblite_tasks_uses_task_toml(tmp_path: Path) -> None:
    _make_task(tmp_path, "b-task")
    _make_task(tmp_path, "a-task")
    (tmp_path / "assets").mkdir()
    (tmp_path / "README.md").write_text("dataset card\n", encoding="utf-8")

    assert tblite.discover_tblite_tasks(tmp_path) == ["a-task", "b-task"]


def test_select_tblite_tasks_validates_requested_names(tmp_path: Path) -> None:
    _make_tasks(tmp_path, tblite.TBLITE_EXPECTED_TASKS)

    assert tblite.select_tblite_tasks(tmp_path, ["task-003"]) == ["task-003"]
    with pytest.raises(ValueError, match="Unknown OpenThoughts-TBLite task"):
        tblite.select_tblite_tasks(tmp_path, ["missing"])


def test_materialize_tblite_uses_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    _make_tasks(snapshot_root, tblite.TBLITE_EXPECTED_TASKS)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr(tblite, "snapshot_download", fake_snapshot_download)

    result = tblite.materialize_tblite(cache_root=tmp_path / "cache", download_method="snapshot")

    assert result == snapshot_root
    assert calls == [
        {
            "repo_id": tblite.TBLITE_DATASET_ID,
            "repo_type": "dataset",
            "revision": tblite.TBLITE_REVISION,
            "local_dir": tmp_path / "cache" / "OpenThoughts-TBLite" / tblite.TBLITE_REVISION,
            "local_files_only": False,
            "headers": {"Accept-Encoding": "identity"},
            "max_workers": 4,
        }
    ]


def test_materialize_tblite_retries_serial_download(monkeypatch, tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    _make_tasks(snapshot_root, tblite.TBLITE_EXPECTED_TASKS)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs["max_workers"])
        if len(calls) == 1:
            raise RuntimeError("transient download failure")
        return str(snapshot_root)

    monkeypatch.setattr(tblite, "snapshot_download", fake_snapshot_download)

    assert (
        tblite.materialize_tblite(cache_root=tmp_path / "cache", download_method="snapshot")
        == snapshot_root
    )
    assert calls == [4, 1]


def test_materialize_tblite_falls_back_to_git(monkeypatch, tmp_path: Path) -> None:
    git_root = tmp_path / "git-snapshot"
    _make_tasks(git_root, tblite.TBLITE_EXPECTED_TASKS)
    calls = []

    def fake_snapshot_download(**kwargs):
        raise RuntimeError("snapshot failed")

    def fake_git_materialize(repo_url, revision, target):
        calls.append((repo_url, revision, target))
        return git_root

    monkeypatch.setattr(tblite, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(tblite, "_git_materialize_tblite", fake_git_materialize)

    assert (
        tblite.materialize_tblite(cache_root=tmp_path / "cache", download_method="snapshot")
        == git_root
    )
    assert calls == [
        (
            tblite.TBLITE_GIT_URL,
            tblite.TBLITE_REVISION,
            tmp_path / "cache" / "OpenThoughts-TBLite" / f"{tblite.TBLITE_REVISION}-git",
        )
    ]


def _make_tasks(root: Path, count: int) -> None:
    for index in range(count):
        _make_task(root, f"task-{index:03d}")


def _make_task(root: Path, name: str) -> None:
    task_dir = root / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n", encoding="utf-8")
