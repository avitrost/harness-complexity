from __future__ import annotations

import subprocess
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - dependency errors are surfaced at runtime.
    snapshot_download = None

TBLITE_DATASET_ID = "open-thoughts/OpenThoughts-TBLite"
TBLITE_GIT_URL = f"https://huggingface.co/datasets/{TBLITE_DATASET_ID}"
TBLITE_REVISION = "7b70111339b4af23cece95d63aeec1c705790868"
TBLITE_EXPECTED_TASKS = 100
TBLITE_TRIALS = 5
TBLITE_CONCURRENCY = 160
TBLITE_SPLIT = "tblite"
TBLITE_CACHE_ROOT = Path("external_datasets")


def materialize_tblite(
    cache_root: Path = TBLITE_CACHE_ROOT,
    revision: str = TBLITE_REVISION,
    dataset_id: str = TBLITE_DATASET_ID,
    local_files_only: bool = False,
    download_workers: int = 4,
    download_method: str = "git",
) -> Path:
    if download_method not in {"git", "snapshot"}:
        raise ValueError(f"unsupported OpenThoughts-TBLite download method: {download_method}")
    if download_method == "snapshot" and snapshot_download is None:
        raise RuntimeError("huggingface_hub is required to download OpenThoughts-TBLite")
    target = cache_root / "OpenThoughts-TBLite" / revision
    target.parent.mkdir(parents=True, exist_ok=True)
    if download_method == "git":
        if local_files_only:
            path = target.parent / f"{revision}-git"
        else:
            path = _git_materialize_tblite(
                repo_url=f"https://huggingface.co/datasets/{dataset_id}",
                revision=revision,
                target=target.parent / f"{revision}-git",
            )
        validate_tblite_dataset(path)
        return path
    try:
        path = _snapshot_download_with_retry(
            repo_id=dataset_id,
            revision=revision,
            local_dir=target,
            local_files_only=local_files_only,
            download_workers=download_workers,
        )
        validate_tblite_dataset(path)
        return path
    except Exception:
        if local_files_only:
            raise
        path = _git_materialize_tblite(
            repo_url=f"https://huggingface.co/datasets/{dataset_id}",
            revision=revision,
            target=target.parent / f"{revision}-git",
        )
    validate_tblite_dataset(path)
    return path


def _snapshot_download_with_retry(
    repo_id: str,
    revision: str,
    local_dir: Path,
    local_files_only: bool,
    download_workers: int,
) -> Path:
    attempts = [max(1, download_workers)]
    if attempts[0] != 1:
        attempts.append(1)
    last_error: Exception | None = None
    for workers in attempts:
        try:
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=local_dir,
                    local_files_only=local_files_only,
                    headers={"Accept-Encoding": "identity"},
                    max_workers=workers,
                )
            )
        except Exception as exc:
            last_error = exc
            if local_files_only:
                break
    assert last_error is not None
    raise last_error


def _git_materialize_tblite(repo_url: str, revision: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".materialize.lock")
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            time.sleep(2)
    try:
        if target.exists() and not (target / ".git").is_dir():
            raise RuntimeError(f"Cannot use non-git OpenThoughts-TBLite fallback path: {target}")
        if not target.exists():
            _run_git(["git", "init", str(target)])
            _run_git(["git", "-C", str(target), "remote", "add", "origin", repo_url])
        index_lock = target / ".git" / "index.lock"
        if index_lock.exists():
            index_lock.unlink()
        _run_git(["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision])
        _run_git(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"])
        return target
    finally:
        lock.rmdir()


def _run_git(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Git dataset materialization failed: {' '.join(command)}\n{result.stderr}"
        )


def discover_tblite_tasks(dataset_path: Path) -> list[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"OpenThoughts-TBLite path does not exist: {dataset_path}")
    return sorted(
        path.name
        for path in dataset_path.iterdir()
        if path.is_dir() and (path / "task.toml").is_file()
    )


def validate_tblite_dataset(
    dataset_path: Path,
    expected_tasks: int = TBLITE_EXPECTED_TASKS,
) -> list[str]:
    tasks = discover_tblite_tasks(dataset_path)
    if len(tasks) != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} OpenThoughts-TBLite tasks in {dataset_path}, "
            f"found {len(tasks)}."
        )
    return tasks


def select_tblite_tasks(dataset_path: Path, requested_tasks: list[str] | None) -> list[str]:
    available = validate_tblite_dataset(dataset_path)
    if not requested_tasks:
        return available
    available_set = set(available)
    missing = sorted(set(requested_tasks) - available_set)
    if missing:
        raise ValueError(f"Unknown OpenThoughts-TBLite task(s): {', '.join(missing)}")
    return list(requested_tasks)
