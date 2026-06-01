import json
import sys

from scripts import run_tb2_core


def test_tb2_core_runner_writes_manifest_and_uses_requested_candidates(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run_candidate(candidate, root, tasks, args):
        calls.append((candidate.name, list(tasks), args.concurrency))
        return {"candidate": candidate.name, "ran": False, "dry_run": True}

    monkeypatch.setattr(run_tb2_core, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tb2_core",
            "--run-id",
            "test",
            "--out-root",
            str(tmp_path),
            "--candidate",
            "seed_codex_400",
            "--candidate",
            "codex_cli",
            "--concurrency",
            "80",
            "--max-candidate-workers",
            "2",
            "--dry-run",
        ],
    )

    assert run_tb2_core.main() == 0
    assert calls == [
        ("seed_codex_400", run_tb2_core.get_tb2_core_tasks(), 80),
        ("codex_cli", run_tb2_core.get_tb2_core_tasks(), 80),
    ]
    manifest = json.loads((tmp_path / "test" / "manifest.json").read_text())
    assert manifest["split"] == "tb2-core"
    assert manifest["trials"] == 10
    assert manifest["concurrency_per_candidate"] == 80
    assert manifest["max_candidate_workers"] == 2
