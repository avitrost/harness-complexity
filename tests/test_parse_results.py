import json
from pathlib import Path

from evaluator.parse_results import parse_records


def test_parse_harbor_job_result(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "fix-git",
                        "trial_name": "fix-git__3",
                        "verifier_result": {"rewards": {"reward": 1}},
                        "started_at": "2026-05-13T00:00:00",
                        "finished_at": "2026-05-13T00:00:05",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert parse_records(job) == [
        {
            "task": "fix-git",
            "trial": 1,
            "reward": 1,
            "status": "success",
            "runtime_sec": 5.0,
        }
    ]


def test_parse_harbor_crash_result(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "trial_name": "fix-git__1",
                "exception_info": {"exception_type": "AgentTimeoutError"},
            }
        ),
        encoding="utf-8",
    )
    [record] = parse_records(trial)
    assert record["status"] == "crash"
    assert record["reward"] == 0


def test_parse_harbor_random_trial_suffixes_do_not_dedupe(tmp_path: Path) -> None:
    for name in ("fix-git__26vwnxY", "fix-git__YmQYwNf"):
        trial = tmp_path / name
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "fix-git",
                    "trial_name": name,
                    "exception_info": {"exception_type": "RuntimeError"},
                }
            ),
            encoding="utf-8",
        )
    records = parse_records(tmp_path)
    assert [record["trial"] for record in records] == [1, 2]
    assert len(records) == 2
