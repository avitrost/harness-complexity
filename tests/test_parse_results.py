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
            "trial": 3,
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
