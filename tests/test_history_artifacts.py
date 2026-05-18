import json
import shutil
from pathlib import Path

from evaluator.history_artifacts import write_history_artifacts


def test_write_history_artifacts_indexes_scores_frontier_and_traces(tmp_path: Path) -> None:
    candidate = tmp_path / "iter_001_cand_01"
    trial = candidate / "2026-05-18__00-00-00" / "fix-git__abc123"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (trial / "trial.log").write_text("trial output\n", encoding="utf-8")
    (agent / "harness-result.json").write_text('{"done": false}\n', encoding="utf-8")
    (agent / "harness-turn-01.json").write_text("{}\n", encoding="utf-8")
    (agent / "model-call-01.json").write_text("{}\n", encoding="utf-8")
    (trial / "exception.txt").write_text("Traceback\nRuntimeError: boom\n", encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_result": {"metadata": {"done": False, "turns": 7, "last_return_code": 1}},
                "started_at": "2026-05-18T00:00:00Z",
                "finished_at": "2026-05-18T00:01:30Z",
            }
        ),
        encoding="utf-8",
    )
    (candidate / "summary.json").write_text(
        json.dumps(
            {
                "split_mean": 0.25,
                "estimated_full_score": 0.25,
                "num_trials": 4,
                "num_successes": 1,
                "num_crashes": 1,
                "per_task": [
                    {
                        "task": "fix-git",
                        "mean": 0.25,
                        "num_successes": 1,
                        "num_crashes": 1,
                        "mean_runtime": 90.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (candidate / "validation.json").write_text(
        json.dumps({"ok": True, "checks": [{"json": {"physical_loc": 128}}]}),
        encoding="utf-8",
    )

    history = tmp_path / "history"
    history.mkdir()
    history_candidate = history / candidate.name
    shutil.copytree(candidate, history_candidate)
    write_history_artifacts(history, [history_candidate])

    index = json.loads((history / "index.json").read_text(encoding="utf-8"))
    assert index[0]["num_successes"] == 1
    assert index[0]["per_task"]["fix-git"]["mean"] == 0.25

    frontier = json.loads((history / "frontier.json").read_text(encoding="utf-8"))
    assert frontier["best_overall"]["candidate"] == "iter_001_cand_01"
    assert frontier["per_task"]["fix-git"]["candidate"] == "iter_001_cand_01"

    traces = json.loads((history / "trace_index.json").read_text(encoding="utf-8"))
    assert traces[0]["status"] == "crash"
    assert traces[0]["runtime_sec"] == 90.0
    assert traces[0]["result"].endswith("/fix-git__abc123/result.json")
    assert traces[0]["turn_logs_glob"].endswith("/agent/harness-turn-*.json")

    failures = (history / "failures.md").read_text(encoding="utf-8")
    assert "RuntimeError: boom" in failures
    assert "history/iter_001_cand_01/" in failures

    summary_lines = (history / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert '"candidate": "iter_001_cand_01"' in summary_lines

    logs = tmp_path / "logs"
    assert (logs / "frontier_val.json").is_file()
    assert (logs / "trace_index.json").is_file()
    assert (logs / "reports" / "failures.md").is_file()
    assert (tmp_path / "jobs" / "iter_001_cand_01").exists()
