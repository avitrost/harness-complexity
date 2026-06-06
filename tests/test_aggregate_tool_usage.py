import json

from scripts.aggregate_tool_usage import aggregate_tool_usage, write_tool_usage_outputs


def test_aggregate_tool_usage_groups_harness_turns(tmp_path) -> None:
    run_root = tmp_path / "run"
    agent_dir = (
        run_root
        / "gpt_5_4_mini"
        / "seed_codex_full"
        / "bn-fit-modify"
        / "attempt_01"
        / "2026-06-02__00-00-00"
        / "bn-fit-modify__abc"
        / "agent"
    )
    agent_dir.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"models": ["gpt-5.4-mini"]}),
        encoding="utf-8",
    )
    (agent_dir / "harness-turn-01.json").write_text(
        json.dumps(
            {
                "command": "pytest",
                "return_code": 0,
                "stdout": "ok",
                "tool_name": "exec_command",
                "metadata": {
                    "unified_exec": {
                        "wall_time_seconds": 1.5,
                        "original_token_count": 12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "harness-turn-02.json").write_text(
        json.dumps(
            {
                "command": "apply_patch",
                "return_code": 1,
                "stderr": "failed",
                "tool_name": "apply_patch",
                "metadata": {"patch_bytes": 99},
            }
        ),
        encoding="utf-8",
    )

    summary = aggregate_tool_usage(run_root)

    assert summary["total_tool_events"] == 2
    assert summary["total_trials_with_tools"] == 1
    by_tool = {row["tool_name"]: row for row in summary["by_tool"]}
    assert by_tool["exec_command"]["events"] == 1
    assert by_tool["exec_command"]["mean_wall_time_seconds"] == 1.5
    assert by_tool["exec_command"]["total_original_token_count"] == 12
    assert by_tool["apply_patch"]["failures"] == 1
    assert by_tool["apply_patch"]["total_patch_bytes"] == 99
    assert summary["by_model"][0]["model"] == "gpt-5.4-mini"


def test_write_tool_usage_outputs(tmp_path) -> None:
    summary = {
        "run_root": str(tmp_path),
        "total_tool_events": 1,
        "total_trials_with_tools": 1,
        "by_tool": [{"tool_name": "exec_command", "events": 1}],
        "by_model": [{"model": "gpt-5.4", "events": 1}],
        "by_candidate": [{"candidate": "seed_codex_full", "events": 1}],
        "by_task": [{"task": "bn-fit-modify", "events": 1}],
        "by_model_candidate_tool": [
            {
                "model": "gpt-5.4",
                "candidate": "seed_codex_full",
                "tool_name": "exec_command",
                "events": 1,
            }
        ],
        "by_task_tool": [{"task": "bn-fit-modify", "tool_name": "exec_command", "events": 1}],
    }

    write_tool_usage_outputs(summary, tmp_path)

    assert (tmp_path / "tool_usage_summary.json").is_file()
    assert (tmp_path / "tool_usage_tool.csv").is_file()
    assert (tmp_path / "tool_usage_model_candidate_tool.csv").is_file()
