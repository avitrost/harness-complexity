from __future__ import annotations

from pathlib import Path

from evaluator.validate_candidate import validate_candidate

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "seeds" / "mini_swe_agent_barebones_v2_context_only" / "harness.py"


def test_context_only_seed_validates() -> None:
    result = validate_candidate(SEED_PATH, max_lines=20, min_lines=3)

    assert result["ok"], result


def test_context_only_keeps_mini_prompt_and_adds_context(monkeypatch) -> None:
    from plumbing import mini_swe_barebones_v2_context_only as impl

    base = impl._base()
    calls = []

    def fake_model(messages, tools):
        calls.append({"messages": messages, "tools": tools})
        return base.ToolModelResult(
            content="",
            tool_calls=[
                base.ModelToolCall(
                    name="bash",
                    arguments={"command": "pwd"},
                    arguments_text='{"command":"pwd"}',
                    call_id="call_1",
                )
            ],
            response_items=[],
        )

    monkeypatch.setattr(base, "call_terminal_model_with_tools", fake_model)
    agent = impl.create_agent()
    agent.next_command(
        base.TaskContext(
            "Do it.",
            working_dir="/work",
            metadata={"agents_md": [{"path": "AGENTS.md", "content": "Use Slurm."}]},
        ),
        [],
    )

    prompt = calls[0]["messages"][1]["content"]
    assert "<environment_context>" in prompt
    assert "<cwd>/work</cwd>" in prompt
    assert "<agents_md path='AGENTS.md'>" in prompt
    assert "Please solve this issue: Do it." in prompt
    assert "Recommended Workflow" not in prompt
