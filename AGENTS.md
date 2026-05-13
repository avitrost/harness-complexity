# Harness Complexity Optimization

You are the meta-optimizer for a TerminalBench 2.0 harness experiment.

Meta-optimizer model:
- Codex GPT-5.5 Medium

Terminal-solving model:
- GPT-5.4 Nano

Your task:
- Improve candidate/harness.py.
- The file must be at most {BUDGET} physical lines after Black formatting.
- The harness should improve validation performance on the selected TerminalBench 2.0 val split.
- Optimize for general terminal-agent behavior, not task-specific hacks.

You may inspect:
- candidate/harness.py
- proposal.md
- validation failure details from this same current attempt if they are provided directly

You may edit:
- candidate/harness.py
- proposal.md

You must not edit:
- plumbing/
- evaluator/
- scripts/
- experience/
- final_test/
- results/
- pyproject.toml

Hard constraints:
- All behavior-changing logic must be in candidate/harness.py.
- Prompt text counts toward the line budget.
- Do not import behavior from uncounted files.
- Do not hardcode task names, task slugs, test names, expected answers, or oracle behavior.
- Do not read benchmark solutions, tests, hidden files, split files, experience logs, or final-test results at runtime.
- Do not inspect parent directories or prior experiment runs.
- Do not use eval, exec, dynamic imports, or generated Python code.
- Do not optimize for validation tasks by name.
- Prefer robust general strategies for terminal problem solving.

Output requirements:
- Write candidate/harness.py.
- Write proposal.md containing:
  - what current workspace files you inspected
  - observed failure modes
  - hypothesis
  - changes made
  - expected benefit
  - risks
