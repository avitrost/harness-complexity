# Terminus-KIRA Reference

Terminus-KIRA is the main reference baseline used by the official Meta-Harness
TerminalBench experiment. Treat it as design inspiration only; do not import it or
depend on this file at runtime.

Useful general patterns:

- Native structured tool calls instead of fragile free-form command parsing.
- A larger but bounded terminal observation cap; KIRA uses about 30 KB.
- Marker-based command polling to avoid wasting time after commands finish early.
- Completion confirmation that checks requirements, robustness, and user intent from
  multiple perspectives before marking a task done.
- Context-overflow recovery through summarization rather than dropping the run.
- Environment bootstrapping: collect a small, timeout-guarded snapshot of the
  working directory, available languages/tools, package managers, and memory before
  the first model turn.

For this experiment, adapt only general ideas that fit inside
`candidate/harness.py` and the active physical line budget.
