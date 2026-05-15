# Proposal

## Current Workspace Files Inspected

- `candidate/harness.py`
- `proposal.md`

External reference read at the user's request:

- `https://www.mihaileric.com/The-Emperor-Has-No-Clothes/`

## Observed Failure Modes

The 60-line harness had the right basic shape, but it left useful agent-loop
behavior implicit. A small terminal model can still repeat failed commands,
return fenced or alternate tool-call formats, stop without evidence, or lose the
connection between recent writes and the need to verify.

## Hypothesis

The article's core pattern is a compact tool loop: tell the model what tools it
has, parse structured calls, execute one step, feed back results. In this
environment the only tool the counted harness can expose is a shell command, so
a stronger one-tool contract plus better parsing and state hints should improve
general TerminalBench behavior without task-specific logic.

## Changes Made

`candidate/harness.py` now:

- uses a deterministic first `pwd && ls -la` inspection;
- frames the model as a one-tool terminal agent;
- carries six clipped, indexed terminal observations;
- adds state hints for failed commands, repeated commands, and recent writes;
- accepts strict JSON, fenced JSON, and simple `tool: run({...})` forms;
- normalizes `action`/`tool`/`name`, `command`/`cmd`, and nested args;
- keeps `done` gated in the prompt by verification evidence.

The file is 127 physical lines after Black formatting.

## Expected Benefit

The terminal model should spend fewer turns on malformed outputs, repeated
failures, and premature `done` actions, while getting a clearer inspect-edit-test
loop.

## Risks

The fixed first inspection costs one turn on tasks where the next action is
obvious. More prompt and parsing logic may help malformed responses but cannot
force the model to choose a good command.
