# Proposal

## Current workspace files inspected

- `candidate/harness.py`
- `proposal.md`

## Observed failure modes

The baseline asked the terminal model to return raw command text. In a smoke run,
that style can leak prose, duplicate commands, or make it ambiguous whether the
model wants to continue or stop.

## Hypothesis

A minimal action protocol should make the harness closer to a real tool-using
agent while staying small. Requiring JSON for either `run` or `done` gives the
model a clearer contract and gives the harness a narrow place to parse responses.

## Changes made

`candidate/harness.py` now asks the terminal model for one JSON action per turn:
`{"action":"run","command":"..."}` or `{"action":"done"}`. The harness parses
that action, keeps recent terminal history compact, and returns the command to
the Harbor adapter.

## Expected benefit

The terminal model should produce fewer malformed shell commands and should be
less likely to mix final-answer prose with executable commands.

## Risks

The model may still ignore the JSON contract. The fallback preserves raw text as
a command so a bad response can still fail at the shell, but the primary path is
now structured and minimal.
