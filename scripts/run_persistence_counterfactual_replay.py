from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.parse_results import parse_records  # noqa: E402
from evaluator.run_val import run_split  # noqa: E402
from plumbing.harbor_adapter import TERMINAL_BENCH_DATASET  # noqa: E402
from scripts.run_tb2_core import TB2_CORE_SPLIT  # noqa: E402

NP_CANDIDATE = "seed_mini_swe_agent_barebones_v2"
PERSISTENT_CANDIDATE = "seed_mini_swe_agent_barebones_v2_persistent"
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@dataclass(frozen=True)
class AttemptRow:
    provider: str
    config_id: str
    model: str
    effort: str
    candidate: str
    task: str
    attempt: str
    reward: float
    status: str
    out_dir: Path


@dataclass(frozen=True)
class ReplaySpec:
    replay_id: str
    source: AttemptRow
    original_peer: AttemptRow
    replay_mode: str
    commands: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id", default="persistence_counterfactual_replay")
    parser.add_argument("--backend", choices=("docker", "slurm-pyxis"), default="slurm-pyxis")
    parser.add_argument("--dataset", default=TERMINAL_BENCH_DATASET)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--slurm-partition", default="m7i-cpu2")
    parser.add_argument("--harbor-bin")
    parser.add_argument("--limit-per-direction", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit_per_direction < 1:
        raise ValueError("--limit-per-direction must be >= 1")
    if args.backend == "slurm-pyxis" and not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to run Harbor/evals outside Slurm. Submit with sbatch/salloc/srun."
        )

    root = args.out_root / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    specs = _replay_specs(_read_attempts(args.attempts_csv), args.limit_per_direction)
    _write_json(root / "replay_manifest.json", [_spec_manifest(spec) for spec in specs])
    _write_replay_harness(root)
    rows = [_run_replay(root, args, spec) for spec in specs]
    _write_csv(root / "counterfactual_replay.csv", rows)
    summary = {
        "run_id": args.run_id,
        "out_root": str(root),
        "replays": len(rows),
        "dry_run": args.dry_run,
        "counterfactual_replay": str(root / "counterfactual_replay.csv"),
    }
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(row.get("returncode") == "0" for row in rows) else 1


def _read_attempts(path: Path) -> list[AttemptRow]:
    rows: list[AttemptRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = str(row.get("candidate") or "")
            if candidate not in {NP_CANDIDATE, PERSISTENT_CANDIDATE}:
                continue
            rows.append(
                AttemptRow(
                    provider=str(row.get("provider") or ""),
                    config_id=str(row.get("config_id") or ""),
                    model=str(row.get("model") or ""),
                    effort=str(row.get("effort") or ""),
                    candidate=candidate,
                    task=str(row.get("task") or ""),
                    attempt=str(row.get("attempt") or ""),
                    reward=_reward(row.get("reward")),
                    status=str(row.get("status") or ""),
                    out_dir=Path(str(row.get("out_dir") or "")),
                )
            )
    return rows


def _replay_specs(rows: list[AttemptRow], limit_per_direction: int) -> list[ReplaySpec]:
    by_cell: dict[tuple[str, str, str, str, str], dict[str, AttemptRow]] = {}
    for row in rows:
        key = (row.provider, row.model, row.effort, row.task, row.attempt)
        by_cell.setdefault(key, {})[row.candidate] = row

    specs: list[ReplaySpec] = []
    direction_counts = {"np_to_persistent": 0, "persistent_to_np": 0}
    for _key, pair in sorted(by_cell.items()):
        np_row = pair.get(NP_CANDIDATE)
        persistent_row = pair.get(PERSISTENT_CANDIDATE)
        if np_row is None or persistent_row is None:
            continue
        if np_row.reward == persistent_row.reward:
            continue
        if np_row.reward > persistent_row.reward:
            direction = "np_to_persistent"
            source = np_row
            peer = persistent_row
            replay_mode = "persistent"
        else:
            direction = "persistent_to_np"
            source = persistent_row
            peer = np_row
            replay_mode = "nonpersistent"
        if direction_counts[direction] >= limit_per_direction:
            continue
        commands = tuple(_extract_bash_commands(source.out_dir))
        if not commands:
            continue
        direction_counts[direction] += 1
        specs.append(
            ReplaySpec(
                replay_id=f"{direction}_{direction_counts[direction]:03d}",
                source=source,
                original_peer=peer,
                replay_mode=replay_mode,
                commands=commands,
            )
        )
    return specs


def _extract_bash_commands(out_dir: Path) -> list[str]:
    agent_dir = _agent_dir(out_dir)
    if agent_dir is None:
        return []
    commands: list[str] = []
    for path in sorted(agent_dir.glob("harness-turn-*.json")):
        payload = _read_json(path)
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        items = metadata.get("mini_swe_agent_v2_response_items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            if str(item.get("name") or "") != "bash":
                continue
            arguments = _json_object(item.get("arguments"))
            command = str(arguments.get("command") or "").strip()
            if command:
                commands.append(command)
                if command == SUBMIT_COMMAND:
                    return commands
    return commands


def _agent_dir(out_dir: Path) -> Path | None:
    if (out_dir / "agent").is_dir():
        return out_dir / "agent"
    for child in sorted(out_dir.glob("**/agent")):
        if child.is_dir():
            return child
    return None


def _run_replay(root: Path, args: argparse.Namespace, spec: ReplaySpec) -> dict[str, str]:
    spec_dir = root / "replay_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{spec.replay_id}.json"
    _write_json(
        spec_path,
        {
            "replay_id": spec.replay_id,
            "mode": spec.replay_mode,
            "commands": list(spec.commands),
        },
    )
    out_dir = root / "replays" / spec.replay_id
    candidate_dir = root / f"_replay_candidate_{spec.replay_mode}"
    summary = run_split(
        split=TB2_CORE_SPLIT,
        candidate_dir=candidate_dir,
        budget=0,
        out_dir=out_dir,
        tasks=[spec.source.task],
        trials=1,
        concurrency=1,
        dry_run=args.dry_run,
        harbor_bin=args.harbor_bin,
        backend=args.backend,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        agent_env=(
            "TERMINAL_MODEL_PROVIDER=openai",
            "OPENAI_AUTH_MODE=codex",
            f"HC_REPLAY_SPEC={spec_path}",
            f"HC_REPLAY_MODE={spec.replay_mode}",
        ),
    )
    records = parse_records(out_dir)
    record = records[0] if records else {}
    return {
        "replay_id": spec.replay_id,
        "source_candidate": spec.source.candidate,
        "source_model": spec.source.model,
        "source_effort": spec.source.effort,
        "source_task": spec.source.task,
        "source_attempt": spec.source.attempt,
        "source_reward": str(spec.source.reward),
        "peer_candidate": spec.original_peer.candidate,
        "peer_reward": str(spec.original_peer.reward),
        "replay_mode": spec.replay_mode,
        "commands": str(len(spec.commands)),
        "status": str(record.get("status") or ""),
        "reward": str(record.get("reward") if record.get("reward") is not None else ""),
        "returncode": str(int(summary.get("returncode", 0))),
        "out_dir": str(out_dir),
        "source_out_dir": str(spec.source.out_dir),
    }


def _write_replay_harness(root: Path) -> None:
    for mode in ("nonpersistent", "persistent"):
        candidate_dir = root / f"_replay_candidate_{mode}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "harness.py").write_text(_replay_harness_source(mode), encoding="utf-8")


def _replay_harness_source(mode: str) -> str:
    persistent_flag = "True" if mode == "persistent" else "False"
    tool_name = "persistent_bash" if mode == "persistent" else "bash"
    return f"""from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from plumbing.base_agent import BaseHarness
from plumbing.types import HarnessToolCall, HarnessTurn

SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ENVIRONMENT_ENV = {{
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}}


class ReplayHarness(BaseHarness):
    wants_persistent_terminal = {persistent_flag}

    def __init__(self):
        spec = json.loads(Path(os.environ["HC_REPLAY_SPEC"]).read_text(encoding="utf-8"))
        self.commands = [str(command) for command in spec.get("commands", [])]

    def next_command(self, task, history):
        if history and history[-1].command.strip() == SUBMIT_COMMAND and history[-1].return_code == 0:
            return HarnessTurn(done=True)
        if len(history) >= len(self.commands):
            return HarnessTurn(done=True)
        command = _runtime_command(self.commands[len(history)])
        args = {{"command": command, "timeout_sec": 30}}
        if {persistent_flag}:
            args["session_id"] = task.metadata["persistent_terminal"]["session_id"]
        return HarnessTurn(
            tool_calls=(HarnessToolCall("{tool_name}", args, f"replay_{{len(history) + 1}}"),),
            metadata={{"sequential_tool_calls": True}},
        )


def create_agent():
    return ReplayHarness()


def _runtime_command(command):
    assignments = " ".join(f"{{key}}={{shlex.quote(value)}}" for key, value in ENVIRONMENT_ENV.items())
    return f"export {{assignments}};\\n{{command}}"
"""


def _spec_manifest(spec: ReplaySpec) -> dict[str, Any]:
    return {
        "replay_id": spec.replay_id,
        "source_candidate": spec.source.candidate,
        "source_model": spec.source.model,
        "source_effort": spec.source.effort,
        "source_task": spec.source.task,
        "source_attempt": spec.source.attempt,
        "source_reward": spec.source.reward,
        "peer_candidate": spec.original_peer.candidate,
        "peer_reward": spec.original_peer.reward,
        "replay_mode": spec.replay_mode,
        "commands": len(spec.commands),
        "source_out_dir": str(spec.source.out_dir),
    }


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _reward(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
