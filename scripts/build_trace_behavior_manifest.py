from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_prior_openai10_aggregate import (  # noqa: E402
    DEFAULT_LOW10_ROOT,
    HARNESS_ALIASES,
    PRIOR_CANDIDATES,
    TASKS,
    attempt_row,
    newest_valid_record,
    slug,
)

DEFAULT_PROVIDER_ATTEMPTS = Path(
    "/wbl-fast/usrs/trost/harness-complexity/final_test/"
    "tb2_provider_matrix_openai_20260607_010143/attempts.csv"
)

MODEL_DIRS = {
    "gpt-5.4-mini": "gpt_5_4_mini",
    "gpt-5.4": "gpt_5_4",
    "gpt-5.5": "gpt_5_5",
}

PROVIDER_MATRIX_CANDIDATES = {
    "seed_mini_swe_agent_barebones",
    "seed_mini_swe_agent_barebones_v2_persistent",
    "seed_mini_swe_agent_barebones_v2_rich_terminal",
    "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples",
    "seed_mini_swe_agent_v2",
}

HARNESS_INFO = {
    "seed_minimal_agent": ("minimal", "minimal-agent", 100),
    "seed_mini_swe_agent_barebones": ("mini_bare", "msa-fully-compressed", 149),
    "seed_codex_400": ("c400", "codex-400", 398),
    "seed_mini_swe_agent_barebones_v2": ("bare_v2", "msa-prompt-compressed", 408),
    "seed_mini_swe_agent_barebones_v2_persistent": (
        "bbv2_bash_persistent",
        "msa-prompt-compressed-hidden-persistent-bash",
        408,
    ),
    "seed_mini_swe_agent_barebones_v2_rich_terminal": (
        "bbv2_rich_terminal",
        "msa-prompt-compressed-rich-terminal",
        408,
    ),
    "seed_mini_swe_agent_barebones_v2_rich_terminal_no_examples": (
        "bbv2_rich_no_examples",
        "msa-prompt-compressed-rich-terminal-no-examples",
        408,
    ),
    "seed_mini_swe_agent_v2": ("mini_v2", "mini-swe-agent", 478),
    "seed_terminus_2_compressed": ("term2_comp", "terminus-2-compressed", 634),
    "seed_codex_700": ("c700", "codex-700", 700),
    "seed_codex_1000": ("c1000", "codex-1000", 1000),
    "seed_codex_1300": ("c1300", "codex-1300", 1300),
    "seed_codex_compressed": ("c_comp", "codex-compressed", 1660),
    "seed_codex_full": ("c_full", "codex-full", 2210),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an attempts manifest for trace analysis that mirrors the gpt-5.4-mini/"
            "gpt-5.4 low cells used by viz/sparkline_tables_v2.html."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low10-root", type=Path, default=DEFAULT_LOW10_ROOT)
    parser.add_argument("--provider-attempts", type=Path, default=DEFAULT_PROVIDER_ATTEMPTS)
    parser.add_argument("--model", action="append", required=True, choices=sorted(MODEL_DIRS))
    parser.add_argument("--effort", default="low")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for model in args.model:
        rows.extend(_prior_rows(args.low10_root, model, args.effort))
        rows.extend(_provider_matrix_rows(args.provider_attempts, model, args.effort))

    rows.sort(
        key=lambda row: (
            str(row.get("model", "")),
            int(row.get("harness_loc") or 0),
            str(row.get("candidate", "")),
            str(row.get("task", "")),
            int(row.get("attempt") or 0),
        )
    )
    _write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    for model in args.model:
        model_rows = [row for row in rows if row.get("model") == model]
        harnesses = sorted({str(row["candidate"]) for row in model_rows})
        print(
            f"{model} {args.effort}: {len(model_rows)} attempts across {len(harnesses)} harnesses"
        )
    return 0


def _prior_rows(root: Path, model: str, effort: str) -> list[dict[str, Any]]:
    model_dir = MODEL_DIRS[model]
    config_id = f"openai_{slug(model)}_{effort}"
    rows: list[dict[str, Any]] = []
    for candidate in PRIOR_CANDIDATES:
        for task in TASKS:
            for attempt in range(1, 11):
                attempt_dir = root / model_dir / candidate / task / f"attempt_{attempt:02d}"
                run_dir, record, _valid_count = newest_valid_record(attempt_dir, task)
                row = attempt_row(
                    effort=effort,
                    model=model,
                    config_id=config_id,
                    candidate=candidate,
                    task=task,
                    attempt=attempt,
                    source_run_id=root.name,
                    selected_run_dir=run_dir,
                    record=record,
                )
                row["out_dir"] = str(run_dir or attempt_dir)
                row["benchmark"] = "tb2-core"
                _add_harness_info(row)
                rows.append(row)
    return rows


def _provider_matrix_rows(path: Path, model: str, effort: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("provider") != "openai":
                continue
            if row.get("model") != model or row.get("effort") != effort:
                continue
            if row.get("candidate") not in PROVIDER_MATRIX_CANDIDATES:
                continue
            row = dict(row)
            row.setdefault("source_run_id", path.parent.name)
            if not row.get("source_run_id"):
                row["source_run_id"] = path.parent.name
            row["selected_run_dir"] = row.get("out_dir", "")
            row["benchmark"] = "tb2-core"
            _add_harness_info(row)
            rows.append(row)
    return rows


def _add_harness_info(row: dict[str, Any]) -> None:
    candidate = str(row.get("candidate", ""))
    alias, name, loc = HARNESS_INFO.get(
        candidate,
        (HARNESS_ALIASES.get(candidate, candidate), candidate, ""),
    )
    row["harness_alias"] = alias
    row["harness_name"] = name
    row["harness_loc"] = loc


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "provider",
        "config_id",
        "model",
        "effort",
        "benchmark",
        "candidate",
        "harness_alias",
        "harness_name",
        "harness_loc",
        "task",
        "attempt",
        "status",
        "reward",
        "failure_class",
        "corrupted",
        "api_input_tokens",
        "api_output_tokens",
        "api_cached_tokens",
        "api_total_tokens",
        "source_run_id",
        "out_dir",
        "selected_run_dir",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
