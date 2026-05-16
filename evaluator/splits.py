from __future__ import annotations

BASE_VAL_TASKS = [
    "fix-git",
    "qemu-alpine-ssh",
    "sparql-university",
    "sqlite-db-truncate",
    "write-compressor",
]

BASE_TEST_TASKS = [
    "adaptive-rejection-sampler",
    "feal-differential-cryptanalysis",
    "fix-code-vulnerability",
    "fix-ocaml-gc",
    "git-multibranch",
    "hf-model-inference",
    "large-scale-text-editing",
    "merge-diff-arc-agi-task",
    "modernize-scientific-stack",
    "openssl-selfsigned-cert",
    "path-tracing-reverse",
    "portfolio-optimization",
    "pytorch-model-cli",
    "regex-log",
    "reshard-c4-data",
]

VAL_TASKS = [*BASE_VAL_TASKS, *BASE_TEST_TASKS]
TEST_TASKS: list[str] = []
VAL_TRIALS = 4
VAL_CONCURRENCY = 10
TEST_TRIALS = 4
TEST_CONCURRENCY = 10


def get_val_tasks() -> list[str]:
    return list(VAL_TASKS)


def get_test_tasks() -> list[str]:
    return list(TEST_TASKS)


def val_estimated_full_score(val_split_mean: float) -> float:
    return val_split_mean


def test_estimated_full_score(test_split_mean: float) -> float:
    return test_split_mean
