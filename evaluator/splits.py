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
HELDOUT_TASKS = [
    "gpt2-codegolf",
    "llm-inference-batching-scheduler",
    "break-filter-js-from-html",
    "winning-avg-corewars",
    "log-summary-date-ranges",
    "largest-eigenval",
    "password-recovery",
    "regex-chess",
    "torch-tensor-parallelism",
    "path-tracing",
    "prove-plus-comm",
    "feal-linear-cryptanalysis",
    "caffe-cifar-10",
    "distribution-search",
    "mteb-retrieve",
    "pypi-server",
    "custom-memory-heap-crash",
    "multi-source-data-merger",
    "configure-git-webserver",
    "crack-7z-hash",
    "chess-best-move",
    "cobol-modernization",
    "overfull-hbox",
    "polyglot-rust-c",
    "compile-compcert",
    "db-wal-recovery",
    "headless-terminal",
    "schemelike-metacircular-eval",
    "qemu-startup",
    "build-pov-ray",
    "train-fasttext",
    "video-processing",
    "kv-store-grpc",
    "install-windows-3.11",
    "make-doom-for-mips",
    "torch-pipeline-parallelism",
    "tune-mjcf",
    "extract-moves-from-video",
    "gcode-to-text",
    "make-mips-interpreter",
    "count-dataset-tokens",
    "circuit-fibsqrt",
    "mteb-leaderboard",
    "mailman",
    "raman-fitting",
    "query-optimize",
    "extract-elf",
    "protein-assembly",
    "code-from-image",
    "financial-document-processor",
    "mcmc-sampling-stan",
    "filter-js-from-html",
    "polyglot-c-py",
    "cancel-async-tasks",
    "bn-fit-modify",
    "git-leak-recovery",
    "build-cython-ext",
    "pytorch-model-recovery",
    "sam-cell-seg",
    "model-extraction-relu-logits",
    "nginx-request-logging",
    "sanitize-git-repo",
    "build-pmars",
    "rstan-to-pystan",
    "sqlite-with-gcov",
    "constraints-scheduling",
    "dna-insert",
    "vulnerable-secret",
    "dna-assembly",
]
VAL_TRIALS = 4
VAL_CONCURRENCY = 160
TEST_TRIALS = 4
TEST_CONCURRENCY = 160
HELDOUT_TRIALS = 2
HELDOUT_CONCURRENCY = 300


def get_val_tasks() -> list[str]:
    return list(VAL_TASKS)


def get_test_tasks() -> list[str]:
    return list(TEST_TASKS)


def get_heldout_tasks() -> list[str]:
    return list(HELDOUT_TASKS)


def val_estimated_full_score(val_split_mean: float) -> float:
    return val_split_mean


def test_estimated_full_score(test_split_mean: float) -> float:
    return test_split_mean
