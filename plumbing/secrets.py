from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SECRETS_FILE = Path.home() / ".config" / "harness-complexity" / "secrets.env"


def get_secret(name: str) -> str | None:
    load_secrets_file()
    value = os.environ.get(name)
    return value if value else None


def require_openai_api_key() -> str:
    return _require_secret("OPENAI_API_KEY")


def require_anthropic_api_key() -> str:
    return _require_secret("ANTHROPIC_API_KEY")


def require_deepseek_api_key() -> str:
    return _require_secret("DEEPSEEK_API_KEY")


def load_secrets_file(path: Path | str | None = None) -> None:
    secrets_path = Path(path).expanduser() if path is not None else _secrets_file_path()
    try:
        lines = secrets_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"failed to read secrets file {secrets_path}: {exc}") from exc
    for line in lines:
        key, value = _parse_secret_line(line)
        if key and (key not in os.environ or not os.environ[key]):
            os.environ[key] = value


def _require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _secrets_file_path() -> Path:
    configured = os.getenv("HARNESS_SECRETS_FILE")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_SECRETS_FILE


def _parse_secret_line(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return "", ""
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    key, separator, value = stripped.partition("=")
    if not separator:
        return "", ""
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return "", ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value
