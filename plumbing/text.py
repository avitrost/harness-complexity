from __future__ import annotations


def compact_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
