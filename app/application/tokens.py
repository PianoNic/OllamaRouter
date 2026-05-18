"""Token estimation. Cheap heuristic, ~4 chars per token."""

from __future__ import annotations


def estimate_tokens(text: object) -> int:
    if text is None:
        return 0
    s = text if isinstance(text, str) else str(text)
    if not s:
        return 0
    return max(1, len(s) // 4)
