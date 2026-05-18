"""Domain entities and value objects. Pure Python, no I/O, no framework deps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class OllamaAccount:
    """An immutable description of an Ollama backend instance."""

    name: str
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    is_cloud: bool = False
    max_requests_per_minute: int = 30


@dataclass
class AccountMetrics:
    """Mutable runtime metrics for one account."""

    name: str
    requests_made: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tool_calls: int = 0
    consecutive_errors: int = 0
    rate_limit_count: int = 0
    is_rate_limited: bool = False
    last_rate_limit_time: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def snapshot(self) -> "AccountMetrics":
        """Return an independent copy for safe cross-thread reads."""
        return replace(self)


@dataclass(frozen=True)
class HttpResult:
    """Minimal value object the application layer uses for non-streaming results."""

    status: int
    json: Optional[dict]
    text: str
