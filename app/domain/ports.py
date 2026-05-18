"""Ports (Protocols) describing what the application needs from the outside world.

Adapters in `app.infrastructure` implement these. The application layer depends
only on these abstractions.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from .entities import AccountMetrics, HttpResult, OllamaAccount


@runtime_checkable
class AccountRepository(Protocol):
    """Loads the configured Ollama accounts."""

    def load_all(self) -> list[OllamaAccount]:
        ...


@runtime_checkable
class MetricsRepository(Protocol):
    """Persists per-account metrics."""

    async def load_all(self) -> dict[str, AccountMetrics]:
        ...

    async def save(self, metrics: AccountMetrics) -> None:
        ...


class OllamaStream(AbstractAsyncContextManager["OllamaStream"], Protocol):
    """A live response stream from an Ollama backend.

    `aiter_lines` yields one JSON-encoded chunk per line as produced by Ollama.
    """

    status: int

    def aiter_lines(self) -> AsyncIterator[str]:
        ...


@runtime_checkable
class OllamaGateway(Protocol):
    """The outgoing HTTP boundary to Ollama instances."""

    async def post_json(self, account: OllamaAccount, path: str, payload: dict) -> HttpResult:
        ...

    async def get_json(self, account: OllamaAccount, path: str, timeout: float = 10.0) -> HttpResult:
        ...

    def stream(self, account: OllamaAccount, path: str, payload: dict) -> AbstractAsyncContextManager[OllamaStream]:
        ...

    async def close(self) -> None:
        ...


@runtime_checkable
class EventBus(Protocol):
    """Publish/subscribe for cross-component notifications."""

    async def publish(self, event: str, payload: Optional[Any] = None) -> None:
        ...

    def subscribe(self) -> "Subscription":
        ...


class Subscription(AbstractAsyncContextManager["Subscription"], Protocol):
    """Subscriber handle. `wait` returns the next event payload."""

    async def wait(self) -> Any:
        ...
