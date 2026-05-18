"""RouterService: account selection, rate-limit bookkeeping, retry orchestration.

Owns the in-memory `AccountMetrics`. Persists changes through `MetricsRepository`.
Publishes `metrics_changed` on every state mutation so the dashboard can react
event-driven instead of polling.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from app.domain.entities import AccountMetrics, HttpResult, OllamaAccount
from app.domain.exceptions import AllInstancesRateLimited, NoAccountsConfigured
from app.domain.ports import EventBus, MetricsRepository, OllamaGateway, OllamaStream

logger = logging.getLogger(__name__)

T = TypeVar("T")

METRICS_CHANGED = "metrics_changed"


class RouterService:
    """Selects an available Ollama account and tracks metrics."""

    def __init__(
        self,
        accounts: list[OllamaAccount],
        metrics_repo: MetricsRepository,
        gateway: OllamaGateway,
        event_bus: EventBus,
        cooldown_seconds: int = 30,
        display_names: dict[str, str] | None = None,
    ) -> None:
        if not accounts:
            raise NoAccountsConfigured("No Ollama accounts configured")
        self._accounts = accounts
        self._metrics_repo = metrics_repo
        self._gateway = gateway
        self._event_bus = event_bus
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._lock = asyncio.Lock()
        self._idx = 0
        self._metrics: dict[str, AccountMetrics] = {a.name: AccountMetrics(name=a.name) for a in accounts}
        self._display_names: dict[str, str] = dict(display_names or {})

    def display_name_of(self, name: str) -> str | None:
        return self._display_names.get(name)

    # ------------------------------------------------------------------ state

    @property
    def accounts(self) -> list[OllamaAccount]:
        return list(self._accounts)

    def metrics_snapshot(self) -> dict[str, AccountMetrics]:
        return {name: m.snapshot() for name, m in self._metrics.items()}

    def get_metrics(self, name: str) -> AccountMetrics | None:
        m = self._metrics.get(name)
        return m.snapshot() if m else None

    async def hydrate_from_repo(self) -> None:
        """Load persisted counters at startup, but force `is_rate_limited` to
        False so a stale flag from a previous run can't keep an account locked
        out forever."""
        persisted = await self._metrics_repo.load_all()
        for name, p in persisted.items():
            if name in self._metrics:
                m = self._metrics[name]
                m.requests_made = p.requests_made
                m.tokens_input = p.tokens_input
                m.tokens_output = p.tokens_output
                m.tool_calls = p.tool_calls
                m.rate_limit_count = p.rate_limit_count
                m.last_error = p.last_error
                m.created_at = p.created_at
                m.is_rate_limited = False
                m.consecutive_errors = 0

    # -------------------------------------------------------------- selection

    def _clear_expired_rate_limits_locked(self) -> list[AccountMetrics]:
        """Caller must hold `_lock`. Returns the metrics objects we reset."""
        now = datetime.now()
        cleared: list[AccountMetrics] = []
        for m in self._metrics.values():
            if (
                m.is_rate_limited
                and m.last_rate_limit_time is not None
                and now - m.last_rate_limit_time >= self._cooldown
            ):
                m.is_rate_limited = False
                m.consecutive_errors = 0
                cleared.append(m)
                logger.info("Cooldown elapsed for %s, clearing rate-limit flag", m.name)
        return cleared

    async def next_account(self) -> OllamaAccount:
        async with self._lock:
            cleared = self._clear_expired_rate_limits_locked()

            n = len(self._accounts)
            for _ in range(n):
                acc = self._accounts[self._idx]
                if not self._metrics[acc.name].is_rate_limited:
                    chosen = acc
                    break
                self._idx = (self._idx + 1) % n
            else:
                raise AllInstancesRateLimited("All instances are currently rate-limited")

        # Persist resets and notify outside the lock.
        for m in cleared:
            await self._metrics_repo.save(m)
        if cleared:
            await self._event_bus.publish(METRICS_CHANGED)

        return chosen

    async def mark_rate_limited(self, name: str, error: str = "") -> None:
        async with self._lock:
            m = self._metrics.get(name)
            if m is None:
                return
            m.is_rate_limited = True
            m.last_rate_limit_time = datetime.now()
            m.rate_limit_count += 1
            m.consecutive_errors += 1
            m.last_error = error or m.last_error
            self._idx = (self._idx + 1) % len(self._accounts)
            snapshot = m.snapshot()
        logger.warning("Marked %s rate-limited (%s)", name, error)
        await self._metrics_repo.save(snapshot)
        await self._event_bus.publish(METRICS_CHANGED)

    async def record_success(
        self,
        name: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: int = 0,
    ) -> None:
        async with self._lock:
            m = self._metrics.get(name)
            if m is None:
                return
            m.requests_made += 1
            m.tokens_input += tokens_in
            m.tokens_output += tokens_out
            m.tool_calls += tool_calls
            m.consecutive_errors = 0
            snapshot = m.snapshot()
        await self._metrics_repo.save(snapshot)
        await self._event_bus.publish(METRICS_CHANGED)

    # ----------------------------------------------------------------- calls

    async def execute(
        self,
        path: str,
        payload: dict,
        is_retryable: Callable[[HttpResult], bool] | None = None,
    ) -> tuple[OllamaAccount, HttpResult]:
        """Run a non-streaming POST with retry across accounts on 429 / errors.

        Raises AllInstancesRateLimited if every account is unavailable.
        """
        if is_retryable is None:
            is_retryable = _default_is_retryable

        last_result: HttpResult | None = None
        last_account: OllamaAccount | None = None
        attempts = len(self._accounts)

        for attempt in range(attempts):
            account = await self.next_account()
            last_account = account
            try:
                result = await self._gateway.post_json(account, path, payload)
            except Exception as exc:  # network, timeout, etc.
                logger.warning("Upstream %s call failed on %s: %s", path, account.name, exc)
                await self.mark_rate_limited(account.name, f"exception:{type(exc).__name__}")
                continue

            if 200 <= result.status < 300:
                return account, result

            if is_retryable(result):
                reason = "rate_limit" if result.status == 429 else f"http_{result.status}"
                await self.mark_rate_limited(account.name, reason)
                last_result = result
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.1 * (attempt + 1), 1.0))
                continue

            # Non-retryable upstream status: surface as-is.
            return account, result

        if last_result is not None and last_account is not None:
            return last_account, last_result
        raise AllInstancesRateLimited("All instances failed to respond")

    @asynccontextmanager
    async def open_stream(self, path: str, payload: dict) -> AsyncIterator[tuple[OllamaAccount, OllamaStream]]:
        """Open a streaming POST against the first account that accepts it.

        Retries account selection on connection / 4xx-before-body failures.
        Once the stream is yielded, the caller owns it until exit; no retry
        happens mid-stream.
        """
        attempts = len(self._accounts)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            account = await self.next_account()
            cm = self._gateway.stream(account, path, payload)
            try:
                stream = await cm.__aenter__()
            except Exception as exc:
                last_exc = exc
                logger.warning("Stream open failed on %s: %s", account.name, exc)
                await self.mark_rate_limited(account.name, f"exception:{type(exc).__name__}")
                continue

            if stream.status == 429 or stream.status >= 500:
                await cm.__aexit__(None, None, None)
                await self.mark_rate_limited(
                    account.name,
                    "rate_limit" if stream.status == 429 else f"http_{stream.status}",
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.1 * (attempt + 1), 1.0))
                continue

            try:
                yield account, stream
            finally:
                await cm.__aexit__(None, None, None)
            return

        raise AllInstancesRateLimited("Could not open a stream to any instance") from last_exc


def _default_is_retryable(result: HttpResult) -> bool:
    return result.status == 429 or result.status >= 500
