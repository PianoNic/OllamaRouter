"""Unit tests for the refactored router."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from app.application.router_service import RouterService
from app.application.tokens import estimate_tokens
from app.domain.entities import AccountMetrics, HttpResult, OllamaAccount
from app.domain.exceptions import AllInstancesRateLimited
from app.infrastructure.asyncio_event_bus import AsyncioEventBus


class FakeMetricsRepo:
    def __init__(self) -> None:
        self.saved: list[AccountMetrics] = []

    async def load_all(self) -> dict[str, AccountMetrics]:
        return {}

    async def save(self, metrics: AccountMetrics) -> None:
        self.saved.append(metrics.snapshot())


class FakeGateway:
    def __init__(self, results: dict[str, list[HttpResult]]) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    async def post_json(self, account, path, payload) -> HttpResult:
        self.calls.append((account.name, path))
        queue = self.results.get(account.name, [])
        if not queue:
            return HttpResult(200, {}, "{}")
        return queue.pop(0)

    async def get_json(self, account, path, timeout=10.0) -> HttpResult:
        return HttpResult(200, {"models": []}, "{}")

    def stream(self, account, path, payload):
        raise NotImplementedError

    async def close(self) -> None:
        pass


def _accounts(*names: str) -> list[OllamaAccount]:
    return [OllamaAccount(name=n) for n in names]


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25


def test_router_rotates_on_rate_limit():
    async def run():
        bus = AsyncioEventBus()
        repo = FakeMetricsRepo()
        gateway = FakeGateway({})
        router = RouterService(_accounts("a", "b"), repo, gateway, bus, cooldown_seconds=60)

        first = await router.next_account()
        assert first.name == "a"

        await router.mark_rate_limited("a", "rate_limit")
        second = await router.next_account()
        assert second.name == "b"

        await router.mark_rate_limited("b", "rate_limit")
        with pytest.raises(AllInstancesRateLimited):
            await router.next_account()

    asyncio.run(run())


def test_router_clears_expired_rate_limit():
    """Regression for: rate limits never resetting."""
    async def run():
        bus = AsyncioEventBus()
        repo = FakeMetricsRepo()
        gateway = FakeGateway({})
        router = RouterService(_accounts("a"), repo, gateway, bus, cooldown_seconds=1)

        await router.mark_rate_limited("a", "rate_limit")
        with pytest.raises(AllInstancesRateLimited):
            await router.next_account()

        m = router._metrics["a"]
        m.last_rate_limit_time = datetime.now() - timedelta(seconds=10)
        acc = await router.next_account()
        assert acc.name == "a"
        assert not router.get_metrics("a").is_rate_limited

    asyncio.run(run())


def test_router_execute_retries_on_429():
    async def run():
        bus = AsyncioEventBus()
        repo = FakeMetricsRepo()
        gateway = FakeGateway({
            "a": [HttpResult(429, None, "")],
            "b": [HttpResult(200, {"ok": True}, "{}")],
        })
        router = RouterService(_accounts("a", "b"), repo, gateway, bus)
        account, result = await router.execute("/v1/messages", {})
        assert account.name == "b"
        assert result.status == 200

    asyncio.run(run())


def test_router_record_success_updates_metrics():
    async def run():
        bus = AsyncioEventBus()
        repo = FakeMetricsRepo()
        gateway = FakeGateway({})
        router = RouterService(_accounts("a"), repo, gateway, bus)
        await router.record_success("a", tokens_in=10, tokens_out=20, tool_calls=2)
        m = router.get_metrics("a")
        assert m.requests_made == 1
        assert m.tokens_input == 10
        assert m.tokens_output == 20
        assert m.tool_calls == 2
        assert repo.saved

    asyncio.run(run())


def test_router_display_names_exposed():
    async def run():
        bus = AsyncioEventBus()
        repo = FakeMetricsRepo()
        gateway = FakeGateway({})
        router = RouterService(
            _accounts("a", "b"),
            repo,
            gateway,
            bus,
            display_names={"a": "alice"},
        )
        assert router.display_name_of("a") == "alice"
        assert router.display_name_of("b") is None

    asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
