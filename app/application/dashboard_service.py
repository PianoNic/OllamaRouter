"""Builds dashboard payloads from RouterService state.

Output shape is preserved verbatim from the original monolith so the existing
dashboard.html keeps working without changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.application.router_service import RouterService


class DashboardService:
    def __init__(self, router: RouterService, rate_limit_per_account: int = 30) -> None:
        self._router = router
        self._rate_limit_per_account = rate_limit_per_account

    def build_snapshot(self) -> dict[str, Any]:
        metrics = self._router.metrics_snapshot()
        accounts_payload: list[dict[str, Any]] = []
        total_requests = 0
        total_tokens_in = 0
        total_tokens_out = 0
        total_tool_calls = 0
        total_errors = 0
        rate_limited = 0
        healthy = 0
        now = datetime.now()

        for account in self._router.accounts:
            m = metrics.get(account.name)
            if m is None:
                continue
            total_requests += m.requests_made
            total_tokens_in += m.tokens_input
            total_tokens_out += m.tokens_output
            total_tool_calls += m.tool_calls
            total_errors += m.rate_limit_count
            if m.is_rate_limited:
                rate_limited += 1
                usage_percent: float = 100.0
            else:
                healthy += 1
                if m.rate_limit_count > 0:
                    usage_percent = 80.0
                else:
                    usage_percent = float(min(50, m.requests_made / 10))

            accounts_payload.append(
                {
                    "name": account.name,
                    "display_name": self._router.display_name_of(account.name),
                    "requests_made": m.requests_made,
                    "tokens_input": m.tokens_input,
                    "tokens_output": m.tokens_output,
                    "tool_calls": m.tool_calls,
                    "is_rate_limited": m.is_rate_limited,
                    "usage_percent": usage_percent,
                    "consecutive_errors": m.consecutive_errors,
                    "last_error": m.last_error or "",
                    "last_rate_limit": m.last_rate_limit_time.isoformat() if m.last_rate_limit_time else None,
                    "uptime_seconds": (now - m.created_at).total_seconds(),
                }
            )

        total_accounts = len(self._router.accounts)
        if rate_limited == 0:
            overall = "healthy"
        elif rate_limited < total_accounts:
            overall = "degraded"
        else:
            overall = "limited"

        return {
            "timestamp": now.isoformat(),
            "summary": {
                "total_accounts": total_accounts,
                "healthy_accounts": healthy,
                "rate_limited_accounts": rate_limited,
                "overall_health": overall,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "tokens_input": total_tokens_in,
                "tokens_output": total_tokens_out,
                "tool_calls": total_tool_calls,
                "estimated_capacity": f"{healthy}/{total_accounts} accounts available",
                "rate_limit_per_account": f"{self._rate_limit_per_account} requests/minute",
                "estimated_total_capacity": f"~{healthy * self._rate_limit_per_account} requests/minute",
            },
            "accounts": sorted(accounts_payload, key=lambda a: a["usage_percent"], reverse=True),
        }
