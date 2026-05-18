"""FastAPI application factory and lifespan-based DI wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.application.dashboard_service import DashboardService
from app.application.router_service import RouterService
from app.infrastructure.asyncio_event_bus import AsyncioEventBus
from app.infrastructure.file_account_repo import FileAccountRepository
from app.infrastructure.httpx_ollama_gateway import HttpxOllamaGateway
from app.infrastructure.peewee_metrics_repo import PeeweeMetricsRepository
from app.infrastructure.settings import Settings
from app.interface.deps import Container
from app.interface.routes import anthropic as anthropic_routes
from app.interface.routes import dashboard as dashboard_routes
from app.interface.routes import ollama as ollama_routes
from app.interface.routes import openai as openai_routes

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_logging()
    resolved = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Initialising container (data_dir=%s)", resolved.data_dir)
        resolved.data_dir.mkdir(parents=True, exist_ok=True)

        account_repo = FileAccountRepository(
            resolved.apikeys_file, default_rate_limit=resolved.per_account_rate_limit
        )
        accounts = account_repo.load_all()
        logger.info("Loaded %d account(s): %s", len(accounts), [a.name for a in accounts])

        metrics_repo = PeeweeMetricsRepository(resolved.db_path)
        gateway = HttpxOllamaGateway(timeout_seconds=resolved.http_timeout_seconds)
        event_bus = AsyncioEventBus()

        display_names = await _resolve_display_names(accounts, gateway)

        router_service = RouterService(
            accounts=accounts,
            metrics_repo=metrics_repo,
            gateway=gateway,
            event_bus=event_bus,
            cooldown_seconds=resolved.cooldown_seconds,
            display_names=display_names,
        )
        await router_service.hydrate_from_repo()

        dashboard_service = DashboardService(
            router_service, rate_limit_per_account=resolved.per_account_rate_limit
        )

        app.state.container = Container(
            settings=resolved,
            router=router_service,
            dashboard=dashboard_service,
            gateway=gateway,
            metrics_repo=metrics_repo,
            event_bus=event_bus,
        )
        try:
            yield
        finally:
            logger.info("Shutting down container")
            await gateway.close()
            await metrics_repo.close()

    app = FastAPI(
        title="Ollama API Router",
        description="Routes Ollama API requests with automatic instance switching on rate limits.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(anthropic_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(ollama_routes.router)
    app.include_router(dashboard_routes.router)

    assets_dir = resolved.frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    favicon = resolved.frontend_dist / "favicon.svg"
    if favicon.exists():
        from fastapi.responses import FileResponse

        @app.get("/favicon.svg", include_in_schema=False, response_model=None)
        async def _favicon():  # pragma: no cover
            return FileResponse(str(favicon), media_type="image/svg+xml")

    return app


async def _resolve_display_names(accounts, gateway) -> dict[str, str]:
    """Best-effort lookup of human-readable names via Ollama Cloud's /api/me.

    The endpoint is undocumented but used by the official client. Failures are
    swallowed; the synthetic account_N name remains the fallback.
    """
    names: dict[str, str] = {}
    for account in accounts:
        if not account.is_cloud or not account.api_key:
            continue
        try:
            result = await gateway.post_json(account, "/api/me", {})
        except Exception as exc:
            logger.info("Identity lookup failed for %s: %s", account.name, exc)
            continue
        if result.status != 200 or not isinstance(result.json, dict):
            continue
        body = result.json
        # Ollama returns PascalCase fields (Name, Email, FirstName, LastName).
        full_name = " ".join(
            part for part in (body.get("FirstName") or "", body.get("LastName") or "") if part
        ).strip()
        candidate = full_name or body.get("Name") or body.get("Email")
        if isinstance(candidate, str) and candidate.strip():
            names[account.name] = candidate.strip()
            logger.info("Resolved %s -> %s", account.name, names[account.name])
    return names
