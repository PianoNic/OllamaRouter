"""Lightweight DI container.

The container is attached to `app.state.container` during the lifespan so route
handlers can pull collaborators via `Depends(container_from_request)` rather
than touching module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.application.dashboard_service import DashboardService
from app.application.router_service import RouterService
from app.domain.ports import EventBus, MetricsRepository, OllamaGateway
from app.infrastructure.settings import Settings


@dataclass
class Container:
    settings: Settings
    router: RouterService
    dashboard: DashboardService
    gateway: OllamaGateway
    metrics_repo: MetricsRepository
    event_bus: EventBus


def container_from_request(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Container not initialised — lifespan did not run")
    return container
