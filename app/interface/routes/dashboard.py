"""Dashboard, health, metrics, instance routes + WebSocket live updates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.interface.deps import Container, container_from_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=None)
async def root(container: Container = Depends(container_from_request)):
    index = container.settings.frontend_dist / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {
        "message": "Ollama Router API. Build the frontend (cd frontend && npm install && npm run build) or use /dashboard, /health, /metrics directly.",
    }


@router.get("/health")
async def health(container: Container = Depends(container_from_request)) -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "instances": len(container.router.accounts),
    }


@router.get("/metrics")
async def metrics(container: Container = Depends(container_from_request)) -> dict:
    snapshot = container.router.metrics_snapshot()
    return {
        "timestamp": datetime.now().isoformat(),
        "instances": {
            name: {
                "name": m.name,
                "requests_made": m.requests_made,
                "is_rate_limited": m.is_rate_limited,
                "consecutive_errors": m.consecutive_errors,
                "last_error": m.last_error,
                "uptime": (datetime.now() - m.created_at).total_seconds(),
            }
            for name, m in snapshot.items()
        },
    }


@router.get("/instances")
async def list_instances(container: Container = Depends(container_from_request)) -> dict:
    accounts = container.router.accounts
    return {
        "total": len(accounts),
        "instances": [
            {
                "name": a.name,
                "base_url": a.base_url,
                "max_requests_per_minute": a.max_requests_per_minute,
            }
            for a in accounts
        ],
    }


@router.get("/instances/{instance_name}/metrics")
async def instance_metrics(instance_name: str, container: Container = Depends(container_from_request)) -> dict:
    m = container.router.get_metrics(instance_name)
    if m is None:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {
        "name": m.name,
        "requests_made": m.requests_made,
        "is_rate_limited": m.is_rate_limited,
        "consecutive_errors": m.consecutive_errors,
        "last_error": m.last_error,
        "last_rate_limit": m.last_rate_limit_time.isoformat() if m.last_rate_limit_time else None,
    }


@router.get("/dashboard")
async def dashboard(container: Container = Depends(container_from_request)) -> JSONResponse:
    return JSONResponse(content=container.dashboard.build_snapshot())


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    container: Container = websocket.app.state.container
    await websocket.accept()
    logger.info("Dashboard WebSocket connected")
    last_hash: str | None = None

    async def send_if_changed() -> str | None:
        snapshot = container.dashboard.build_snapshot()
        comparable = {
            "summary": snapshot["summary"],
            "accounts": [
                {k: v for k, v in acc.items() if k != "uptime_seconds"}
                for acc in snapshot["accounts"]
            ],
        }
        h = hashlib.md5(
            json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if h != last_hash:
            await websocket.send_json(snapshot)
            return h
        return last_hash

    try:
        async with container.event_bus.subscribe() as queue:
            last_hash = await send_if_changed()
            while True:
                # Wait either for an event or a long heartbeat to refresh uptime fields.
                try:
                    await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass
                last_hash = await send_if_changed()
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket disconnected")
    except Exception as exc:
        logger.error("Dashboard WebSocket error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
