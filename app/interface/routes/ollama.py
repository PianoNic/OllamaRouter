"""Native Ollama-compatible endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.application.tokens import estimate_tokens
from app.domain.exceptions import AllInstancesRateLimited
from app.interface.deps import Container, container_from_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/chat", response_model=None)
async def chat(request: Request, container: Container = Depends(container_from_request)):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    stream = bool(body.get("stream", False))
    return await _proxy(container, "/api/chat", body, stream, input_text=str(body.get("messages", [])))


@router.post("/api/generate", response_model=None)
async def generate(request: Request, container: Container = Depends(container_from_request)):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    stream = bool(body.get("stream", False))
    return await _proxy(container, "/api/generate", body, stream, input_text=str(body.get("prompt", "")))


@router.get("/api/tags")
async def list_models(container: Container = Depends(container_from_request)) -> dict:
    all_models: set[str] = set()
    for account in container.router.accounts:
        try:
            result = await container.gateway.get_json(account, "/api/tags", timeout=10.0)
        except Exception as exc:
            logger.warning("Could not list models on %s: %s", account.name, exc)
            continue
        if result.status != 200 or not isinstance(result.json, dict):
            continue
        for model in result.json.get("models", []) or []:
            name = model.get("name")
            if isinstance(name, str):
                all_models.add(name)
    return {
        "models": [
            {"name": name, "size": 0, "digest": "", "modified_at": ""}
            for name in sorted(all_models)
        ]
    }


async def _proxy(container: Container, path: str, body: dict, stream: bool, *, input_text: str):
    input_tokens = estimate_tokens(input_text)

    if stream:
        return StreamingResponse(
            _stream_passthrough(container, path, body, input_tokens),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        account, result = await container.router.execute(path, body)
    except AllInstancesRateLimited as exc:
        return JSONResponse(status_code=429, content={"error": str(exc)})

    if result.status >= 400 or result.json is None:
        return JSONResponse(
            status_code=result.status if result.status >= 400 else 502,
            content={"error": "Upstream Ollama error", "status": result.status, "body": result.text[:500]},
        )

    output_tokens = _extract_output_tokens(path, result.json)
    tool_calls_count = _extract_tool_call_count(path, result.json)
    await container.router.record_success(account.name, input_tokens, output_tokens, tool_calls_count)
    return JSONResponse(content=result.json, headers={"X-Instance-Used": account.name})


async def _stream_passthrough(container: Container, path: str, body: dict, input_tokens: int):
    output_tokens = 0
    tool_calls_count = 0
    account_name: str | None = None
    try:
        async with container.router.open_stream(path, body) as (account, stream):
            account_name = account.name
            async for line in stream.aiter_lines():
                if not line:
                    continue
                output_tokens += estimate_tokens(line)
                yield line + "\n"
                # Best-effort tool_call accounting from chat streaming.
                if path == "/api/chat":
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = parsed.get("message") or {}
                    calls = msg.get("tool_calls")
                    if calls:
                        tool_calls_count = max(tool_calls_count, len(calls))
    except AllInstancesRateLimited as exc:
        logger.warning("Streaming aborted: %s", exc)
        return

    if account_name is not None:
        await container.router.record_success(account_name, input_tokens, output_tokens, tool_calls_count)


def _extract_output_tokens(path: str, data: dict) -> int:
    if path == "/api/chat":
        msg = data.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return estimate_tokens(content)
        return estimate_tokens(str(content)) if content else 0
    return estimate_tokens(data.get("response", ""))


def _extract_tool_call_count(path: str, data: dict) -> int:
    if path != "/api/chat":
        return 0
    msg = data.get("message") or {}
    calls = msg.get("tool_calls") or []
    return len(calls) if isinstance(calls, list) else 0
