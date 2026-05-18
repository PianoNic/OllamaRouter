"""Anthropic-compatible endpoints.

As of Ollama v0.14, Ollama itself exposes a native `/v1/messages` endpoint
(https://docs.ollama.com/api/anthropic-compatibility). We proxy through it
unchanged — no translation, no SSE re-assembly, no model rewriting. The router
only adds account rotation, rate-limit failover, and metrics.

`/v1/messages/count_tokens` isn't part of Ollama's surface, so we keep a local
char-based estimator for that one path.
"""

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


@router.post("/v1/messages", response_model=None)
async def messages_v1(request: Request, container: Container = Depends(container_from_request)):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    is_stream = bool(body.get("stream", False))
    input_tokens = estimate_tokens(json.dumps(body.get("messages", "")))

    if is_stream:
        return StreamingResponse(
            _stream(container, body, input_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        account, result = await container.router.execute("/v1/messages", body)
    except AllInstancesRateLimited as exc:
        return JSONResponse(
            status_code=429,
            content={"type": "error", "error": {"type": "rate_limit_error", "message": str(exc)}},
        )

    if result.status >= 400 or result.json is None:
        return JSONResponse(
            status_code=result.status if result.status >= 400 else 502,
            content={
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "message": "Upstream Ollama returned an error",
                    "status": result.status,
                    "body": result.text[:500],
                },
            },
        )

    output_tokens, tool_calls = _account_for_usage(result.json)
    await container.router.record_success(account.name, input_tokens, output_tokens, tool_calls)
    return JSONResponse(content=result.json, headers={"X-Instance-Used": account.name})


async def _stream(container: Container, body: dict, input_tokens: int):
    output_tokens = 0
    tool_calls = 0
    account_name: str | None = None
    try:
        async with container.router.open_stream("/v1/messages", body) as (account, stream):
            account_name = account.name
            async for line in stream.aiter_lines():
                if not line:
                    yield "\n"
                    continue
                yield line + "\n"
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "message_delta":
                        usage = event.get("usage") or {}
                        candidate = usage.get("output_tokens")
                        if isinstance(candidate, int) and candidate > output_tokens:
                            output_tokens = candidate
                    elif event.get("type") == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            tool_calls += 1
    except AllInstancesRateLimited as exc:
        payload = json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": str(exc)}})
        yield f"event: error\ndata: {payload}\n\n"
        return

    if account_name is not None:
        if output_tokens == 0:
            # Upstream didn't report usage — leave at 0 rather than guess.
            pass
        await container.router.record_success(account_name, input_tokens, output_tokens, tool_calls)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict:
    body = await request.json()
    messages = body.get("messages", []) or []
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return {
        "input_tokens": max(1, total_chars // 4),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _account_for_usage(data: dict) -> tuple[int, int]:
    usage = data.get("usage") or {}
    output = usage.get("output_tokens")
    output_tokens = output if isinstance(output, int) else 0

    tool_calls = 0
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls += 1
    return output_tokens, tool_calls
