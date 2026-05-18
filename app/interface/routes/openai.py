"""OpenAI-compatible endpoints.

Ollama already ships an OpenAI-compatible surface at /v1/chat/completions,
/v1/completions, /v1/embeddings, and /v1/models. We proxy through the same
account-rotation logic so OpenAI SDK clients benefit from rate-limit failover.

See https://github.com/ollama/ollama/blob/main/docs/openai.md
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


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request, container: Container = Depends(container_from_request)):
    return await _proxy_openai(container, "/v1/chat/completions", request, input_field="messages")


@router.post("/v1/completions", response_model=None)
async def completions(request: Request, container: Container = Depends(container_from_request)):
    return await _proxy_openai(container, "/v1/completions", request, input_field="prompt")


@router.post("/v1/embeddings", response_model=None)
async def embeddings(request: Request, container: Container = Depends(container_from_request)):
    return await _proxy_openai(container, "/v1/embeddings", request, input_field="input")


@router.get("/v1/models", response_model=None)
async def list_models(container: Container = Depends(container_from_request)):
    """Aggregate `/v1/models` from every account."""
    seen: dict[str, dict] = {}
    for account in container.router.accounts:
        try:
            result = await container.gateway.get_json(account, "/v1/models", timeout=10.0)
        except Exception as exc:
            logger.warning("OpenAI /v1/models failed on %s: %s", account.name, exc)
            continue
        if result.status != 200 or not isinstance(result.json, dict):
            continue
        for entry in result.json.get("data", []) or []:
            mid = entry.get("id")
            if isinstance(mid, str):
                seen[mid] = entry
    return {"object": "list", "data": list(seen.values())}


async def _proxy_openai(container: Container, path: str, request: Request, *, input_field: str):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    is_stream = bool(body.get("stream", False))
    input_tokens = estimate_tokens(json.dumps(body.get(input_field, "")))

    if is_stream:
        return StreamingResponse(
            _stream_openai(container, path, body, input_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        account, result = await container.router.execute(path, body)
    except AllInstancesRateLimited as exc:
        return JSONResponse(status_code=429, content={"error": {"message": str(exc), "type": "rate_limit_exceeded"}})

    if result.status >= 400 or result.json is None:
        return JSONResponse(
            status_code=result.status if result.status >= 400 else 502,
            content={"error": {"message": "Upstream error", "status": result.status, "body": result.text[:500]}},
        )

    output_tokens = _extract_openai_output_tokens(path, result.json)
    await container.router.record_success(account.name, input_tokens, output_tokens, 0)
    return JSONResponse(content=result.json, headers={"X-Instance-Used": account.name})


async def _stream_openai(container: Container, path: str, body: dict, input_tokens: int):
    output_tokens = 0
    account_name: str | None = None
    try:
        async with container.router.open_stream(path, body) as (account, stream):
            account_name = account.name
            async for line in stream.aiter_lines():
                if not line:
                    continue
                output_tokens += estimate_tokens(line)
                yield line + "\n"
    except AllInstancesRateLimited as exc:
        yield f'data: {{"error":"{exc}"}}\n\n'
        return

    if account_name is not None:
        await container.router.record_success(account_name, input_tokens, output_tokens, 0)


def _extract_openai_output_tokens(path: str, data: dict) -> int:
    usage = data.get("usage") or {}
    explicit = usage.get("completion_tokens") or usage.get("total_tokens")
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    if path == "/v1/chat/completions":
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return estimate_tokens(content)
    if path == "/v1/completions":
        for choice in data.get("choices") or []:
            text = choice.get("text", "")
            if isinstance(text, str):
                return estimate_tokens(text)
    return 0
