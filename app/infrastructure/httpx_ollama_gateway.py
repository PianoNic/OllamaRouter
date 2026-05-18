"""Httpx-backed Ollama gateway.

Uses a single long-lived AsyncClient so:
- Streaming responses stay alive until the SSE generator finishes (the original
  monolith closed the client before the generator started iterating, which
  silently broke streams).
- Connection pooling cuts latency.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from app.domain.entities import HttpResult, OllamaAccount

logger = logging.getLogger(__name__)


class _HttpxStreamAdapter:
    """Wraps an httpx.Response so it satisfies the `OllamaStream` protocol."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code

    def aiter_lines(self) -> AsyncIterator[str]:
        return self._response.aiter_lines()

    async def aread_text(self) -> str:
        body = await self._response.aread()
        try:
            return body.decode("utf-8", errors="replace")
        except Exception:
            return ""

    async def __aenter__(self) -> "_HttpxStreamAdapter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class HttpxOllamaGateway:
    def __init__(self, timeout_seconds: float = 300.0, list_models_timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._list_models_timeout = list_models_timeout

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _headers(account: OllamaAccount) -> dict[str, str]:
        return {"Authorization": f"Bearer {account.api_key}"} if account.api_key else {}

    async def post_json(self, account: OllamaAccount, path: str, payload: dict) -> HttpResult:
        url = account.base_url.rstrip("/") + path
        response = await self._client.post(url, json=payload, headers=self._headers(account))
        return self._materialise(response)

    async def get_json(self, account: OllamaAccount, path: str, timeout: float = 10.0) -> HttpResult:
        url = account.base_url.rstrip("/") + path
        response = await self._client.get(url, headers=self._headers(account), timeout=timeout)
        return self._materialise(response)

    @asynccontextmanager
    async def stream(self, account: OllamaAccount, path: str, payload: dict):
        url = account.base_url.rstrip("/") + path
        request = self._client.build_request("POST", url, json=payload, headers=self._headers(account))
        response = await self._client.send(request, stream=True)
        adapter = _HttpxStreamAdapter(response)
        try:
            yield adapter
        finally:
            await response.aclose()

    @staticmethod
    def _materialise(response: httpx.Response) -> HttpResult:
        text = response.text
        parsed: dict | None
        try:
            parsed = response.json() if text else None
        except json.JSONDecodeError:
            parsed = _parse_last_json_object(text)
        return HttpResult(status=response.status_code, json=parsed, text=text)


def _parse_last_json_object(text: str) -> dict | None:
    """Some Ollama backends return newline-delimited JSON even when stream=false.

    Walk the lines in reverse and return the last valid JSON object.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
