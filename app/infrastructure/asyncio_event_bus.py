"""In-process asyncio event bus.

Per-subscriber bounded queues. Slow subscribers get newer events dropped (not
the whole stream) so a stuck WebSocket cannot back-pressure the request path.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class AsyncioEventBus:
    def __init__(self, queue_size: int = 16) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._queue_size = queue_size
        self._lock = asyncio.Lock()

    async def publish(self, event: str, payload: Any = None) -> None:
        async with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait((event, payload))
            except asyncio.QueueFull:
                logger.debug("Event bus subscriber is full; dropping event")

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers.append(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
