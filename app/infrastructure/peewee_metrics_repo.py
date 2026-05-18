"""Peewee-backed MetricsRepository.

The Peewee model is private to this module. The application layer only sees
`AccountMetrics` domain entities.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    IntegerField,
    Model,
    SqliteDatabase,
)

from app.domain.entities import AccountMetrics

logger = logging.getLogger(__name__)


def _build_model(db: SqliteDatabase) -> type[Model]:
    class TokenMetrics(Model):
        account_name = CharField(unique=True)
        tokens_input = IntegerField(default=0)
        tokens_output = IntegerField(default=0)
        requests_made = IntegerField(default=0)
        tool_calls = IntegerField(default=0)
        rate_limit_count = IntegerField(default=0)
        last_error = CharField(default="")
        is_rate_limited = BooleanField(default=False)
        created_at = DateTimeField(default=datetime.now)
        updated_at = DateTimeField(default=datetime.now)

        class Meta:
            database = db
            table_name = "token_metrics"

    return TokenMetrics


class PeeweeMetricsRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = SqliteDatabase(str(db_path))
        self._model = _build_model(self._db)
        self._db.connect(reuse_if_open=True)
        self._db.create_tables([self._model], safe=True)
        # All Peewee calls run in a thread to avoid blocking the event loop.
        self._executor_lock = asyncio.Lock()
        logger.info("Metrics DB initialised at %s", db_path)

    async def close(self) -> None:
        if not self._db.is_closed():
            self._db.close()

    async def load_all(self) -> dict[str, AccountMetrics]:
        rows = await asyncio.to_thread(lambda: list(self._model.select()))
        return {
            row.account_name: AccountMetrics(
                name=row.account_name,
                requests_made=row.requests_made,
                tokens_input=row.tokens_input,
                tokens_output=row.tokens_output,
                tool_calls=row.tool_calls,
                rate_limit_count=row.rate_limit_count,
                is_rate_limited=row.is_rate_limited,
                last_error=row.last_error or None,
                created_at=row.created_at or datetime.now(),
            )
            for row in rows
        }

    async def save(self, metrics: AccountMetrics) -> None:
        async with self._executor_lock:
            await asyncio.to_thread(self._save_sync, metrics)

    def _save_sync(self, metrics: AccountMetrics) -> None:
        row, _ = self._model.get_or_create(
            account_name=metrics.name,
            defaults={"created_at": metrics.created_at},
        )
        row.requests_made = metrics.requests_made
        row.tokens_input = metrics.tokens_input
        row.tokens_output = metrics.tokens_output
        row.tool_calls = metrics.tool_calls
        row.rate_limit_count = metrics.rate_limit_count
        row.is_rate_limited = metrics.is_rate_limited
        row.last_error = metrics.last_error or ""
        row.updated_at = datetime.now()
        row.save()
