"""Loads OllamaAccount entries from apikeys.txt / OLLAMA_INSTANCES env / fallback."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.domain.entities import OllamaAccount

logger = logging.getLogger(__name__)


class FileAccountRepository:
    def __init__(self, apikeys_file: Path, default_rate_limit: int = 30) -> None:
        self._apikeys_file = apikeys_file
        self._default_rate_limit = default_rate_limit

    def load_all(self) -> list[OllamaAccount]:
        accounts = self._from_apikeys_file()
        if accounts:
            return accounts

        accounts = self._from_env()
        if accounts:
            return accounts

        logger.info("Falling back to default localhost Ollama instance")
        return [OllamaAccount(name="default")]

    def _from_apikeys_file(self) -> list[OllamaAccount]:
        if not self._apikeys_file.exists():
            return []
        try:
            raw = self._apikeys_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.error("Failed to read %s: %s", self._apikeys_file, exc)
            return []

        out: list[OllamaAccount] = []
        for idx, line in enumerate(raw, start=1):
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            out.append(
                OllamaAccount(
                    name=f"account_{idx}",
                    base_url="https://ollama.com",
                    api_key=key,
                    is_cloud=True,
                    max_requests_per_minute=self._default_rate_limit,
                )
            )
        if out:
            logger.info("Loaded %d API keys from %s", len(out), self._apikeys_file)
        return out

    def _from_env(self) -> list[OllamaAccount]:
        raw = os.getenv("OLLAMA_INSTANCES")
        if not raw:
            return []
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("OLLAMA_INSTANCES is not valid JSON: %s", exc)
            return []
        out: list[OllamaAccount] = []
        for entry in entries:
            try:
                out.append(OllamaAccount(**entry))
            except TypeError as exc:
                logger.error("Skipping invalid OLLAMA_INSTANCES entry %r: %s", entry, exc)
        if out:
            logger.info("Loaded %d instances from OLLAMA_INSTANCES env", len(out))
        return out
