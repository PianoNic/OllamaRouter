#!/usr/bin/env python3
"""Local dev launcher with autoreload."""

from __future__ import annotations

import uvicorn

from app.infrastructure.settings import Settings


def main() -> None:
    settings = Settings.from_env()
    print("=" * 60)
    print("Ollama API Router")
    print("=" * 60)
    print(f"Listening on {settings.server_host}:{settings.server_port}")
    print(f"Data dir:    {settings.data_dir}")
    print(f"API keys:    {settings.apikeys_file}")
    print("=" * 60)
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
