"""Ollama Router application package (onion architecture).

Layers, dependencies point inward only:
    interface  -> infrastructure -> application -> domain

- domain:         pure entities, value objects, exceptions, ports (protocols)
- application:    use-case services and pure transformations
- infrastructure: concrete adapters (httpx, peewee, filesystem, asyncio)
- interface:      FastAPI app, routes, DI wiring
"""

from app.interface.app_factory import create_app

__all__ = ["create_app"]
