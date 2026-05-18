"""Module entry point. Keeps backwards compatibility with `uvicorn main:app`."""

from app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    from app.infrastructure.settings import Settings

    settings = Settings.from_env()
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
