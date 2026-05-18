"""Application settings sourced from environment variables.

Imports must not perform I/O; resolution happens when `from_env()` is called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_APIKEYS_FILE = REPO_ROOT / "apikeys.txt"
DEFAULT_FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    apikeys_file: Path
    frontend_dist: Path
    db_filename: str
    default_model: str
    cooldown_seconds: int
    per_account_rate_limit: int
    http_timeout_seconds: float
    server_host: str
    server_port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
        apikeys_file = Path(os.getenv("APIKEYS_FILE", str(DEFAULT_APIKEYS_FILE)))
        frontend_dist = Path(os.getenv("FRONTEND_DIST", str(DEFAULT_FRONTEND_DIST)))
        return cls(
            data_dir=data_dir,
            apikeys_file=apikeys_file,
            frontend_dist=frontend_dist,
            db_filename=os.getenv("DB_FILENAME", "ollama_metrics.db"),
            default_model=os.getenv("DEFAULT_MODEL", "glm-4.7:cloud"),
            cooldown_seconds=int(os.getenv("RATE_LIMIT_COOLDOWN_SECONDS", "30")),
            per_account_rate_limit=int(os.getenv("PER_ACCOUNT_RATE_LIMIT", "30")),
            http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "300")),
            server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("SERVER_PORT", "8000")),
        )
