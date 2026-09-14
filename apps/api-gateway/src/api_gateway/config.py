"""
Application settings loaded from environment variables or .env file.

pydantic-settings reads variables in this order (highest priority first):
  1. Environment variables (export JWT_PRIVATE_KEY=...)
  2. .env file in the working directory
  3. Repository defaults for local development/test
  4. Default values defined in the class

Usage:
    from api_gateway.config import settings
    print(settings.app_name)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache
def _repo_root() -> Path:
    """Locate the monorepo root from the installed package path."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "apps").exists():
            return candidate
    raise RuntimeError("Could not locate CogniBoiler repository root.")


def _read_dev_key(filename: str) -> str:
    """Load repo-owned development JWT keys for local runs and tests."""
    key_path = _repo_root() / "certs" / filename
    if not key_path.exists():
        return ""
    return key_path.read_text(encoding="utf-8").strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "CogniBoiler API Gateway"
    app_version: str = "0.1.0"
    debug: bool = False

    # JWT
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # RSA keys — env/.env override these defaults.
    # The checked-in dev pair keeps local runs and tests reproducible.
    jwt_private_key: str = Field(
        default_factory=lambda: _read_dev_key("dev-jwt-private.pem")
    )
    jwt_public_key: str = Field(
        default_factory=lambda: _read_dev_key("dev-jwt-public.pem")
    )

    # Database
    database_url: str = (
        "postgresql+asyncpg://cogniboiler:cogniboiler@localhost:5432/cogniboiler"
    )
    auto_init_db: bool = False

    # MQTT (for WebSocket streaming)
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883

    # gRPC upstreams
    physics_grpc_target: str = "localhost:50052"
    plc_grpc_target: str = "localhost:50051"

    # Historian / InfluxDB
    influx_url: str = "http://localhost:8086"
    influx_token: str = "cogniboiler-dev-token"
    influx_org: str = "cogniboiler"
    influx_bucket: str = "sensors"


settings = Settings()
