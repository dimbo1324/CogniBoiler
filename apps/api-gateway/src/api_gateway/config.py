"""
Application settings loaded from environment variables or .env file.

pydantic-settings reads variables in this order (highest priority first):
  1. Environment variables (export JWT_PRIVATE_KEY=...)
  2. .env file in the working directory
  3. Default values defined in the class

Usage:
    from api_gateway.config import settings
    print(settings.app_name)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[misc]
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

    # RSA keys — read from env or .env file as PEM strings.
    # Generate with: scripts/gen_keys.py
    jwt_private_key: str = ""
    jwt_public_key: str = ""

    # Database
    database_url: str = (
        "postgresql+asyncpg://cogniboiler:cogniboiler@localhost:5432/cogniboiler"
    )

    # MQTT (for WebSocket streaming)
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883


settings = Settings()
