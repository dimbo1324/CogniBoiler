"""
Application settings loaded from environment variables or .env file.

pydantic-settings reads variables in this order (highest priority first):
  1. Environment variables (export JWT_PRIVATE_KEY=...)
  2. .env file in the working directory
  3. Default values defined in the class

Secrets (JWT keys, database password, InfluxDB token, demo user passwords) have no
usable default: they come from .env, created by
`python dev_tools_scripts_runner.py dev-secrets`, or from the container environment.

Usage:
    from api_gateway.config import settings
    print(settings.app_name)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "CogniBoiler API Gateway"
    app_version: str = "0.1.0"
    debug: bool = False

    # JWT
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    jwt_private_key: str = ""
    jwt_public_key: str = ""

    # Database
    database_url: str = (
        "postgresql+asyncpg://cogniboiler:cogniboiler@localhost:5432/cogniboiler"
    )
    auto_init_db: bool = False

    # Demo users seeded by AUTO_INIT_DB; an empty password skips that user.
    demo_admin_password: str = ""
    demo_engineer_password: str = ""
    demo_operator_password: str = ""
    demo_viewer_password: str = ""

    # MQTT (for WebSocket streaming)
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883

    # gRPC upstreams
    physics_grpc_target: str = "localhost:50052"
    plc_grpc_target: str = "localhost:50051"

    # Historian / InfluxDB
    influx_url: str = "http://localhost:8086"
    influx_token: str = ""
    influx_org: str = "cogniboiler"
    influx_bucket: str = "sensors"


settings = Settings()
