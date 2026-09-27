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

from typing import Literal

from pydantic import Field
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

    # Browser origins allowed to call the API cross-origin, with credentials. The console
    # is served from the gateway's own origin (Vite proxy, nginx), so none is needed;
    # set CORS_ALLOWED_ORIGINS only for a development tool that runs elsewhere.
    cors_allowed_origins: list[str] = []

    # JWT: the private key signs, the public key verifies; no other algorithm.
    jwt_algorithm: Literal["RS256"] = "RS256"
    jwt_access_token_expire_minutes: int = Field(default=15, ge=1, le=60)
    jwt_refresh_token_expire_days: int = 7
    jwt_private_key: str = ""
    jwt_public_key: str = ""

    # Sessions: a refresh token presented again after this grace period closes its
    # whole session; within it (two browser tabs refreshing at once) it is only refused.
    refresh_reuse_grace_s: float = Field(default=5.0, ge=0.0, le=30.0)
    refresh_cookie_name: str = "cogniboiler_refresh"
    refresh_cookie_path: str = "/auth"
    refresh_cookie_secure: bool = True

    # Sign-in throttling, per account name and per client address.
    login_max_failures_per_account: int = Field(default=5, ge=1)
    login_max_failures_per_client: int = Field(default=20, ge=1)
    login_failure_window_s: float = Field(default=900.0, gt=0.0)
    # Argon2 runs allowed at once (64 MiB each); further sign-ins wait their turn.
    login_max_concurrent_hashes: int = Field(default=4, ge=1)

    # Database: DATABASE_URL carries the account and its password; the default names no
    # account, so a gateway started without it cannot sign in anywhere by accident.
    database_url: str = "postgresql+asyncpg://localhost:5432/cogniboiler"
    auto_init_db: bool = False

    # Demo users seeded by AUTO_INIT_DB; an empty password skips that user.
    demo_admin_password: str = ""
    demo_engineer_password: str = ""
    demo_operator_password: str = ""
    demo_viewer_password: str = ""

    # MQTT: PLC events and alarm changes for the WebSocket channels
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = "api-gateway"
    mqtt_password: str = ""

    # gRPC upstreams
    physics_grpc_target: str = "localhost:50052"
    plc_grpc_target: str = "localhost:50051"
    alarm_grpc_target: str = "localhost:50053"

    # Historian / InfluxDB
    influx_url: str = "http://localhost:8086"
    influx_token: str = ""
    influx_org: str = "cogniboiler"
    influx_bucket: str = "sensors"
    influx_aggregate_bucket: str = "sensors_1m"
    influx_raw_retention_days: int = 7

    # WebSocket
    ws_auth_timeout_s: float = 5.0
    ws_max_rate_hz: float = 10.0
    ws_plc_status_interval_s: float = 1.0
    ws_send_queue_size: int = 256
    # Authenticated connections at once, all users together; one more is closed 1013.
    ws_max_connections: int = Field(default=200, ge=1)

    # Readiness probe
    ready_check_timeout_s: float = 2.0


settings = Settings()
