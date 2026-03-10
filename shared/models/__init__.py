"""Shared domain models for CogniBoiler platform."""

from shared.models.base import (
    BaseTimestampedModel,
    HealthCheckResponse,
    SensorQuality,
    SensorReading,
    ServiceStatus,
)

__all__ = [
    "BaseTimestampedModel",
    "HealthCheckResponse",
    "SensorQuality",
    "SensorReading",
    "ServiceStatus",
]
