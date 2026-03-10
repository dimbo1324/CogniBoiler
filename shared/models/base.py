"""Base domain models shared across all CogniBoiler services.

This module defines foundational data structures used throughout
the platform. All models use Pydantic v2 for validation and
serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ServiceStatus(str, Enum):
    """Operational status of a microservice."""

    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class SensorQuality(str, Enum):
    """Data quality indicator for sensor readings.

    Follows OPC UA quality codes convention:
    GOOD   — sensor is functioning normally
    UNCERTAIN — sensor may be inaccurate
    BAD    — sensor failure or communication lost
    """

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


class BaseTimestampedModel(BaseModel):
    """Base model with automatic UTC timestamp.

    All domain models that represent time-series data
    should inherit from this class.
    """

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        populate_by_name=True,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the measurement",
    )


class SensorReading(BaseTimestampedModel):
    """A single sensor measurement with metadata.

    Attributes:
        sensor_id: Unique identifier of the sensor (e.g. 'boiler.pressure.drum')
        value: Measured value in the specified unit
        unit: Engineering unit (e.g. 'bar', 'degC', 'MW', 'm3/h')
        quality: Data quality indicator
        tag: Optional process tag for SCADA/OPC UA compatibility
    """

    sensor_id: str = Field(
        min_length=1,
        max_length=128,
        description="Unique sensor identifier",
        examples=["boiler.pressure.drum", "turbine.power.electrical"],
    )
    value: float = Field(
        description="Measured value in engineering units",
    )
    unit: str = Field(
        min_length=1,
        max_length=32,
        description="Engineering unit of measurement",
        examples=["bar", "degC", "MW", "t/h", "m"],
    )
    quality: SensorQuality = Field(
        default=SensorQuality.GOOD,
        description="OPC UA-compatible data quality indicator",
    )
    tag: str | None = Field(
        default=None,
        description="Optional SCADA/OPC UA process tag",
    )


class HealthCheckResponse(BaseModel):
    """Standard health check response for all services.

    Used by Kubernetes liveness and readiness probes,
    and by Grafana infrastructure dashboard.
    """

    model_config = ConfigDict(use_enum_values=True)

    service: str = Field(description="Service name")
    status: ServiceStatus = Field(description="Current operational status")
    version: str = Field(description="Service version from pyproject.toml")
    uptime_seconds: float = Field(
        ge=0.0,
        description="Service uptime in seconds",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional service-specific health details",
    )
