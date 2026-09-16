"""
Retention and downsampling of the time-series store.

At start the historian makes InfluxDB match the storage policy, idempotently:
  - the raw bucket keeps telemetry for raw_retention_days (7 by default);
  - the aggregate bucket keeps one-minute mean, min and max of every float field of
    boiler_sensors, turbine_sensors and plant_status for aggregate_retention_days (90),
    tagged agg = mean | min | max;
  - an InfluxDB task fills the aggregate bucket every minute. It rereads the last two
    whole minutes, so data that arrives a little late is still aggregated; rewriting a
    window replaces the same points.

InfluxDB may not be ready, or the token may lack the rights; then raw ingestion goes on
and the setup is retried.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules
from influxdb_client.domain.task_create_request import TaskCreateRequest

from historian.points import AGGREGATED_MEASUREMENTS

logger = logging.getLogger(__name__)

DOWNSAMPLE_TASK_NAME = "cogniboiler-downsample-1m"
RETRY_DELAY_S = 60.0
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True, slots=True)
class StoragePolicy:
    org: str
    raw_bucket: str
    aggregate_bucket: str
    raw_retention_days: int = 7
    aggregate_retention_days: int = 90


def downsample_flux(policy: StoragePolicy) -> str:
    measurements = " or ".join(
        f'r._measurement == "{name}"' for name in AGGREGATED_MEASUREMENTS
    )
    return f"""import "types"

option task = {{name: "{DOWNSAMPLE_TASK_NAME}", every: 1m, offset: 10s}}

data = from(bucket: "{policy.raw_bucket}")
  |> range(start: -2m)
  |> filter(fn: (r) => {measurements})
  |> filter(fn: (r) => types.isType(v: r._value, type: "float"))
  |> group(columns: ["_measurement", "_field"])

union(tables: [
  data |> aggregateWindow(every: 1m, fn: mean, createEmpty: false) |> set(key: "agg", value: "mean"),
  data |> aggregateWindow(every: 1m, fn: min, createEmpty: false) |> set(key: "agg", value: "min"),
  data |> aggregateWindow(every: 1m, fn: max, createEmpty: false) |> set(key: "agg", value: "max"),
])
  |> to(bucket: "{policy.aggregate_bucket}", org: "{policy.org}")
"""


def _retention(days: int) -> list[BucketRetentionRules]:
    return [
        BucketRetentionRules(  # type: ignore[no-untyped-call]
            type="expire", every_seconds=days * SECONDS_PER_DAY
        )
    ]


def _ensure_bucket(client: Any, org_id: str, name: str, days: int) -> None:
    buckets = client.buckets_api()
    bucket = buckets.find_bucket_by_name(name)
    wanted = days * SECONDS_PER_DAY
    if bucket is None:
        buckets.create_bucket(
            bucket_name=name, org_id=org_id, retention_rules=_retention(days)
        )
        logger.info("Created bucket %s with %d-day retention", name, days)
        return
    current = [rule.every_seconds for rule in bucket.retention_rules or []]
    if current != [wanted]:
        bucket.retention_rules = _retention(days)
        buckets.update_bucket(bucket)
        logger.info(
            "Bucket %s retention set to %d days (was %s s)", name, days, current
        )


def _ensure_task(client: Any, org_id: str, flux: str) -> None:
    tasks = client.tasks_api()
    existing = tasks.find_tasks(name=DOWNSAMPLE_TASK_NAME)
    if any(task.flux == flux and task.status == "active" for task in existing):
        return
    for task in existing:
        tasks.delete_task(task.id)
    tasks.create_task(
        task_create_request=TaskCreateRequest(  # type: ignore[no-untyped-call]
            org_id=org_id,
            flux=flux,
            status="active",
            description="One-minute mean, min and max of CogniBoiler telemetry",
        )
    )
    logger.info("Downsampling task %s installed", DOWNSAMPLE_TASK_NAME)


def apply_policy(url: str, token: str, policy: StoragePolicy) -> None:
    """Make buckets and the downsampling task match the policy. Blocking."""
    client: Any = InfluxDBClient(url=url, token=token, org=policy.org)
    try:
        organizations = client.organizations_api().find_organizations(org=policy.org)
        if not organizations:
            raise LookupError(f"organization {policy.org!r} not found")
        org_id = str(organizations[0].id)
        _ensure_bucket(client, org_id, policy.raw_bucket, policy.raw_retention_days)
        _ensure_bucket(
            client, org_id, policy.aggregate_bucket, policy.aggregate_retention_days
        )
        _ensure_task(client, org_id, downsample_flux(policy))
    finally:
        client.close()


async def ensure_storage(url: str, token: str, policy: StoragePolicy) -> None:
    """Apply the storage policy, retrying until it succeeds."""
    while True:
        try:
            await asyncio.to_thread(apply_policy, url, token, policy)
        except Exception as exc:
            logger.warning(
                "Storage policy not applied (%s); retrying in %.0f s",
                exc,
                RETRY_DELAY_S,
            )
            await asyncio.sleep(RETRY_DELAY_S)
            continue
        logger.info(
            "Storage policy applied: %s %d d raw, %s %d d one-minute aggregates",
            policy.raw_bucket,
            policy.raw_retention_days,
            policy.aggregate_bucket,
            policy.aggregate_retention_days,
        )
        return
