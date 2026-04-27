"""Health/readiness API for platform and ingestion checks."""

from __future__ import annotations

import socket
from datetime import datetime
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Gauge, generate_latest

from nexus_alpha.config import NexusConfig, load_config
from nexus_alpha.data.feature_store import InMemoryFeatureStore
from nexus_alpha.data.streaming import FeatureStreamingLoop
from nexus_alpha.schema_types import ExchangeName

app = FastAPI(title="NEXUS-ALPHA Control Plane", version="0.1.0")
config = load_config()
streaming_loop = FeatureStreamingLoop.from_config(config, prefer_kafka=True)
started_at = datetime.utcnow()

_METRIC_TICK_P99 = Gauge("nexus_tick_latency_p99_ms", "P99 ingestion tick latency in ms")
_METRIC_QUALITY_FAILURES_HOUR = Gauge(
    "nexus_quality_failures_last_hour",
    "Ingestion data-quality failures in the last hour",
)
_METRIC_FEATURE_STALENESS = Gauge(
    "nexus_feature_staleness_seconds",
    "Feature staleness in seconds from latest snapshot timestamp",
)
_METRIC_SLO_OK = Gauge("nexus_pipeline_slo_ok", "Pipeline SLO status (1=ok, 0=not ok)")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    cfg = load_config()
    report = _build_dependency_report(cfg)
    if report["ready"]:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "dependencies": report["checks"]},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "dependencies": report["checks"]},
    )


@app.get("/v1/pipeline/metrics")
def pipeline_metrics() -> dict[str, object]:
    metrics = streaming_loop.metrics()
    _update_prometheus_metrics(metrics)
    return {
        "uptime_seconds": int((datetime.utcnow() - started_at).total_seconds()),
        "ingestion": metrics["pipeline"],
        "slo": metrics["slo"],
        "feature_store": metrics["feature_store"],
    }


@app.get("/v1/pipeline/slo")
def pipeline_slo() -> dict[str, object]:
    metrics = streaming_loop.metrics()
    _update_prometheus_metrics(metrics)
    return metrics["slo"]


@app.get("/v1/feature-store/point-in-time")
def feature_store_point_in_time(
    symbol: Annotated[str, Query(min_length=3)],
    exchange: ExchangeName,
    as_of: datetime,
) -> JSONResponse:
    store = _extract_feature_store(streaming_loop)
    snapshot = store.get_point_in_time(symbol=symbol, exchange=exchange, as_of=as_of)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="no_snapshot_at_or_before_timestamp")
    return JSONResponse(status_code=200, content=snapshot.model_dump(mode="json"))


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    _update_prometheus_metrics(streaming_loop.metrics())
    return PlainTextResponse(generate_latest().decode("utf-8"))


def _build_dependency_report(cfg: NexusConfig) -> dict[str, object]:
    db_url = cfg.database.timescaledb_url.get_secret_value()
    db_parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    redis_parsed = urlparse(cfg.database.redis_url)

    kafka_host = cfg.kafka.bootstrap_servers.split(",")[0].strip()
    kafka_parts = kafka_host.split(":")
    kafka_port = int(kafka_parts[1]) if len(kafka_parts) > 1 else 9092
    kafka_target = kafka_parts[0]

    checks = {
        "timescaledb": _tcp_check(db_parsed.hostname, db_parsed.port or 5432),
        "redis": _tcp_check(redis_parsed.hostname, redis_parsed.port or 6379),
        "kafka": _tcp_check(kafka_target, kafka_port),
    }
    return {"ready": all(checks.values()), "checks": checks}


def _extract_feature_store(loop: FeatureStreamingLoop) -> InMemoryFeatureStore:
    store = getattr(loop, "_feature_store", None)
    if not isinstance(store, InMemoryFeatureStore):
        raise RuntimeError("feature_store_unavailable")
    return store


def _update_prometheus_metrics(metrics: dict[str, object]) -> None:
    slo = metrics["slo"]
    measurements = slo["measurements"]
    tick_p99 = measurements["tick_latency_p99_ms"]
    failures_last_hour = measurements["failures_last_hour"]
    feature_staleness = measurements["feature_staleness_seconds"]
    _METRIC_TICK_P99.set(float(tick_p99) if tick_p99 is not None else 0.0)
    _METRIC_QUALITY_FAILURES_HOUR.set(float(failures_last_hour))
    _METRIC_FEATURE_STALENESS.set(
        float(feature_staleness) if feature_staleness is not None else 0.0
    )
    _METRIC_SLO_OK.set(1.0 if bool(slo["ok"]) else 0.0)


def _tcp_check(host: str | None, port: int, timeout: float = 0.5) -> bool:
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
