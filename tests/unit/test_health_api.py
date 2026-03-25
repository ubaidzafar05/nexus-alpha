from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from nexus_alpha.api import health as health_api
from nexus_alpha.api.health import app
from nexus_alpha.data.contracts import FeatureSnapshotPayload
from nexus_alpha.types import ExchangeName


def test_health_and_readiness_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        health_api,
        "_build_dependency_report",
        lambda _cfg: {"ready": True, "checks": {"timescaledb": True, "redis": True, "kafka": True}},
    )
    client = TestClient(app)
    health = client.get("/healthz")
    ready = client.get("/readyz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_pipeline_metrics_shape() -> None:
    client = TestClient(app)
    resp = client.get("/v1/pipeline/metrics")
    assert resp.status_code == 200
    payload = resp.json()
    assert "uptime_seconds" in payload
    assert "ingestion" in payload
    assert "slo" in payload
    assert "feature_store" in payload
    assert "events_total" in payload["ingestion"]
    assert "publisher_total_events" in payload["ingestion"]


def test_pipeline_slo_shape() -> None:
    client = TestClient(app)
    resp = client.get("/v1/pipeline/slo")
    assert resp.status_code == 200
    payload = resp.json()
    assert "ok" in payload
    assert "slo" in payload
    assert "targets" in payload
    assert "measurements" in payload


def test_prometheus_metrics_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "nexus_tick_latency_p99_ms" in resp.text
    assert "nexus_pipeline_slo_ok" in resp.text


def test_feature_store_point_in_time_endpoint() -> None:
    store = health_api._extract_feature_store(health_api.streaming_loop)  # noqa: SLF001
    now = datetime.utcnow()
    store.upsert_snapshot(
        FeatureSnapshotPayload(
            symbol="BTCUSDT",
            exchange=ExchangeName.BINANCE,
            timestamp=now,
            features={"mid_price": 100.0},
        )
    )
    client = TestClient(app)
    resp = client.get(
        "/v1/feature-store/point-in-time",
        params={
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "as_of": now.isoformat(),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["features"]["mid_price"] == 100.0


def test_readiness_returns_503_when_dependency_down(monkeypatch) -> None:
    monkeypatch.setattr(
        health_api,
        "_build_dependency_report",
        lambda _cfg: {
            "ready": False,
            "checks": {"timescaledb": False, "redis": True, "kafka": True},
        },
    )
    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"
