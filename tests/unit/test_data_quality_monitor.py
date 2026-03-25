from __future__ import annotations

from datetime import datetime, timedelta

from nexus_alpha.data.contracts import EventKind, IngestionEvent, TickEventPayload
from nexus_alpha.data.quality import DataQualityMonitor, QualityThresholds
from nexus_alpha.types import ExchangeName


def _build_tick_event(
    timestamp: datetime,
    bid: float = 100.0,
    ask: float = 101.0,
) -> IngestionEvent:
    payload = TickEventPayload(
        symbol="BTCUSDT",
        exchange=ExchangeName.BINANCE,
        timestamp=timestamp,
        bid=bid,
        ask=ask,
        last_price=(bid + ask) / 2,
        volume_24h=1_000.0,
        bid_size=1.0,
        ask_size=1.0,
    )
    return IngestionEvent(
        event_id="abcd1234",
        kind=EventKind.TICK,
        partition_key="binance:BTCUSDT",
        produced_at=timestamp + timedelta(milliseconds=5),
        payload=payload,
    )


def test_quality_monitor_tracks_failure_and_slo_status() -> None:
    monitor = DataQualityMonitor(QualityThresholds(max_quality_failures_per_hour=0))
    now = datetime.utcnow()
    bad_event = _build_tick_event(timestamp=now, bid=102.0, ask=101.0)
    outcome = monitor.validate(bad_event, now=now)
    assert not outcome.accepted
    metrics = monitor.metrics(now=now)
    assert metrics["failures_last_hour"] == 1
    assert metrics["slo"]["quality_failures_ok"] is False


def test_quality_monitor_computes_tick_latency_p99() -> None:
    monitor = DataQualityMonitor(QualityThresholds(max_tick_latency_p99_ms=10.0))
    now = datetime.utcnow()
    for lag_ms in [1, 2, 3, 4, 5]:
        event_ts = now - timedelta(milliseconds=lag_ms)
        event = _build_tick_event(timestamp=event_ts)
        event = event.model_copy(update={"produced_at": now})
        monitor.validate(event, now=now)
    metrics = monitor.metrics(now=now)
    assert metrics["tick_latency_p99_ms"] is not None
    assert metrics["slo"]["tick_latency_p99_ok"] is True
