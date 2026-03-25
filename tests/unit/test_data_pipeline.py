from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nexus_alpha.config import NexusConfig
from nexus_alpha.data import ingestion
from nexus_alpha.data.contracts import EventKind
from nexus_alpha.data.ingestion import IngestionPipeline, InMemoryEventBus
from nexus_alpha.types import ExchangeName


def test_tick_ingestion_event_contract() -> None:
    pipeline = IngestionPipeline()
    event = pipeline.publish_tick(
        {
            "symbol": "BTCUSDT",
            "exchange": ExchangeName.BINANCE,
            "timestamp": datetime.utcnow(),
            "bid": 65000.0,
            "ask": 65001.0,
            "last_price": 65000.5,
            "volume_24h": 1_000_000.0,
            "bid_size": 4.0,
            "ask_size": 3.0,
        }
    )
    assert event.kind == EventKind.TICK
    assert event.partition_key == "binance:BTCUSDT"
    assert pipeline.metrics()["events_total"]["tick"] == 1


def test_invalid_tick_rejected() -> None:
    pipeline = IngestionPipeline()
    with pytest.raises(Exception):
        pipeline.publish_tick(
            {
                "symbol": "BTCUSDT",
                "exchange": ExchangeName.BINANCE,
                "timestamp": datetime.utcnow(),
                "bid": -1.0,
                "ask": 65001.0,
                "last_price": 65000.5,
                "volume_24h": 1_000_000.0,
            }
        )


def test_pipeline_from_config_falls_back_to_in_memory(monkeypatch) -> None:
    class _BrokenKafkaPublisher:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("kafka down")

    monkeypatch.setattr(ingestion, "KafkaEventPublisher", _BrokenKafkaPublisher)

    cfg = NexusConfig()
    pipeline = IngestionPipeline.from_config(cfg, prefer_kafka=True)
    assert isinstance(pipeline._bus, InMemoryEventBus)  # noqa: SLF001


def test_tick_with_future_timestamp_is_rejected() -> None:
    pipeline = IngestionPipeline()
    with pytest.raises(ValueError):
        pipeline.publish_tick(
            {
                "symbol": "BTCUSDT",
                "exchange": ExchangeName.BINANCE,
                "timestamp": datetime.utcnow() + timedelta(seconds=30),
                "bid": 65000.0,
                "ask": 65001.0,
                "last_price": 65000.5,
                "volume_24h": 1_000_000.0,
            }
        )
    metrics = pipeline.metrics()
    assert metrics["events_rejected_total"] == 1
    assert metrics["quality"]["failures_last_hour"] >= 1
