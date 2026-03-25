from __future__ import annotations

from datetime import datetime

from nexus_alpha.config import NexusConfig
from nexus_alpha.data import consumer as consumer_module
from nexus_alpha.data.consumer import InMemoryEventConsumer, build_tick_consumer
from nexus_alpha.data.features import FeatureMaterializationWorker
from nexus_alpha.data.ingestion import IngestionPipeline, InMemoryEventBus
from nexus_alpha.types import ExchangeName


def _publish_tick(pipeline: IngestionPipeline) -> None:
    pipeline.publish_tick(
        {
            "symbol": "BTCUSDT",
            "exchange": ExchangeName.BINANCE,
            "timestamp": datetime.utcnow(),
            "bid": 65000.0,
            "ask": 65002.0,
            "last_price": 65001.0,
            "volume_24h": 1_000_000.0,
            "bid_size": 12.0,
            "ask_size": 8.0,
        }
    )


def test_feature_worker_emits_snapshot_from_tick() -> None:
    bus = InMemoryEventBus()
    pipeline = IngestionPipeline(bus=bus)
    _publish_tick(pipeline)
    consumer = InMemoryEventConsumer(bus)
    worker = FeatureMaterializationWorker(consumer=consumer, pipeline=pipeline)
    stats = worker.run_once()
    assert stats.processed_ticks == 1
    assert stats.emitted_snapshots == 1
    assert pipeline.metrics()["events_total"]["feature_snapshot"] == 1


def test_feature_worker_deduplicates_seen_events() -> None:
    bus = InMemoryEventBus()
    pipeline = IngestionPipeline(bus=bus)
    _publish_tick(pipeline)
    consumer = InMemoryEventConsumer(bus)
    worker = FeatureMaterializationWorker(consumer=consumer, pipeline=pipeline)
    first = worker.run_once()
    second = worker.run_once()
    assert first.emitted_snapshots == 1
    assert second.emitted_snapshots == 1
    assert pipeline.metrics()["events_total"]["feature_snapshot"] == 1


def test_build_tick_consumer_falls_back_to_in_memory(monkeypatch) -> None:
    class _BrokenKafkaConsumer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("kafka down")

    monkeypatch.setattr(consumer_module, "KafkaEventConsumer", _BrokenKafkaConsumer)
    cfg = NexusConfig()
    bus = InMemoryEventBus()
    consumer = build_tick_consumer(cfg, in_memory_bus=bus, prefer_kafka=True)
    assert isinstance(consumer, InMemoryEventConsumer)

