"""End-to-end streaming loop for tick ingestion to feature snapshots."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from nexus_alpha.config import NexusConfig
from nexus_alpha.data.consumer import InMemoryEventConsumer, build_tick_consumer
from nexus_alpha.data.contracts import IngestionEvent
from nexus_alpha.data.feature_store import InMemoryFeatureStore
from nexus_alpha.data.features import FeatureMaterializationWorker, FeatureWorkerStats
from nexus_alpha.data.ingestion import IngestionPipeline, InMemoryEventBus
from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import ExchangeName

logger = get_logger(__name__)


class FeatureStreamingLoop:
    """Coordinates event publishing, consumption, and feature materialization."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        worker: FeatureMaterializationWorker,
        feature_store: InMemoryFeatureStore,
        mode: str,
    ):
        self._pipeline = pipeline
        self._worker = worker
        self._feature_store = feature_store
        self.mode = mode

    @classmethod
    def from_config(cls, config: NexusConfig, prefer_kafka: bool = True) -> FeatureStreamingLoop:
        pipeline = IngestionPipeline.from_config(config, prefer_kafka=prefer_kafka)
        publisher = pipeline.publisher

        in_memory_bus = publisher if isinstance(publisher, InMemoryEventBus) else None
        consumer = build_tick_consumer(
            config=config,
            in_memory_bus=in_memory_bus,
            prefer_kafka=prefer_kafka,
        )
        mode = "in_memory" if isinstance(consumer, InMemoryEventConsumer) else "kafka"
        feature_store = InMemoryFeatureStore()
        worker = FeatureMaterializationWorker(
            consumer=consumer,
            pipeline=pipeline,
            feature_store=feature_store,
        )
        logger.info("feature_streaming_loop_initialized", mode=mode)
        return cls(
            pipeline=pipeline,
            worker=worker,
            feature_store=feature_store,
            mode=mode,
        )

    def publish_tick(self, raw_tick: dict[str, Any]) -> IngestionEvent:
        return self._pipeline.publish_tick(raw_tick)

    def seed_demo_ticks(
        self,
        symbol: str = "BTCUSDT",
        exchange: ExchangeName = ExchangeName.BINANCE,
        n: int = 1,
        start_price: float = 65000.0,
    ) -> None:
        now = datetime.utcnow()
        for i in range(max(n, 0)):
            price = start_price + float(i)
            self.publish_tick(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timestamp": now + timedelta(milliseconds=i),
                    "bid": price - 0.5,
                    "ask": price + 0.5,
                    "last_price": price,
                    "volume_24h": 1_000_000.0 + i,
                    "bid_size": 10.0 + i,
                    "ask_size": 9.0 + i,
                }
            )
        self._pipeline.flush()

    def run_cycle(self, max_messages: int = 200) -> FeatureWorkerStats:
        return self._worker.run_once(max_messages=max_messages)

    def run_for(
        self,
        cycles: int,
        interval_seconds: float = 1.0,
        max_messages: int = 200,
    ) -> dict[str, Any]:
        total_cycles = max(cycles, 1)
        for _ in range(total_cycles):
            self.run_cycle(max_messages=max_messages)
            time.sleep(interval_seconds)
        return self.metrics()

    def metrics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "worker": self._worker.metrics(),
            "pipeline": self._pipeline.metrics(),
            "slo": self._pipeline.slo_report(),
            "feature_store": self._feature_store.metrics(),
        }

    def close(self) -> None:
        self._worker.close()

    def recent_events(self, limit: int = 100) -> list[IngestionEvent]:
        publisher = self._pipeline.publisher
        if isinstance(publisher, InMemoryEventBus):
            return publisher.latest(limit=limit)
        return []
