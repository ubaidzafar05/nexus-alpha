"""Feature materialization worker from tick events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nexus_alpha.data.consumer import EventConsumer
from nexus_alpha.data.contracts import EventKind, FeatureSnapshotPayload, TickEventPayload
from nexus_alpha.data.feature_store import FeatureStore
from nexus_alpha.data.ingestion import IngestionPipeline
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureWorkerStats:
    polled_events: int = 0
    processed_ticks: int = 0
    emitted_snapshots: int = 0
    skipped_events: int = 0


class FeatureMaterializationWorker:
    """Converts TICK events into FEATURE_SNAPSHOT events."""

    def __init__(
        self,
        consumer: EventConsumer,
        pipeline: IngestionPipeline,
        feature_store: FeatureStore | None = None,
    ):
        self._consumer = consumer
        self._pipeline = pipeline
        self._feature_store = feature_store
        self._stats = FeatureWorkerStats()

    def run_once(self, max_messages: int = 200) -> FeatureWorkerStats:
        events = self._consumer.poll(max_messages=max_messages, timeout=0.1)
        self._stats.polled_events += len(events)

        for event in events:
            if event.kind != EventKind.TICK:
                self._stats.skipped_events += 1
                continue
            tick = TickEventPayload.model_validate(event.payload)
            features = self._compute_features(tick)
            snapshot_event = self._pipeline.publish_feature_snapshot(
                {
                    "symbol": tick.symbol,
                    "exchange": tick.exchange,
                    "timestamp": tick.timestamp,
                    "features": features,
                    "source": "tick_worker",
                }
            )
            if self._feature_store is not None:
                snapshot_payload = FeatureSnapshotPayload.model_validate(snapshot_event.payload)
                self._feature_store.upsert_snapshot(snapshot_payload)
            self._stats.processed_ticks += 1
            self._stats.emitted_snapshots += 1

        logger.info(
            "feature_worker_cycle_complete",
            polled=len(events),
            processed=self._stats.processed_ticks,
            emitted=self._stats.emitted_snapshots,
        )
        return self._stats

    def metrics(self) -> dict[str, Any]:
        return {
            "polled_events": self._stats.polled_events,
            "processed_ticks": self._stats.processed_ticks,
            "emitted_snapshots": self._stats.emitted_snapshots,
            "skipped_events": self._stats.skipped_events,
        }

    def close(self) -> None:
        self._pipeline.flush()
        self._consumer.close()

    @staticmethod
    def _compute_features(tick: TickEventPayload) -> dict[str, float]:
        mid_price = (tick.bid + tick.ask) / 2
        spread_bps = ((tick.ask - tick.bid) / max(mid_price, 1e-9)) * 10_000
        depth_total = tick.bid_size + tick.ask_size
        order_book_imbalance = (
            (tick.bid_size - tick.ask_size) / depth_total if depth_total > 0 else 0.0
        )
        last_to_mid_bps = ((tick.last_price - mid_price) / max(mid_price, 1e-9)) * 10_000
        return {
            "mid_price": float(mid_price),
            "spread_bps": float(spread_bps),
            "order_book_imbalance": float(order_book_imbalance),
            "last_to_mid_bps": float(last_to_mid_bps),
            "volume_24h": float(tick.volume_24h),
        }
