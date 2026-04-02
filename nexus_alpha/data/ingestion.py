"""Ingestion and normalization primitives for market events."""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from nexus_alpha.data.contracts import (
    EventKind,
    FeatureSnapshotPayload,
    IngestionEvent,
    OHLCVEventPayload,
    TickEventPayload,
)
from nexus_alpha.data.kafka_admin import ensure_topics_exist
from nexus_alpha.data.quality import DataQualityMonitor, QualityThresholds
from nexus_alpha.logging import get_logger

if TYPE_CHECKING:
    from nexus_alpha.config import NexusConfig

logger = get_logger(__name__)


class EventPublisher(Protocol):
    def publish(self, event: IngestionEvent) -> None: ...
    def flush(self, timeout: float = 5.0) -> None: ...
    @property
    def total_events(self) -> int | None: ...
    @property
    def failed_events(self) -> int: ...


class InMemoryEventBus:
    """
    Lightweight in-memory event sink.

    This is the local development fallback until Kafka producers are wired.
    """

    def __init__(self, maxlen: int = 10_000):
        self._events: deque[IngestionEvent] = deque(maxlen=maxlen)
        self._failed_events = 0

    def publish(self, event: IngestionEvent) -> None:
        self._events.append(event)

    def flush(self, timeout: float = 5.0) -> None:
        del timeout

    def latest(self, limit: int = 100) -> list[IngestionEvent]:
        return list(self._events)[-limit:]

    def snapshot(self) -> list[IngestionEvent]:
        return list(self._events)

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def failed_events(self) -> int:
        return self._failed_events


class KafkaEventPublisher:
    """Kafka-backed event publisher with best-effort delivery semantics."""

    def __init__(
        self,
        bootstrap_servers: str,
        tick_topic: str,
        signal_topic: str,
    ) -> None:
        try:
            from confluent_kafka import Producer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("confluent_kafka is not available") from exc

        ensure_topics_exist(bootstrap_servers, [tick_topic, signal_topic])
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._topics = {
            EventKind.TICK: tick_topic,
            EventKind.OHLCV: tick_topic,
            EventKind.FEATURE_SNAPSHOT: signal_topic,
        }
        self._total_events = 0
        self._failed_events = 0

    def publish(self, event: IngestionEvent) -> None:
        topic = self._topics[event.kind]
        payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
        try:
            self._producer.produce(topic, key=event.partition_key, value=payload)
            self._producer.poll(0)
            self._total_events += 1
        except Exception:
            self._failed_events += 1
            raise

    def flush(self, timeout: float = 5.0) -> None:
        self._producer.flush(timeout=timeout)

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def failed_events(self) -> int:
        return self._failed_events


class IngestionPipeline:
    """Normalization + validation + event envelope generation."""

    def __init__(
        self,
        bus: EventPublisher | None = None,
        quality_monitor: DataQualityMonitor | None = None,
    ):
        self._bus = bus or InMemoryEventBus()
        self._counters: dict[str, int] = {"tick": 0, "ohlcv": 0, "feature_snapshot": 0}
        self._rejected_events = 0
        self._last_event_at: datetime | None = None
        self._quality = quality_monitor or DataQualityMonitor(QualityThresholds())

    @classmethod
    def from_config(cls, config: NexusConfig, prefer_kafka: bool = True) -> IngestionPipeline:
        """Build a pipeline from root config with Kafka auto-fallback."""
        quality_thresholds = QualityThresholds(
            max_future_skew_seconds=config.data_quality.max_future_skew_seconds,
            max_quality_failures_per_hour=config.data_quality.max_quality_failures_per_hour,
            max_tick_latency_p99_ms=config.data_quality.max_tick_latency_p99_ms,
            max_feature_staleness_seconds=config.data_quality.max_feature_staleness_seconds,
            max_quality_check_p99_ms=config.data_quality.max_quality_check_p99_ms,
        )
        if prefer_kafka:
            try:
                bus = KafkaEventPublisher(
                    bootstrap_servers=config.kafka.bootstrap_servers,
                    tick_topic=config.kafka.tick_topic,
                    signal_topic=config.kafka.signal_topic,
                )
                logger.info("ingestion_bus_selected", publisher="kafka")
                return cls(bus=bus, quality_monitor=DataQualityMonitor(quality_thresholds))
            except Exception as exc:
                logger.warning("ingestion_kafka_unavailable_fallback", reason=str(exc))

        logger.info("ingestion_bus_selected", publisher="in_memory")
        return cls(
            bus=InMemoryEventBus(),
            quality_monitor=DataQualityMonitor(quality_thresholds),
        )

    def publish_tick(self, raw: dict[str, Any]) -> IngestionEvent:
        payload = TickEventPayload.model_validate(raw)
        event = self._build_event(
            EventKind.TICK,
            payload,
            partition_key=f"{payload.exchange.value}:{payload.symbol}",
        )
        self._accept(event)
        return event

    def publish_ohlcv(self, raw: dict[str, Any]) -> IngestionEvent:
        payload = OHLCVEventPayload.model_validate(raw)
        partition_key = f"{payload.exchange.value}:{payload.symbol}:{payload.timeframe}"
        event = self._build_event(EventKind.OHLCV, payload, partition_key=partition_key)
        self._accept(event)
        return event

    def publish_feature_snapshot(self, raw: dict[str, Any]) -> IngestionEvent:
        payload = FeatureSnapshotPayload.model_validate(raw)
        partition_key = f"{payload.exchange.value}:{payload.symbol}"
        event = self._build_event(EventKind.FEATURE_SNAPSHOT, payload, partition_key=partition_key)
        self._accept(event)
        return event

    def metrics(self) -> dict[str, Any]:
        return {
            "events_total": dict(self._counters),
            "events_rejected_total": self._rejected_events,
            "publisher_total_events": self._bus.total_events,
            "publisher_failed_events": self._bus.failed_events,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "quality": self._quality.metrics(),
        }

    def slo_report(self) -> dict[str, Any]:
        quality_metrics = self._quality.metrics()
        slo = quality_metrics["slo"]
        return {
            "ok": all(bool(flag) for flag in slo.values()),
            "slo": slo,
            "targets": quality_metrics["targets"],
            "measurements": {
                "failures_last_hour": quality_metrics["failures_last_hour"],
                "tick_latency_p99_ms": quality_metrics["tick_latency_p99_ms"],
                "quality_check_p99_ms": quality_metrics["quality_check_p99_ms"],
                "feature_staleness_seconds": quality_metrics["feature_staleness_seconds"],
            },
        }

    @property
    def publisher(self) -> EventPublisher:
        return self._bus

    def flush(self, timeout: float = 5.0) -> None:
        self._bus.flush(timeout=timeout)

    def _build_event(
        self,
        kind: EventKind,
        payload: TickEventPayload | OHLCVEventPayload | FeatureSnapshotPayload,
        partition_key: str,
    ) -> IngestionEvent:
        return IngestionEvent(
            event_id=uuid.uuid4().hex,
            kind=kind,
            partition_key=partition_key,
            payload=payload,
        )

    def _accept(self, event: IngestionEvent) -> None:
        validation = self._quality.validate(event)
        if not validation.accepted:
            self._rejected_events += 1
            logger.warning(
                "ingestion_event_rejected",
                kind=event.kind.value,
                partition_key=event.partition_key,
                event_id=event.event_id,
                reasons=list(validation.reasons),
                validation_ms=validation.validation_ms,
            )
            raise ValueError(f"event_rejected: {','.join(validation.reasons)}")
        self._bus.publish(event)
        self._counters[event.kind.value] += 1
        self._last_event_at = event.produced_at
        logger.info(
            "ingestion_event_accepted",
            kind=event.kind.value,
            partition_key=event.partition_key,
            event_id=event.event_id,
        )
