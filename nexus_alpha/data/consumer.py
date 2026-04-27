"""Event consumer abstractions for in-memory and Kafka-backed streams."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Protocol

from nexus_alpha.data.contracts import EventKind, IngestionEvent
from nexus_alpha.data.kafka_admin import ensure_topics_exist
from nexus_alpha.log_config import get_logger

if TYPE_CHECKING:
    from nexus_alpha.config import NexusConfig
    from nexus_alpha.data.ingestion import InMemoryEventBus

logger = get_logger(__name__)


class EventConsumer(Protocol):
    def poll(self, max_messages: int = 100, timeout: float = 0.2) -> list[IngestionEvent]: ...
    def close(self) -> None: ...


class InMemoryEventConsumer:
    """Consumer over in-memory event bus with id-based deduplication."""

    def __init__(self, bus: InMemoryEventBus, kinds: set[EventKind] | None = None):
        self._bus = bus
        self._seen_event_ids: set[str] = set()
        self._kinds = kinds

    def poll(self, max_messages: int = 100, timeout: float = 0.2) -> list[IngestionEvent]:
        del timeout
        events = self._bus.latest(limit=max_messages * 3)
        pending: list[IngestionEvent] = []
        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            if self._kinds and event.kind not in self._kinds:
                continue
            self._seen_event_ids.add(event.event_id)
            pending.append(event)
            if len(pending) >= max_messages:
                break
        return pending

    def close(self) -> None:
        return None


class KafkaEventConsumer:
    """Kafka consumer for ingestion events serialized as JSON envelopes."""

    def __init__(self, bootstrap_servers: str, topics: list[str], group_id: str):
        try:
            from confluent_kafka import Consumer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("confluent_kafka is not available") from exc

        ensure_topics_exist(bootstrap_servers, topics)
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe(topics)
        self._topics = topics
        self._wait_for_assignment()

    def _wait_for_assignment(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._consumer.poll(0.1)
            if self._consumer.assignment():
                return
        logger.warning("kafka_consumer_assignment_timeout", topics=self._topics)

    def poll(self, max_messages: int = 100, timeout: float = 0.2) -> list[IngestionEvent]:
        events: list[IngestionEvent] = []
        for _ in range(max_messages):
            msg = self._consumer.poll(timeout)
            if msg is None:
                break
            if msg.error():
                logger.warning("kafka_consumer_error", error=str(msg.error()))
                continue
            if msg.value() is None:
                continue
            try:
                payload = json.loads(msg.value().decode("utf-8"))
                events.append(IngestionEvent.model_validate(payload))
            except Exception as exc:
                logger.warning("kafka_consumer_decode_failed", reason=str(exc))
        return events

    def close(self) -> None:
        self._consumer.close()


def build_tick_consumer(
    config: NexusConfig,
    in_memory_bus: InMemoryEventBus | None = None,
    prefer_kafka: bool = True,
) -> EventConsumer:
    """Build tick-event consumer with Kafka auto-fallback."""
    if prefer_kafka:
        try:
            consumer = KafkaEventConsumer(
                bootstrap_servers=config.kafka.bootstrap_servers,
                topics=[config.kafka.tick_topic],
                group_id=f"{config.kafka.consumer_group}-feature-worker-{uuid.uuid4().hex[:8]}",
            )
            logger.info("event_consumer_selected", type="kafka", topics=[config.kafka.tick_topic])
            return consumer
        except Exception as exc:
            logger.warning("event_consumer_kafka_fallback", reason=str(exc))

    if in_memory_bus is None:
        raise RuntimeError("in_memory_bus is required when Kafka is unavailable")
    logger.info("event_consumer_selected", type="in_memory")
    return InMemoryEventConsumer(in_memory_bus, kinds={EventKind.TICK})
