"""Kafka admin helpers for topic management."""

from __future__ import annotations

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


def ensure_topics_exist(bootstrap_servers: str, topics: list[str], timeout: float = 10.0) -> None:
    """Create missing Kafka topics with simple single-broker defaults."""
    try:
        from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("confluent_kafka admin client is not available") from exc

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(timeout=timeout)
    existing = set(metadata.topics)
    missing = [topic for topic in topics if topic not in existing]
    if not missing:
        return

    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=1, replication_factor=1) for topic in missing]
    )
    for topic, future in futures.items():
        try:
            future.result(timeout=timeout)
            logger.info("kafka_topic_created", topic=topic)
        except Exception as exc:  # topic may already exist due to a race
            if "TOPIC_ALREADY_EXISTS" in str(exc):
                continue
            raise
