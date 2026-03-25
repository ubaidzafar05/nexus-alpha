from __future__ import annotations

import time

import pytest

from nexus_alpha.config import NexusConfig
from nexus_alpha.data.contracts import EventKind
from nexus_alpha.data.streaming import FeatureStreamingLoop


@pytest.mark.kafka_integration
def test_kafka_tick_to_feature_snapshot_path(kafka_stack) -> None:
    del kafka_stack
    cfg = NexusConfig()
    loop = FeatureStreamingLoop.from_config(cfg, prefer_kafka=True)
    if loop.mode != "kafka":
        pytest.skip("Kafka mode unavailable in current runtime.")

    loop.seed_demo_ticks(n=3, start_price=72000.0)

    emitted = 0
    for _ in range(15):
        stats = loop.run_cycle(max_messages=100)
        emitted = stats.emitted_snapshots
        if emitted >= 3:
            break
        time.sleep(0.5)

    assert emitted >= 3
    metrics = loop.metrics()
    assert metrics["mode"] == "kafka"
    assert metrics["worker"]["processed_ticks"] >= 3
    assert metrics["pipeline"]["publisher_total_events"] >= 6
    assert metrics["pipeline"]["publisher_failed_events"] == 0


@pytest.mark.kafka_integration
def test_kafka_streaming_still_emits_feature_kind(kafka_stack) -> None:
    del kafka_stack
    cfg = NexusConfig()
    loop = FeatureStreamingLoop.from_config(cfg, prefer_kafka=True)
    if loop.mode != "kafka":
        pytest.skip("Kafka mode unavailable in current runtime.")

    loop.seed_demo_ticks(n=1, start_price=73000.0)
    loop.run_cycle(max_messages=50)

    # Kafka mode has no local event history; verify event accounting and worker metrics instead.
    metrics = loop.metrics()
    assert metrics["worker"]["emitted_snapshots"] >= 1
    assert EventKind.FEATURE_SNAPSHOT.value in metrics["pipeline"]["events_total"]

