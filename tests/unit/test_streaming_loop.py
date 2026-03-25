from __future__ import annotations

from nexus_alpha.config import NexusConfig
from nexus_alpha.data.contracts import EventKind
from nexus_alpha.data.streaming import FeatureStreamingLoop


def test_streaming_loop_fallback_end_to_end() -> None:
    loop = FeatureStreamingLoop.from_config(NexusConfig(), prefer_kafka=True)
    assert loop.mode in {"in_memory", "kafka"}

    # For deterministic local harness, force in-memory mode.
    loop = FeatureStreamingLoop.from_config(NexusConfig(), prefer_kafka=False)
    assert loop.mode == "in_memory"

    loop.seed_demo_ticks(n=3)
    stats = loop.run_cycle(max_messages=10)
    assert stats.processed_ticks == 3
    assert stats.emitted_snapshots == 3

    events = loop.recent_events(limit=100)
    feature_events = [event for event in events if event.kind == EventKind.FEATURE_SNAPSHOT]
    assert len(feature_events) == 3


def test_streaming_loop_metrics_shape() -> None:
    loop = FeatureStreamingLoop.from_config(NexusConfig(), prefer_kafka=False)
    loop.seed_demo_ticks(n=1)
    loop.run_cycle()
    metrics = loop.metrics()
    assert "mode" in metrics
    assert "worker" in metrics
    assert "pipeline" in metrics
    assert "slo" in metrics
    assert "feature_store" in metrics
    assert metrics["worker"]["emitted_snapshots"] >= 1
