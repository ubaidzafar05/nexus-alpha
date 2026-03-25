from __future__ import annotations

from nexus_alpha.config import NexusConfig
from nexus_alpha.data.streaming import FeatureStreamingLoop


def test_feature_streaming_harness_in_memory_path() -> None:
    """
    Integration harness:
    in-memory tick publish -> consumer poll -> feature materialization -> snapshot emit.
    """
    loop = FeatureStreamingLoop.from_config(NexusConfig(), prefer_kafka=False)
    loop.seed_demo_ticks(n=5, start_price=70000.0)
    metrics = loop.run_for(cycles=1, interval_seconds=0.0, max_messages=100)
    assert metrics["worker"]["processed_ticks"] == 5
    assert metrics["worker"]["emitted_snapshots"] == 5

