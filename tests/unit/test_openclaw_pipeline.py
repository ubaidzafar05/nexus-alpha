from __future__ import annotations

from datetime import datetime, timedelta

from nexus_alpha.intelligence.openclaw_agents import (
    IntelligenceCategory,
    IntelligenceReport,
    Urgency,
)
from nexus_alpha.intelligence.openclaw_pipeline import OpenClawProcessingPipeline


class _StaticConnector:
    def __init__(self, reports: list[IntelligenceReport]) -> None:
        self._reports = reports

    def fetch(self) -> list[IntelligenceReport]:
        return list(self._reports)


def test_openclaw_pipeline_quarantines_and_signalizes() -> None:
    valid = IntelligenceReport(
        report_id="r1",
        category=IntelligenceCategory.BREAKING_NEWS,
        urgency=Urgency.HIGH,
        headline="Large inflow",
        summary="Exchange inflow increased",
        sentiment_score=-0.8,
        confidence=0.9,
        affected_symbols=["BTCUSDT"],
        source_urls=["https://example.com"],
        raw_data={},
        timestamp=datetime.utcnow(),
    )
    low_conf = IntelligenceReport(
        report_id="r2",
        category=IntelligenceCategory.ON_CHAIN,
        urgency=Urgency.MEDIUM,
        headline="Noise",
        summary="Unclear signal",
        sentiment_score=0.2,
        confidence=0.1,
        affected_symbols=["ETHUSDT"],
        source_urls=["https://example.com"],
        raw_data={},
        timestamp=datetime.utcnow(),
    )
    stale = IntelligenceReport(
        report_id="r3",
        category=IntelligenceCategory.SOCIAL_PULSE,
        urgency=Urgency.CRITICAL,
        headline="Old event",
        summary="Outdated",
        sentiment_score=0.7,
        confidence=0.95,
        affected_symbols=["SOLUSDT"],
        source_urls=["https://example.com"],
        raw_data={},
        timestamp=datetime.utcnow() - timedelta(minutes=30),
    )

    pipeline = OpenClawProcessingPipeline(connectors=[_StaticConnector([valid, low_conf, stale])])
    signals, quarantined = pipeline.run_once()

    assert len(signals) == 1
    assert signals[0].symbol == "BTCUSDT"
    assert signals[0].direction < 0
    reasons = {record.reason for record in quarantined}
    assert "low_confidence" in reasons
    assert "stale_intelligence" in reasons
