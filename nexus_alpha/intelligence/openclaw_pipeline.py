"""Phase 3 processing pipeline for OpenClaw source ingestion and signalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from nexus_alpha.intelligence.openclaw_agents import IntelligenceReport, Urgency
from nexus_alpha.log_config import get_logger
from nexus_alpha.signals.contracts import SignalCandidate

logger = get_logger(__name__)


@dataclass(frozen=True)
class QuarantineRecord:
    report_id: str
    reason: str
    timestamp: datetime


class IntelligenceSourceConnector(Protocol):
    def fetch(self) -> list[IntelligenceReport]: ...


class OpenClawQuarantine:
    """Quarantine low-quality or stale intelligence before signalization."""

    def __init__(
        self,
        min_confidence: float = 0.45,
        max_age_seconds: int = 300,
    ):
        self.min_confidence = min_confidence
        self.max_age_seconds = max_age_seconds
        self.records: list[QuarantineRecord] = []

    def check(self, report: IntelligenceReport) -> str | None:
        if report.confidence < self.min_confidence:
            return "low_confidence"
        if datetime.utcnow() - report.timestamp > timedelta(seconds=self.max_age_seconds):
            return "stale_intelligence"
        if not report.affected_symbols:
            return "missing_symbol_scope"
        return None

    def quarantine(self, report: IntelligenceReport, reason: str) -> None:
        self.records.append(
            QuarantineRecord(
                report_id=report.report_id,
                reason=reason,
                timestamp=datetime.utcnow(),
            )
        )


class OpenClawSignalizer:
    """Converts intelligence reports to signal candidates."""

    _URGENCY_WEIGHTS = {
        Urgency.LOW: 0.4,
        Urgency.MEDIUM: 0.6,
        Urgency.HIGH: 0.85,
        Urgency.CRITICAL: 1.0,
    }

    def to_candidates(self, report: IntelligenceReport) -> list[SignalCandidate]:
        score = report.sentiment_score * report.confidence * self._URGENCY_WEIGHTS[report.urgency]
        direction = max(-1.0, min(score, 1.0))
        confidence = max(0.0, min(abs(score), 1.0))
        if confidence < 0.05:
            return []

        return [
            SignalCandidate(
                signal_id=f"ocl-{report.report_id}-{idx}",
                source=f"openclaw:{report.category.value}",
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                timestamp=report.timestamp,
                features_used=["sentiment_score", "urgency", "source_confidence"],
            )
            for idx, symbol in enumerate(report.affected_symbols)
        ]


class OpenClawProcessingPipeline:
    """Fetch -> quarantine -> signalize intelligence flow."""

    def __init__(
        self,
        connectors: list[IntelligenceSourceConnector],
        quarantine: OpenClawQuarantine | None = None,
        signalizer: OpenClawSignalizer | None = None,
    ):
        self._connectors = connectors
        self._quarantine = quarantine or OpenClawQuarantine()
        self._signalizer = signalizer or OpenClawSignalizer()

    def run_once(self) -> tuple[list[SignalCandidate], list[QuarantineRecord]]:
        signals: list[SignalCandidate] = []
        for connector in self._connectors:
            reports = connector.fetch()
            for report in reports:
                reason = self._quarantine.check(report)
                if reason is not None:
                    self._quarantine.quarantine(report, reason)
                    continue
                signals.extend(self._signalizer.to_candidates(report))

        logger.info(
            "openclaw_pipeline_cycle",
            signals=len(signals),
            quarantined=len(self._quarantine.records),
        )
        return signals, list(self._quarantine.records)
