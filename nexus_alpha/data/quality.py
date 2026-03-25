"""Data-quality validation and SLO accounting for ingestion events."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from nexus_alpha.data.contracts import (
    EventKind,
    FeatureSnapshotPayload,
    IngestionEvent,
    TickEventPayload,
)


@dataclass(frozen=True)
class QualityThresholds:
    max_future_skew_seconds: float = 1.0
    max_quality_failures_per_hour: int = 10
    max_tick_latency_p99_ms: float = 10.0
    max_feature_staleness_seconds: float = 300.0
    max_quality_check_p99_ms: float = 2.0


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: bool
    reasons: tuple[str, ...]
    validation_ms: float


class DataQualityMonitor:
    """Validates events and tracks quality/latency SLO metrics."""

    def __init__(self, thresholds: QualityThresholds, window_size: int = 10_000) -> None:
        self._thresholds = thresholds
        self._failures: deque[datetime] = deque(maxlen=window_size)
        self._failure_reasons: Counter[str] = Counter()
        self._tick_latencies_ms: deque[float] = deque(maxlen=window_size)
        self._check_latencies_ms: deque[float] = deque(maxlen=window_size)
        self._last_feature_timestamp: datetime | None = None

    def validate(self, event: IngestionEvent, now: datetime | None = None) -> ValidationOutcome:
        started = time.perf_counter()
        current = now or datetime.utcnow()
        reasons = list(self._validate_common(event, current))
        reasons.extend(self._validate_kind_specific(event, current))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._check_latencies_ms.append(elapsed_ms)

        if event.kind == EventKind.TICK:
            tick = TickEventPayload.model_validate(event.payload)
            tick_latency_ms = max((event.produced_at - tick.timestamp).total_seconds() * 1000, 0.0)
            self._tick_latencies_ms.append(tick_latency_ms)
        if event.kind == EventKind.FEATURE_SNAPSHOT:
            feature = FeatureSnapshotPayload.model_validate(event.payload)
            self._last_feature_timestamp = feature.timestamp

        if reasons:
            self._record_failure(current, reasons)
            return ValidationOutcome(
                accepted=False,
                reasons=tuple(reasons),
                validation_ms=elapsed_ms,
            )
        return ValidationOutcome(accepted=True, reasons=tuple(), validation_ms=elapsed_ms)

    def metrics(self, now: datetime | None = None) -> dict[str, object]:
        current = now or datetime.utcnow()
        failures_last_hour = self._failures_last_hour(current)
        tick_p99 = _p99(self._tick_latencies_ms)
        check_p99 = _p99(self._check_latencies_ms)
        feature_staleness = self._feature_staleness_seconds(current)
        return {
            "failures_last_hour": failures_last_hour,
            "failure_reasons": dict(self._failure_reasons),
            "tick_latency_p99_ms": tick_p99,
            "quality_check_p99_ms": check_p99,
            "feature_staleness_seconds": feature_staleness,
            "slo": {
                "tick_latency_p99_ok": tick_p99 is None
                or tick_p99 <= self._thresholds.max_tick_latency_p99_ms,
                "quality_failures_ok": failures_last_hour
                <= self._thresholds.max_quality_failures_per_hour,
                "feature_freshness_ok": feature_staleness is None
                or feature_staleness <= self._thresholds.max_feature_staleness_seconds,
                "quality_check_p99_ok": check_p99 is None
                or check_p99 <= self._thresholds.max_quality_check_p99_ms,
            },
            "targets": {
                "tick_latency_p99_ms": self._thresholds.max_tick_latency_p99_ms,
                "quality_failures_per_hour": self._thresholds.max_quality_failures_per_hour,
                "feature_staleness_seconds": self._thresholds.max_feature_staleness_seconds,
                "quality_check_p99_ms": self._thresholds.max_quality_check_p99_ms,
            },
        }

    def _validate_common(self, event: IngestionEvent, now: datetime) -> tuple[str, ...]:
        payload_ts = getattr(event.payload, "timestamp", None)
        if not isinstance(payload_ts, datetime):
            return ("missing_payload_timestamp",)
        if payload_ts > now + timedelta(seconds=self._thresholds.max_future_skew_seconds):
            return ("future_timestamp",)
        return tuple()

    def _validate_kind_specific(self, event: IngestionEvent, now: datetime) -> tuple[str, ...]:
        if event.kind == EventKind.TICK:
            return self._validate_tick(TickEventPayload.model_validate(event.payload), now)
        if event.kind == EventKind.FEATURE_SNAPSHOT:
            return self._validate_feature(FeatureSnapshotPayload.model_validate(event.payload))
        return tuple()

    def _validate_tick(self, tick: TickEventPayload, now: datetime) -> tuple[str, ...]:
        reasons: list[str] = []
        if tick.bid > tick.ask:
            reasons.append("bid_gt_ask")
        if tick.timestamp > now + timedelta(seconds=self._thresholds.max_future_skew_seconds):
            reasons.append("future_tick")
        return tuple(reasons)

    def _validate_feature(self, feature: FeatureSnapshotPayload) -> tuple[str, ...]:
        bad_keys = [key for key, value in feature.features.items() if not math.isfinite(value)]
        if not bad_keys:
            return tuple()
        return ("non_finite_feature_values",)

    def _record_failure(self, now: datetime, reasons: list[str]) -> None:
        self._failures.append(now)
        self._failure_reasons.update(reasons)

    def _failures_last_hour(self, now: datetime) -> int:
        cutoff = now - timedelta(hours=1)
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        return len(self._failures)

    def _feature_staleness_seconds(self, now: datetime) -> float | None:
        if self._last_feature_timestamp is None:
            return None
        return max((now - self._last_feature_timestamp).total_seconds(), 0.0)


def _p99(values: deque[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.ceil(len(ordered) * 0.99)) - 1
    return float(ordered[max(index, 0)])
