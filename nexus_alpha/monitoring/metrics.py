"""Shared Prometheus metrics for Nexus-Alpha.
Provides counters with no-op fallback if prometheus_client isn't installed.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge

    RETRAIN_ATTEMPTS = Counter("nexus_retrain_attempts_total", "Online retrain attempts")
    RETRAIN_ACCEPTED = Counter("nexus_retrain_accepted_total", "Retrains accepted/promoted")
    RETRAIN_REJECTED = Counter("nexus_retrain_rejected_total", "Retrains rejected")

    LLM_FALLBACKS = Counter("nexus_llm_fallbacks_total", "LLM fallback occurrences", ["from_model", "to_model"])
except Exception:
    class _Noop:
        def inc(self, *_, **__):
            return None

    RETRAIN_ATTEMPTS = RETRAIN_ACCEPTED = RETRAIN_REJECTED = _Noop()
    LLM_FALLBACKS = _Noop()
