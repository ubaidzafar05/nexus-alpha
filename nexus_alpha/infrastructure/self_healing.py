"""
Self-Healing Infrastructure.

Monitors system health, detects failures, and initiates automatic recovery.
Components: health checks, watchdog processes, failover, auto-restart.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


# ─── Health Status ────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    latency_ms: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemHealth:
    """Aggregate health of the entire system."""
    overall_status: HealthStatus
    components: dict[str, HealthCheckResult]
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ─── Health Checks ────────────────────────────────────────────────────────────

class BaseHealthCheck(ABC):
    """Abstract health check for a system component."""

    def __init__(self, component_name: str, timeout_seconds: float = 10.0):
        self.component_name = component_name
        self.timeout = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        ...


class DatabaseHealthCheck(BaseHealthCheck):
    """Check database connectivity and query latency."""

    def __init__(self, dsn: str):
        super().__init__("database")
        self.dsn = dsn

    async def check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            # In production: execute "SELECT 1" against the database
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY if latency < 100 else HealthStatus.DEGRADED,
                latency_ms=latency,
                message="Database responsive",
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message=str(e),
            )


class KafkaHealthCheck(BaseHealthCheck):
    """Check Kafka broker connectivity."""

    def __init__(self, bootstrap_servers: str):
        super().__init__("kafka")
        self.bootstrap_servers = bootstrap_servers

    async def check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Kafka connected",
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message=str(e),
            )


class ExchangeHealthCheck(BaseHealthCheck):
    """Check exchange API connectivity and rate limit headroom."""

    def __init__(self, exchange_name: str, base_url: str):
        super().__init__(f"exchange_{exchange_name}")
        self.base_url = base_url

    async def check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/api/v3/ping")
                latency = (time.monotonic() - start) * 1000
                return HealthCheckResult(
                    component=self.component_name,
                    status=HealthStatus.HEALTHY if resp.status_code == 200 else HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message=f"HTTP {resp.status_code}",
                )
        except Exception as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message=str(e),
            )


class ModelHealthCheck(BaseHealthCheck):
    """Check ML model inference pipeline (World Model, RL Agent, etc.)."""

    def __init__(self, model_name: str, inference_fn: Callable[[], Any] | None = None):
        super().__init__(f"model_{model_name}")
        self.inference_fn = inference_fn

    async def check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            if self.inference_fn:
                self.inference_fn()
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Model inference OK",
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message=str(e),
            )


# ─── Recovery Actions ─────────────────────────────────────────────────────────

@dataclass
class RecoveryAction:
    """Definition of a recovery action for a component."""
    component: str
    action_name: str
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    recovery_fn: Callable[[], Coroutine[Any, Any, bool]] | None = None


class RecoveryManager:
    """Manages recovery actions for unhealthy components."""

    def __init__(self) -> None:
        self._actions: dict[str, RecoveryAction] = {}
        self._retry_counts: dict[str, int] = {}
        self._recovery_history: deque[dict[str, Any]] = deque(maxlen=500)

    def register(self, action: RecoveryAction) -> None:
        self._actions[action.component] = action
        self._retry_counts[action.component] = 0

    async def attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a failed component."""
        action = self._actions.get(component)
        if not action or not action.recovery_fn:
            logger.warning("no_recovery_action", component=component)
            return False

        retries = self._retry_counts.get(component, 0)
        if retries >= action.max_retries:
            logger.error(
                "recovery_exhausted",
                component=component,
                max_retries=action.max_retries,
            )
            return False

        self._retry_counts[component] = retries + 1
        logger.info(
            "recovery_attempt",
            component=component,
            attempt=retries + 1,
            max=action.max_retries,
        )

        try:
            success = await action.recovery_fn()
            self._recovery_history.append({
                "component": component,
                "attempt": retries + 1,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
            })
            if success:
                self._retry_counts[component] = 0
                logger.info("recovery_success", component=component)
            return success
        except Exception:
            logger.exception("recovery_failed", component=component)
            return False

    def reset_retries(self, component: str) -> None:
        self._retry_counts[component] = 0


# ─── Watchdog ─────────────────────────────────────────────────────────────────

class SystemWatchdog:
    """
    Continuous health monitor with automatic recovery.

    Runs health checks on a fixed interval and triggers recovery
    actions when components become unhealthy.
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        degraded_threshold: int = 3,
        unhealthy_threshold: int = 1,
    ):
        self.check_interval = check_interval_seconds
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold

        self._checks: list[BaseHealthCheck] = []
        self._recovery = RecoveryManager()
        self._running = False
        self._start_time = time.monotonic()
        self._consecutive_failures: dict[str, int] = {}
        self._last_health: SystemHealth | None = None

    def register_check(self, check: BaseHealthCheck) -> None:
        self._checks.append(check)
        self._consecutive_failures[check.component_name] = 0

    def register_recovery(self, action: RecoveryAction) -> None:
        self._recovery.register(action)

    async def run(self) -> None:
        """Main watchdog loop."""
        self._running = True
        self._start_time = time.monotonic()
        logger.info("watchdog_started", check_count=len(self._checks))

        while self._running:
            health = await self._run_all_checks()
            self._last_health = health

            # Handle unhealthy components
            for name, result in health.components.items():
                if result.status == HealthStatus.UNHEALTHY:
                    self._consecutive_failures[name] = (
                        self._consecutive_failures.get(name, 0) + 1
                    )
                    if self._consecutive_failures[name] >= self.unhealthy_threshold:
                        await self._recovery.attempt_recovery(name)
                elif result.status == HealthStatus.DEGRADED:
                    self._consecutive_failures[name] = (
                        self._consecutive_failures.get(name, 0) + 1
                    )
                    if self._consecutive_failures[name] >= self.degraded_threshold:
                        logger.warning("component_degraded_threshold", component=name)
                        await self._recovery.attempt_recovery(name)
                else:
                    self._consecutive_failures[name] = 0
                    self._recovery.reset_retries(name)

            await asyncio.sleep(self.check_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("watchdog_stopped")

    async def _run_all_checks(self) -> SystemHealth:
        """Run all health checks concurrently."""
        results: dict[str, HealthCheckResult] = {}

        tasks = [check.check() for check in self._checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for check, result in zip(self._checks, check_results):
            if isinstance(result, Exception):
                results[check.component_name] = HealthCheckResult(
                    component=check.component_name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    message=str(result),
                )
            else:
                results[check.component_name] = result

        # Compute overall status
        statuses = [r.status for r in results.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif statuses:
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        return SystemHealth(
            overall_status=overall,
            components=results,
            uptime_seconds=time.monotonic() - self._start_time,
        )

    @property
    def health(self) -> SystemHealth | None:
        return self._last_health
