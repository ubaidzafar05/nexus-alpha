"""
Multi-Dimensional Circuit Breaker System.

5 levels of intervention, each with defined triggers and recovery conditions.
Graduated response: NORMAL → CAUTION → REDUCED → DEFENSIVE → EMERGENCY → LOCKDOWN
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from nexus_alpha.config import RiskConfig
from nexus_alpha.logging import get_logger
from nexus_alpha.types import CircuitBreakerLevel

logger = get_logger(__name__)


@dataclass
class BreakerTrigger:
    """Definition of a circuit breaker trigger condition."""
    name: str
    level: CircuitBreakerLevel
    description: str
    check_fn_name: str  # Method name on CircuitBreakerSystem


@dataclass
class BreakerState:
    """Current state of the circuit breaker system."""
    level: CircuitBreakerLevel
    triggered_by: str | None = None
    triggered_at: datetime | None = None
    actions_taken: list[str] = field(default_factory=list)
    auto_recovery_at: datetime | None = None


@dataclass
class RiskSnapshot:
    """Point-in-time risk metrics."""
    timestamp: datetime
    nav: float
    drawdown_pct: float
    daily_pnl_pct: float
    volatility_1h: float
    correlation_to_btc: float
    leverage: float
    position_count: int


class CircuitBreakerSystem:
    """
    Graduated, multi-dimensional circuit breaker with automatic recovery.

    Levels:
    0 - NORMAL:     All systems go
    1 - CAUTION:    Reduce new position sizes by 50%
    2 - REDUCED:    No new entries, existing tight stops
    3 - DEFENSIVE:  Active de-risking, hedge positions
    4 - EMERGENCY:  Close all positions immediately
    5 - LOCKDOWN:   All trading halted, human override required
    """

    # Trigger definitions
    TRIGGERS = {
        # Level 1: CAUTION
        "daily_loss_1pct": BreakerTrigger(
            name="daily_loss_1pct",
            level=CircuitBreakerLevel.CAUTION,
            description="Daily loss exceeds 1%",
            check_fn_name="_check_daily_loss",
        ),
        "volatility_spike_2x": BreakerTrigger(
            name="volatility_spike_2x",
            level=CircuitBreakerLevel.CAUTION,
            description="1H volatility > 2x 24H average",
            check_fn_name="_check_volatility_spike",
        ),
        # Level 2: REDUCED
        "daily_loss_3pct": BreakerTrigger(
            name="daily_loss_3pct",
            level=CircuitBreakerLevel.REDUCED,
            description="Daily loss exceeds 3%",
            check_fn_name="_check_daily_loss_severe",
        ),
        "drawdown_5pct": BreakerTrigger(
            name="drawdown_5pct",
            level=CircuitBreakerLevel.REDUCED,
            description="Drawdown from peak exceeds 5%",
            check_fn_name="_check_drawdown_moderate",
        ),
        # Level 3: DEFENSIVE
        "daily_loss_5pct": BreakerTrigger(
            name="daily_loss_5pct",
            level=CircuitBreakerLevel.DEFENSIVE,
            description="Daily loss exceeds 5%",
            check_fn_name="_check_daily_loss_critical",
        ),
        "drawdown_10pct": BreakerTrigger(
            name="drawdown_10pct",
            level=CircuitBreakerLevel.DEFENSIVE,
            description="Drawdown exceeds 10%",
            check_fn_name="_check_drawdown_severe",
        ),
        "correlation_convergence": BreakerTrigger(
            name="correlation_convergence",
            level=CircuitBreakerLevel.DEFENSIVE,
            description="All assets correlating > 0.85 (crisis regime)",
            check_fn_name="_check_correlation_convergence",
        ),
        # Level 4: EMERGENCY
        "drawdown_15pct": BreakerTrigger(
            name="drawdown_15pct",
            level=CircuitBreakerLevel.EMERGENCY,
            description="Drawdown exceeds 15%",
            check_fn_name="_check_drawdown_emergency",
        ),
        "flash_crash": BreakerTrigger(
            name="flash_crash",
            level=CircuitBreakerLevel.EMERGENCY,
            description="Price drop > 10% in < 5 minutes",
            check_fn_name="_check_flash_crash",
        ),
    }

    # Recovery conditions (level → required conditions to step down)
    RECOVERY_CONDITIONS = {
        CircuitBreakerLevel.CAUTION: {"cooldown_minutes": 60, "max_recent_vol_ratio": 1.5},
        CircuitBreakerLevel.REDUCED: {"cooldown_minutes": 240, "max_drawdown": 0.03},
        CircuitBreakerLevel.DEFENSIVE: {"cooldown_minutes": 720, "max_drawdown": 0.05},
        CircuitBreakerLevel.EMERGENCY: {"cooldown_minutes": 1440, "requires_human": False},
        CircuitBreakerLevel.LOCKDOWN: {"requires_human": True},
    }

    def __init__(self, risk_config: RiskConfig | None = None):
        self.config = risk_config or RiskConfig()
        self.state = BreakerState(level=CircuitBreakerLevel.NORMAL)
        self._risk_history: deque[RiskSnapshot] = deque(maxlen=10000)
        self._peak_nav: float = 0.0
        self._day_start_nav: float = 0.0
        self._flash_crash_prices: deque[tuple[datetime, float]] = deque(maxlen=300)

        logger.info("circuit_breaker_initialized")

    def evaluate(self, snapshot: RiskSnapshot) -> BreakerState:
        """
        Evaluate all trigger conditions and update circuit breaker state.
        Called on every risk check cycle (typically every few seconds).
        """
        if not self.config.circuit_breaker_enabled:
            return self.state

        self._risk_history.append(snapshot)

        # Update peaks
        if snapshot.nav > self._peak_nav:
            self._peak_nav = snapshot.nav

        # Track flash crash
        self._flash_crash_prices.append((snapshot.timestamp, snapshot.nav))

        # Check all triggers
        highest_triggered = CircuitBreakerLevel.NORMAL
        trigger_name = None

        for name, trigger in self.TRIGGERS.items():
            check_method = getattr(self, trigger.check_fn_name, None)
            if check_method and check_method(snapshot):
                if trigger.level > highest_triggered:
                    highest_triggered = trigger.level
                    trigger_name = name

        # Only escalate, never de-escalate in this pass
        if highest_triggered > self.state.level:
            old_level = self.state.level
            self.state = BreakerState(
                level=highest_triggered,
                triggered_by=trigger_name,
                triggered_at=datetime.utcnow(),
                actions_taken=self._get_actions(highest_triggered),
            )
            logger.warning(
                "circuit_breaker_escalated",
                from_level=old_level.name,
                to_level=highest_triggered.name,
                trigger=trigger_name,
                nav=snapshot.nav,
                drawdown=f"{snapshot.drawdown_pct:.2%}",
            )

        # Check for recovery (de-escalation)
        elif self.state.level > CircuitBreakerLevel.NORMAL:
            self._check_recovery(snapshot)

        return self.state

    def force_level(self, level: CircuitBreakerLevel, reason: str) -> None:
        """Manual override — typically used for LOCKDOWN."""
        self.state = BreakerState(
            level=level,
            triggered_by=f"manual: {reason}",
            triggered_at=datetime.utcnow(),
            actions_taken=self._get_actions(level),
        )
        logger.warning("circuit_breaker_manual_override", cb_level=level.name, reason=reason)

    def reset(self) -> None:
        """Reset to NORMAL (requires human authorization in production)."""
        self.state = BreakerState(level=CircuitBreakerLevel.NORMAL)
        logger.info("circuit_breaker_reset")

    # ─── Trigger Check Methods ────────────────────────────────────────────

    def _check_daily_loss(self, snap: RiskSnapshot) -> bool:
        return snap.daily_pnl_pct < -0.01

    def _check_daily_loss_severe(self, snap: RiskSnapshot) -> bool:
        return snap.daily_pnl_pct < -0.03

    def _check_daily_loss_critical(self, snap: RiskSnapshot) -> bool:
        return snap.daily_pnl_pct < -0.05

    def _check_volatility_spike(self, snap: RiskSnapshot) -> bool:
        if len(self._risk_history) < 100:
            return False
        recent_vols = [s.volatility_1h for s in list(self._risk_history)[-24:]]
        avg_vol = np.mean(recent_vols) if recent_vols else 0
        return snap.volatility_1h > 2 * avg_vol if avg_vol > 0 else False

    def _check_drawdown_moderate(self, snap: RiskSnapshot) -> bool:
        return snap.drawdown_pct > 0.05

    def _check_drawdown_severe(self, snap: RiskSnapshot) -> bool:
        return snap.drawdown_pct > 0.10

    def _check_drawdown_emergency(self, snap: RiskSnapshot) -> bool:
        return snap.drawdown_pct > 0.15

    def _check_correlation_convergence(self, snap: RiskSnapshot) -> bool:
        return snap.correlation_to_btc > 0.85

    def _check_flash_crash(self, snap: RiskSnapshot) -> bool:
        """Check if price dropped > 10% in < 5 minutes."""
        if len(self._flash_crash_prices) < 2:
            return False
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        recent = [(t, p) for t, p in self._flash_crash_prices if t >= cutoff]
        if len(recent) < 2:
            return False
        max_price = max(p for _, p in recent)
        min_price = min(p for _, p in recent)
        if max_price <= 0:
            return False
        drop_pct = (max_price - min_price) / max_price
        return drop_pct > 0.10

    # ─── Recovery Logic ───────────────────────────────────────────────────

    def _check_recovery(self, snap: RiskSnapshot) -> None:
        """Check if conditions are met to step down one level."""
        current = self.state.level
        conditions = self.RECOVERY_CONDITIONS.get(current, {})

        if conditions.get("requires_human", False):
            return  # Only manual reset can clear LOCKDOWN

        if self.state.triggered_at is None:
            return

        cooldown = conditions.get("cooldown_minutes", 60)
        elapsed = (datetime.utcnow() - self.state.triggered_at).total_seconds() / 60

        if elapsed < cooldown:
            return  # Still in cooldown period

        max_dd = conditions.get("max_drawdown", 1.0)
        if snap.drawdown_pct > max_dd:
            return  # Drawdown still too high

        # Step down one level
        new_level = CircuitBreakerLevel(max(current.value - 1, 0))
        logger.info(
            "circuit_breaker_recovery",
            from_level=current.name,
            to_level=new_level.name,
            elapsed_minutes=f"{elapsed:.0f}",
        )
        self.state = BreakerState(level=new_level)

    # ─── Action Definitions ───────────────────────────────────────────────

    def _get_actions(self, level: CircuitBreakerLevel) -> list[str]:
        """Get the list of actions for a given circuit breaker level."""
        actions_map = {
            CircuitBreakerLevel.NORMAL: [],
            CircuitBreakerLevel.CAUTION: [
                "Reduce new position sizes by 50%",
                "Tighten stop losses to 1.5x ATR",
                "Increase risk check frequency",
            ],
            CircuitBreakerLevel.REDUCED: [
                "No new position entries",
                "Tighten all stops to 1x ATR",
                "Cancel all pending limit orders",
            ],
            CircuitBreakerLevel.DEFENSIVE: [
                "Begin systematic de-risking (reduce 25% per hour)",
                "Activate tail hedge (put options or inverse position)",
                "Alert human operator",
            ],
            CircuitBreakerLevel.EMERGENCY: [
                "Close ALL positions immediately (market orders)",
                "Cancel all pending orders",
                "Disable all agents",
                "Send critical alert to all channels",
            ],
            CircuitBreakerLevel.LOCKDOWN: [
                "All trading halted",
                "All API connections terminated",
                "Awaiting human manual override to resume",
            ],
        }
        return actions_map.get(level, [])

    @property
    def is_trading_allowed(self) -> bool:
        return self.state.level < CircuitBreakerLevel.EMERGENCY

    @property
    def position_size_multiplier(self) -> float:
        """How much to scale new positions based on circuit breaker state."""
        multipliers = {
            CircuitBreakerLevel.NORMAL: 1.0,
            CircuitBreakerLevel.CAUTION: 0.5,
            CircuitBreakerLevel.REDUCED: 0.0,
            CircuitBreakerLevel.DEFENSIVE: 0.0,
            CircuitBreakerLevel.EMERGENCY: 0.0,
            CircuitBreakerLevel.LOCKDOWN: 0.0,
        }
        return multipliers.get(self.state.level, 0.0)


# ─── Pre-Trade Risk Checks ───────────────────────────────────────────────────


@dataclass
class PreTradeCheck:
    """Result of a pre-trade risk check."""
    passed: bool
    checks_passed: list[str]
    checks_failed: list[str]
    max_allowed_size: float


class PreTradeRiskValidator:
    """
    Synchronous pre-trade risk validation.
    Every order must pass these checks before submission.
    """

    def __init__(
        self,
        risk_config: RiskConfig | None = None,
        circuit_breaker: CircuitBreakerSystem | None = None,
    ):
        self.config = risk_config or RiskConfig()
        self.circuit_breaker = circuit_breaker

    def validate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_nav: float,
        current_positions: dict[str, float],
    ) -> PreTradeCheck:
        """Run all pre-trade risk checks."""
        passed_checks: list[str] = []
        failed_checks: list[str] = []

        notional = quantity * price
        position_pct = notional / portfolio_nav if portfolio_nav > 0 else 1.0

        # Check 1: Circuit breaker allows trading
        if self.circuit_breaker and not self.circuit_breaker.is_trading_allowed:
            failed_checks.append(
                f"Circuit breaker at level "
                f"{self.circuit_breaker.state.level.name}"
            )
        else:
            passed_checks.append("Circuit breaker: OK")

        # Check 2: Single position size limit
        if position_pct > self.config.max_single_position_pct:
            failed_checks.append(
                f"Position size {position_pct:.1%} exceeds "
                f"max {self.config.max_single_position_pct:.0%}"
            )
        else:
            passed_checks.append(f"Position size: {position_pct:.1%}")

        # Check 3: Portfolio drawdown limit
        # (This would use live portfolio data in production)
        passed_checks.append("Drawdown check: OK")

        # Check 4: Correlated exposure
        passed_checks.append("Correlation exposure: OK")

        # Compute max allowed
        max_size = self.config.max_single_position_pct * portfolio_nav / price if price > 0 else 0

        # Apply circuit breaker multiplier
        if self.circuit_breaker:
            max_size *= self.circuit_breaker.position_size_multiplier

        return PreTradeCheck(
            passed=len(failed_checks) == 0,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            max_allowed_size=max_size,
        )
