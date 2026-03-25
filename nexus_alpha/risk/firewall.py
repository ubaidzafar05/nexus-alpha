"""Phase 5 risk firewall orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from nexus_alpha.config import RiskConfig
from nexus_alpha.infrastructure.adversarial import AdversarialTestRunner
from nexus_alpha.logging import get_logger
from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem, PreTradeRiskValidator
from nexus_alpha.risk.contracts import DeploymentGateResult, RiskAction, RiskDecision
from nexus_alpha.types import Order, Portfolio

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskContext:
    portfolio: Portfolio
    recent_volatility: float
    expected_tail_loss: float


class RiskFirewall:
    """Central risk firewall around pre-trade checks and tail-risk controls."""

    def __init__(self, risk_config: RiskConfig | None = None):
        self._config = risk_config or RiskConfig()
        self._breaker = CircuitBreakerSystem(self._config)
        self._validator = PreTradeRiskValidator(self._config, circuit_breaker=self._breaker)

    def decide(self, order: Order, context: RiskContext) -> RiskDecision:
        order_price = order.price or 0.0
        if order_price <= 0:
            return RiskDecision(
                action=RiskAction.BLOCK,
                reason_codes=["invalid_order_price"],
                requested_size=order.quantity,
                approved_size=0.0,
                reduction_factor=0.0,
                timestamp=datetime.utcnow(),
            )
        current_positions = {
            position.symbol: float(position.quantity) for position in context.portfolio.positions
        }
        result = self._validator.validate(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=order_price,
            portfolio_nav=context.portfolio.nav,
            current_positions=current_positions,
        )
        reason_codes: list[str] = []

        if not result.passed:
            reason_codes.extend(_sanitize_reason(code) for code in result.checks_failed)
            return RiskDecision(
                action=RiskAction.BLOCK,
                reason_codes=reason_codes,
                requested_size=order.quantity,
                approved_size=0.0,
                reduction_factor=0.0,
                timestamp=datetime.utcnow(),
            )

        reduction = 1.0
        if context.recent_volatility > 0.04:
            reduction *= 0.5
            reason_codes.append("high_volatility_reduce")
        if context.expected_tail_loss < -0.02:
            reduction *= 0.5
            reason_codes.append("tail_loss_reduce")

        approved_size = order.quantity * reduction
        if reduction < 0.2:
            action = RiskAction.BLOCK
            approved_size = 0.0
            reason_codes.append("size_reduced_below_minimum")
        elif reduction < 1.0:
            action = RiskAction.REDUCE
        else:
            action = RiskAction.ALLOW
            if not reason_codes:
                reason_codes.append("all_checks_passed")

        decision = RiskDecision(
            action=action,
            reason_codes=reason_codes,
            requested_size=order.quantity,
            approved_size=approved_size,
            reduction_factor=reduction,
            timestamp=datetime.utcnow(),
        )
        logger.info(
            "risk_firewall_decision",
            action=decision.action.value,
            approved_size=decision.approved_size,
            reasons=decision.reason_codes,
        )
        return decision


class AdversarialDeploymentGate:
    """Phase 5 deployment gate using adversarial stress suite."""

    def __init__(
        self,
        max_failed_scenarios: int = 0,
        max_drawdown_threshold: float = 0.20,
    ):
        self.max_failed_scenarios = max_failed_scenarios
        self.max_drawdown_threshold = max_drawdown_threshold

    def evaluate(self, base_price: float = 65000.0) -> DeploymentGateResult:
        runner = AdversarialTestRunner()
        runner.run_all(base_price=base_price)
        report = runner.report()
        failed = int(report["failed"])
        worst_drawdown_str = str(report["worst_drawdown"]).rstrip("%")
        worst_drawdown = float(worst_drawdown_str) / 100 if worst_drawdown_str else 0.0

        reason_codes: list[str] = []
        if failed > self.max_failed_scenarios:
            reason_codes.append("adversarial_failures_exceeded")
        if worst_drawdown > self.max_drawdown_threshold:
            reason_codes.append("worst_drawdown_exceeded")

        passed = len(reason_codes) == 0
        if passed:
            reason_codes.append("adversarial_gate_passed")

        return DeploymentGateResult(
            passed=passed,
            reason_codes=reason_codes,
            scenarios_total=int(report["total_scenarios"]),
            scenarios_failed=failed,
            worst_drawdown=worst_drawdown,
            generated_at=datetime.utcnow(),
        )


def _sanitize_reason(text: str) -> str:
    return text.lower().replace(" ", "_").replace(".", "")
