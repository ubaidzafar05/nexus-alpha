from __future__ import annotations

from datetime import datetime

from nexus_alpha.risk.firewall import AdversarialDeploymentGate, RiskContext, RiskFirewall
from nexus_alpha.types import ExchangeName, Order, OrderSide, OrderType, Portfolio


def _order(quantity: float) -> Order:
    return Order(
        order_id="order1234",
        symbol="BTCUSDT",
        exchange=ExchangeName.BINANCE,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=65000.0,
    )


def _portfolio(nav: float = 100000.0) -> Portfolio:
    return Portfolio(nav=nav, cash=nav, positions=[])


def test_risk_firewall_reduces_when_tail_risk_is_high() -> None:
    firewall = RiskFirewall()
    decision = firewall.decide(
        order=_order(quantity=1.0),
        context=RiskContext(
            portfolio=_portfolio(),
            recent_volatility=0.05,
            expected_tail_loss=-0.03,
        ),
    )
    assert decision.action in {"reduce", "block"}
    assert decision.approved_size <= decision.requested_size


def test_adversarial_deployment_gate_returns_structured_result() -> None:
    gate = AdversarialDeploymentGate(max_failed_scenarios=999)
    result = gate.evaluate(base_price=65000.0)
    assert result.scenarios_total >= 1
    assert result.scenarios_failed >= 0
    assert result.generated_at <= datetime.utcnow()
