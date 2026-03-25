from __future__ import annotations

from datetime import datetime

import numpy as np

from nexus_alpha.execution.benchmark import ExecutionBenchmarkHarness
from nexus_alpha.execution.contracts import ExecutionReport, OrderRequest, RouteDecision
from nexus_alpha.types import ExchangeName, OrderSide, OrderType


def test_execution_contract_models_validate() -> None:
    request = OrderRequest(
        order_id="abcd1234",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.5,
        created_at=datetime.utcnow(),
    )
    decision = RouteDecision(
        order_id=request.order_id,
        primary_exchange=ExchangeName.BINANCE,
        allocations={ExchangeName.BINANCE: 1.0},
        estimated_cost_bps=4.2,
        reasoning="Best depth and latency",
        decided_at=datetime.utcnow(),
    )
    report = ExecutionReport(
        order_id=request.order_id,
        symbol=request.symbol,
        side=request.side,
        requested_qty=request.quantity,
        filled_qty=request.quantity,
        avg_fill_price=65010.0,
        arrival_price=65000.0,
        implementation_shortfall_bps=1.53,
        completed_at=datetime.utcnow(),
    )
    assert decision.primary_exchange == ExchangeName.BINANCE
    assert report.filled_qty == request.quantity


def test_execution_benchmark_harness_runs() -> None:
    harness = ExecutionBenchmarkHarness()
    prices = np.array([65000.0, 65005.0, 65010.0, 65020.0, 65015.0], dtype=np.float64)
    volumes = np.array([100, 120, 130, 110, 90], dtype=np.float64)
    report = harness.run(
        side=OrderSide.BUY,
        quantity=3.0,
        prices=prices,
        volumes=volumes,
        urgency=0.7,
    )
    assert report["winner"] in {"adaptive_ac", "twap", "vwap"}
    assert len(report["results"]) == 3
