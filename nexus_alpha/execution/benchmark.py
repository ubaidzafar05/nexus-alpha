"""Execution benchmark harness against TWAP/VWAP baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from nexus_alpha.execution.execution_engine import AlmgrenChrissOptimizer
from nexus_alpha.logging import get_logger
from nexus_alpha.types import OrderSide

logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    strategy: str
    avg_fill_price: float
    implementation_shortfall_bps: float


class ExecutionBenchmarkHarness:
    """Compares adaptive schedule execution against TWAP and VWAP."""

    def __init__(self) -> None:
        self._optimizer = AlmgrenChrissOptimizer()

    def run(
        self,
        side: OrderSide,
        quantity: float,
        prices: np.ndarray,
        volumes: np.ndarray,
        urgency: float = 0.5,
    ) -> dict[str, object]:
        self._validate_inputs(quantity=quantity, prices=prices, volumes=volumes)
        arrival_price = float(prices[0])
        twap = self._simulate_twap(side, quantity, prices, arrival_price)
        vwap = self._simulate_vwap(side, quantity, prices, volumes, arrival_price)
        adaptive = self._simulate_adaptive(side, quantity, prices, arrival_price, urgency=urgency)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "arrival_price": arrival_price,
            "results": [
                self._to_payload(adaptive),
                self._to_payload(twap),
                self._to_payload(vwap),
            ],
            "winner": min(
                [adaptive, twap, vwap],
                key=lambda item: abs(item.implementation_shortfall_bps),
            ).strategy,
        }
        logger.info("execution_benchmark_complete", winner=report["winner"])
        return report

    def _simulate_twap(
        self,
        side: OrderSide,
        quantity: float,
        prices: np.ndarray,
        arrival_price: float,
    ) -> BenchmarkResult:
        weights = np.full(len(prices), 1 / len(prices))
        avg_fill = float(np.sum(prices * weights))
        is_bps = self._implementation_shortfall_bps(side, avg_fill, arrival_price)
        return BenchmarkResult(
            strategy="twap",
            avg_fill_price=avg_fill,
            implementation_shortfall_bps=is_bps,
        )

    def _simulate_vwap(
        self,
        side: OrderSide,
        quantity: float,
        prices: np.ndarray,
        volumes: np.ndarray,
        arrival_price: float,
    ) -> BenchmarkResult:
        weights = volumes / (np.sum(volumes) + 1e-10)
        avg_fill = float(np.sum(prices * weights))
        is_bps = self._implementation_shortfall_bps(side, avg_fill, arrival_price)
        return BenchmarkResult(
            strategy="vwap",
            avg_fill_price=avg_fill,
            implementation_shortfall_bps=is_bps,
        )

    def _simulate_adaptive(
        self,
        side: OrderSide,
        quantity: float,
        prices: np.ndarray,
        arrival_price: float,
        urgency: float,
    ) -> BenchmarkResult:
        schedule = self._optimizer.compute_schedule(
            total_quantity=quantity,
            current_price=arrival_price,
            n_slices=len(prices),
            urgency=urgency,
        )
        raw = np.array([max(slice_.quantity, 0.0) for slice_ in schedule.slices])
        weights = raw / (np.sum(raw) + 1e-10)

        impact = np.linspace(0.0, 0.0008, len(prices))
        if side == OrderSide.BUY:
            exec_prices = prices * (1 + impact)
        else:
            exec_prices = prices * (1 - impact)

        avg_fill = float(np.sum(exec_prices * weights))
        is_bps = self._implementation_shortfall_bps(side, avg_fill, arrival_price)
        return BenchmarkResult(
            strategy="adaptive_ac",
            avg_fill_price=avg_fill,
            implementation_shortfall_bps=is_bps,
        )

    def _implementation_shortfall_bps(
        self,
        side: OrderSide,
        avg_fill_price: float,
        arrival_price: float,
    ) -> float:
        if arrival_price <= 0:
            return 0.0
        signed = avg_fill_price - arrival_price
        if side == OrderSide.SELL:
            signed = -signed
        return float((signed / arrival_price) * 10_000)

    def _validate_inputs(self, quantity: float, prices: np.ndarray, volumes: np.ndarray) -> None:
        if quantity <= 0:
            raise ValueError("quantity_must_be_positive")
        if len(prices) == 0 or len(volumes) == 0:
            raise ValueError("price_volume_series_required")
        if len(prices) != len(volumes):
            raise ValueError("price_volume_length_mismatch")
        if np.any(prices <= 0) or np.any(volumes < 0):
            raise ValueError("invalid_price_or_volume_values")

    def _to_payload(self, result: BenchmarkResult) -> dict[str, float | str]:
        return {
            "strategy": result.strategy,
            "avg_fill_price": result.avg_fill_price,
            "implementation_shortfall_bps": result.implementation_shortfall_bps,
        }
