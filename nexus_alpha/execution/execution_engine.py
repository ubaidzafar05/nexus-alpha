"""
Execution Intelligence — Optimal Control Theory + RL-based Execution.

Combines classical Almgren-Chriss optimal execution with RL-based
real-time adaptation. The RL agent learns to deviate from the textbook
trajectory when the order book gives signals that deviation is profitable.

Also includes intelligent cross-exchange routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from nexus_alpha.config import NexusConfig
from nexus_alpha.logging import get_logger
from nexus_alpha.types import ExchangeName, Order, OrderSide, OrderType

logger = get_logger(__name__)


# ─── Almgren-Chriss Optimal Execution ────────────────────────────────────────


@dataclass
class ExecutionSlice:
    """Single slice of an execution schedule."""
    time_index: int
    quantity: float
    target_time: datetime
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    slippage: float = 0.0
    is_complete: bool = False


@dataclass
class ExecutionSchedule:
    """Full execution plan from Almgren-Chriss."""
    total_quantity: float
    direction: str  # "buy" or "sell"
    n_slices: int
    time_horizon_minutes: float
    slices: list[ExecutionSlice] = field(default_factory=list)
    expected_cost: float = 0.0
    urgency: float = 0.5  # 0=patient, 1=aggressive


class AlmgrenChrissOptimizer:
    """
    Classical Almgren-Chriss optimal execution framework.
    Minimizes E[cost] + lambda * Var[cost] where cost = market impact + risk.

    Parameters:
    - sigma: volatility (annualized)
    - eta: temporary impact coefficient
    - gamma_perm: permanent impact coefficient
    - lambda_risk: risk aversion parameter (higher = more aggressive schedule)
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 2.5e-7,
        gamma_perm: float = 2.5e-7,
        lambda_risk: float = 1e-6,
    ):
        self.sigma = sigma
        self.eta = eta
        self.gamma_perm = gamma_perm
        self.lambda_risk = lambda_risk

    def compute_schedule(
        self,
        total_quantity: float,
        current_price: float,
        n_slices: int = 20,
        time_horizon_minutes: float = 60.0,
        urgency: float = 0.5,
    ) -> ExecutionSchedule:
        """
        Compute the optimal execution trajectory.

        Higher urgency → more front-loaded (like VWAP).
        Lower urgency → more evenly spread (reduces impact).
        """
        tau = time_horizon_minutes / n_slices  # Time per slice in minutes

        # Almgren-Chriss kappa (trade-off between risk and cost)
        # Higher kappa → more aggressive execution
        kappa = np.sqrt(self.lambda_risk * self.sigma ** 2 / self.eta) * (1 + urgency)

        # Optimal trajectory: x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
        T = n_slices
        trajectory = np.zeros(n_slices + 1)
        trajectory[0] = total_quantity

        for k in range(1, n_slices + 1):
            trajectory[k] = total_quantity * np.sinh(kappa * (T - k)) / (np.sinh(kappa * T) + 1e-10)

        # Trade sizes = difference between inventory levels
        trade_sizes = -np.diff(trajectory)

        # Expected cost
        temp_cost = self.eta * np.sum(trade_sizes ** 2 / tau)
        perm_cost = 0.5 * self.gamma_perm * total_quantity ** 2
        expected_cost = (temp_cost + perm_cost) * current_price

        now = datetime.utcnow()
        slices = [
            ExecutionSlice(
                time_index=i,
                quantity=float(trade_sizes[i]),
                target_time=datetime.utcnow(),  # Will be set by scheduler
            )
            for i in range(n_slices)
        ]

        schedule = ExecutionSchedule(
            total_quantity=total_quantity,
            direction="buy" if total_quantity > 0 else "sell",
            n_slices=n_slices,
            time_horizon_minutes=time_horizon_minutes,
            slices=slices,
            expected_cost=expected_cost,
            urgency=urgency,
        )

        logger.info(
            "execution_schedule_computed",
            quantity=total_quantity,
            n_slices=n_slices,
            horizon_min=time_horizon_minutes,
            expected_cost=f"{expected_cost:.2f}",
        )

        return schedule


# ─── Intelligent Exchange Routing ─────────────────────────────────────────────


@dataclass
class ExchangeLiquidity:
    """Real-time liquidity snapshot from one exchange."""
    exchange: ExchangeName
    bid_depth: float  # Total bid depth in base units
    ask_depth: float
    spread_bps: float  # Spread in basis points
    maker_fee_bps: float
    taker_fee_bps: float
    latency_ms: float
    recent_fill_quality: float  # 0-1, based on recent slippage
    is_available: bool = True


@dataclass
class RoutingDecision:
    """Decision on how to route an order across exchanges."""
    primary_exchange: ExchangeName
    allocation: dict[ExchangeName, float]  # Exchange → fraction of order
    estimated_cost_bps: float
    reasoning: str


class IntelligentExchangeRouter:
    """
    Real-time selection of optimal execution venue based on:
    - Current liquidity depth
    - Fee structure (maker vs taker)
    - Recent fill quality (slippage history)
    - Latency to exchange
    """

    # Default fee schedules (basis points)
    DEFAULT_FEES = {
        ExchangeName.BINANCE: {"maker": 1.0, "taker": 5.0},
        ExchangeName.BYBIT: {"maker": 1.0, "taker": 6.0},
        ExchangeName.KRAKEN: {"maker": 2.5, "taker": 4.0},
        ExchangeName.HYPERLIQUID: {"maker": 0.0, "taker": 3.0},
    }

    def __init__(self):
        self._liquidity_cache: dict[ExchangeName, ExchangeLiquidity] = {}
        self._fill_quality_history: dict[ExchangeName, list[float]] = {
            e: [] for e in ExchangeName
        }

    def update_liquidity(self, exchange: ExchangeName, liquidity: ExchangeLiquidity) -> None:
        """Update liquidity snapshot for an exchange."""
        self._liquidity_cache[exchange] = liquidity

    def route_order(self, order: Order) -> RoutingDecision:
        """
        Determine optimal routing for an order.
        Uses cost-minimization with liquidity and quality weighting.
        """
        available = {
            ex: liq for ex, liq in self._liquidity_cache.items()
            if liq.is_available
        }

        if not available:
            # Fallback to primary
            return RoutingDecision(
                primary_exchange=ExchangeName.BINANCE,
                allocation={ExchangeName.BINANCE: 1.0},
                estimated_cost_bps=5.0,
                reasoning="No liquidity data available — defaulting to primary exchange.",
            )

        # Score each exchange
        scores: dict[ExchangeName, float] = {}
        for exchange, liq in available.items():
            # Lower is better
            fee = liq.taker_fee_bps if order.order_type == OrderType.MARKET else liq.maker_fee_bps
            spread_cost = liq.spread_bps / 2  # Half spread
            latency_penalty = liq.latency_ms / 1000  # Normalized
            quality_bonus = liq.recent_fill_quality * 2  # Higher is better

            # Depth score (deeper is better)
            relevant_depth = liq.bid_depth if order.side == OrderSide.BUY else liq.ask_depth
            depth_score = min(relevant_depth / (order.quantity + 1e-10), 1.0)

            # Total score (higher is better)
            score = (
                -fee
                - spread_cost
                - latency_penalty
                + quality_bonus
                + depth_score * 3
            )
            scores[exchange] = score

        # Allocate proportionally to scores (softmax-style)
        min_score = min(scores.values())
        adj_scores = {k: v - min_score + 1 for k, v in scores.items()}
        total = sum(adj_scores.values())
        allocation = {k: v / total for k, v in adj_scores.items()}

        # Primary = highest allocation
        primary = max(allocation, key=allocation.get)  # type: ignore

        # Estimate total cost
        est_cost = sum(
            alloc * (available[ex].taker_fee_bps + available[ex].spread_bps / 2)
            for ex, alloc in allocation.items()
            if ex in available
        )

        decision = RoutingDecision(
            primary_exchange=primary,
            allocation=allocation,
            estimated_cost_bps=est_cost,
            reasoning=f"Optimal split across {len(allocation)} venues. "
                      f"Primary: {primary.value} ({allocation[primary]:.0%}).",
        )

        logger.info(
            "order_routed",
            symbol=order.symbol,
            primary=primary.value,
            allocation={k.value: f"{v:.2%}" for k, v in allocation.items()},
            est_cost_bps=f"{est_cost:.1f}",
        )

        return decision


# ─── Order Management System ─────────────────────────────────────────────────


class OrderManagementSystem:
    """
    Central OMS: manages order lifecycle, tracks fills, computes slippage.
    """

    def __init__(self):
        self._pending_orders: dict[str, Order] = {}
        self._executed_orders: list[Order] = []
        self._router = IntelligentExchangeRouter()
        self._ac_optimizer = AlmgrenChrissOptimizer()

    def submit_order(self, order: Order) -> RoutingDecision:
        """Submit an order through the routing engine."""
        self._pending_orders[order.order_id] = order
        routing = self._router.route_order(order)

        logger.info(
            "order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            primary_exchange=routing.primary_exchange.value,
        )

        return routing

    def create_execution_plan(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
        urgency: float = 0.5,
    ) -> ExecutionSchedule:
        """Create an optimal execution plan for a large order."""
        return self._ac_optimizer.compute_schedule(
            total_quantity=quantity,
            current_price=current_price,
            urgency=urgency,
        )

    def record_fill(self, order_id: str, filled_price: float, filled_quantity: float) -> None:
        """Record a fill for a pending order."""
        if order_id not in self._pending_orders:
            logger.warning("fill_for_unknown_order", order_id=order_id)
            return

        order = self._pending_orders[order_id]
        order.filled_quantity += filled_quantity
        order.filled_price = filled_price  # Simplified: should be VWAP of all fills
        order.filled_at = datetime.utcnow()

        if order.filled_quantity >= order.quantity:
            order.status = order.status.__class__("filled")
            order.slippage = abs(filled_price - (order.price or filled_price)) / filled_price
            self._executed_orders.append(order)
            del self._pending_orders[order_id]

            logger.info(
                "order_filled",
                order_id=order_id,
                filled_price=filled_price,
                slippage_bps=f"{order.slippage * 10000:.1f}",
            )

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    @property
    def avg_slippage_bps(self) -> float:
        if not self._executed_orders:
            return 0.0
        return np.mean([o.slippage * 10000 for o in self._executed_orders[-100:]])
