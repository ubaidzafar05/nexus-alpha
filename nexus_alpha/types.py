"""Shared data types used across the entire NEXUS-ALPHA system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


# ─── Market Regimes ──────────────────────────────────────────────────────────

class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


# ─── Order / Trade Types ─────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExchangeName(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    KRAKEN = "kraken"
    HYPERLIQUID = "hyperliquid"


# ─── Circuit Breaker Levels ──────────────────────────────────────────────────

class CircuitBreakerLevel(int, Enum):
    NORMAL = 0
    CAUTION = 1       # Reduce new position sizes by 50%
    REDUCED = 2       # No new entries, existing tight stops
    DEFENSIVE = 3     # Active de-risking, hedge positions
    EMERGENCY = 4     # Close all positions immediately
    LOCKDOWN = 5      # All trading halted, human override required


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tick:
    """Single market tick."""
    symbol: str
    exchange: ExchangeName
    timestamp: datetime
    bid: float
    ask: float
    last_price: float
    volume_24h: float
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass
class OHLCV:
    """Candlestick bar."""
    symbol: str
    exchange: ExchangeName
    timestamp: datetime
    timeframe: str  # "1m", "5m", "1h", "4h", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0


@dataclass
class Order:
    """Order to be submitted to an exchange."""
    order_id: str
    symbol: str
    exchange: ExchangeName
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current portfolio position."""
    symbol: str
    exchange: ExchangeName
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    stop_loss: float | None = None
    take_profit: float | None = None

    @property
    def notional_value(self) -> float:
        return abs(self.quantity * self.current_price)

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class Portfolio:
    """Full portfolio state."""
    nav: float
    cash: float
    positions: list[Position]
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(p.notional_value for p in self.positions)

    @property
    def leverage(self) -> float:
        return self.gross_exposure / self.nav if self.nav > 0 else 0.0


@dataclass
class Signal:
    """Trading signal from any agent or model."""
    signal_id: str
    source: str  # Agent/model name
    symbol: str
    direction: float  # -1.0 (strong sell) to 1.0 (strong buy)
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    timeframe: str
    features_used: list[str] = field(default_factory=list)
    causal_validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeState:
    """Current market regime detection result."""
    regime: MarketRegime
    confidence: float
    changepoint_probability: float
    volatility: float
    trend_strength: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    hmm_state: int = -1
    transition_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class WorldModelOutput:
    """Output from the World Model."""
    quantile_predictions: dict[float, np.ndarray]
    epistemic_uncertainty: float
    regime_context: MarketRegime
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeExplanation:
    """SHAP-based explanation for a trade decision."""
    trade_id: str
    signal_source: str
    top_features: list[tuple[str, float]]  # (feature_name, shap_value)
    causal_chain: list[str]
    risk_checks_passed: list[str]
    risk_checks_failed: list[str]
    llm_explanation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentPerformance:
    """Rolling performance metrics for a tournament agent."""
    agent_id: str
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win_loss_ratio: float
    total_trades: int
    pnl: float
    evaluation_window_days: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DebateVerdict:
    """Result of a multi-agent debate."""
    proposed_trade: Signal
    proposal_strength: float
    challenge_strength: float
    synthesis_recommendation: str  # "proceed", "reduce_size", "reject"
    adjusted_confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlphaFreshnessReport:
    """Tracks whether alpha is being replenished faster than it decays."""
    active_strategy_count: int
    avg_strategy_age_days: float
    alpha_decay_rate: float
    alpha_replenishment_rate: float
    is_self_sustaining: bool
    oldest_still_profitable_days: int
    newest_strategy_age_days: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
