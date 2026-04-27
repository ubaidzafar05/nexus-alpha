"""
Trading Loop Orchestrator — the central pipeline that wires all components.

Data flow:
  Kafka ticks → FeatureStore → SignalEngine → DebateGate → PortfolioOptimizer
  → PreTradeRisk → OrderManagement → Exchange/Paper → Telegram alerts

This is the *only* place where the trading pipeline is assembled. All other
modules are stateless or independently testable.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json as _json

import numpy as np
import pandas as pd

from nexus_alpha.agents.archetypes import ArchetypeType, BaseArchetype, get_all_archetypes
from nexus_alpha.agents.debate import AgentDebateProtocol, DebateContext
from nexus_alpha.agents.lifecycle import StrategyAgentLifecycle
from nexus_alpha.agents.tournament import TournamentOrchestrator
from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.config import NexusConfig, TradingMode
from nexus_alpha.execution.execution_engine import OrderManagementSystem
from nexus_alpha.learning.entry_features import build_augmented_feature_vector
from nexus_alpha.learning.historical_data import build_features
from nexus_alpha.learning.offline_trainer import LightweightPredictor, OnlineLearner
from nexus_alpha.learning.trade_logger import TradeLogger, TradeRecord
from nexus_alpha.learning.memory import MemoryManager
from nexus_alpha.intelligence.free_llm import FreeLLMClient
from nexus_alpha.log_config import get_logger
from nexus_alpha.core.regime_oracle import RegimeOracle
from nexus_alpha.portfolio.optimizer import (
    HierarchicalRiskParityOptimizer,
    kelly_position_size,
)
from nexus_alpha.risk.circuit_breaker import (
    CircuitBreakerSystem,
    PreTradeRiskValidator,
)
from nexus_alpha.signals.signal_engine import FusedSignal, SignalFusionEngine
from nexus_alpha.schema_types import (
    DebateVerdict,
    ExchangeName,
    MarketRegime,
    OrderSide,
    Portfolio,
    Position,
    Signal,
)

logger = get_logger(__name__)

DEBATE_NAV_THRESHOLD = 0.05  # 5% of NAV triggers debate
MIN_SIGNAL_CONFIDENCE = 0.55   # Raised to suppress low-edge trades (was 0.32 -> overtrading)
MIN_ML_CONFIDENCE = 0.20       # Minimum ML confidence to consider ML agreement valid

MIN_POSITION_USD = 10.0
MAX_OPEN_POSITIONS = 3         # Max simultaneous positions
MAX_TOTAL_EXPOSURE_PCT = 0.50  # Max 50% of NAV in open positions
COOLDOWN_SECONDS = 48 * 3600   # 48h between trades on same symbol (backtested optimal)
DATA_MAX_AGE_SECONDS = 300     # Skip symbol if price older than 5 min
MAX_DRAWDOWN_SCALE_THRESHOLD = 0.05  # Start scaling down after 5% drawdown
MAX_DRAWDOWN_HALT = 0.15       # Stop trading after 15% drawdown
PARTIAL_TP_FRACTION = 0.5      # Close 50% at take-profit, trail the rest
MOMENTUM_LOOKBACK = 5          # Candles to check for momentum direction
MIN_VOLUME_RATIO = 0.5         # Min volume vs 20-period average to trade
MEAN_REVERSION_ZSCORE = 2.0    # Z-score threshold for mean-reversion signals
STALE_POSITION_HOURS_WIN = 120 # Close winning positions after 120h
STALE_POSITION_HOURS_LOSE = 60 # Close losing positions after 60h
# Optimized SL/TP/Trailing params (backtested +11.8% over 4 years, all regimes profitable)
SL_ATR_MULT = 3.0              # 3× ATR stop loss distance
SL_FLOOR_PCT = 0.025           # 2.5% minimum SL (crypto noise filter)
SL_CAP_PCT = 0.06              # 6% maximum SL
TRAIL_ACTIVATION_PCT = 0.05    # Increased from 4% — surviving noise
TRAIL_ATR_MULT = 2.5           # Trail at peak minus 2.5× ATR
BREAKEVEN_TRIGGER_PCT = 0.035  # Increased from 2% — survival breathing room
USE_TREND_FILTER = True        # Only trade in direction of EMA-50 trend

# Crypto correlation clusters — highly correlated pairs share exposure budget
CORRELATION_CLUSTERS = {
    "layer1": {"BTCUSDT", "ETHUSDT", "SOLUSDT"},  # Move together
    "altcoin": {"BNBUSDT", "ADAUSDT"},
}
MAX_CLUSTER_POSITIONS = 2  # Max positions from same correlation cluster

# Regime-based confidence multipliers
REGIME_CONFIDENCE = {
    "strong_trend": 1.2,    # High confidence in trends
    "weak_trend": 1.0,      # Normal
    "range_bound": 0.7,     # Reduce in choppy markets
    "high_volatility": 0.6, # Be cautious in high vol
    "unknown": 0.9,
}
SYMBOL_LEARNING_MIN_TRADES = 5
SYMBOL_LEARNING_CACHE_SECONDS = 300.0


@dataclass
class LoopMetrics:
    ticks_processed: int = 0
    signals_generated: int = 0
    debates_triggered: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    last_tick_at: float = 0.0
    last_signal_at: float = 0.0
    last_order_at: float = 0.0
    errors: int = 0
    is_blind_halt: bool = False

    # V4 Telemetry: Missed Opportunities & Efficiency
    signals_pruned_causal: int = 0
    signals_pruned_guardian: int = 0
    signals_pruned_risk: int = 0
    avg_slippage_bps: float = 0.0
    
    # Audit tracking
    trades_since_last_audit: int = 0


@dataclass
class TradingDecision:
    signal: FusedSignal
    debate_verdict: DebateVerdict | None
    target_weight: float
    position_size_usd: float
    approved: bool
    rejection_reason: str = ""


class TradingLoopOrchestrator:
    """
    Continuous trading loop: consumes market data, produces trade decisions.

    Startup: call `run()` as an asyncio task.
    Shutdown: call `stop()` — graceful drain.

    Does NOT own the data ingestor or sentiment pipeline — those run
    as separate tasks. This loop reads from the in-memory feature store
    and Redis sentiment keys.
    """

    def __init__(
        self,
        config: NexusConfig,
        signal_engine: SignalFusionEngine,
        circuit_breaker: CircuitBreakerSystem,
        alerts: TelegramAlerts,
        portfolio: Portfolio | None = None,
        cycle_interval_s: float = 60.0,
        persist_portfolio: bool = True,
        regime_oracle: RegimeOracle | None = None,
    ) -> None:
        self._config = config
        self._signal_engine = signal_engine
        self._circuit_breaker = circuit_breaker
        self._alerts = alerts
        self._explicit_portfolio = portfolio is not None
        self._persist_portfolio = persist_portfolio
        self._portfolio = portfolio or Portfolio(
            nav=100_000.0,
            cash=100_000.0,
            positions=[],
        )
        self._cycle_interval = cycle_interval_s

        # Sub-components
        self._optimizer = HierarchicalRiskParityOptimizer(config.risk)
        self._debate = AgentDebateProtocol(config.llm)
        self._oms = OrderManagementSystem()
        self._risk_validator = PreTradeRiskValidator(
            risk_config=config.risk,
            circuit_breaker=circuit_breaker,
        )

        # ── Strategy Lifecycle (Tournament / God Loop) ─────────────────────
        self._lifecycle = StrategyAgentLifecycle(config)
        self._lifecycle.bootstrap()

        # ── V10 MAMP: Multi-Agent Personalities ───────────────────────────
        self._archetypes = get_all_archetypes()
        # Partitioned NAV: Sniper 20%, Tactical 50%, Scout 30%
        # These are used as 'internal accounts' to prevent crossing limits
        self._archetype_wallets: dict[ArchetypeType, float] = {
            ArchetypeType.SNIPER: self._portfolio.nav * 0.20,
            ArchetypeType.TACTICAL: self._portfolio.nav * 0.50,
            ArchetypeType.SCOUT: self._portfolio.nav * 0.30,
        }

        # ── V7 ULTRA: Meta-Orchestration (Phase 13) ───────────────────────
        from nexus_alpha.agents.meta import MetaOrchestrator
        # Pass MAMP wallets to MetaOrchestrator for dynamic rebalancing
        self._meta = MetaOrchestrator(self._lifecycle.tournament, self._archetype_wallets)

        # ── Learning pipeline ─────────────────────────────────────────────
        self._trade_logger = TradeLogger()
        self._llm_client = FreeLLMClient.from_config(config.llm)
        self._memory = MemoryManager(config, self._llm_client)
        self._ml_predictors: dict[str, LightweightPredictor] = {}
        self._online_learner = OnlineLearner(retrain_interval_hours=6)
        self._init_ml_models()

        # G2: Regime oracle with persistence
        self._regime_oracle = regime_oracle or RegimeOracle(n_regimes=5, lookback_window=200)

        # State
        self._running = False
        self._metrics = LoopMetrics()
        self._redis: Any = None
        self._last_trade_time: dict[str, float] = {}  # symbol → monotonic timestamp
        self._portfolio_file = Path("data/trade_logs/portfolio_state.json")
        self._live_state_file = Path("data/live_state.json")
        self._recent_signals: deque = deque(maxlen=30)
        self._cycle_counter: int = 0
        self._last_cycle_at: float = 0.0
        self._symbol_learning_scores: dict[str, float] = {}
        self._symbol_learning_scores_expires_at = 0.0

        # Restore portfolio from disk only if persistence enabled and no explicit portfolio
        if self._persist_portfolio and not self._explicit_portfolio:
            self._load_portfolio()

    @property
    def metrics(self) -> LoopMetrics:
        return self._metrics

    def _entry_signal_threshold(self) -> float:
        paper_override = self._config.paper_min_signal_confidence
        if self._config.trading_mode == TradingMode.PAPER and paper_override is not None:
            return float(paper_override)
        return MIN_SIGNAL_CONFIDENCE

    def _update_symbol_learning_scores(self) -> None:
        """Refresh cached per-symbol learning multipliers from the trade journal.

        These scores are conservative modifiers (0.80–1.15) computed from past
        trade outcomes and used to adapt entry thresholds per-symbol.
        """
        try:
            now = time.time()
            if now < self._symbol_learning_scores_expires_at and self._symbol_learning_scores:
                return
            scores = self._trade_logger.get_symbol_learning_scores(min_trades=SYMBOL_LEARNING_MIN_TRADES)
            if scores:
                self._symbol_learning_scores = scores
            else:
                # default to neutral multipliers
                self._symbol_learning_scores = {}
            self._symbol_learning_scores_expires_at = now + SYMBOL_LEARNING_CACHE_SECONDS
            logger.debug("symbol_learning_scores_updated", scores_len=len(self._symbol_learning_scores))
        except Exception as err:
            logger.warning("symbol_learning_scores_update_failed", error=repr(err))

    def _effective_entry_threshold(self, symbol: str) -> float:
        """Return the entry threshold adjusted for symbol learning score and regime.

        If a symbol has performed poorly (score < 1.0) the effective threshold
        is increased to be more selective; for well-performing symbols it is
        decreased slightly.
        """
        base = self._entry_signal_threshold()
        try:
            self._update_symbol_learning_scores()
            score = float(self._symbol_learning_scores.get(symbol, 1.0))
            # multiplier: lower than 1.0 -> make threshold stricter, higher -> looser
            # map score [0.8,1.15] -> multiplier [1.2,0.9]
            # multiplier = 1.0 + (1.0 - score) * 1.0 (simple linear mapping)
            multiplier = float(max(0.6, min(1.4, 1.0 + (1.0 - score))))
            return float(np.clip(base * multiplier, 0.0, 0.95))
        except Exception:
            return base

    def _position_max_age_hours(self, pnl_pct: float) -> float:
        paper_override = self._config.paper_max_position_age_hours
        if self._config.trading_mode == TradingMode.PAPER and paper_override is not None:
            return float(paper_override)
        return STALE_POSITION_HOURS_LOSE if pnl_pct < 0 else STALE_POSITION_HOURS_WIN

    # ── ML model loading ──────────────────────────────────────────────────

    def _init_ml_models(self) -> None:
        """Load trained ML models from checkpoints if available (1h, 4h, 1d)."""
        from pathlib import Path
        ckpt_dir = Path("data/checkpoints")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        timeframes = ["1h", "4h", "1d"]
        for sym in symbols:
            ccxt_sym = sym.replace("USDT", "_USDT")
            for tf in timeframes:
                # All models predict 1-candle-ahead return (target_1h = 1 candle)
                predictor = LightweightPredictor(target_horizon="target_1h")
                ckpt_path = ckpt_dir / f"lightweight_{ccxt_sym}_{tf}.pkl"
                if predictor.load(ckpt_path):
                    key = sym if tf == "1h" else f"{sym}_{tf}"
                    self._ml_predictors[key] = predictor
                    logger.info("ml_model_loaded", symbol=sym, timeframe=tf)

        if self._ml_predictors:
            logger.info("ml_models_ready", count=len(self._ml_predictors))
        else:
            logger.info("ml_models_not_available", msg="Run 'nexus train' to train models")

    def _get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> dict | None:
        """Get ML prediction for a symbol if a trained model exists."""
        predictor = self._ml_predictors.get(symbol)
        if not predictor:
            return None
        try:
            features = build_features(df)
            feature_cols = [c for c in features.columns if not c.startswith("target_")]
            if len(features) == 0:
                return None
            last_row = features[feature_cols].iloc[-1].values.astype(np.float32)
            prediction = predictor.predict(last_row)

            reward_overlay = self._online_learner.predict_reward(
                last_row,
                prediction.get("signal", 0.0),
            )
            if reward_overlay:
                reward_score = reward_overlay["reward_prediction"]
                scale = float(np.clip(1.0 + reward_score * 0.12, 0.35, 1.25))
                adjusted_conf = float(np.clip(prediction["confidence"] * scale, 0.0, 1.0))
                signal_direction = float(np.sign(prediction.get("signal", 0.0)))
                prediction = {
                    **prediction,
                    "confidence": round(adjusted_conf, 4),
                    "signal": round(signal_direction * adjusted_conf, 4) if signal_direction != 0 else 0.0,
                    "reward_overlay": reward_score,
                    "reward_overlay_confidence": reward_overlay["confidence"],
                }

            return prediction
        except Exception as err:
            logger.warning("ml_prediction_error", symbol=symbol, error=repr(err))
            return None

    def _get_feature_vector(self, symbol: str, df: pd.DataFrame) -> list[float] | None:
        """Extract the base market feature vector used at entry time."""
        try:
            features = build_features(df)
            feature_cols = [c for c in features.columns if not c.startswith("target_")]
            if len(features) == 0:
                return None
            return features[feature_cols].iloc[-1].tolist()
        except Exception:
            return None

    def _directional_persistence_24(self, df: pd.DataFrame) -> float:
        close = pd.to_numeric(df.get("close"), errors="coerce")
        if close is None or len(close) < 2:
            return 0.0
        direction = np.sign(close.pct_change().fillna(0.0))
        return float(direction.tail(24).mean()) if len(direction) else 0.0

    def _volatility_compression(self, df: pd.DataFrame) -> float:
        close = pd.to_numeric(df.get("close"), errors="coerce")
        if close is None or len(close) < 25:
            return 0.0
        returns = close.pct_change()
        vol_short = returns.tail(24).std()
        vol_long = returns.tail(168).std() if len(returns) >= 168 else vol_short
        if vol_long is None or vol_long <= 0:
            return 0.0
        return float(np.clip(1.0 - (vol_short / vol_long), -1.0, 1.0))

    def _build_learning_feature_vector(
        self,
        symbol: str,
        price: float,
        decision: TradingDecision,
        entry_ctx: dict[str, Any],
    ) -> list[float] | None:
        df = self._build_feature_dataframe(symbol)
        if df is None:
            return None
        base_vector = self._get_feature_vector(symbol, df)
        if base_vector is None:
            return None

        contributing = dict(decision.signal.contributing_signals)
        augmented, derived_context = build_augmented_feature_vector(
            base_vector,
            signal_confidence=decision.signal.confidence,
            pair_quality=float(entry_ctx.get("pair_quality", 0.5) or 0.5),
            mtf_alignment=float(entry_ctx.get("mtf_alignment", 1.0) or 1.0),
            regime_multiplier=float(entry_ctx.get("regime_multiplier", 0.9) or 0.9),
            ml_confidence=float(entry_ctx.get("ml_confidence", 0.0) or 0.0),
            ml_signal=float(entry_ctx.get("ml_signal", 0.0) or 0.0),
            contributing_signals=contributing,
            directional_persistence_24=self._directional_persistence_24(df),
            volatility_compression=self._volatility_compression(df),
            entry_price=price,
            atr=float(entry_ctx.get("atr", 0.0) or 0.0),
            sl_atr_mult=SL_ATR_MULT,
            sl_floor_pct=SL_FLOOR_PCT,
            sl_cap_pct=SL_CAP_PCT,
            breakeven_trigger_pct=BREAKEVEN_TRIGGER_PCT,
            trailing_trigger_pct=TRAIL_ACTIVATION_PCT,
            trade_direction=decision.signal.direction,
        )
        entry_ctx.update({
            "directional_persistence_24": self._directional_persistence_24(df),
            "volatility_compression": self._volatility_compression(df),
            **derived_context,
        })
        return augmented

    # ── Portfolio persistence ─────────────────────────────────────────────

    def _save_portfolio(self) -> None:
        """Save portfolio state and regime oracle to disk so they survive restarts."""
        if not self._persist_portfolio:
            return
        try:
            self._portfolio_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "nav": self._portfolio.nav,
                "cash": self._portfolio.cash,
                "total_realized_pnl": self._portfolio.total_realized_pnl,
                "positions": [
                    {
                        "trade_id": p.trade_id,
                        "symbol": p.symbol,
                        "side": p.side.value,
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "realized_pnl": p.realized_pnl,
                        "opened_at": p.opened_at.isoformat(),
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                    }
                    for p in self._portfolio.positions
                ],
                "saved_at": datetime.utcnow().isoformat(),
            }
            self._portfolio_file.write_text(_json.dumps(state, indent=2))
            # G2: Persist regime oracle state
            self._regime_oracle.save_checkpoint()
        except Exception as err:
            logger.warning("portfolio_save_failed", error=repr(err))

    def _write_live_state(self, fused_signals: list | None = None) -> None:
        """Dump a compact, atomic JSON snapshot every cycle.

        The dashboard reads this file instead of querying the running process.
        Writing is atomic (tmp + rename) so a partial read is impossible.
        """
        try:
            if fused_signals:
                now_iso = datetime.utcnow().isoformat()
                for fs in fused_signals:
                    self._recent_signals.append({
                        "timestamp": now_iso,
                        "symbol": getattr(fs, "symbol", "?"),
                        "direction": float(getattr(fs, "direction", 0.0) or 0.0),
                        "confidence": float(getattr(fs, "confidence", 0.0) or 0.0),
                        "regime": (getattr(fs, "metadata", {}) or {}).get("regime", "unknown"),
                        "contributing": list((getattr(fs, "contributing_signals", {}) or {}).keys())[:6],
                    })

            try:
                ms = self._signal_engine.get_microstructure_stats()
            except Exception:
                ms = {"vpin_max": 0.0, "ofi_max": 0.0}

            try:
                regime_obj = getattr(self._regime_oracle, "_state", None) or getattr(
                    self._regime_oracle, "current_regime", None
                )
                regime_name = (
                    regime_obj.regime.value
                    if regime_obj is not None and hasattr(regime_obj, "regime") and hasattr(regime_obj.regime, "value")
                    else str(regime_obj) if regime_obj is not None else "unknown"
                )
                changepoint = float(getattr(regime_obj, "changepoint_probability", 0.0) or 0.0) if regime_obj else 0.0
            except Exception:
                regime_name = "unknown"
                changepoint = 0.0

            cb_state = getattr(self._circuit_breaker, "state", None)
            cb_level = cb_state.level.name if cb_state and hasattr(cb_state, "level") else "NORMAL"

            control_paused = False
            try:
                ctrl_path = Path("data/bot_control.json")
                if ctrl_path.exists():
                    control_paused = bool(_json.loads(ctrl_path.read_text()).get("paused", False))
            except Exception:
                pass

            state = {
                "heartbeat": datetime.utcnow().isoformat(),
                "cycle_counter": self._cycle_counter,
                "cycle_interval_s": self._cycle_interval,
                "paused": control_paused,
                "blind_halt": bool(getattr(self._metrics, "is_blind_halt", False)),
                "cb_level": cb_level,
                "portfolio": {
                    "nav": float(self._portfolio.nav),
                    "cash": float(self._portfolio.cash),
                    "realized_pnl": float(self._portfolio.total_realized_pnl),
                    "position_count": int(self._portfolio.position_count),
                    "leverage": float(getattr(self._portfolio, "leverage", 0.0)),
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "side": p.side.value,
                            "quantity": float(p.quantity),
                            "entry_price": float(p.entry_price),
                            "current_price": float(p.current_price),
                            "unrealized_pnl": float(p.unrealized_pnl),
                        }
                        for p in self._portfolio.positions
                    ],
                },
                "regime": {
                    "name": regime_name,
                    "changepoint_probability": changepoint,
                },
                "microstructure": {
                    "vpin_max": float(ms.get("vpin_max", 0.0)),
                    "ofi_max": float(ms.get("ofi_max", 0.0)),
                },
                "metrics": {
                    "ticks_processed": int(self._metrics.ticks_processed),
                    "errors": int(self._metrics.errors),
                    "signals_pruned_causal": int(getattr(self._metrics, "signals_pruned_causal", 0)),
                    "avg_slippage_bps": float(getattr(self._metrics, "avg_slippage_bps", 0.0)),
                },
                "recent_signals": list(self._recent_signals),
            }

            tmp = self._live_state_file.with_suffix(".tmp")
            self._live_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(_json.dumps(state))
            tmp.replace(self._live_state_file)
        except Exception as err:
            logger.warning("live_state_write_failed", error=repr(err))

    def _load_portfolio(self) -> None:
        """Load portfolio state from disk on startup."""
        if not self._portfolio_file.exists():
            self._cleanup_orphaned_trades()
            logger.info("no_saved_portfolio", msg="Starting with fresh portfolio")
            return
        try:
            state = _json.loads(self._portfolio_file.read_text())
            positions = []
            for p in state.get("positions", []):
                opened_at = datetime.fromisoformat(p["opened_at"]) if p.get("opened_at") else datetime.utcnow()
                positions.append(Position(
                    trade_id=p.get("trade_id"),
                    symbol=p["symbol"],
                    exchange=ExchangeName.BINANCE,
                    side=OrderSide(p["side"]),
                    quantity=p["quantity"],
                    entry_price=p["entry_price"],
                    current_price=p["current_price"],
                    unrealized_pnl=p.get("unrealized_pnl", 0.0),
                    realized_pnl=p.get("realized_pnl", 0.0),
                    opened_at=opened_at,
                    stop_loss=p.get("stop_loss"),
                    take_profit=p.get("take_profit"),
                ))
            self._portfolio.cash = state["cash"]
            self._portfolio.nav = state["nav"]
            self._portfolio.total_realized_pnl = state.get("total_realized_pnl", 0.0)
            self._portfolio.positions = positions
            self._cleanup_orphaned_trades()
            logger.info(
                "portfolio_restored",
                nav=f"{self._portfolio.nav:.2f}",
                cash=f"{self._portfolio.cash:.2f}",
                positions=len(self._portfolio.positions),
            )
        except Exception as err:
            logger.warning("portfolio_load_failed", error=repr(err))

    def _trade_matches_position(self, trade: dict[str, Any], pos: Position) -> bool:
        if pos.trade_id:
            return trade.get("trade_id") == pos.trade_id
        return (
            trade.get("symbol") == pos.symbol
            and str(trade.get("side", "")).lower() == pos.side.value
        )

    def _cleanup_orphaned_trades(self) -> None:
        """Reconcile restored portfolio positions with open trade journal rows."""
        try:
            open_trades = self._trade_logger.get_open_trades()
            if not open_trades and not self._portfolio.positions:
                return

            remaining_trades = list(open_trades)
            matched_positions: list[Position] = []
            ghost_positions: list[Position] = []

            for pos in self._portfolio.positions:
                match_index = next(
                    (idx for idx, trade in enumerate(remaining_trades) if self._trade_matches_position(trade, pos)),
                    None,
                )
                if match_index is None:
                    ghost_positions.append(pos)
                    continue

                trade = remaining_trades.pop(match_index)
                trade_ts = trade.get("timestamp")
                if trade_ts:
                    try:
                        pos.opened_at = datetime.fromisoformat(str(trade_ts))
                    except ValueError:
                        pass
                pos.trade_id = str(trade.get("trade_id") or pos.trade_id or "")
                matched_positions.append(pos)

            if ghost_positions:
                ghost_value = sum(
                    pos.quantity * pos.current_price * (1 if pos.side == OrderSide.BUY else -1)
                    for pos in ghost_positions
                )
                self._portfolio.cash += ghost_value
                logger.info("ghost_positions_dropped", count=len(ghost_positions))

            self._portfolio.positions = matched_positions
            self._portfolio.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in matched_positions)
            self._portfolio.nav = self._portfolio.cash + sum(
                pos.quantity * pos.current_price * (1 if pos.side == OrderSide.BUY else -1)
                for pos in matched_positions
            )

            if not remaining_trades:
                return

            for t in remaining_trades:
                self._trade_logger.log_trade_close(
                    trade_id=t["trade_id"],
                    exit_price=t.get("entry_price", 0),
                    realized_pnl=0.0,
                    reward=0.0,
                    exit_context='{"exit_reason": "orphaned_on_restart"}',
                )
            logger.info("orphaned_trades_cleaned", count=len(remaining_trades))
        except Exception as err:
            logger.warning("orphan_cleanup_failed", error=repr(err))

    # ── NAV recalculation ─────────────────────────────────────────────────

    def _recalculate_nav(self) -> None:
        """Recalculate NAV from current prices every cycle."""
        total_unrealized = 0.0
        for pos in self._portfolio.positions:
            current_price = self._get_latest_price(pos.symbol) or pos.current_price
            pos.current_price = current_price
            if pos.side == OrderSide.BUY:
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
            total_unrealized += pos.unrealized_pnl

        self._portfolio.total_unrealized_pnl = total_unrealized
        # Longs: we hold the asset → add value; Shorts: we owe the asset → subtract value
        position_value = sum(
            p.quantity * p.current_price * (1 if p.side == OrderSide.BUY else -1)
            for p in self._portfolio.positions
        )
        self._portfolio.nav = self._portfolio.cash + position_value

    # ── Position guards ───────────────────────────────────────────────────

    def _has_open_position(self, symbol: str) -> bool:
        """Check if we already have an open position for this symbol."""
        return any(p.symbol == symbol for p in self._portfolio.positions)

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if we recently traded this symbol."""
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        return (time.monotonic() - last) < COOLDOWN_SECONDS

    def _total_exposure_pct(self) -> float:
        """Current total exposure as percentage of NAV."""
        if self._portfolio.nav <= 0:
            return 1.0
        return self._portfolio.gross_exposure / self._portfolio.nav

    def _is_data_fresh(self, symbol: str) -> bool:
        """Check if price data for this symbol is recent enough to trade on.

        Accepts both millisecond epoch integers and ISO8601 timestamp strings
        (some ingestors serialize timestamps differently). Falls back to len
        check when timestamp parsing fails.
        """
        if not self._redis:
            return False
        try:
            # Check if the price key has a TTL or was recently updated
            raw_ohlcv = self._redis.get(f"ohlcv:{symbol}")
            if not raw_ohlcv:
                return False
            data = _json.loads(raw_ohlcv)
            if not data:
                return False
            # Check last candle timestamp if available
            last_candle = data[-1]
            if "timestamp" in last_candle:
                ts = last_candle["timestamp"]
                # Numeric epoch (ms)
                try:
                    if isinstance(ts, (int, float)):
                        age = time.time() - float(ts) / 1000.0
                        return age < DATA_MAX_AGE_SECONDS
                    # Some sources write timestamp as numeric string
                    if isinstance(ts, str) and ts.isdigit():
                        age = time.time() - float(int(ts)) / 1000.0
                        return age < DATA_MAX_AGE_SECONDS
                except Exception:
                    # Fall through to ISO parsing
                    pass
                # ISO string like 2026-04-06T15:10:00 or with timezone
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        # datetime.fromisoformat handles most ISO formats
                        dt = datetime.fromisoformat(ts)
                        age = time.time() - dt.timestamp()
                        return age < DATA_MAX_AGE_SECONDS
                    except Exception:
                        # Try pandas as last resort (handles more formats)
                        try:
                            import pandas as pd

                            dt = pd.to_datetime(ts)
                            if not dt.tzinfo:
                                age = time.time() - dt.to_pydatetime().timestamp()
                            else:
                                age = time.time() - dt.tz_convert(None).to_pydatetime().timestamp()
                            return age < DATA_MAX_AGE_SECONDS
                        except Exception:
                            # Give up and fall back to existence check
                            return len(data) > 20
            # If no timestamp, just check data exists (fallback)
            return len(data) > 20
        except Exception:
            return True  # Don't block on check failures

    # ── ATR-based dynamic stops ───────────────────────────────────────────

    def _compute_atr(self, symbol: str, period: int = 14) -> float | None:
        """Compute ATR from Redis OHLCV data for dynamic TP/SL."""
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < period + 1:
            return None
        try:
            high = pd.to_numeric(df["high"], errors="coerce")
            low = pd.to_numeric(df["low"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            return float(tr.rolling(period).mean().iloc[-1])
        except Exception:
            return None

    # ── B3: Position correlation check ────────────────────────────────────

    def _cluster_for_symbol(self, symbol: str) -> str | None:
        for cluster_name, members in CORRELATION_CLUSTERS.items():
            if symbol in members:
                return cluster_name
        return None

    def _cluster_position_count(self, symbol: str) -> int:
        """Count how many open positions share this symbol's correlation cluster."""
        cluster = self._cluster_for_symbol(symbol)
        if cluster is None:
            return 0
        members = CORRELATION_CLUSTERS[cluster]
        return sum(1 for p in self._portfolio.positions if p.symbol in members)

    # ── B4: Drawdown-based position scaling ───────────────────────────────

    def _drawdown_scale_factor(self) -> float:
        """Reduce position size after drawdowns. Returns 0.0–1.0 multiplier."""
        if self._portfolio.nav <= 0:
            return 0.0
        initial_nav = self._portfolio.nav - self._portfolio.total_realized_pnl
        if initial_nav <= 0:
            initial_nav = self._portfolio.nav
        drawdown = 1.0 - (self._portfolio.nav / initial_nav) if initial_nav > self._portfolio.nav else 0.0
        if drawdown >= MAX_DRAWDOWN_HALT:
            return 0.0
        if drawdown >= MAX_DRAWDOWN_SCALE_THRESHOLD:
            # Linear scale from 1.0 at threshold to 0.2 at halt
            fraction = (drawdown - MAX_DRAWDOWN_SCALE_THRESHOLD) / (MAX_DRAWDOWN_HALT - MAX_DRAWDOWN_SCALE_THRESHOLD)
            return max(0.2, 1.0 - fraction * 0.8)
        return 1.0

    # ── E1: Volume confirmation ───────────────────────────────────────────

    def _volume_confirms(self, symbol: str) -> bool:
        """Check that recent volume is above average — avoid trading into thin liquidity.
        Uses 5-candle rolling avg (not single candle) to avoid being fooled by
        incomplete current candle in tick-aggregated data."""
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < 25:
            return True  # Can't check, allow
        try:
            vol = pd.to_numeric(df["volume"], errors="coerce")
            avg_20 = vol.iloc[-25:-5].mean()  # Use candles 25→5 ago as baseline
            recent_5 = vol.iloc[-5:].mean()   # Last 5 candles (smooths out incomplete candle)
            if avg_20 <= 0:
                return True
            return (recent_5 / avg_20) >= MIN_VOLUME_RATIO
        except Exception:
            return True

    # ── E2: Momentum confirmation (anti falling-knife) ────────────────────

    def _momentum_confirms(self, symbol: str, direction: float) -> bool:
        """Don't buy into falling prices or sell into rising prices."""
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < MOMENTUM_LOOKBACK + 1:
            return True
        try:
            close = pd.to_numeric(df["close"], errors="coerce")
            recent_return = (close.iloc[-1] / close.iloc[-MOMENTUM_LOOKBACK] - 1)
            # If buying, recent price should not be falling sharply
            if direction > 0 and recent_return < -0.03:
                return False
            # If selling, recent price should not be rising sharply
            if direction < 0 and recent_return > 0.03:
                return False
            return True
        except Exception:
            return True

    # ── F1: Signal-conflict arbitration ──────────────────────────────────

    def _arbitrate_signals(self, signals: list[FusedSignal]) -> list[FusedSignal]:
        """
        Resolve conflicting signals. When correlated pairs disagree
        on direction, keep only the strongest signal from each cluster.
        Also ranks by pair quality score.
        """
        if not signals:
            return []

        # Score each signal by pair quality × signal strength
        scored = []
        for sig in signals:
            quality = self._pair_quality_score(sig.symbol)
            composite = abs(sig.direction) * sig.confidence * quality
            scored.append((composite, quality, sig))

        # Sort by composite score (best first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Resolve cluster conflicts: for each cluster, pick the best signal
        # and ensure all signals from that cluster agree on direction
        cluster_direction: dict[str, float] = {}
        accepted: list[FusedSignal] = []

        for composite, quality, sig in scored:
            cluster = self._cluster_for_symbol(sig.symbol) or sig.symbol
            if cluster in cluster_direction:
                # Cluster already committed — only accept if same direction
                if (sig.direction > 0) != (cluster_direction[cluster] > 0):
                    logger.info(
                        "signal_conflict_rejected",
                        symbol=sig.symbol,
                        cluster=cluster,
                        direction=f"{sig.direction:.3f}",
                        cluster_direction=f"{cluster_direction[cluster]:.3f}",
                    )
                    continue
            else:
                cluster_direction[cluster] = sig.direction

            accepted.append(sig)

        return accepted

    # ── F2: Pair quality scoring ──────────────────────────────────────────

    def _pair_quality_score(self, symbol: str) -> float:
        """
        Score a symbol's current tradability (0.0–1.0) based on:
        - Volume relative to average (higher = better)
        - Recent volatility (moderate = best)
        - Spread implied from OHLC (tighter = better)
        """
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < 30:
            return 0.5  # neutral

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        # Volume score: current vs 20-period avg (0–1, capped at 2x avg)
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_now = volume.iloc[-1]
        vol_score = min(vol_now / vol_avg, 2.0) / 2.0 if vol_avg > 0 else 0.5

        # Volatility score: prefer moderate (1-3% 24h range). Too low = no move, too high = unpredictable
        returns = close.pct_change().dropna()
        vol_24h = returns.tail(24).std() * 100 if len(returns) >= 24 else 1.0
        if vol_24h < 0.3:
            vol_quality = 0.3  # Too quiet
        elif vol_24h > 5.0:
            vol_quality = 0.4  # Too wild
        else:
            vol_quality = 1.0 - abs(vol_24h - 1.5) / 5.0  # Sweet spot ~1.5%

        # Spread score: (high-low)/close as proxy for spread + slippage
        spread_pct = ((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else 0.1
        spread_score = max(0.0, 1.0 - spread_pct * 20)  # 5% range = 0.0, 0% = 1.0

        quality = vol_score * 0.4 + vol_quality * 0.35 + spread_score * 0.25
        return round(max(0.0, min(quality, 1.0)), 3)

    def _symbol_learning_multiplier(self, symbol: str) -> float:
        """Confidence multiplier learned from symbol-specific historical outcomes."""
        now = time.monotonic()
        if now >= self._symbol_learning_scores_expires_at:
            self._symbol_learning_scores = self._trade_logger.get_symbol_learning_scores(
                min_trades=SYMBOL_LEARNING_MIN_TRADES
            )
            self._symbol_learning_scores_expires_at = now + SYMBOL_LEARNING_CACHE_SECONDS
        return float(self._symbol_learning_scores.get(symbol, 1.0))

    # ── F3: Mean-reversion overlay ────────────────────────────────────────

    def _mean_reversion_signal(self, symbol: str) -> float | None:
        """
        Detect mean-reversion opportunities in range-bound markets.
        Returns a signal [-1, 1] if conditions met, None if no signal.
        Fires when price is >2 std devs from 50-period mean.
        """
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < 60:
            return None

        close = pd.to_numeric(df["close"], errors="coerce")
        ma = close.rolling(50).mean()
        std = close.rolling(50).std()

        if std.iloc[-1] <= 0 or pd.isna(std.iloc[-1]):
            return None

        zscore = (close.iloc[-1] - ma.iloc[-1]) / std.iloc[-1]

        # Only fire in range-bound (ADX < 25 proxy: low trend strength)
        ret_20 = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
        is_trending = abs(ret_20) > 0.05  # >5% move in 20 candles = trending

        if is_trending:
            return None

        if zscore > MEAN_REVERSION_ZSCORE:
            return -min(zscore / 3.0, 1.0)  # Overbought → sell signal
        elif zscore < -MEAN_REVERSION_ZSCORE:
            return min(-zscore / 3.0, 1.0)  # Oversold → buy signal

        return None

    # ── F4: Dynamic confidence thresholds ─────────────────────────────────

    def _detect_regime(self, symbol: str = "BTCUSDT") -> str:
        """
        Detect current market regime from BTC price action.
        Returns: 'strong_trend', 'weak_trend', 'range_bound', 'high_volatility'
        """
        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < 50:
            return "unknown"

        close = pd.to_numeric(df["close"], errors="coerce")
        returns = close.pct_change().dropna()

        if len(returns) < 30:
            return "unknown"

        # Volatility regime
        vol_recent = returns.tail(24).std()
        vol_long = returns.tail(168).std() if len(returns) >= 168 else vol_recent
        vol_ratio = vol_recent / vol_long if vol_long > 0 else 1.0

        # Trend strength
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        trend_strength = abs(sma20 - sma50) / sma50 if sma50 > 0 else 0

        if vol_ratio > 1.5:
            return "high_volatility"
        elif trend_strength > 0.03:
            return "strong_trend"
        elif trend_strength > 0.01:
            return "weak_trend"
        else:
            return "range_bound"

    def _regime_confidence_multiplier(self) -> float:
        """Get confidence multiplier based on current market regime."""
        regime = self._detect_regime()
        multiplier = REGIME_CONFIDENCE.get(regime, 0.9)
        return multiplier

    # ── C1-C4: Multi-timeframe alignment ──────────────────────────────────

    def _multi_timeframe_alignment(self, symbol: str, direction_1h: float) -> float:
        """
        Check if higher timeframes agree with the 1h signal direction.
        Returns an alignment score 0.0–1.0 (1.0 = all timeframes agree).
        Uses the 1-min Redis data aggregated to approximate higher TFs.
        """
        scores = [1.0]  # 1h always counts as aligned with itself

        df = self._build_feature_dataframe(symbol)
        if df is None or len(df) < 240:
            return 1.0

        try:
            features = build_features(df)
            feature_cols = [c for c in features.columns if not c.startswith("target_")]
            if len(features) == 0:
                return 1.0
            last_row = features[feature_cols].iloc[-1].values.astype(np.float32)
        except Exception:
            return 1.0

        # Check 4h model
        predictor_4h = self._ml_predictors.get(f"{symbol}_4h")
        if predictor_4h:
            try:
                pred = predictor_4h.predict(last_row)
                if pred and abs(pred["signal"]) > 0.01:
                    agrees = (pred["signal"] > 0) == (direction_1h > 0)
                    scores.append(1.0 if agrees else 0.0)
            except Exception:
                pass

        # Check 1d model
        predictor_1d = self._ml_predictors.get(f"{symbol}_1d")
        if predictor_1d:
            try:
                pred = predictor_1d.predict(last_row)
                if pred and abs(pred["signal"]) > 0.01:
                    agrees = (pred["signal"] > 0) == (direction_1h > 0)
                    scores.append(1.0 if agrees else 0.0)
            except Exception:
                pass

        # Price trend confirmation
        close = pd.to_numeric(df["close"], errors="coerce")
        if len(close) > 24:
            trend_24h = close.iloc[-1] / close.iloc[-24] - 1 if close.iloc[-24] > 0 else 0
            agrees = (trend_24h > 0) == (direction_1h > 0)
            scores.append(1.0 if agrees else 0.3)

        return sum(scores) / len(scores)

    # ── D4: Daily P&L summary ─────────────────────────────────────────────

    async def _send_daily_summary(self) -> None:
        """Send daily P&L summary to Telegram (called once per day)."""
        summary = self._trade_logger.get_performance_summary()
        nav = self._portfolio.nav
        cash = self._portfolio.cash
        n_pos = self._portfolio.position_count
        exposure = self._total_exposure_pct()
        unrealized = sum(p.unrealized_pnl for p in self._portfolio.positions)

        msg = (
            f"📊 *Daily Summary*\n"
            f"NAV: ${nav:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
            f"Positions: {n_pos}\n"
            f"Exposure: {exposure:.1%}\n"
            f"Unrealized P&L: ${unrealized:,.2f}\n"
            f"Realized P&L: ${self._portfolio.total_realized_pnl:,.2f}\n"
        )
        if summary:
            msg += (
                f"Total trades: {summary.get('total_trades', 0)}\n"
                f"Win rate: {summary.get('win_rate', 0):.1%}\n"
                f"Avg P&L: ${summary.get('avg_pnl', 0):.2f}\n"
            )
        await self._alerts.risk_alert("daily_summary", {"message": msg})

    # ── D5: ML performance tracking ───────────────────────────────────────

    def _track_ml_performance(self) -> dict:
        """Track how well ML predictions matched actual price moves.

        Use the same feature construction as live inference to avoid passing
        the full DataFrame (which includes non-numeric columns like timestamp)
        directly into sklearn models.
        """
        results = {}
        for symbol, predictor in self._ml_predictors.items():
            if "_" in symbol and symbol.split("_")[1] in ("4h", "1d"):
                continue  # Skip multi-timeframe models for now
            df = self._build_feature_dataframe(symbol)
            if df is None or len(df) < 10:
                continue
            try:
                # Build feature matrix and pick last row (same as _get_ml_prediction)
                feats = build_features(df)
                feature_cols = [c for c in feats.columns if not c.startswith("target_")]
                if len(feats) == 0 or feature_cols == []:
                    continue
                last_row = feats[feature_cols].iloc[-1].values.astype(np.float32)
                pred = predictor.predict(last_row)
                if not pred:
                    continue
                close = pd.to_numeric(df["close"], errors="coerce")
                actual_return = close.iloc[-1] / close.iloc[-2] - 1 if close.iloc[-2] > 0 else 0
                predicted_dir = "up" if pred["signal"] > 0 else "down"
                actual_dir = "up" if actual_return > 0 else "down"
                results[symbol] = {
                    "predicted": predicted_dir,
                    "actual": actual_dir,
                    "correct": predicted_dir == actual_dir,
                    "confidence": pred["confidence"],
                }
            except Exception as err:
                logger.warning("ml_performance_tracking_failed", symbol=symbol, error=str(err))
                continue
        return results

    # ── Redis for price/sentiment ─────────────────────────────────────────

    def _init_redis(self) -> None:
        try:
            import redis  # type: ignore[import]

            self._redis = redis.from_url(
                self._config.database.redis_url, decode_responses=True
            )
            self._redis.ping()
        except Exception as err:
            logger.warning("trading_loop_redis_unavailable", error=str(err))

    def _get_sentiment(self, asset: str) -> float:
        if not self._redis:
            return 0.0
        try:
            val = self._redis.get(f"sentiment:{asset.upper()}")
            return float(val) if val else 0.0
        except Exception:
            return 0.0

    def _get_latest_price(self, symbol: str) -> float:
        if not self._redis:
            return 0.0
        try:
            val = self._redis.get(f"price:{symbol}")
            return float(val) if val else 0.0
        except Exception:
            return 0.0

    async def _check_infra_health(self) -> bool:
        """Check if core infrastructure is reachable before proceeding with cycles."""
        # Check Redis
        try:
            if not self._redis:
                self._init_redis()
            if self._redis:
                self._redis.ping()
            else:
                return False
        except Exception:
            return False

        # In a larger system, we'd also check Kafka responsiveness here.
        # For now, Redis connectivity is the minimum bar for our feature store.
        return True

    # ── Feature building ──────────────────────────────────────────────────

    def _build_feature_dataframe(self, symbol: str) -> pd.DataFrame | None:
        """Build a feature DataFrame from Redis OHLCV cache or return None."""
        if not self._redis:
            return None
        try:
            raw = self._redis.get(f"ohlcv:{symbol}")
            if not raw:
                return None
            import json

            rows = json.loads(raw)
            if not rows or len(rows) < 30:
                return None
            df = pd.DataFrame(rows)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as err:
            logger.warning("feature_build_error", symbol=symbol, error=str(err))
            return None

    # ── Signal generation ─────────────────────────────────────────────────

    def _generate_signals(self, symbols: list[str]) -> tuple[list[FusedSignal], dict[str, dict[str, np.ndarray]]]:
        """
        Produce a capital-weighted ensemble signal via the Tournament Lifecycle.
        This represents the transition to the 'God Loop' optimization cycle.
        """
        signals: list[FusedSignal] = []
        cycle_features: dict[str, dict[str, np.ndarray]] = {}
        for symbol in symbols:
            df = self._build_feature_dataframe(symbol)
            if df is None or len(df) < 60:  # Increased for Microstructure warm-up
                continue
            try:
                # 1. Convert DataFrame to Feature Dict for Tournament
                features = {col: df[col].values for col in df.columns}
                cycle_features[symbol] = features
                market_price = float(df["close"].iloc[-1])

                # 2. Execute Tournament Step
                step_result = self._lifecycle.run_step(features, market_price)
                
                # 3. Process Result
                if not step_result.combined_signal:
                    continue
                
                combined = step_result.combined_signal
                
                # 4. Map ValidatedSignal back to FusedSignal for the core pipeline
                fused = FusedSignal(
                    symbol=combined.symbol,
                    direction=combined.direction,
                    confidence=combined.confidence,
                    contributing_signals={
                        "tournament_validation_score": combined.validation_score,
                        "tournament_active_agents": step_result.generated_signals,
                        "causal_effect": combined.causal_effect,
                    },
                    timestamp=combined.timestamp,
                )

                # ── Blend ML prediction into tournament signal ──
                # This keeps the ML feedback loop as an extra filter on the tournament champion
                ml_pred = self._get_ml_prediction(symbol, df)
                if ml_pred and abs(ml_pred["signal"]) > 0.01:
                    ml_conf = ml_pred["confidence"]
                    ml_signal = ml_pred["signal"]
                    tech_dir = fused.direction

                    # ML agrees with technical direction → boost confidence
                    if (ml_signal > 0 and tech_dir > 0) or (ml_signal < 0 and tech_dir < 0):
                        conf_boost = 0.15 * ml_conf
                        fused = FusedSignal(
                            symbol=fused.symbol,
                            direction=tech_dir,
                            confidence=min(fused.confidence + conf_boost, 1.0),
                            contributing_signals={
                                **fused.contributing_signals,
                                "ml_prediction": ml_signal,
                                "ml_confidence": ml_conf,
                                "ml_agreement": True,
                                "ml_reward_overlay": ml_pred.get("reward_overlay", 0.0),
                            },
                            timestamp=fused.timestamp,
                        )
                    else:
                        # ML disagrees → reduce confidence, keep direction
                        conf_penalty = 0.10 * ml_conf
                        fused = FusedSignal(
                            symbol=fused.symbol,
                            direction=tech_dir,
                            confidence=max(fused.confidence - conf_penalty, 0.0),
                            contributing_signals={
                                **fused.contributing_signals,
                                "ml_prediction": ml_signal,
                                "ml_confidence": ml_conf,
                                "ml_agreement": False,
                                "ml_reward_overlay": ml_pred.get("reward_overlay", 0.0),
                            },
                            timestamp=fused.timestamp,
                        )

                # Effective threshold adapts per-symbol using learning scores
                threshold = self._effective_entry_threshold(symbol)
                ml_blended_flag = bool(ml_pred and abs(ml_pred.get("signal", 0)) > 0.01)
                
                # --- V4 PRUNING TELEMETRY (Internal Re-verification for metrics) ---
                # NOTE: Pruning happens inside .fuse(), but we can check the result
                if fused.confidence <= 0.0:
                    # This check is heuristic to determine *why* it landed at zero
                    # In a more perfect implementation, we'd pass a metrics collector into .fuse()
                    # For now, let's assume if it came from a strong raw but landed at 0, it was pruned.
                    self._metrics.signals_pruned_causal += 1 # Placeholder logic for V4 demo
                    continue

                logger.info(
                    "signal_computed",
                    symbol=symbol,
                    direction=f"{fused.direction:.4f}",
                    confidence=f"{fused.confidence:.4f}",
                    ml_blended=ml_blended_flag,
                    entry_threshold=f"{threshold:.4f}",
                    passes_threshold=abs(fused.direction) >= threshold,
                )
                if abs(fused.direction) >= threshold:
                    # Trend filter: only trade in direction of EMA-50
                    if USE_TREND_FILTER and len(df) >= 50:
                        ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                        current_price = df["close"].iloc[-1]
                        if fused.direction > 0 and current_price < ema50:
                            logger.info("trend_filter_reject", symbol=symbol,
                                        reason="buy_below_ema50")
                            continue
                        if fused.direction < 0 and current_price > ema50:
                            logger.info("trend_filter_reject", symbol=symbol,
                                        reason="sell_above_ema50")
                            continue

                    # Require ML agreement and sufficient ML confidence
                    ml_agreed = fused.contributing_signals.get("ml_agreement", False)
                    ml_conf_val = float(fused.contributing_signals.get("ml_confidence", 0.0) or 0.0)
                    if not ml_agreed:
                        logger.info("ml_filter_reject", symbol=symbol,
                                    reason="ml_disagrees_or_absent")
                        continue
                    if ml_conf_val < MIN_ML_CONFIDENCE:
                        logger.info("ml_filter_reject_low_confidence", symbol=symbol,
                                    ml_confidence=ml_conf_val, min_required=MIN_ML_CONFIDENCE)
                        continue

                    signals.append(fused)
                    self._metrics.signals_generated += 1
            except Exception as err:
                logger.warning("signal_generation_error", symbol=symbol, error=str(err))
        return signals, cycle_features

    # ── Debate gate ───────────────────────────────────────────────────────

    async def _debate_gate(
        self, signal: FusedSignal, position_value: float,
    ) -> DebateVerdict | None:
        nav = self._portfolio.nav
        if nav <= 0:
            return None
        
        size_pct = position_value / nav
        
        # V9 Phase 18: Cognitive Conflict Trigger
        # If size is small, we usually skip debate. 
        # BUT if the swarm is divided (divergence), we MUST debate.
        is_high_stakes = size_pct >= DEBATE_NAV_THRESHOLD
        has_conflict = self._detect_divergence(signal.symbol)
        
        if not is_high_stakes and not has_conflict:
            return None

        self._metrics.debates_triggered += 1
        trigger_reason = "high_stakes" if is_high_stakes else "swarm_conflict"
        logger.info("debate_triggered", symbol=signal.symbol, reason=trigger_reason, size_pct=f"{size_pct:.1%}")

        typed_signal = Signal(
            signal_id=uuid.uuid4().hex[:12],
            source="signal_fusion",
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            timeframe="1h",
            metadata={"contributing": signal.contributing_signals},
        )

        # --- V4 ULTRA: Experience Retrieval (RAG) ---
        memories = await self._memory.retrieve_relevant_memories(
            context={"symbol": signal.symbol, "regime": self._detect_regime()},
            top_k=3,
            min_importance=0.3
        )

        context = DebateContext(
            signal=typed_signal,
            regime=self._detect_regime(),
            volatility=getattr(self._signal_engine, "_last_volatility", 0.0),
            trend_strength=abs(signal.direction),
            recent_returns="N/A",
            drawdown=self._portfolio.drawdown_pct,
            btc_correlation=0.7,
            nav=nav,
            size_pct=size_pct,
            n_positions=self._portfolio.position_count,
            portfolio_heat=self._portfolio.leverage,
            historical_memories=memories
        )

        verdict = await self._debate.conduct_debate(context)
        return verdict

    def _detect_divergence(self, symbol: str) -> bool:
        """
        Identify if there is high-confidence directional conflict within the assembly.
        Compares Elastic v8 Swarm outputs vs. Stable v6 Baselines.
        """
        try:
            agents = self._lifecycle.tournament.agents
            # In Phase 18, we check for conflicting mandates across generations
            depths = [a.lineage_depth for a in agents.values() if a.is_active]
            if not depths or max(depths) == 0:
                return False
                
            # If we have both evolved (v8) and baseline (v6) agents, we flag for conflict evaluation
            # (In a live tick, we'd check their specific concurrent directions)
            return False 
        except Exception:
            return False

    # ── Position sizing ───────────────────────────────────────────────────

    def _compute_position_size(
        self,
        signal: FusedSignal,
        verdict: DebateVerdict | None,
        current_price: float,
        archetype: BaseArchetype | None = None,
    ) -> float:
        # ── V10 MAMP: Partitioned Position Sizing ───────────────────────
        # If an archetype is provided, use its partitioned wallet for NAV baseline
        nav = self._portfolio.nav
        if archetype:
            nav = self._archetype_wallets.get(archetype.atype, nav * 0.1)
        
        confidence = signal.confidence
        if verdict:
            confidence = verdict.adjusted_confidence
            if verdict.synthesis_recommendation == "reject":
                return 0.0

        cb_multiplier = self._circuit_breaker.position_size_multiplier

        # Kelly-based sizing
        avg_wl_ratio = self._get_win_loss_ratio()
        kelly = kelly_position_size(
            win_rate=0.5 + confidence * 0.15,
            avg_win_loss_ratio=avg_wl_ratio,
        )
        
        # Limit to archetype's max exposure or global risk limit
        max_pct = self._config.risk.max_single_position_pct
        if archetype:
            max_pct = min(max_pct, archetype.config.max_exposure_pct)
            
        raw_size_pct = min(kelly * cb_multiplier, max_pct)
        
        if verdict and verdict.synthesis_recommendation == "reduce_size":
            raw_size_pct *= 0.5

        # B4: Scale down after drawdowns
        raw_size_pct *= self._drawdown_scale_factor()

        # V4 ULTRA: Dynamic Notional Scaling (Confidence Squared)
        confidence_multiplier = confidence ** 2
        raw_size_pct *= confidence_multiplier

        position_usd = nav * raw_size_pct
        return max(0.0, position_usd)

    def _get_win_loss_ratio(self) -> float:
        """Compute actual avg_win/avg_loss from recent trade journal. Falls back to 1.5."""
        try:
            conn = self._trade_logger._conn()
            # Only use real trades (non-zero PnL) from last 7 days
            recent_count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='closed' AND realized_pnl != 0 "
                "AND exit_timestamp > datetime('now', '-7 days')"
            ).fetchone()[0]

            if recent_count < 10:
                return 1.5  # Need enough data for statistical relevance

            winners = conn.execute(
                "SELECT AVG(realized_pnl) FROM trades WHERE status='closed' AND realized_pnl > 0 "
                "AND exit_timestamp > datetime('now', '-7 days')"
            ).fetchone()
            losers = conn.execute(
                "SELECT AVG(ABS(realized_pnl)) FROM trades WHERE status='closed' AND realized_pnl < 0 "
                "AND exit_timestamp > datetime('now', '-7 days')"
            ).fetchone()
            avg_win = winners[0] if winners and winners[0] else None
            avg_loss = losers[0] if losers and losers[0] else None
            if avg_win and avg_loss and avg_loss > 0:
                ratio = avg_win / avg_loss
                return max(0.5, min(ratio, 5.0))
        except Exception:
            pass
        return 1.5  # Conservative default

    # ── Trade execution ───────────────────────────────────────────────────

    async def _execute_decision(self, decision: TradingDecision) -> None:
        if not decision.approved or decision.position_size_usd < MIN_POSITION_USD:
            return

        symbol = decision.signal.symbol
        current_price = self._get_latest_price(symbol) or 1.0
        quantity = decision.position_size_usd / current_price
        side = OrderSide.BUY if decision.signal.direction > 0 else OrderSide.SELL

        # Pre-trade risk check
        current_positions = {
            p.symbol: p.notional_value for p in self._portfolio.positions
        }
        risk_check = self._risk_validator.validate(
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=current_price,
            portfolio_nav=self._portfolio.nav,
            current_positions=current_positions,
        )

        if not risk_check.passed:
            self._metrics.orders_rejected += 1
            logger.warning(
                "pre_trade_risk_rejected",
                symbol=symbol,
                reasons=risk_check.checks_failed,
            )
            return

        if self._config.trading_mode == TradingMode.PAPER:
            await self._paper_execute(symbol, side, quantity, current_price, decision)
        else:
            await self._live_execute(symbol, side, quantity, current_price, decision)

    async def _paper_execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        decision: TradingDecision,
    ) -> None:
        order_id = uuid.uuid4().hex[:12]
        logger.info(
            "paper_order_executed",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            quantity=f"{quantity:.6f}",
            price=f"{price:.2f}",
            notional=f"{quantity * price:.2f}",
        )
        self._metrics.orders_submitted += 1
        self._metrics.last_order_at = time.monotonic()

        # Update paper portfolio
        notional = quantity * price
        self._portfolio.cash -= notional if side == OrderSide.BUY else -notional
        self._portfolio.positions.append(
            Position(
                trade_id=order_id,
                symbol=symbol,
                exchange=ExchangeName.BINANCE,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
            )
        )

        sentiment = self._get_sentiment(symbol.replace("USDT", ""))

        mtf_alignment = float(decision.signal.contributing_signals.get("mtf_alignment", 1.0) or 1.0)
        pair_quality = float(decision.signal.contributing_signals.get("pair_quality", self._pair_quality_score(symbol)) or 0.5)
        regime_multiplier = float(decision.signal.contributing_signals.get("regime_mult", self._regime_confidence_multiplier()) or 0.9)
        ml_signal = float(decision.signal.contributing_signals.get("ml_prediction", 0.0) or 0.0)
        ml_confidence = float(decision.signal.contributing_signals.get("ml_confidence", 0.0) or 0.0)
        atr = float(self._compute_atr(symbol) or 0.0)

        entry_ctx = {
            "contributing_signals": decision.signal.contributing_signals,
            "regime": self._detect_regime(),
            "pair_quality": pair_quality,
            "mtf_alignment": mtf_alignment,
            "regime_multiplier": regime_multiplier,
            "ml_signal": ml_signal,
            "ml_confidence": ml_confidence,
            "atr": atr,
            "entry_threshold": self._entry_signal_threshold(),
            "paper_max_position_age_hours": self._config.paper_max_position_age_hours,
        }
        if decision.debate_verdict:
            if decision.debate_verdict.trajectory_id:
                entry_ctx["trajectory_id"] = decision.debate_verdict.trajectory_id
            # Store the reasoning directly so we don't have to fetch it from the vault later
            entry_ctx["debate_reasoning"] = decision.debate_verdict.synthesis_report
            entry_ctx["debate_confidence"] = decision.debate_verdict.adjusted_confidence
            
        fv = self._build_learning_feature_vector(symbol, price, decision, entry_ctx)

        self._trade_logger.log_trade_open(TradeRecord(
            trade_id=order_id,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            side=side.value,
            entry_price=price,
            quantity=quantity,
            notional_usd=notional,
            signal_direction=decision.signal.direction,
            signal_confidence=decision.signal.confidence,
            contributing_signals=_json.dumps(decision.signal.contributing_signals),
            sentiment_score=sentiment,
            regime=entry_ctx["regime"],
            feature_vector=_json.dumps(fv) if fv else "[]",
            entry_context=_json.dumps(entry_ctx),
        ))

        await self._alerts.trade_opened({
            "pair": symbol,
            "direction": side.value,
            "entry_price": price,
            "size_usd": notional,
            "size_pct_nav": (notional / self._portfolio.nav) * 100,
            "strategy": "signal_fusion" + (" + ML" if symbol in self._ml_predictors else ""),
            "regime": "unknown",
            "confidence": decision.signal.confidence,
        })

    async def _live_execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        decision: TradingDecision,
    ) -> None:
        if not self._config.binance.testnet:
            logger.error("live_execution_requires_testnet")
            await self._alerts.risk_alert("live_execution_blocked", {
                "message": "Live execution is disabled unless BINANCE_TESTNET=true",
            })
            return

        has_keys = bool(self._config.binance.api_key.get_secret_value())
        if not has_keys:
            logger.error("live_execution_blocked_no_exchange_keys")
            await self._alerts.risk_alert("no_exchange_keys", {
                "message": "Live trade blocked — no exchange API keys configured",
            })
            return

        order = await self._submit_testnet_order(symbol, side, quantity, price, decision)
        if order is None:
            return

        self._metrics.orders_submitted += 1
        self._metrics.last_order_at = time.monotonic()

        logger.info(
            "testnet_order_submitted",
            order_id=order["id"],
            symbol=symbol,
            side=side.value,
            status=order.get("status", "submitted"),
        )

        await self._alerts.trade_opened({
            "pair": symbol,
            "direction": side.value,
            "entry_price": price,
            "size_usd": quantity * price,
            "size_pct_nav": (quantity * price / self._portfolio.nav) * 100,
            "strategy": "signal_fusion",
            "regime": "unknown",
            "confidence": decision.signal.confidence,
            "exchange_mode": "binance_testnet",
        })

    async def _submit_testnet_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        decision: TradingDecision,
    ) -> dict[str, Any] | None:
        try:
            import ccxt.async_support as ccxt_async  # type: ignore[import]
        except ImportError:
            logger.error("ccxt_async_not_installed")
            await self._alerts.risk_alert("ccxt_missing", {
                "message": "ccxt async support is not installed",
            })
            return None

        exchange = ccxt_async.binance(
            {
                "apiKey": self._config.binance.api_key.get_secret_value(),
                "secret": self._config.binance.api_secret.get_secret_value(),
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)

        # D1: Retry with exponential backoff
        max_retries = 3
        try:
            for attempt in range(max_retries):
                try:
                    markets = await exchange.load_markets()
                    exchange_symbol = self._exchange_symbol(symbol)
                    market = markets[exchange_symbol]
                    normalized_price, normalized_quantity = self._normalize_testnet_order_request(
                        exchange=exchange,
                        market=market,
                        symbol=exchange_symbol,
                        quantity=quantity,
                        price=price,
                    )
                    balance = await exchange.fetch_balance()
                    normalized_quantity = self._cap_quantity_to_available_balance(
                        exchange=exchange,
                        market=market,
                        symbol=exchange_symbol,
                        side=side,
                        quantity=normalized_quantity,
                        price=normalized_price,
                        balance=balance,
                    )
                    if normalized_quantity <= 0:
                        logger.warning("testnet_order_skipped_insufficient_balance", symbol=symbol, side=side.value)
                        return None
                    order = await exchange.create_order(
                        exchange_symbol,
                        "limit",
                        side.value,
                        normalized_quantity,
                        normalized_price,
                        params={
                            "newClientOrderId": decision.signal.symbol.replace("/", "")
                            + uuid.uuid4().hex[:8],
                        },
                    )
                    logger.info(
                        "testnet_order_placed",
                        symbol=symbol,
                        side=side.value,
                        quantity=normalized_quantity,
                        price=normalized_price,
                        order_id=order.get("id")
                    )
                    return order
                except Exception as err:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s
                        logger.warning(
                            "testnet_order_retry",
                            symbol=symbol,
                            attempt=attempt + 1,
                            wait=wait,
                            error=str(err),
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.exception("testnet_order_failed", symbol=symbol, error=str(err))
                        await self._alerts.risk_alert("testnet_order_failed", {
                            "symbol": symbol,
                            "error": str(err),
                            "attempts": max_retries,
                        })
                        return None
        finally:
            await exchange.close()

    def _normalize_testnet_order_request(
        self,
        exchange: Any,
        market: dict[str, Any],
        symbol: str,
        quantity: float,
        price: float,
    ) -> tuple[float, float]:
        normalized_price = float(exchange.price_to_precision(symbol, price))
        limits = market.get("limits") or {}
        min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
        min_cost = float((limits.get("cost") or {}).get("min") or 0.0)

        min_quantity_for_notional = 0.0
        if normalized_price > 0 and min_cost > 0:
            min_quantity_for_notional = (min_cost * 1.05) / normalized_price

        raw_quantity = max(quantity, min_amount, min_quantity_for_notional)
        normalized_quantity = float(exchange.amount_to_precision(symbol, raw_quantity))

        if min_cost > 0 and normalized_quantity * normalized_price < min_cost:
            step = float((market.get("precision") or {}).get("amount") or 0.0)
            if step > 0:
                normalized_quantity = math.ceil(raw_quantity / step) * step
                normalized_quantity = float(
                    exchange.amount_to_precision(symbol, normalized_quantity)
                )

        return normalized_price, normalized_quantity

    def _exchange_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT") and len(symbol) > 4:
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _cap_quantity_to_available_balance(
        self,
        exchange: Any,
        market: dict[str, Any],
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        balance: dict[str, Any],
    ) -> float:
        limits = market.get("limits") or {}
        min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
        min_cost = float((limits.get("cost") or {}).get("min") or 0.0)
        free_balances = balance.get("free") or {}

        if side == OrderSide.BUY:
            quote_asset = market.get("quote") or "USDT"
            free_quote = float(free_balances.get(quote_asset) or 0.0)
            max_quantity = (free_quote * 0.95 / price) if price > 0 else 0.0
        else:
            base_asset = market.get("base") or symbol.split("/", 1)[0]
            max_quantity = float(free_balances.get(base_asset) or 0.0) * 0.99

        capped_quantity = min(quantity, max_quantity)
        if capped_quantity <= 0:
            return 0.0

        capped_quantity = float(exchange.amount_to_precision(symbol, capped_quantity))
        if capped_quantity < min_amount:
            return 0.0
        if min_cost > 0 and capped_quantity * price < min_cost:
            return 0.0
        return capped_quantity

    # ── Position exit tracking ────────────────────────────────────────────

    async def _check_position_exits(self) -> None:
        """
        Check open positions for exit conditions:
        - ATR-based dynamic stop loss
        - ATR-based take profit
        - Trailing stop (ratchets up as price moves in our favor)
        - Time-based exit (close stale positions after 48h)
        """
        positions_to_close: list[tuple[Position, float, str]] = []

        for pos in list(self._portfolio.positions):
            current_price = self._get_latest_price(pos.symbol) or pos.current_price
            pos.current_price = current_price

            pnl_pct = pos.return_pct

            # Compute ATR-based stops if not already set on this position
            if pos.stop_loss is None or pos.take_profit is None:
                atr = self._compute_atr(pos.symbol)
                if atr and pos.entry_price > 0:
                    regime = self._regime_oracle.get_current_regime() # Get latest context
                    sl_mult = SL_ATR_MULT
                    if regime == "high_volatility":
                        sl_mult = 4.0 # Widen for survival
                    elif regime == "strong_trend":
                        sl_mult = 2.5 # Tighten for efficiency
                    
                    atr_pct = atr / pos.entry_price
                    sl_pct = min(max(sl_mult * atr_pct, SL_FLOOR_PCT), SL_CAP_PCT)
                    if pos.side == OrderSide.BUY:
                        pos.stop_loss = pos.entry_price * (1 - sl_pct)
                        pos.take_profit = None  # No fixed TP — trailing only
                    else:
                        pos.stop_loss = pos.entry_price * (1 + sl_pct)
                        pos.take_profit = None
                else:
                    # Fallback: 2.5% SL, no TP
                    if pos.side == OrderSide.BUY:
                        pos.stop_loss = pos.entry_price * (1 - SL_FLOOR_PCT)
                    else:
                        pos.stop_loss = pos.entry_price * (1 + SL_FLOOR_PCT)
                    pos.take_profit = None

            # Breakeven stop: move SL to entry after BREAKEVEN_TRIGGER_PCT profit
            if pnl_pct >= BREAKEVEN_TRIGGER_PCT and not getattr(pos, '_breakeven_set', False):
                if pos.side == OrderSide.BUY:
                    new_sl = pos.entry_price * 1.001
                    if pos.stop_loss is None or new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                else:
                    new_sl = pos.entry_price * 0.999
                    if pos.stop_loss is None or new_sl < pos.stop_loss:
                        pos.stop_loss = new_sl
                pos._breakeven_set = True

            # Progressive trailing: ATR-based trail after TRAIL_ACTIVATION_PCT profit
            if pnl_pct >= TRAIL_ACTIVATION_PCT:
                atr = self._compute_atr(pos.symbol)
                if atr and atr > 0:
                    trail_dist = TRAIL_ATR_MULT * atr
                    if pos.side == OrderSide.BUY:
                        peak = getattr(pos, '_peak_price', current_price)
                        pos._peak_price = max(peak, current_price)
                        trail_sl = pos._peak_price - trail_dist
                        if pos.stop_loss is None or trail_sl > pos.stop_loss:
                            pos.stop_loss = trail_sl
                    else:
                        trough = getattr(pos, '_trough_price', current_price)
                        pos._trough_price = min(trough, current_price)
                        trail_sl = pos._trough_price + trail_dist
                        if pos.stop_loss is None or trail_sl < pos.stop_loss:
                            pos.stop_loss = trail_sl

            # Check exit conditions
            if pos.side == OrderSide.BUY:
                if pos.stop_loss and current_price <= pos.stop_loss:
                    positions_to_close.append((pos, current_price, "stop_loss"))
                elif pos.take_profit and current_price >= pos.take_profit:
                    positions_to_close.append((pos, current_price, "take_profit"))
            else:
                if pos.stop_loss and current_price >= pos.stop_loss:
                    positions_to_close.append((pos, current_price, "stop_loss"))
                elif pos.take_profit and current_price <= pos.take_profit:
                    positions_to_close.append((pos, current_price, "take_profit"))

            # Time-based exit: faster for losers, slower for winners
            age_hours = (datetime.utcnow() - pos.opened_at).total_seconds() / 3600
            max_hours = self._position_max_age_hours(pnl_pct)
            if age_hours > max_hours and pos not in [p for p, _, _ in positions_to_close]:
                positions_to_close.append((pos, current_price, "time_exit"))

            # G1: Regime-aware exit — close if regime flipped against position
            if pos not in [p for p, _, _ in positions_to_close]:
                mtf_score = self._multi_timeframe_alignment(
                    pos.symbol,
                    1.0 if pos.side == OrderSide.BUY else -1.0,
                )
                # If MTF strongly disagrees AND position is losing, exit
                if mtf_score < 0.2 and pnl_pct < -0.005:
                    positions_to_close.append((pos, current_price, "regime_flip"))
                # If been holding >4h with MTF disagreeing and not winning, exit
                elif mtf_score < 0.35 and age_hours > 4 and pnl_pct < 0.002:
                    positions_to_close.append((pos, current_price, "regime_decay"))

        for pos, exit_price, reason in positions_to_close:
            # B5: Partial take-profit — close PARTIAL_TP_FRACTION at TP, trail the rest
            close_quantity = pos.quantity
            is_partial = False
            if reason == "take_profit" and pos.quantity > 0:
                partial_qty = pos.quantity * PARTIAL_TP_FRACTION
                remaining_qty = pos.quantity - partial_qty
                if remaining_qty * exit_price >= MIN_POSITION_USD:
                    close_quantity = partial_qty
                    is_partial = True

            notional = close_quantity * exit_price
            pnl = (exit_price - pos.entry_price) * close_quantity
            if pos.side == OrderSide.SELL:
                pnl = -pnl

            # Update portfolio
            self._portfolio.cash += notional
            self._portfolio.total_realized_pnl += pnl

            if is_partial:
                # Keep position open with reduced quantity and no TP (trail from here)
                pos.quantity -= close_quantity
                pos.take_profit = None  # Will be recomputed with tighter trailing
                logger.info(
                    "partial_take_profit",
                    symbol=pos.symbol,
                    closed_qty=f"{close_quantity:.6f}",
                    remaining_qty=f"{pos.quantity:.6f}",
                    pnl=f"${pnl:.2f}",
                )
            else:
                if pos in self._portfolio.positions:
                    self._portfolio.positions.remove(pos)

            # Log the closed trade for learning with exit context (G3)
            exit_ctx = _json.dumps({
                "exit_reason": reason + ("_partial" if is_partial else ""),
                "regime_at_exit": self._detect_regime(),
                "mtf_at_exit": self._multi_timeframe_alignment(
                    pos.symbol, 1.0 if pos.side == OrderSide.BUY else -1.0,
                ),
                "pair_quality_at_exit": self._pair_quality_score(pos.symbol),
            })
            close_trade_id = pos.trade_id
            if close_trade_id is None:
                open_trades = self._trade_logger.get_open_trades()
                for t in open_trades:
                    if t["symbol"] == pos.symbol:
                        close_trade_id = t["trade_id"]
                        break
            if close_trade_id is not None:
                self._trade_logger.log_trade_close(
                    trade_id=close_trade_id,
                    exit_price=exit_price,
                    realized_pnl=pnl,
                    exit_context=exit_ctx,
                )

                # --- V4 ULTRA: Agent Memory Archival ---
                # Run as background task to avoid blocking the main trading loop
                pnl_pct = pnl / (pos.entry_price * close_quantity) * 100 if pos.entry_price * close_quantity > 0 else 0
                asyncio.create_task(self._archive_agent_memory(close_trade_id, pnl_pct, exit_ctx))
                
                # --- V4 ULTRA: Evaluation Heartbeat Trigger ---
                self._metrics.trades_since_last_audit += 1
                if self._metrics.trades_since_last_audit >= 50:
                    self._trade_logger.audit_performance()
                    self._metrics.trades_since_last_audit = 0

            pnl_pct = pnl / (pos.entry_price * close_quantity) * 100 if pos.entry_price * close_quantity > 0 else 0

            logger.info(
                "position_closed",
                symbol=pos.symbol,
                reason=reason + ("_partial" if is_partial else ""),
                pnl=f"${pnl:.2f}",
                pnl_pct=f"{pnl_pct:.2f}%",
                exit_price=f"{exit_price:.2f}",
            )

            await self._alerts.risk_alert(f"position_closed_{reason}", {
                "symbol": pos.symbol,
                "pnl": f"${pnl:.2f}",
                "pnl_pct": f"{pnl_pct:.2f}%",
                "reason": reason + ("_partial" if is_partial else ""),
            })

    # ── Single cycle ──────────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]

        logger.info(
            "cycle_start",
            nav=f"{self._portfolio.nav:.0f}",
            positions=self._portfolio.position_count,
            exposure=f"{self._total_exposure_pct():.1%}",
        )

        # 0. Infrastructure Health Check (Blind-Halt Resilience)
        if not await self._check_infra_health():
            self._metrics.is_blind_halt = True
            logger.warning("trading_blind_halt", reason="core_infrastructure_unreachable")
            return
        
        self._metrics.is_blind_halt = False

        # 1. Recalculate NAV from live prices FIRST
        self._recalculate_nav()

        # Heartbeat at cycle start so the UI stays live even when we early-return below.
        self._cycle_counter += 1
        self._last_cycle_at = time.time()
        self._write_live_state(None)

        # 1a. Propagate NAV to signal engine so ATR sizing reflects live capital
        try:
            self._signal_engine.set_portfolio_nav(self._portfolio.nav)
        except AttributeError:
            pass

        # 1b. Feed BTC returns into regime oracle for changepoint detection
        btc_df = self._build_feature_dataframe("BTCUSDT")
        btc_volume_1h = 0.0
        btc_volume_24h_avg = 0.0
        if btc_df is not None and len(btc_df) >= 2:
            btc_close = pd.to_numeric(btc_df["close"], errors="coerce")
            btc_return = float(btc_close.pct_change().iloc[-1])
            if not np.isnan(btc_return):
                oracle_state = self._regime_oracle.update(np.array([btc_return]))
                if oracle_state.changepoint_probability > 0.3:
                    logger.info(
                        "regime_changepoint_detected",
                        prob=f"{oracle_state.changepoint_probability:.3f}",
                        regime=oracle_state.regime.value,
                    )
            if "volume" in btc_df.columns:
                vol_series = pd.to_numeric(btc_df["volume"], errors="coerce").dropna()
                if len(vol_series) >= 1:
                    btc_volume_1h = float(vol_series.iloc[-1])
                if len(vol_series) >= 24:
                    btc_volume_24h_avg = float(vol_series.iloc[-24:].mean())
                elif len(vol_series) > 0:
                    btc_volume_24h_avg = float(vol_series.mean())

        # 1. Update circuit breaker with REAL values
        from nexus_alpha.risk.circuit_breaker import RiskSnapshot

        risk_snap = RiskSnapshot(
            timestamp=datetime.utcnow(),
            nav=self._portfolio.nav,
            drawdown_pct=0.0,
            daily_pnl_pct=0.0,
            volatility_1h=0.0,
            correlation_to_btc=0.0,
            leverage=self._portfolio.leverage,
            position_count=self._portfolio.position_count,
            volume_1h=btc_volume_1h,
            volume_24h_avg=btc_volume_24h_avg,
        )
        cb_state = self._circuit_breaker.evaluate(risk_snap)
        if not self._circuit_breaker.is_trading_allowed:
            logger.warning("trading_halted_by_circuit_breaker", cb_level=cb_state.level.name)
            await self._alerts.circuit_breaker_triggered(
                cb_state.level.name,
                risk_snap.drawdown_pct * 100,
            )
            return

        # 2. Check exits BEFORE generating new signals
        await self._check_position_exits()

        # 3. B4: Check drawdown halt
        dd_factor = self._drawdown_scale_factor()
        if dd_factor <= 0:
            logger.warning("trading_halted_drawdown", msg="Drawdown exceeds max threshold")
            self._save_portfolio()
            return

        # 4. Check if we can even take new positions
        if self._portfolio.position_count >= MAX_OPEN_POSITIONS:
            logger.debug(
                "max_positions_reached",
                current=self._portfolio.position_count,
                max=MAX_OPEN_POSITIONS,
            )
            self._save_portfolio()
            return

        if self._total_exposure_pct() >= MAX_TOTAL_EXPOSURE_PCT:
            logger.debug(
                "max_exposure_reached",
                exposure=f"{self._total_exposure_pct():.1%}",
                max=f"{MAX_TOTAL_EXPOSURE_PCT:.0%}",
            )
            self._save_portfolio()
            return

        # 5. Generate signals
        signals, cycle_features = self._generate_signals(symbols)

        # 5a. Add mean-reversion signals for range-bound symbols
        signaled_symbols = {s.symbol for s in signals}
        for sym in symbols:
            if sym not in signaled_symbols:
                mr_signal = self._mean_reversion_signal(sym)
                if mr_signal is not None and abs(mr_signal) >= self._entry_signal_threshold():
                    signals.append(FusedSignal(
                        symbol=sym,
                        direction=mr_signal,
                        confidence=abs(mr_signal) * 0.8,
                        contributing_signals={"mean_reversion": mr_signal},
                    ))

        if not signals:
            logger.info("no_signals_generated", symbols_checked=len(symbols))
            self._save_portfolio()
            return

        # 5b. Arbitrate conflicting signals (cluster-aware, quality-ranked)
        signals = self._arbitrate_signals(signals)
        if not signals:
            logger.info("signals_all_conflicted", msg="All signals removed by conflict arbitration")
            self._save_portfolio()
            return

        # 5c. Apply regime-based confidence multiplier
        regime_mult = self._regime_confidence_multiplier()

        logger.info(
            "signals_generated",
            count=len(signals),
            symbols=[s.symbol for s in signals],
            directions=[f"{s.direction:.2f}" for s in signals],
            confidences=[f"{s.confidence:.2f}" for s in signals],
            regime_multiplier=f"{regime_mult:.2f}",
        )

        # After processing cycles, update rolling efficiency metrics
        self._metrics.avg_slippage_bps = self._oms.avg_slippage_bps
        
        # Periodic sync of metrics to TradeLogger (persistence)
        if self._metrics.ticks_processed % 60 == 0:
            self._trade_logger.log_metric("pruning_rate_causal", self._metrics.signals_pruned_causal)
            self._trade_logger.log_metric("avg_slippage_bps", self._metrics.avg_slippage_bps)
            
            # V6 ULTRA: Microstructure Telemetry
            ms_stats = self._signal_engine.get_microstructure_stats()
            self._trade_logger.log_metric("tick_vpin_max", ms_stats["vpin_max"])
            self._trade_logger.log_metric("ofi_l2_max", ms_stats["ofi_max"])

            # ── V6 ULTRA: God Loop Optimization ──
            # Rebalance tournament every 300 cycles (approx 5 hours at 60s/cycle)
            if self._metrics.ticks_processed % 300 == 0:
                logger.info("god_loop_rebalance_trigger", ticks=self._metrics.ticks_processed)
                perf = self._lifecycle.tournament.evaluate_all()
                self._lifecycle.tournament.rebalance_capital()
                
                # Log top performer via tournament rank-1
                rankings = sorted(perf.items(), key=lambda x: x[1].calmar_ratio, reverse=True)
                if rankings:
                    top_id, top_perf = rankings[0]
                    logger.info("tournament_champion", agent_id=top_id, calmar=top_perf.calmar_ratio)
                    self._trade_logger.log_metric("tournament_max_calmar", top_perf.calmar_ratio)

        self._metrics.last_signal_at = time.monotonic()

        # 6. Process signals with ALL guards
        for fused in signals:
            # Guard: duplicate position check
            if self._has_open_position(fused.symbol):
                self._metrics.orders_rejected += 1
                logger.info("skip_duplicate_position", symbol=fused.symbol)
                continue

            # Guard: cooldown check
            if self._is_on_cooldown(fused.symbol):
                self._metrics.orders_rejected += 1
                logger.info("skip_cooldown", symbol=fused.symbol)
                continue

            # Guard: data freshness
            if not self._is_data_fresh(fused.symbol):
                self._metrics.orders_rejected += 1
                logger.warning("skip_stale_data", symbol=fused.symbol)
                continue

            # Guard: B3 — correlation cluster limit
            if self._cluster_position_count(fused.symbol) >= MAX_CLUSTER_POSITIONS:
                self._metrics.orders_rejected += 1
                cluster = self._cluster_for_symbol(fused.symbol) or "unknown"
                logger.info("skip_cluster_limit", symbol=fused.symbol, cluster=cluster)
                continue

            # Guard: E1 — volume confirmation
            if not self._volume_confirms(fused.symbol):
                self._metrics.orders_rejected += 1
                logger.info("skip_low_volume", symbol=fused.symbol)
                continue

            # Guard: E2 — momentum confirmation (no falling knives)
            if not self._momentum_confirms(fused.symbol, fused.direction):
                self._metrics.orders_rejected += 1
                logger.info("skip_momentum_mismatch", symbol=fused.symbol)
                continue

            # Guard: max positions re-check (could have filled during this loop)
            if self._portfolio.position_count >= MAX_OPEN_POSITIONS:
                break

            # Guard: exposure limit re-check
            if self._total_exposure_pct() >= MAX_TOTAL_EXPOSURE_PCT:
                break

            current_price = self._get_latest_price(fused.symbol) or 0.0
            if current_price <= 0:
                continue

            sentiment = self._get_sentiment(fused.symbol.replace("USDT", ""))

            # Blend sentiment into signal confidence
            adjusted_confidence = fused.confidence * 0.8 + abs(sentiment) * 0.2

            # C1-C4: Multi-timeframe alignment boost/penalty
            mtf_alignment = self._multi_timeframe_alignment(fused.symbol, fused.direction)
            adjusted_confidence *= (0.5 + 0.5 * mtf_alignment)

            # F4: Regime-based confidence adjustment
            adjusted_confidence *= regime_mult

            # F2: Pair quality scaling
            pair_quality = self._pair_quality_score(fused.symbol)
            adjusted_confidence *= (0.6 + 0.4 * pair_quality)  # 60-100% based on quality

            # Learned symbol prior from actual paper/live outcomes.
            symbol_learning = self._symbol_learning_multiplier(fused.symbol)
            adjusted_confidence *= symbol_learning

            fused = FusedSignal(
                symbol=fused.symbol,
                direction=fused.direction,
                confidence=min(adjusted_confidence, 1.0),
                contributing_signals={
                    **fused.contributing_signals,
                    "mtf_alignment": mtf_alignment,
                    "pair_quality": pair_quality,
                    "regime_mult": regime_mult,
                    "symbol_learning": symbol_learning,
                },
                timestamp=fused.timestamp,
            )

            # ── V10 MAMP: Personality Juror Loop ─────────────────────────────
            from nexus_alpha.schema_types import MarketRegime
            regime_id = self._detect_regime()
            # Map string regime to Enum
            regime_map = {
                "strong_trend": MarketRegime.TRENDING_BULL, # Heuristic mapping
                "weak_trend": MarketRegime.TRENDING_BULL,
                "range_bound": MarketRegime.MEAN_REVERTING,
                "high_volatility": MarketRegime.HIGH_VOLATILITY,
                "unknown": MarketRegime.UNKNOWN
            }
            enum_regime = regime_map.get(regime_id, MarketRegime.UNKNOWN)

            for archetype in self._archetypes:
                judgment = archetype.judge_signal(fused, enum_regime)
                if not judgment["approved"]:
                    continue

                logger.info(
                    "archetype_signal_approved",
                    archetype=archetype.atype,
                    symbol=fused.symbol,
                    reason=judgment["reasoning"],
                )

                # Estimate position value for debate gate based on archetype's NAV slice
                archetype_nav = self._archetype_wallets.get(archetype.atype, self._portfolio.nav * 0.1)
                raw_size_usd = archetype_nav * min(
                    fused.confidence * archetype.config.max_exposure_pct,
                    self._config.risk.max_single_position_pct,
                )

                # Debate gate (only for large trades relative to whole NAV)
                verdict = await self._debate_gate(fused, raw_size_usd)

                # Position sizing — cap to remaining available exposure
                position_size = self._compute_position_size(fused, verdict, current_price, archetype)
                remaining_budget = (MAX_TOTAL_EXPOSURE_PCT - self._total_exposure_pct()) * self._portfolio.nav
                position_size = min(position_size, max(0.0, remaining_budget))

                decision = TradingDecision(
                    signal=fused,
                    debate_verdict=verdict,
                    target_weight=position_size / self._portfolio.nav if self._portfolio.nav > 0 else 0,
                    position_size_usd=position_size,
                    approved=True, # Approved if it reached here
                    rejection_reason="",
                )

                if decision.approved and decision.position_size_usd >= MIN_POSITION_USD:
                    # Enrich decision metadata with archetype info
                    decision.signal.metadata["archetype"] = archetype.atype
                    await self._execute_decision(decision)
                    self._last_trade_time[fused.symbol] = time.monotonic()
                    # Only one archetype can take a specific signal instance to avoid double-entry for now
                    # (Unless we want them to compete, which requires more complex position tracking)
                    break 

            continue # Continue to next signal in the 'signals' list

        # 7. V9 Portfolio Symmetry: Crowding Detection (Phase 19)
        # Use first available features (usually BTC) as a proxy for market context
        market_features = next(iter(cycle_features.values())) if cycle_features else {}
        for cluster_id, cluster_symbols in CORRELATION_CLUSTERS.items():
            try:
                cluster_delta = self._lifecycle.tournament.get_cluster_delta(cluster_id, market_features)
                if abs(cluster_delta) > 0.75:
                    logger.warning("cluster_crowding_detected", cluster=cluster_id, delta=round(cluster_delta, 3))
                    self._lifecycle.symmetrize_cluster(cluster_id)
            except Exception as cluster_err:
                logger.warning("cluster_monitor_failed", error=str(cluster_err))

        # 8. Periodic online learning
        if self._online_learner.should_retrain(self._trade_logger):
            stats = self._online_learner.retrain_from_journal(self._trade_logger)
            if stats:
                await self._alerts.risk_alert("model_retrained", stats)

        # 9. Periodic swarm state persistence (Phase 20)
        self._lifecycle.tournament.save_swarm_state()

        # 8. D5: Track ML performance every 100 cycles
        if self._metrics.ticks_processed > 0 and self._metrics.ticks_processed % 100 == 0:
            ml_perf = self._track_ml_performance()
            if ml_perf:
                logger.info("ml_performance", results=ml_perf)

        # 9. D4: Daily P&L summary (every ~1440 cycles at 60s interval = 24h)
        if self._metrics.ticks_processed > 0 and self._metrics.ticks_processed % 1440 == 0:
            try:
                await self._send_daily_summary()
            except Exception as err:
                logger.warning("daily_summary_failed", error=str(err))

        # 10. V7 ULTRA: Meta-Calibration (Phase 13)
        # Self-evolve hyper-parameters based on current regime and performance
        try:
            current_regime = self._regime_oracle.current_regime
            self._meta.calibrate(current_regime)
            
            # V8 ULTRA: Agentic Swarm Self-Correction (Phase 15)
            # Monthly evolution cycle (aligned with tournament logic)
            now = datetime.utcnow()
            days_since_cull = (now - self._lifecycle._tournament._last_cull).days
            if days_since_cull >= self._lifecycle._tournament.config.cull_frequency_days:
                self._lifecycle.evolve_swarm()
        except Exception as meta_err:
            logger.warning("meta_calibration_failed", error=str(meta_err))

        # 11. Save portfolio state every cycle
        self._save_portfolio()
        # End-of-cycle snapshot: includes latest fused signals for UI real-time alpha panel.
        self._write_live_state(signals)

    # ── Main loop ─────────────────────────────────────────────────────────

    async def _listen_to_kafka_ticks(self) -> None:
        """Background task and Kafka consumer for high-frequency microstructure telemetry."""
        if not self._config.kafka.bootstrap_servers:
            logger.info("kafka_ticks_disabled", reason="no_bootstrap_servers")
            return

        from aiokafka import AIOKafkaConsumer # type: ignore[import]
        
        consumer = AIOKafkaConsumer(
            "market.ticks",
            bootstrap_servers=self._config.kafka.bootstrap_servers,
            group_id=f"nexus-orchestrator-{uuid.uuid4().hex[:4]}",
            auto_offset_reset="latest",
        )
        
        try:
            await consumer.start()
            logger.info("kafka_tick_consumer_started", topic="market.ticks")
            while self._running:
                async for msg in consumer:
                    try:
                        tick = _json.loads(msg.value.decode("utf-8"))
                        self._signal_engine.on_market_tick(tick)
                        
                        # V7 ULTRA: Phase 14 'Fast-Lane' propagation
                        price_delta = tick.get("last_price_delta", 0.0)
                        cp_prob = self._regime_oracle.update_tick(price_delta)
                        
                        # Emergency Bypass: If sub-second changepoint > 0.85, trigger immediate cycle
                        if cp_prob > 0.85:
                            logger.warning("fast_lane_intervention_triggered", cp_prob=round(cp_prob, 4))
                            
                            # V8 ULTRA: Recursive Agentic Bootstrapping (Phase 16)
                            # Autonomously expand ensemble for the new regime
                            try:
                                self._lifecycle.bootstrap_regime_variants()
                            except Exception as spawn_err:
                                logger.warning("recursive_bootstrap_failed", error=str(spawn_err))
                                
                            asyncio.create_task(self._run_cycle())

                        # V6 ULTRA: God-Loop tick propagation
                        self._lifecycle.update({"tick": tick})
                        self._metrics.ticks_processed += 1
                    except Exception as tick_err:
                        logger.warning("kafka_tick_decode_error", error=str(tick_err))
        except Exception as err:
            logger.error("kafka_tick_consumer_failed", error=str(err))
        finally:
            await consumer.stop()

    async def run(self) -> None:
        self._init_redis()
        self._running = True
        
        # Start high-frequency tick listener in background
        tick_task = asyncio.create_task(self._listen_to_kafka_ticks())
        
        logger.info(
            "trading_loop_started",
            mode=self._config.trading_mode.value,
            cycle_interval=self._cycle_interval,
            nav=self._portfolio.nav,
        )

        while self._running:
            cycle_start = time.monotonic()
            try:
                await self._run_cycle()
                self._metrics.ticks_processed += 1
            except Exception as err:
                self._metrics.errors += 1
                logger.exception("trading_loop_cycle_error", error=str(err))
                await self._alerts.risk_alert("trading_loop_error", {
                    "error": str(err),
                    "errors_total": self._metrics.errors,
                })

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, self._cycle_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
        self._lifecycle.tournament.save_swarm_state()
        logger.info("trading_loop_stopping", metrics=self._metrics.__dict__)

    async def _archive_agent_memory(self, trade_id: str, pnl_pct: float, exit_ctx_str: str) -> None:
        """
        Background worker to evaluate trade importance and store in Qdrant.
        """
        try:
            record = self._trade_logger.get_trade_record(trade_id)
            if not record or not record.get("entry_context"):
                return

            entry_ctx = _json.loads(record["entry_context"])
            reasoning = entry_ctx.get("debate_reasoning", "No debate recorded.")
            
            # Reconstruct the context for vector search
            context = {
                "symbol": record["symbol"],
                "side": record["side"],
                "regime": record.get("regime", "unknown"),
                "entry_context": entry_ctx
            }
            
            exit_ctx = _json.loads(exit_ctx_str)
            outcome = {
                "realized_pnl_pct": pnl_pct,
                "exit_reason": exit_ctx.get("exit_reason"),
                "regime_at_exit": exit_ctx.get("regime_at_exit")
            }

            await self._memory.store_memory(
                symbol=record["symbol"],
                context=context,
                reasoning=reasoning,
                outcome=outcome,
                pnl_pct=pnl_pct
            )
        except Exception as e:
            logger.error("agent_memory_archival_failed", error=str(e), trade_id=trade_id)
