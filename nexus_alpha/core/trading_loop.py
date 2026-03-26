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
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from nexus_alpha.agents.debate import AgentDebateProtocol, DebateContext
from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.config import NexusConfig, TradingMode
from nexus_alpha.execution.execution_engine import OrderManagementSystem
from nexus_alpha.logging import get_logger
from nexus_alpha.portfolio.optimizer import (
    HierarchicalRiskParityOptimizer,
    kelly_position_size,
)
from nexus_alpha.risk.circuit_breaker import (
    CircuitBreakerSystem,
    PreTradeRiskValidator,
)
from nexus_alpha.signals.signal_engine import FusedSignal, SignalFusionEngine
from nexus_alpha.types import (
    DebateVerdict,
    ExchangeName,
    MarketRegime,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
    Signal,
)

logger = get_logger(__name__)

DEBATE_NAV_THRESHOLD = 0.05  # 5% of NAV triggers debate
MIN_SIGNAL_CONFIDENCE = 0.15
MIN_POSITION_USD = 10.0


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
    ) -> None:
        self._config = config
        self._signal_engine = signal_engine
        self._circuit_breaker = circuit_breaker
        self._alerts = alerts
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

        # State
        self._running = False
        self._metrics = LoopMetrics()
        self._redis: Any = None

    @property
    def metrics(self) -> LoopMetrics:
        return self._metrics

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

    def _generate_signals(self, symbols: list[str]) -> list[FusedSignal]:
        signals: list[FusedSignal] = []
        for symbol in symbols:
            df = self._build_feature_dataframe(symbol)
            if df is None or len(df) < 30:
                continue
            try:
                fused = self._signal_engine.fuse(df, symbol)
                if abs(fused.direction) >= MIN_SIGNAL_CONFIDENCE:
                    signals.append(fused)
                    self._metrics.signals_generated += 1
            except Exception as err:
                logger.warning("signal_generation_error", symbol=symbol, error=str(err))
        return signals

    # ── Debate gate ───────────────────────────────────────────────────────

    async def _debate_gate(
        self, signal: FusedSignal, position_value: float,
    ) -> DebateVerdict | None:
        nav = self._portfolio.nav
        if nav <= 0:
            return None
        size_pct = position_value / nav
        if size_pct < DEBATE_NAV_THRESHOLD:
            return None

        self._metrics.debates_triggered += 1
        logger.info("debate_triggered", symbol=signal.symbol, size_pct=f"{size_pct:.1%}")

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

        context = DebateContext(
            signal=typed_signal,
            regime=MarketRegime.UNKNOWN.value,
            volatility=0.0,
            trend_strength=abs(signal.direction),
            recent_returns="N/A",
            drawdown=0.0,
            btc_correlation=0.7,
            nav=nav,
            size_pct=size_pct,
            n_positions=self._portfolio.position_count,
            portfolio_heat=self._portfolio.leverage,
        )

        verdict = await self._debate.conduct_debate(context)
        return verdict

    # ── Position sizing ───────────────────────────────────────────────────

    def _compute_position_size(
        self,
        signal: FusedSignal,
        verdict: DebateVerdict | None,
        current_price: float,
    ) -> float:
        nav = self._portfolio.nav
        confidence = signal.confidence
        if verdict:
            confidence = verdict.adjusted_confidence
            if verdict.synthesis_recommendation == "reject":
                return 0.0

        cb_multiplier = self._circuit_breaker.position_size_multiplier
        kelly = kelly_position_size(
            win_rate=0.5 + confidence * 0.15,
            avg_win_loss_ratio=0.02 / 0.015,  # ~1.33 win/loss ratio
        )
        raw_size_pct = min(kelly * cb_multiplier, self._config.risk.max_single_position_pct)
        if verdict and verdict.synthesis_recommendation == "reduce_size":
            raw_size_pct *= 0.5

        position_usd = nav * raw_size_pct
        return max(0.0, position_usd)

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
                symbol=symbol,
                exchange=ExchangeName.BINANCE,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
            )
        )

        await self._alerts.trade_opened({
            "pair": symbol,
            "direction": side.value,
            "entry_price": price,
            "size_usd": notional,
            "size_pct_nav": (notional / self._portfolio.nav) * 100,
            "strategy": "signal_fusion",
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
        has_keys = bool(self._config.binance.api_key.get_secret_value())
        if not has_keys:
            logger.error("live_execution_blocked_no_exchange_keys")
            await self._alerts.risk_alert("no_exchange_keys", {
                "message": "Live trade blocked — no exchange API keys configured",
            })
            return

        order = Order(
            order_id=uuid.uuid4().hex[:12],
            symbol=symbol,
            exchange=ExchangeName.BINANCE,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            metadata={
                "source": "trading_loop",
                "confidence": decision.signal.confidence,
                "debate": decision.debate_verdict is not None,
            },
        )
        routing = self._oms.submit_order(order)
        self._metrics.orders_submitted += 1
        self._metrics.last_order_at = time.monotonic()

        logger.info(
            "live_order_submitted",
            order_id=order.order_id,
            symbol=symbol,
            side=side.value,
            primary_exchange=routing.primary_exchange.value,
            est_cost_bps=f"{routing.estimated_cost_bps:.1f}",
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
        })

    # ── Single cycle ──────────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]

        # 1. Update circuit breaker
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
        )
        cb_state = self._circuit_breaker.evaluate(risk_snap)
        if not self._circuit_breaker.is_trading_allowed:
            logger.warning("trading_halted_by_circuit_breaker", cb_level=cb_state.level.name)
            await self._alerts.circuit_breaker_triggered(
                cb_state.level.name,
                risk_snap.drawdown_pct * 100,
            )
            return

        # 2. Generate signals
        signals = self._generate_signals(symbols)
        if not signals:
            return

        self._metrics.last_signal_at = time.monotonic()

        # 3. Process each signal through debate → sizing → risk → execution
        for fused in signals:
            current_price = self._get_latest_price(fused.symbol) or 1.0
            sentiment = self._get_sentiment(fused.symbol.replace("USDT", ""))

            # Blend sentiment into signal confidence
            adjusted_confidence = fused.confidence * 0.8 + abs(sentiment) * 0.2
            fused = FusedSignal(
                symbol=fused.symbol,
                direction=fused.direction,
                confidence=min(adjusted_confidence, 1.0),
                contributing_signals=fused.contributing_signals,
                timestamp=fused.timestamp,
            )

            # Estimate position value for debate gate
            raw_size_usd = self._portfolio.nav * min(
                fused.confidence * 0.1,
                self._config.risk.max_single_position_pct,
            )

            # Debate gate (only for large trades)
            verdict = await self._debate_gate(fused, raw_size_usd)

            # Position sizing
            position_size = self._compute_position_size(fused, verdict, current_price)

            approved = position_size >= MIN_POSITION_USD
            rejection_reason = ""
            if verdict and verdict.synthesis_recommendation == "reject":
                approved = False
                rejection_reason = f"debate_rejected: {verdict.reasoning}"

            decision = TradingDecision(
                signal=fused,
                debate_verdict=verdict,
                target_weight=position_size / self._portfolio.nav if self._portfolio.nav > 0 else 0,
                position_size_usd=position_size,
                approved=approved,
                rejection_reason=rejection_reason,
            )

            if decision.approved:
                await self._execute_decision(decision)
            else:
                self._metrics.orders_rejected += 1
                if rejection_reason:
                    logger.info("trade_rejected", symbol=fused.symbol, reason=rejection_reason)

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._init_redis()
        self._running = True
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
        logger.info("trading_loop_stopping", metrics=self._metrics.__dict__)
