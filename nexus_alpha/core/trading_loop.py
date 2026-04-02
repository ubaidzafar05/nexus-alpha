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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import json as _json

import numpy as np
import pandas as pd

from nexus_alpha.agents.debate import AgentDebateProtocol, DebateContext
from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.config import NexusConfig, TradingMode
from nexus_alpha.execution.execution_engine import OrderManagementSystem
from nexus_alpha.learning.historical_data import build_features
from nexus_alpha.learning.offline_trainer import LightweightPredictor, OnlineLearner
from nexus_alpha.learning.trade_logger import TradeLogger, TradeRecord
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
    OrderSide,
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

        # ── Learning pipeline ─────────────────────────────────────────────
        self._trade_logger = TradeLogger()
        self._ml_predictors: dict[str, LightweightPredictor] = {}
        self._online_learner = OnlineLearner(retrain_interval_hours=6)
        self._init_ml_models()

        # State
        self._running = False
        self._metrics = LoopMetrics()
        self._redis: Any = None

    @property
    def metrics(self) -> LoopMetrics:
        return self._metrics

    # ── ML model loading ──────────────────────────────────────────────────

    def _init_ml_models(self) -> None:
        """Load trained ML models from checkpoints if available."""
        from pathlib import Path
        ckpt_dir = Path("data/checkpoints")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        for sym in symbols:
            ccxt_sym = sym.replace("USDT", "_USDT")
            predictor = LightweightPredictor(target_horizon="target_1h")
            ckpt_path = ckpt_dir / f"lightweight_{ccxt_sym}_1h.pkl"
            if predictor.load(ckpt_path):
                self._ml_predictors[sym] = predictor
                logger.info("ml_model_loaded", symbol=sym)

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
            return predictor.predict(last_row)
        except Exception as err:
            logger.warning("ml_prediction_error", symbol=symbol, error=repr(err))
            return None

    def _get_feature_vector(self, symbol: str, df: pd.DataFrame) -> list[float] | None:
        """Extract the current feature vector for trade logging."""
        try:
            features = build_features(df)
            feature_cols = [c for c in features.columns if not c.startswith("target_")]
            if len(features) == 0:
                return None
            return features[feature_cols].iloc[-1].tolist()
        except Exception:
            return None

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

                # ── Blend ML prediction into signal ──
                ml_pred = self._get_ml_prediction(symbol, df)
                if ml_pred and abs(ml_pred["signal"]) > 0.01:
                    ml_weight = 0.3 * ml_pred["confidence"]
                    rule_weight = 1.0 - ml_weight
                    blended_dir = (
                        fused.direction * rule_weight + ml_pred["signal"] * ml_weight
                    )
                    blended_conf = (
                        fused.confidence * rule_weight + ml_pred["confidence"] * ml_weight
                    )
                    fused = FusedSignal(
                        symbol=fused.symbol,
                        direction=float(np.clip(blended_dir, -1, 1)),
                        confidence=min(blended_conf, 1.0),
                        contributing_signals={
                            **fused.contributing_signals,
                            "ml_prediction": ml_pred["signal"],
                        },
                        timestamp=fused.timestamp,
                    )

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

        # ── Log trade for learning ──
        df = self._build_feature_dataframe(symbol)
        fv = self._get_feature_vector(symbol, df) if df is not None else None
        sentiment = self._get_sentiment(symbol.replace("USDT", ""))

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
            regime="unknown",
            feature_vector=_json.dumps(fv) if fv else "[]",
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
            return order
        except Exception as err:
            logger.exception("testnet_order_failed", symbol=symbol, error=str(err))
            await self._alerts.risk_alert("testnet_order_failed", {
                "symbol": symbol,
                "error": str(err),
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
        Check open positions for exit conditions (take-profit / stop-loss).
        When a position exits, log the outcome for learning.
        """
        positions_to_close: list[tuple[Position, float, str]] = []

        for pos in list(self._portfolio.positions):
            current_price = self._get_latest_price(pos.symbol) or pos.current_price
            pos.current_price = current_price

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            if pos.side == OrderSide.SELL:
                pnl_pct = -pnl_pct

            # Take profit at 3%, stop loss at -2%
            if pnl_pct >= 0.03:
                positions_to_close.append((pos, current_price, "take_profit"))
            elif pnl_pct <= -0.02:
                positions_to_close.append((pos, current_price, "stop_loss"))

        for pos, exit_price, reason in positions_to_close:
            notional = pos.quantity * exit_price
            pnl = (exit_price - pos.entry_price) * pos.quantity
            if pos.side == OrderSide.SELL:
                pnl = -pnl

            # Log the closed trade for learning
            open_trades = self._trade_logger.get_open_trades()
            for t in open_trades:
                if t["symbol"] == pos.symbol:
                    self._trade_logger.log_trade_close(
                        trade_id=t["trade_id"],
                        exit_price=exit_price,
                        realized_pnl=pnl,
                    )
                    break

            # Update portfolio
            self._portfolio.cash += notional
            if pos in self._portfolio.positions:
                self._portfolio.positions.remove(pos)

            logger.info(
                "position_closed",
                symbol=pos.symbol,
                reason=reason,
                pnl=f"{pnl:.2f}",
                exit_price=f"{exit_price:.2f}",
            )

            await self._alerts.risk_alert(f"position_closed_{reason}", {
                "symbol": pos.symbol,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (pos.entry_price * pos.quantity) * 100, 2),
                "reason": reason,
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

        # ── Check for position exits and log outcomes ──
        await self._check_position_exits()

        # ── Periodic online learning ──
        if self._online_learner.should_retrain():
            stats = self._online_learner.retrain_from_journal(self._trade_logger)
            if stats:
                await self._alerts.risk_alert("model_retrained", stats)

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
