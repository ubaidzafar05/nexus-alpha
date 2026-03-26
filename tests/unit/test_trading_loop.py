"""
Unit tests for the Trading Loop Orchestrator.

Covers:
- Signal generation and fusion
- Debate gate triggering
- Position sizing via Kelly criterion
- Circuit breaker integration
- Paper trade execution
- Heartbeat coroutine
- End-to-end cycle
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus_alpha.config import NexusConfig, TradingMode
from nexus_alpha.risk.circuit_breaker import (
    BreakerState,
    CircuitBreakerSystem,
)
from nexus_alpha.signals.signal_engine import FusedSignal, SignalFusionEngine
from nexus_alpha.types import CircuitBreakerLevel, Portfolio


def _make_config(**overrides) -> NexusConfig:
    defaults = {"trading_mode": TradingMode.PAPER}
    defaults.update(overrides)
    return NexusConfig(**defaults)


def _make_fused_signal(
    symbol: str = "BTCUSDT",
    direction: float = 0.6,
    confidence: float = 0.7,
) -> FusedSignal:
    return FusedSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        contributing_signals={"rsi_14": 0.3, "macd": 0.5},
        timestamp=datetime.utcnow(),
    )


class TestTradingLoopInit:
    def test_creates_with_defaults(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        signal_engine = SignalFusionEngine()
        cb = CircuitBreakerSystem()
        alerts = MagicMock()

        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=signal_engine,
            circuit_breaker=cb,
            alerts=alerts,
        )
        assert loop._portfolio.nav == 100_000.0
        assert loop._cycle_interval == 60.0
        assert loop.metrics.ticks_processed == 0

    def test_custom_portfolio_and_interval(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        portfolio = Portfolio(nav=50_000.0, cash=50_000.0, positions=[])
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
            portfolio=portfolio,
            cycle_interval_s=30.0,
        )
        assert loop._portfolio.nav == 50_000.0
        assert loop._cycle_interval == 30.0


class TestPositionSizing:
    def test_kelly_sizing_returns_positive(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
        )
        signal = _make_fused_signal(confidence=0.7)
        size = loop._compute_position_size(signal, None, 65000.0)
        assert size > 0
        assert size <= loop._portfolio.nav

    def test_sizing_zero_on_debate_reject(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
        from nexus_alpha.types import DebateVerdict, Signal

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
        )
        signal = _make_fused_signal()
        verdict = DebateVerdict(
            proposed_trade=Signal(
                signal_id="test",
                source="test",
                symbol="BTCUSDT",
                direction=0.5,
                confidence=0.5,
                timestamp=datetime.utcnow(),
                timeframe="1h",
            ),
            proposal_strength=0.5,
            challenge_strength=0.8,
            synthesis_recommendation="reject",
            adjusted_confidence=0.1,
            reasoning="Too risky",
        )
        size = loop._compute_position_size(signal, verdict, 65000.0)
        assert size == 0.0

    def test_sizing_reduced_on_debate_reduce_size(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
        from nexus_alpha.types import DebateVerdict, Signal

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
        )
        signal = _make_fused_signal(confidence=0.7)
        base_size = loop._compute_position_size(signal, None, 65000.0)

        verdict = DebateVerdict(
            proposed_trade=Signal(
                signal_id="test",
                source="test",
                symbol="BTCUSDT",
                direction=0.5,
                confidence=0.7,
                timestamp=datetime.utcnow(),
                timeframe="1h",
            ),
            proposal_strength=0.6,
            challenge_strength=0.5,
            synthesis_recommendation="reduce_size",
            adjusted_confidence=0.5,
            reasoning="Cut in half",
        )
        reduced = loop._compute_position_size(signal, verdict, 65000.0)
        assert reduced < base_size


class TestDebateGate:
    @pytest.mark.asyncio
    async def test_no_debate_for_small_position(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
        )
        signal = _make_fused_signal()
        # 1% of NAV = well below 5% threshold
        verdict = await loop._debate_gate(signal, 1000.0)
        assert verdict is None

    @pytest.mark.asyncio
    async def test_debate_triggered_for_large_position(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
        )
        signal = _make_fused_signal()
        # Mock the debate to avoid LLM calls
        mock_verdict = MagicMock()
        mock_verdict.synthesis_recommendation = "proceed"
        mock_verdict.adjusted_confidence = 0.6
        loop._debate = AsyncMock()
        loop._debate.conduct_debate = AsyncMock(return_value=mock_verdict)

        # 10% of NAV = triggers debate
        verdict = await loop._debate_gate(signal, 10_000.0)
        assert verdict is not None
        assert loop.metrics.debates_triggered == 1


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_cycle_halted_when_circuit_breaker_emergency(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        cb = CircuitBreakerSystem()
        cb.force_level(CircuitBreakerLevel.EMERGENCY, "test")
        alerts = MagicMock()
        alerts.circuit_breaker_triggered = AsyncMock()

        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=cb,
            alerts=alerts,
        )
        # Mock redis to avoid connection
        loop._redis = None

        await loop._run_cycle()
        # No signals should be generated when CB blocks
        assert loop.metrics.signals_generated == 0
        alerts.circuit_breaker_triggered.assert_called_once()


class TestPaperExecution:
    @pytest.mark.asyncio
    async def test_paper_execute_updates_portfolio(self):
        from nexus_alpha.core.trading_loop import (
            TradingDecision,
            TradingLoopOrchestrator,
        )

        config = _make_config()
        alerts = MagicMock()
        alerts.trade_opened = AsyncMock()

        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=alerts,
        )

        signal = _make_fused_signal(direction=0.6)
        decision = TradingDecision(
            signal=signal,
            debate_verdict=None,
            target_weight=0.05,
            position_size_usd=5000.0,
            approved=True,
        )

        # Simulate a price available
        loop._get_latest_price = MagicMock(return_value=65000.0)

        # Mock the risk validator to pass
        loop._risk_validator = MagicMock()
        loop._risk_validator.validate.return_value = MagicMock(
            passed=True, checks_failed=[]
        )

        await loop._execute_decision(decision)
        assert loop.metrics.orders_submitted == 1
        assert len(loop._portfolio.positions) == 1
        alerts.trade_opened.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_execute_requires_testnet(self):
        from nexus_alpha.core.trading_loop import (
            TradingDecision,
            TradingLoopOrchestrator,
        )

        config = _make_config(trading_mode=TradingMode.SMALL_LIVE)
        alerts = MagicMock()
        alerts.trade_opened = AsyncMock()
        alerts.risk_alert = AsyncMock()

        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=alerts,
        )

        loop._get_latest_price = MagicMock(return_value=65000.0)
        loop._risk_validator = MagicMock()
        loop._risk_validator.validate.return_value = MagicMock(
            passed=True, checks_failed=[]
        )

        decision = TradingDecision(
            signal=_make_fused_signal(direction=0.6),
            debate_verdict=None,
            target_weight=0.05,
            position_size_usd=5000.0,
            approved=True,
        )

        await loop._execute_decision(decision)

        alerts.risk_alert.assert_called_once()
        alerts.trade_opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_execute_uses_binance_testnet(self):
        from nexus_alpha.core.trading_loop import (
            TradingDecision,
            TradingLoopOrchestrator,
        )

        config = _make_config(
            trading_mode=TradingMode.SMALL_LIVE,
            binance={
                "testnet": True,
                "api_key": "test-key",
                "api_secret": "test-secret",
            },
        )
        alerts = MagicMock()
        alerts.trade_opened = AsyncMock()
        alerts.risk_alert = AsyncMock()

        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=alerts,
        )

        loop._get_latest_price = MagicMock(return_value=65000.0)
        loop._risk_validator = MagicMock()
        loop._risk_validator.validate.return_value = MagicMock(
            passed=True, checks_failed=[]
        )
        loop._submit_testnet_order = AsyncMock(
            return_value={"id": "test-123", "status": "open"}
        )

        decision = TradingDecision(
            signal=_make_fused_signal(direction=0.6),
            debate_verdict=None,
            target_weight=0.05,
            position_size_usd=5000.0,
            approved=True,
        )

        await loop._execute_decision(decision)

        loop._submit_testnet_order.assert_awaited_once()
        alerts.trade_opened.assert_called_once()
        alerts.risk_alert.assert_not_called()


class TestRunAndStop:
    @pytest.mark.asyncio
    async def test_stop_terminates_loop(self):
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

        config = _make_config()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(),
            alerts=MagicMock(),
            cycle_interval_s=0.1,
        )

        async def stop_after_delay():
            await asyncio.sleep(0.3)
            loop.stop()

        asyncio.create_task(stop_after_delay())
        await loop.run()
        assert loop.metrics.ticks_processed >= 1


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_sends_metrics(self):
        from nexus_alpha.cli import _heartbeat
        from nexus_alpha.core.trading_loop import LoopMetrics

        alerts = AsyncMock()
        trading_loop = MagicMock()
        trading_loop.metrics = LoopMetrics(
            ticks_processed=10,
            signals_generated=3,
            orders_submitted=1,
        )

        cb = MagicMock()
        cb.state = BreakerState(level=CircuitBreakerLevel.NORMAL)

        config = _make_config()

        # Run heartbeat with tiny interval, cancel after first send
        task = asyncio.create_task(
            _heartbeat(alerts, trading_loop, cb, config, interval_s=0.1)
        )
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert alerts.send.call_count >= 1
        msg = alerts.send.call_args_list[0][0][0]
        assert "Heartbeat" in msg
        assert "Cycles: `10`" in msg
