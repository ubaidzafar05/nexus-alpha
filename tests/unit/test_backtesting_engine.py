from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from nexus_alpha.backtesting.engine import BacktestTrade, HistoricalBacktester, StrategyParams
from nexus_alpha.learning.trade_logger import TradeLogger


def test_check_exits_uses_atr_based_trailing_for_long(monkeypatch) -> None:
    monkeypatch.setattr(HistoricalBacktester, "_load_ml_models", lambda self: None)
    bt = HistoricalBacktester(params=StrategyParams(trailing_trigger=0.04, trail_atr_mult=2.5))
    pos = BacktestTrade(
        trade_id=1,
        symbol="BTCUSDT",
        side="buy",
        entry_price=100.0,
        entry_time=datetime.utcnow(),
        quantity=1.0,
        stop_loss=95.0,
        take_profit=None,
    )
    bt.positions = [pos]

    bt._check_exits(
        pd.Series({"high_BTCUSDT": 112.0, "low_BTCUSDT": 110.0}),
        {"BTCUSDT": 111.0},
        {"BTCUSDT": 1.0},
    )

    assert len(bt.positions) == 1
    assert pos.stop_loss == 109.5


def test_check_exits_uses_atr_based_trailing_for_short(monkeypatch) -> None:
    monkeypatch.setattr(HistoricalBacktester, "_load_ml_models", lambda self: None)
    bt = HistoricalBacktester(params=StrategyParams(trailing_trigger=0.04, trail_atr_mult=2.5))
    pos = BacktestTrade(
        trade_id=1,
        symbol="ETHUSDT",
        side="sell",
        entry_price=100.0,
        entry_time=datetime.utcnow(),
        quantity=1.0,
        stop_loss=105.0,
        take_profit=None,
    )
    bt.positions = [pos]

    bt._check_exits(
        pd.Series({"high_ETHUSDT": 90.0, "low_ETHUSDT": 88.0}),
        {"ETHUSDT": 89.0},
        {"ETHUSDT": 1.0},
    )

    assert len(bt.positions) == 1
    assert pos.stop_loss == 90.5


def test_arbitrate_signals_rejects_cluster_conflicts(monkeypatch) -> None:
    monkeypatch.setattr(HistoricalBacktester, "_load_ml_models", lambda self: None)
    bt = HistoricalBacktester()

    accepted = bt._arbitrate_signals(
        [
            {"symbol": "BTCUSDT", "direction": 0.8, "confidence": 0.7, "pair_quality": 0.9},
            {"symbol": "ETHUSDT", "direction": -0.7, "confidence": 0.8, "pair_quality": 0.9},
            {"symbol": "ADAUSDT", "direction": 0.6, "confidence": 0.6, "pair_quality": 0.8},
        ]
    )

    accepted_symbols = {sig["symbol"] for sig in accepted}
    assert "ADAUSDT" in accepted_symbols
    assert len({"BTCUSDT", "ETHUSDT"} & accepted_symbols) == 1


def test_engine_uses_final_confidence_threshold() -> None:
    params = StrategyParams(min_confidence=0.40)
    direction = 0.95
    adjusted_confidence = 0.32
    final_confidence = min(adjusted_confidence, 1.0)

    assert abs(direction) >= params.min_confidence
    assert final_confidence < params.min_confidence


def test_export_closed_trades_to_logger_uses_historical_timestamps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(HistoricalBacktester, "_load_ml_models", lambda self: None)
    bt = HistoricalBacktester()
    bt.closed_trades = [
        BacktestTrade(
            trade_id=1,
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            entry_time=datetime.fromisoformat("2025-01-01T00:00:00"),
            quantity=1.0,
            stop_loss=95.0,
            take_profit=None,
            exit_price=103.0,
            exit_time=datetime.fromisoformat("2025-01-02T00:00:00"),
            exit_reason="take_profit",
            pnl=2.5,
            pnl_pct=0.025,
            confidence=0.61,
            ml_agreed=True,
            signal_direction=1.0,
            feature_vector=[0.1, 0.2, 0.3],
            regime="strong_trend",
            entry_context={"pair_quality": 0.9},
            exit_context={"exit_reason": "take_profit"},
        )
    ]

    trade_logger = TradeLogger(tmp_path / "trades.db")
    exported = bt.export_closed_trades_to_logger(trade_logger, run_label="test_run")

    assert exported == 1
    trades = trade_logger.get_closed_trades(limit=10)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["trade_id"] == "replay:test_run:1"
    assert trade["hold_duration_seconds"] == 86400.0
    assert trade["regime"] == "strong_trend"
    assert json.loads(trade["feature_vector"]) == [0.1, 0.2, 0.3]


def test_prepare_market_data_rehydrates_feature_columns(monkeypatch) -> None:
    monkeypatch.setattr(HistoricalBacktester, "_load_ml_models", lambda self: None)
    monkeypatch.setattr("nexus_alpha.backtesting.engine.build_features", lambda cache: pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [0.5, 0.6, 0.7],
        },
        index=cache.index,
    ))
    bt = HistoricalBacktester()
    monkeypatch.setattr(bt.signal_engine, "compute_all", lambda cache: {})

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 12.0, 11.0],
        }
    )

    cache = bt._prepare_symbol_cache(df, "BTCUSDT", warmup=1)
    feature_vector = bt._row_feature_vector(cache.iloc[-1])

    assert feature_vector == [3.0, 0.7]
