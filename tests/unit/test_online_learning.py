from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nexus_alpha.config import NexusConfig, TradingMode
from nexus_alpha.learning.entry_features import CONTEXT_FEATURE_NAMES, build_augmented_feature_vector
from nexus_alpha.learning.offline_trainer import (
    benchmark_regime_slices,
    diagnose_learning_features,
    OnlineLearner,
    benchmark_learning_targets,
    benchmark_trade_bucket_models,
    benchmark_trade_outcome_models,
)
from nexus_alpha.learning.trade_logger import TradeLogger, TradeRecord
from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
from nexus_alpha.signals.signal_engine import SignalFusionEngine


def _make_config(**overrides) -> NexusConfig:
    defaults = {"trading_mode": TradingMode.PAPER}
    defaults.update(overrides)
    return NexusConfig(**defaults)


def _seed_closed_trades(trade_logger: TradeLogger, count: int = 60, start: int = 0) -> None:
    for i in range(start, start + count):
        record = TradeRecord(
            trade_id=f"t{i}",
            timestamp=f"2026-01-01T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy" if i % 2 == 0 else "sell",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0 if i % 2 == 0 else -1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([
                1.0 if i % 3 != 0 else -1.0,
                0.2 * (i % 3),
                0.3 * (i % 7),
            ]),
        )
        trade_logger.log_trade_open(record)
        realized_pnl = 2.0 if i % 3 != 0 else -1.0
        reward = 0.8 if realized_pnl > 0 else -0.6
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=102.0 if realized_pnl > 0 else 99.0,
            realized_pnl=realized_pnl,
            reward=reward,
        )


def test_online_learner_retrains_only_after_new_closed_trades(tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    model_path = tmp_path / "online.pkl"
    trade_logger = TradeLogger(db_path)
    _seed_closed_trades(trade_logger, count=60, start=0)

    learner = OnlineLearner(
        retrain_interval_hours=0,
        min_new_trades=10,
        min_total_trades=50,
        min_direction_accuracy=0.0,
        min_balanced_accuracy=0.0,
        model_path=model_path,
    )

    assert learner.should_retrain(trade_logger) is True
    stats = learner.retrain_from_journal(trade_logger)
    assert stats is not None
    assert model_path.exists()
    assert stats["target_type"] == "binary"
    assert learner.predictor.classifier is not None
    assert learner.should_retrain(trade_logger) is False

    _seed_closed_trades(trade_logger, count=12, start=60)
    assert learner.should_retrain(trade_logger) is True


def test_trade_logger_symbol_learning_scores(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=12, start=0)

    for i in range(12, 24):
        record = TradeRecord(
            trade_id=f"ada{i}",
            timestamp=f"2026-01-02T00:{i % 60:02d}:00",
            symbol="ADAUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(record.trade_id, exit_price=99.0, realized_pnl=-1.0, reward=-0.8)

    scores = trade_logger.get_symbol_learning_scores(min_trades=5)

    assert scores["BTCUSDT"] > 1.0
    assert scores["ADAUSDT"] < 1.0


def test_trade_logger_get_training_data_filters_inconsistent_feature_shapes(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=55, start=0)

    for i in range(55, 60):
        record = TradeRecord(
            trade_id=f"bad{i}",
            timestamp=f"2026-01-03T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector="[]",
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(record.trade_id, exit_price=101.0, realized_pnl=1.0, reward=0.4)

    training_data = trade_logger.get_training_data(min_trades=50)

    assert training_data is not None
    assert training_data["features"].shape == (55, 3)


def test_trade_logger_get_training_data_filters_all_zero_feature_vectors(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=55, start=0)

    for i in range(55, 60):
        record = TradeRecord(
            trade_id=f"zero{i}",
            timestamp=f"2026-01-03T01:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.0, 0.0, 0.0]),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(record.trade_id, exit_price=101.0, realized_pnl=1.0, reward=0.4)

    training_data = trade_logger.get_training_data(min_trades=50)

    assert training_data is not None
    assert training_data["features"].shape == (55, 3)


def test_trade_logger_prefers_richer_feature_schema_once_min_trades_is_met(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    for i in range(3):
        record = TradeRecord(
            trade_id=f"short{i}",
            timestamp=f"2026-01-03T02:{i:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.7,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([1.0, 0.5, 0.2]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 1.0}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(record.trade_id, exit_price=103.0, realized_pnl=3.0, reward=0.5)

    for i in range(3):
        record = TradeRecord(
            trade_id=f"long{i}",
            timestamp=f"2026-01-04T02:{i:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.7,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([1.0, 0.5, 0.2, 0.8, 0.4]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 1.0}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(record.trade_id, exit_price=103.0, realized_pnl=3.0, reward=0.5)

    dataset = trade_logger.build_learning_dataset(min_trades=3, target_mode="binary")

    assert dataset is not None
    assert dataset["features"].shape == (3, 5)


def test_trade_logger_get_training_data_filters_tiny_and_excluded_exits(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")

    for i in range(55):
        record = TradeRecord(
            trade_id=f"good{i}",
            timestamp=f"2026-01-04T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=101.0,
            realized_pnl=1.0,
            reward=0.4,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    for i in range(55, 60):
        record = TradeRecord(
            trade_id=f"tiny{i}",
            timestamp=f"2026-01-05T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=100.05,
            realized_pnl=0.05,
            reward=0.01,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    for i in range(60, 65):
        record = TradeRecord(
            trade_id=f"excluded{i}",
            timestamp=f"2026-01-06T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=101.0,
            realized_pnl=1.0,
            reward=0.4,
            exit_context=json.dumps({"exit_reason": "orphaned_on_restart"}),
        )

    training_data = trade_logger.get_training_data(min_trades=50)

    assert training_data is not None
    assert training_data["features"].shape == (55, 3)


def test_trade_logger_get_training_data_returns_chronological_order(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    for trade_id, ts, fv in [
        ("late", "2026-01-10T00:00:00", [9.0, 9.0, 9.0]),
        ("early", "2026-01-01T00:00:00", [1.0, 1.0, 1.0]),
    ]:
        record = TradeRecord(
            trade_id=trade_id,
            timestamp=ts,
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps(fv),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            trade_id,
            exit_price=103.0,
            realized_pnl=3.0,
            reward=0.5,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    training_data = trade_logger.get_training_data(min_trades=2, min_abs_pnl_pct=0.001)

    assert training_data is not None
    assert training_data["features"][0].tolist() == [1.0, 1.0, 1.0]
    assert training_data["features"][1].tolist() == [9.0, 9.0, 9.0]


def test_trade_logger_get_training_data_applies_replay_mtf_filter_only_to_replay(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")

    for i in range(30):
        record = TradeRecord(
            trade_id=f"replay_good{i}",
            timestamp=f"2026-01-07T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 1.0}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=101.0,
            realized_pnl=1.0,
            reward=0.4,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    for i in range(30, 60):
        record = TradeRecord(
            trade_id=f"replay_bad{i}",
            timestamp=f"2026-01-08T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 0.5}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=101.0,
            realized_pnl=1.0,
            reward=0.4,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    for i in range(60, 70):
        record = TradeRecord(
            trade_id=f"live{i}",
            timestamp=f"2026-01-09T00:{i % 60:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.6,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
            entry_context=json.dumps({"pair_quality": 0.8}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=101.0,
            realized_pnl=1.0,
            reward=0.4,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    training_data = trade_logger.get_training_data(min_trades=30)

    assert training_data is not None
    assert training_data["features"].shape == (40, 3)


def test_trade_logger_build_learning_dataset_uses_lower_min_move_for_live_samples(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")

    for i in range(20):
        record = TradeRecord(
            trade_id=f"replay{i}",
            timestamp=f"2026-01-10T00:{i:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.7,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 1.0}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=100.2,
            realized_pnl=0.2,
            reward=0.1,
            exit_context=json.dumps({"exit_reason": "time_exit"}),
        )

    for i, pnl_pct in enumerate([0.08, -0.17, 0.46]):
        record = TradeRecord(
            trade_id=f"live_small{i}",
            timestamp=f"2026-01-11T00:{i:02d}:00",
            symbol="ETHUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.7,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.4, 0.5, 0.6]),
            entry_context=json.dumps({"source": "live", "pair_quality": 0.8}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=100.0 * (1 + pnl_pct / 100.0),
            realized_pnl=pnl_pct,
            reward=pnl_pct / 100.0,
            exit_context=json.dumps({"exit_reason": "time_exit"}),
        )

    dataset = trade_logger.build_learning_dataset(min_trades=3, target_mode="binary")

    assert dataset is not None
    assert dataset["n_trades"] == 3
    assert dataset["sources"] == ["live", "live", "live"]


def test_trade_logger_build_learning_dataset_supports_quality_and_balancing(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    slices = ["bear", "recovery", "bull", "mixed"]
    for i in range(40):
        pnl = 3.0 if i % 2 == 0 else -3.0
        record = TradeRecord(
            trade_id=f"slice{i}",
            timestamp=f"2026-01-{(i % 28) + 1:02d}T00:00:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.85,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3]),
            entry_context=json.dumps(
                {
                    "source": "historical_replay",
                    "mtf_alignment": 1.0,
                    "pair_quality": 0.9,
                    "regime_multiplier": 1.1,
                    "regime_slice": slices[i % len(slices)],
                }
            ),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=103.0 if pnl > 0 else 97.0,
            realized_pnl=pnl,
            reward=0.6 if pnl > 0 else -0.6,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    dataset = trade_logger.build_learning_dataset(
        min_trades=16,
        target_mode="quaternary",
        strong_move_pct=0.02,
        min_quality_score=0.5,
        balanced=True,
    )

    assert dataset is not None
    assert dataset["n_trades"] == 40
    assert set(dataset["slice_counts"]) == {"bear", "recovery", "bull", "mixed"}
    assert float(dataset["quality_scores"].min()) >= 0.5


def test_trade_logger_build_learning_dataset_supports_risk_multiple_targets(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    for i, pnl_pct in enumerate([0.05, -0.03, 0.01, -0.01]):
        record = TradeRecord(
            trade_id=f"risk{i}",
            timestamp=f"2026-01-11T00:{i:02d}:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.8,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([0.1, 0.2, 0.3, 0.4]),
            entry_context=json.dumps({"source": "historical_replay", "mtf_alignment": 1.0, "ctx_stop_distance_pct": 0.02}),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=100.0 * (1 + pnl_pct),
            realized_pnl=100.0 * pnl_pct,
            reward=pnl_pct,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    dataset = trade_logger.build_learning_dataset(
        min_trades=4,
        target_mode="quaternary",
        target_metric="risk_multiple",
        target_threshold=1.0,
    )

    assert dataset is not None
    assert dataset["target_metric"] == "risk_multiple"
    assert dataset["targets"].tolist() == [3, 0, 2, 1]


def test_build_augmented_feature_vector_appends_context_features() -> None:
    augmented, context = build_augmented_feature_vector(
        [0.1, 0.2, 0.3],
        signal_confidence=0.72,
        pair_quality=0.81,
        mtf_alignment=1.0,
        regime_multiplier=1.1,
        ml_confidence=0.64,
        ml_signal=0.45,
        contributing_signals={"rsi_14": 0.4, "macd": 0.2, "bollinger_position": -0.1},
        directional_persistence_24=0.35,
        volatility_compression=0.22,
        entry_price=100.0,
        atr=1.4,
        sl_atr_mult=3.0,
        sl_floor_pct=0.025,
        sl_cap_pct=0.06,
        breakeven_trigger_pct=0.02,
        trailing_trigger_pct=0.04,
        trade_direction=1.0,
    )

    assert len(augmented) == 3 + len(CONTEXT_FEATURE_NAMES)
    assert context["ctx_signal_agreement_ratio"] > 0.0
    assert context["ctx_stop_distance_pct"] > 0.0


@pytest.mark.asyncio
async def test_trading_loop_ml_prediction_applies_online_reward_overlay(monkeypatch) -> None:
    from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

    loop = TradingLoopOrchestrator(
        config=_make_config(),
        signal_engine=SignalFusionEngine(),
        circuit_breaker=CircuitBreakerSystem(),
        alerts=MagicMock(),
    )

    class DummyPredictor:
        def predict(self, features):
            return {"prediction": 0.01, "confidence": 0.8, "signal": 0.8, "std": 0.01}

    loop._ml_predictors["BTCUSDT"] = DummyPredictor()
    loop._online_learner.predict_reward = lambda features, trade_direction: {
        "reward_prediction": -2.0,
        "confidence": 0.9,
        "std": 0.1,
    }

    monkeypatch.setattr(
        "nexus_alpha.core.trading_loop.build_features",
        lambda df: pd.DataFrame({"f1": [0.1], "f2": [0.2]}),
    )

    prediction = loop._get_ml_prediction("BTCUSDT", pd.DataFrame({"close": [1.0]}))

    assert prediction is not None
    assert prediction["reward_overlay"] == -2.0
    assert prediction["confidence"] < 0.8
    assert prediction["signal"] > 0


def test_trading_loop_symbol_learning_multiplier_uses_logger_scores(monkeypatch) -> None:
    from nexus_alpha.core.trading_loop import TradingLoopOrchestrator

    loop = TradingLoopOrchestrator(
        config=_make_config(),
        signal_engine=SignalFusionEngine(),
        circuit_breaker=CircuitBreakerSystem(),
        alerts=MagicMock(),
    )
    loop._trade_logger.get_symbol_learning_scores = lambda min_trades=5: {"BTCUSDT": 1.08}
    loop._symbol_learning_scores_expires_at = 0.0

    assert loop._symbol_learning_multiplier("BTCUSDT") == 1.08
    assert loop._symbol_learning_multiplier("ETHUSDT") == 1.0


def test_online_learner_predict_reward_uses_profit_classifier() -> None:
    learner = OnlineLearner(retrain_interval_hours=0, min_new_trades=1, min_total_trades=1)

    class DummyClassifier:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    learner.predictor.classifier = DummyClassifier()
    learner.predictor.model = None
    learner.predictor.training_stats = {"target_type": "profit_classification"}

    result = learner.predict_reward(np.array([0.1, 0.2, 0.3]), trade_direction=1.0)

    assert result is not None
    assert result["reward_prediction"] == 0.6
    assert result["confidence"] == 0.6
    assert result["win_probability"] == 0.8


def test_benchmark_trade_outcome_models_returns_candidates(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=60, start=0)

    results = benchmark_trade_outcome_models(trade_logger, min_trades=30)

    assert results is not None
    assert results["n_trades"] == 60
    assert "logistic_regression" in results["models"]
    assert "random_forest" in results["models"]
    assert "gradient_boosting" in results["models"]
    assert "baseline_balanced_accuracy" in results


def test_benchmark_trade_bucket_models_returns_candidates(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=60, start=0)

    results = benchmark_trade_bucket_models(trade_logger, min_trades=30, strong_move_pct=0.01)

    assert results is not None
    assert results["n_trades"] == 60
    assert "logistic_regression" in results["models"]
    assert "random_forest" in results["models"]
    assert "majority_class_balanced_accuracy" in results


def test_benchmark_learning_targets_returns_variants(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=60, start=0)

    results = benchmark_learning_targets(trade_logger, min_trades=30, strong_move_pct=0.01)

    assert results is not None
    assert "binary_chronological" in results["variants"]
    assert "quaternary_balanced" in results["variants"]


def test_benchmark_regime_slices_returns_results(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    slices = ["bear", "recovery", "bull", "mixed"]
    for i in range(48):
        slice_name = slices[i % len(slices)]
        cycle = i // len(slices)
        pnl = 3.0 if cycle % 2 == 0 else -2.0
        record = TradeRecord(
            trade_id=f"slicebench{i}",
            timestamp=f"2026-02-{(i % 28) + 1:02d}T00:00:00",
            symbol="BTCUSDT",
            side="buy",
            entry_price=100.0,
            quantity=1.0,
            notional_usd=100.0,
            signal_direction=1.0,
            signal_confidence=0.8,
            contributing_signals="{}",
            sentiment_score=0.0,
            regime="trend",
            feature_vector=json.dumps([1.0 if pnl > 0 else -1.0, 0.2, 0.3, 0.4]),
            entry_context=json.dumps(
                {
                    "source": "historical_replay",
                    "mtf_alignment": 1.0,
                    "regime_slice": slice_name,
                    "ctx_stop_distance_pct": 0.02,
                }
            ),
        )
        trade_logger.log_trade_open(record)
        trade_logger.log_trade_close(
            record.trade_id,
            exit_price=103.0 if pnl > 0 else 98.0,
            realized_pnl=pnl,
            reward=0.5 if pnl > 0 else -0.3,
            exit_context=json.dumps({"exit_reason": "stop_loss"}),
        )

    results = benchmark_regime_slices(trade_logger, min_trades=10, target_mode="binary")

    assert results is not None
    assert set(results["slices"]) == {"bear", "recovery", "bull", "mixed"}


def test_diagnose_learning_features_returns_ranked_features(tmp_path: Path) -> None:
    trade_logger = TradeLogger(tmp_path / "trades.db")
    _seed_closed_trades(trade_logger, count=60, start=0)

    results = diagnose_learning_features(trade_logger, min_trades=30, target_mode="binary", top_n=3)

    assert results is not None
    assert len(results["top_importance_features"]) == 3
    assert len(results["top_separation_features"]) == 3
