from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from nexus_alpha.cli import cli
from nexus_alpha.learning.walk_forward import (
    LearningPolicyRecommendation,
    WalkForwardSummary,
    WalkForwardWindowResult,
    _rolling_window_starts,
    recommend_learning_policy,
    run_walk_forward_df,
)


def _synthetic_ohlcv(rows: int = 520) -> pd.DataFrame:
    ts0 = datetime(2023, 1, 1)
    index = np.arange(rows, dtype=float)
    close = 100 + 0.04 * index + np.sin(index / 12.0) * 2.5
    open_ = close + np.sin(index / 7.0) * 0.2
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1000 + (np.cos(index / 9.0) + 1.5) * 200
    return pd.DataFrame(
        {
            "timestamp": [ts0 + timedelta(hours=int(i)) for i in index],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_rolling_window_starts_returns_expected_sequence() -> None:
    starts = _rolling_window_starts(total_samples=1000, train_bars=300, test_bars=100, step_bars=100)
    assert starts == [0, 100, 200, 300, 400, 500, 600]


def test_run_walk_forward_df_returns_summary() -> None:
    summary = run_walk_forward_df(
        _synthetic_ohlcv(),
        symbol="BTC/USDT",
        timeframe="1h",
        train_bars=180,
        test_bars=40,
        step_bars=40,
        min_confidence=0.05,
        n_estimators=60,
        max_depth=3,
        learning_rate=0.08,
    )

    assert summary.symbol == "BTC/USDT"
    assert summary.windows > 0
    assert summary.traded_samples > 0
    assert 0.0 <= summary.avg_direction_accuracy <= 1.0
    assert isinstance(summary.window_results[0].net_return_pct, float)


def test_walk_forward_cli_prints_summary(monkeypatch, tmp_path: Path) -> None:
    sample = WalkForwardSummary(
        symbol="BTC/USDT",
        timeframe="1h",
        target_col="target_1h",
        train_bars=1500,
        test_bars=250,
        step_bars=250,
        min_confidence=0.15,
        fee_pct=0.00075,
        total_samples=3000,
        windows=3,
        traded_samples=120,
        traded_coverage=0.16,
        avg_direction_accuracy=0.57,
        avg_traded_direction_accuracy=0.61,
        avg_confidence=0.23,
        total_gross_return_pct=4.2,
        total_net_return_pct=3.1,
        avg_window_net_return_pct=1.03,
        avg_mae=0.0123,
        window_results=[
            WalkForwardWindowResult(
                window_index=1,
                train_start="2024-01-01T00:00:00",
                train_end="2024-03-01T00:00:00",
                test_start="2024-03-02T00:00:00",
                test_end="2024-03-15T00:00:00",
                train_samples=1500,
                test_samples=250,
                coverage=0.2,
                traded_samples=50,
                direction_accuracy=0.56,
                traded_direction_accuracy=0.60,
                gross_return_pct=1.5,
                net_return_pct=1.2,
                avg_confidence=0.22,
                mae=0.01,
            ),
            WalkForwardWindowResult(
                window_index=2,
                train_start="2024-03-16T00:00:00",
                train_end="2024-05-15T00:00:00",
                test_start="2024-05-16T00:00:00",
                test_end="2024-05-30T00:00:00",
                train_samples=1500,
                test_samples=250,
                coverage=0.18,
                traded_samples=45,
                direction_accuracy=0.58,
                traded_direction_accuracy=0.62,
                gross_return_pct=2.1,
                net_return_pct=1.8,
                avg_confidence=0.24,
                mae=0.011,
            ),
        ],
    )

    monkeypatch.setattr(
        "nexus_alpha.learning.walk_forward.run_walk_forward",
        lambda **kwargs: sample,
    )

    out_path = tmp_path / "wf.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["walk-forward", "--symbol", "BTC/USDT", "--save-json", str(out_path)],
    )

    assert result.exit_code == 0
    assert "Walk-forward evaluation" in result.output
    assert "total net return: 3.10%" in result.output
    assert "Suggested learning policy" in result.output
    assert out_path.exists()


def test_recommend_learning_policy_raises_threshold_for_weak_results() -> None:
    summary = WalkForwardSummary(
        symbol="BTC/USDT",
        timeframe="1h",
        target_col="target_1h",
        train_bars=1000,
        test_bars=200,
        step_bars=200,
        min_confidence=0.15,
        fee_pct=0.00075,
        total_samples=2200,
        windows=4,
        traded_samples=20,
        traded_coverage=0.04,
        avg_direction_accuracy=0.51,
        avg_traded_direction_accuracy=0.52,
        avg_confidence=0.18,
        total_gross_return_pct=-1.0,
        total_net_return_pct=-2.5,
        avg_window_net_return_pct=-0.6,
        avg_mae=0.02,
        window_results=[],
    )

    policy = recommend_learning_policy(summary)

    assert isinstance(policy, LearningPolicyRecommendation)
    assert policy.min_confidence > 0.15
    assert policy.min_new_trades >= 30
    assert policy.retrain_interval_hours >= 12
