"""
Walk-forward evaluation for lightweight ML predictors.

This answers the practical question:
"If we keep feeding the bot more free market data and retrain periodically,
does predictive quality actually improve out-of-sample?"

The evaluator trains on a rolling historical window, predicts the next window,
and aggregates metrics across all windows. It is intentionally model-centric:
we validate forecast quality and signal edge first before relying on full
portfolio backtests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from nexus_alpha.learning.historical_data import build_features, load_ohlcv
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardWindowResult:
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    coverage: float
    traded_samples: int
    direction_accuracy: float
    traded_direction_accuracy: float
    gross_return_pct: float
    net_return_pct: float
    avg_confidence: float
    mae: float


@dataclass
class WalkForwardSummary:
    symbol: str
    timeframe: str
    target_col: str
    train_bars: int
    test_bars: int
    step_bars: int
    min_confidence: float
    fee_pct: float
    total_samples: int
    windows: int
    traded_samples: int
    traded_coverage: float
    avg_direction_accuracy: float
    avg_traded_direction_accuracy: float
    avg_confidence: float
    total_gross_return_pct: float
    total_net_return_pct: float
    avg_window_net_return_pct: float
    avg_mae: float
    window_results: list[WalkForwardWindowResult]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["window_results"] = [asdict(window) for window in self.window_results]
        return payload

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass
class LearningPolicyRecommendation:
    min_confidence: float
    min_new_trades: int
    retrain_interval_hours: int
    rationale: list[str]


def _build_dataset(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    features = build_features(df)
    feature_cols = [c for c in features.columns if not c.startswith("target_")]
    if target_col not in features.columns:
        raise ValueError(f"Unknown target column: {target_col}")
    X = features[feature_cols].values.astype(np.float32)
    y = features[target_col].values.astype(np.float32)
    timestamps = pd.to_datetime(df.loc[features.index, "timestamp"]).reset_index(drop=True)
    return X, y, feature_cols, timestamps


def _rolling_window_starts(
    total_samples: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[int]:
    if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
        raise ValueError("train_bars, test_bars, and step_bars must be positive")
    last_trainable_start = total_samples - train_bars - test_bars
    if last_trainable_start < 0:
        return []
    return list(range(0, last_trainable_start + 1, step_bars))


def run_walk_forward_df(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str = "1h",
    target_col: str = "target_1h",
    train_bars: int = 1500,
    test_bars: int = 250,
    step_bars: int = 250,
    min_confidence: float = 0.15,
    fee_pct: float = 0.00075,
    slippage_pct: float = 0.0002,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> WalkForwardSummary:
    """
    Run rolling walk-forward evaluation on a dataframe of OHLCV candles.

    Returns out-of-sample forecasting quality and simple trade-edge metrics:
    - direction accuracy on all future candles
    - direction accuracy only on sufficiently confident predictions
    - gross/net returns of a naive directional trade on confident candles
    """
    X, y, _feature_names, timestamps = _build_dataset(df, target_col=target_col)
    starts = _rolling_window_starts(len(X), train_bars, test_bars, step_bars)
    if not starts:
        raise ValueError(
            f"Not enough data for walk-forward: need at least {train_bars + test_bars} "
            f"samples after feature warmup, got {len(X)}"
        )

    window_results: list[WalkForwardWindowResult] = []

    for idx, start in enumerate(starts, start=1):
        train_end = start + train_bars
        test_end = train_end + test_bars
        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        y_train_dir = (y_train > 0).astype(int)
        y_test_dir = (y_test > 0).astype(int)

        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )
        regressor = GradientBoostingRegressor(
            n_estimators=max(50, n_estimators // 2),
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )

        classifier.fit(X_train, y_train_dir)
        regressor.fit(X_train, y_train)

        dir_proba = classifier.predict_proba(X_test)
        up_prob = dir_proba[:, 1] if dir_proba.shape[1] > 1 else np.full(len(X_test), 0.5)
        predicted_side = np.where(up_prob >= 0.5, 1.0, -1.0)
        predicted_dir = (predicted_side > 0).astype(int)
        confidence = np.abs(up_prob - 0.5) * 2.0
        traded_mask = confidence >= min_confidence

        gross_returns = predicted_side * y_test
        # Subtract both fees and estimated slippage per round-trip
        net_returns = gross_returns - (2.0 * (fee_pct + slippage_pct))
        traded_gross = gross_returns[traded_mask]
        traded_net = net_returns[traded_mask]
        traded_test_dir = y_test_dir[traded_mask]
        traded_predicted_dir = predicted_dir[traded_mask]

        direction_accuracy = float(accuracy_score(y_test_dir, predicted_dir))
        traded_direction_accuracy = (
            float(accuracy_score(traded_test_dir, traded_predicted_dir))
            if traded_mask.any()
            else 0.0
        )
        predictions = regressor.predict(X_test)
        mae = float(mean_absolute_error(y_test, predictions))
        coverage = float(traded_mask.mean())

        result = WalkForwardWindowResult(
            window_index=idx,
            train_start=timestamps.iloc[start].isoformat(),
            train_end=timestamps.iloc[train_end - 1].isoformat(),
            test_start=timestamps.iloc[train_end].isoformat(),
            test_end=timestamps.iloc[test_end - 1].isoformat(),
            train_samples=len(X_train),
            test_samples=len(X_test),
            coverage=round(coverage, 4),
            traded_samples=int(traded_mask.sum()),
            direction_accuracy=round(direction_accuracy, 4),
            traded_direction_accuracy=round(traded_direction_accuracy, 4),
            gross_return_pct=round(float(traded_gross.sum() * 100), 4),
            net_return_pct=round(float(traded_net.sum() * 100), 4),
            avg_confidence=round(float(confidence.mean()), 4),
            mae=round(mae, 6),
        )
        window_results.append(result)

    traded_samples = sum(window.traded_samples for window in window_results)
    total_test_samples = sum(window.test_samples for window in window_results)

    summary = WalkForwardSummary(
        symbol=symbol,
        timeframe=timeframe,
        target_col=target_col,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        min_confidence=min_confidence,
        fee_pct=fee_pct,
        total_samples=len(X),
        windows=len(window_results),
        traded_samples=traded_samples,
        traded_coverage=round(traded_samples / total_test_samples, 4) if total_test_samples else 0.0,
        avg_direction_accuracy=round(float(np.mean([w.direction_accuracy for w in window_results])), 4),
        avg_traded_direction_accuracy=round(
            float(np.mean([w.traded_direction_accuracy for w in window_results])),
            4,
        ),
        avg_confidence=round(float(np.mean([w.avg_confidence for w in window_results])), 4),
        total_gross_return_pct=round(float(sum(w.gross_return_pct for w in window_results)), 4),
        total_net_return_pct=round(float(sum(w.net_return_pct for w in window_results)), 4),
        avg_window_net_return_pct=round(float(np.mean([w.net_return_pct for w in window_results])), 4),
        avg_mae=round(float(np.mean([w.mae for w in window_results])), 6),
        window_results=window_results,
    )

    logger.info(
        "walk_forward_complete",
        symbol=symbol,
        timeframe=timeframe,
        windows=summary.windows,
        traded_samples=summary.traded_samples,
        avg_accuracy=summary.avg_direction_accuracy,
        net_return_pct=summary.total_net_return_pct,
    )
    return summary


def run_walk_forward(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    **kwargs,
) -> WalkForwardSummary:
    """Load historical data from disk and run walk-forward evaluation."""
    df = load_ohlcv(symbol=symbol, timeframe=timeframe)
    return run_walk_forward_df(df, symbol=symbol, timeframe=timeframe, **kwargs)


def recommend_learning_policy(summary: WalkForwardSummary) -> LearningPolicyRecommendation:
    """
    Derive conservative runtime learning knobs from out-of-sample performance.

    The intent is not to chase performance, but to avoid over-reacting:
    weaker walk-forward results => higher confidence threshold and slower retraining.
    """
    rationale: list[str] = []
    min_confidence = summary.min_confidence
    min_new_trades = 20
    retrain_interval_hours = 6

    if summary.total_net_return_pct <= 0:
        min_confidence = round(min(0.5, summary.min_confidence + 0.10), 2)
        min_new_trades = 35
        retrain_interval_hours = 12
        rationale.append("Out-of-sample net return is non-positive; raise selectivity and retrain slower.")
    elif summary.avg_traded_direction_accuracy < 0.56:
        min_confidence = round(min(0.5, summary.min_confidence + 0.05), 2)
        min_new_trades = 30
        retrain_interval_hours = 12
        rationale.append("Traded accuracy is modest; require more evidence before updating models.")
    else:
        rationale.append("Walk-forward edge is positive; current confidence threshold is acceptable.")

    if summary.traded_coverage < 0.08:
        min_new_trades = max(min_new_trades, 30)
        retrain_interval_hours = max(retrain_interval_hours, 12)
        rationale.append("Coverage is sparse; aggregate more closed trades before retraining.")
    elif summary.traded_coverage > 0.20 and summary.avg_traded_direction_accuracy >= 0.58:
        min_new_trades = max(20, min_new_trades - 5)
        rationale.append("Coverage is healthy and accuracy strong; moderate retrain cadence is reasonable.")

    return LearningPolicyRecommendation(
        min_confidence=min_confidence,
        min_new_trades=min_new_trades,
        retrain_interval_hours=retrain_interval_hours,
        rationale=rationale,
    )
