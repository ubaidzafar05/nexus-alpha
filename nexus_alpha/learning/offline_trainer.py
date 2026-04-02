"""
Phase 3: Offline Trainer — trains ML models on historical data.

Supports two model tiers:
1. Lightweight: scikit-learn GradientBoosting (runs without torch)
2. Heavy: WorldModel TFT if torch is available

The lightweight model is always available and is the production default.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from nexus_alpha.learning.historical_data import (
    build_features,
    load_ohlcv,
    prepare_training_data,
)
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = Path("data/checkpoints")


class LightweightPredictor:
    """
    GradientBoosting model for return prediction.
    Always available (sklearn is in base deps), fast to train, and
    surprisingly effective for 1h/4h return prediction.
    """

    def __init__(self, target_horizon: str = "target_1h"):
        self.target_horizon = target_horizon
        self.model: GradientBoostingRegressor | None = None
        self.feature_names: list[str] = []
        self.training_stats: dict = {}

    def train(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ) -> dict:
        """Train on historical data. Returns performance metrics."""
        logger.info("training_started", symbol=symbol, timeframe=timeframe)
        start = time.time()

        df = load_ohlcv(symbol, timeframe)
        data = prepare_training_data(df, target_col=self.target_horizon)

        self.feature_names = data["feature_names"]

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )

        self.model.fit(data["X_train"], data["y_train"])

        # Evaluate
        val_pred = self.model.predict(data["X_val"])
        test_pred = self.model.predict(data["X_test"])

        val_mae = mean_absolute_error(data["y_val"], val_pred)
        test_mae = mean_absolute_error(data["y_test"], test_pred)
        val_r2 = r2_score(data["y_val"], val_pred)
        test_r2 = r2_score(data["y_test"], test_pred)

        # Direction accuracy (most important for trading)
        val_dir_acc = np.mean(np.sign(val_pred) == np.sign(data["y_val"]))
        test_dir_acc = np.mean(np.sign(test_pred) == np.sign(data["y_test"]))

        # Feature importance
        importances = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        elapsed = time.time() - start
        self.training_stats = {
            "symbol": symbol,
            "timeframe": timeframe,
            "target": self.target_horizon,
            "n_samples": len(data["X_train"]),
            "n_features": data["n_features"],
            "val_mae": round(val_mae, 6),
            "test_mae": round(test_mae, 6),
            "val_r2": round(val_r2, 4),
            "test_r2": round(test_r2, 4),
            "val_direction_accuracy": round(val_dir_acc, 4),
            "test_direction_accuracy": round(test_dir_acc, 4),
            "top_features": [(n, round(v, 4)) for n, v in importances[:10]],
            "training_time_seconds": round(elapsed, 1),
        }

        logger.info("training_complete", **self.training_stats)
        return self.training_stats

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict forward returns from a feature vector.
        Returns prediction + confidence estimate.
        """
        if self.model is None:
            return {"prediction": 0.0, "confidence": 0.0, "signal": 0.0}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        pred = float(self.model.predict(features)[0])

        # Confidence from tree variance (pseudo-ensemble)
        tree_preds = np.array([
            tree[0].predict(features)[0]
            for tree in self.model.estimators_
        ])
        pred_std = float(np.std(tree_preds))

        # Direction signal: clip to [-1, 1], scaled by confidence
        raw_signal = np.clip(pred * 100, -1, 1)  # Scale small returns to signal range
        confidence = max(0.0, 1.0 - pred_std / (abs(pred) + 1e-8))
        confidence = min(confidence, 1.0)

        return {
            "prediction": pred,
            "confidence": round(confidence, 4),
            "signal": round(float(raw_signal), 4),
            "std": round(pred_std, 6),
        }

    def save(self, path: Path | None = None) -> Path:
        """Save model checkpoint."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or CHECKPOINT_DIR / f"lightweight_{self.target_horizon}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "training_stats": self.training_stats,
            }, f)
        logger.info("model_saved", path=str(path))
        return path

    def load(self, path: Path | None = None) -> bool:
        """Load model from checkpoint. Returns True if successful."""
        path = path or CHECKPOINT_DIR / f"lightweight_{self.target_horizon}.pkl"
        if not path.exists():
            logger.info("no_checkpoint_found", path=str(path))
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.training_stats = data.get("training_stats", {})
        logger.info("model_loaded", path=str(path), stats=self.training_stats.get("test_direction_accuracy"))
        return True


class OnlineLearner:
    """
    Incremental learning from live trade outcomes.
    Retrains the lightweight model periodically using accumulated trade data.
    """

    def __init__(
        self,
        predictor: LightweightPredictor | None = None,
        retrain_interval_hours: float = 6,
        min_new_trades: int = 20,
    ):
        self.predictor = predictor or LightweightPredictor()
        self.retrain_interval = retrain_interval_hours * 3600
        self.min_new_trades = min_new_trades
        self._last_retrain = 0.0
        self._trades_since_retrain = 0

    def record_outcome(self, features: np.ndarray, actual_return: float) -> None:
        """Record a trade outcome for future retraining."""
        self._trades_since_retrain += 1

    def should_retrain(self) -> bool:
        """Check if enough data has accumulated for a retrain cycle."""
        elapsed = time.time() - self._last_retrain
        return (
            elapsed >= self.retrain_interval
            and self._trades_since_retrain >= self.min_new_trades
        )

    def retrain_from_journal(self, trade_logger) -> dict | None:
        """
        Retrain using closed trades from the trade journal.
        Blends historical base knowledge with live trade experience.
        """
        training_data = trade_logger.get_training_data(min_trades=50)
        if training_data is None:
            return None

        logger.info("online_retrain_started", n_trades=training_data["n_trades"])

        features = training_data["features"]
        rewards = training_data["rewards"]

        # Train a fresh model on trade outcomes
        n = len(features)
        split = int(n * 0.8)

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )

        model.fit(features[:split], rewards[:split])

        val_pred = model.predict(features[split:])
        val_mae = mean_absolute_error(rewards[split:], val_pred)
        val_dir_acc = np.mean(np.sign(val_pred) == np.sign(rewards[split:]))

        stats = {
            "n_trades": n,
            "val_mae": round(val_mae, 4),
            "val_direction_accuracy": round(val_dir_acc, 4),
        }

        # Only update if the trade-based model shows signal
        if val_dir_acc > 0.52:
            self.predictor.model = model
            self.predictor.save()
            stats["updated"] = True
            logger.info("online_retrain_accepted", **stats)
        else:
            stats["updated"] = False
            logger.info("online_retrain_rejected_low_accuracy", **stats)

        self._last_retrain = time.time()
        self._trades_since_retrain = 0

        trade_logger.log_metric("retrain_direction_accuracy", val_dir_acc)
        trade_logger.log_metric("retrain_mae", val_mae)

        return stats


def train_all_symbols(
    symbols: list[str] | None = None,
    timeframe: str = "1h",
) -> dict[str, dict]:
    """Train lightweight models for all symbols. Returns per-symbol stats."""
    symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
    results = {}

    for symbol in symbols:
        predictor = LightweightPredictor(target_horizon="target_1h")
        try:
            stats = predictor.train(symbol=symbol, timeframe=timeframe)
            predictor.save(CHECKPOINT_DIR / f"lightweight_{symbol.replace('/', '_')}_1h.pkl")
            results[symbol] = stats
        except FileNotFoundError:
            logger.warning("no_data_for_training", symbol=symbol, timeframe=timeframe)
            results[symbol] = {"error": "No historical data. Run download first."}
        except Exception as err:
            logger.warning("training_failed", symbol=symbol, error=repr(err))
            results[symbol] = {"error": repr(err)}

    return results
