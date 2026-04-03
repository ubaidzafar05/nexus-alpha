"""
Phase 3: Offline Trainer — trains ML models on historical data.

Supports two model tiers:
1. Lightweight: scikit-learn GradientBoosting (runs without torch)
2. Heavy: WorldModel TFT if torch is available

The lightweight model uses GradientBoostingClassifier for direction prediction
(up/down) which is what trading actually needs, plus a regressor for magnitude.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

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
    Dual-model predictor: GradientBoosting classifier for direction + regressor for magnitude.
    The classifier is the primary decision maker; the regressor provides confidence scaling.
    """

    def __init__(self, target_horizon: str = "target_1h"):
        self.target_horizon = target_horizon
        self.model: GradientBoostingRegressor | None = None
        self.classifier: GradientBoostingClassifier | None = None
        self.feature_names: list[str] = []
        self.training_stats: dict = {}

    def train(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_estimators: int = 800,
        max_depth: int = 5,
        learning_rate: float = 0.03,
    ) -> dict:
        """Train on historical data. Returns performance metrics."""
        logger.info("training_started", symbol=symbol, timeframe=timeframe)
        start = time.time()

        df = load_ohlcv(symbol, timeframe)
        data = prepare_training_data(df, target_col=self.target_horizon)

        self.feature_names = data["feature_names"]

        # Direction classifier (up/down — the core trading decision)
        y_train_dir = (data["y_train"] > 0).astype(int)
        y_val_dir = (data["y_val"] > 0).astype(int)
        y_test_dir = (data["y_test"] > 0).astype(int)

        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )
        self.classifier.fit(data["X_train"], y_train_dir)

        # Magnitude regressor (how much — for confidence scaling)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators // 2,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )
        self.model.fit(data["X_train"], data["y_train"])

        # Evaluate classifier
        val_dir_pred = self.classifier.predict(data["X_val"])
        test_dir_pred = self.classifier.predict(data["X_test"])
        val_dir_acc = accuracy_score(y_val_dir, val_dir_pred)
        test_dir_acc = accuracy_score(y_test_dir, test_dir_pred)

        # Evaluate regressor
        val_pred = self.model.predict(data["X_val"])
        test_pred = self.model.predict(data["X_test"])
        val_mae = mean_absolute_error(data["y_val"], val_pred)
        test_mae = mean_absolute_error(data["y_test"], test_pred)
        val_r2 = r2_score(data["y_val"], val_pred)
        test_r2 = r2_score(data["y_test"], test_pred)

        # Feature importance (from classifier — direction is king)
        importances = sorted(
            zip(self.feature_names, self.classifier.feature_importances_),
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
        Predict direction and magnitude from a feature vector.
        Uses classifier for direction, regressor for magnitude/confidence.
        Falls back to regressor-only if classifier not available (old models).
        """
        if self.model is None and self.classifier is None:
            return {"prediction": 0.0, "confidence": 0.0, "signal": 0.0}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Direction from classifier (if available)
        if self.classifier is not None:
            dir_proba = self.classifier.predict_proba(features)[0]
            # proba[1] = probability of "up", proba[0] = probability of "down"
            up_prob = float(dir_proba[1]) if len(dir_proba) > 1 else 0.5
            direction = 1.0 if up_prob > 0.5 else -1.0
            classifier_confidence = abs(up_prob - 0.5) * 2  # 0-1 scale
        else:
            direction = 0.0
            classifier_confidence = 0.0

        # Magnitude from regressor
        if self.model is not None:
            pred = float(self.model.predict(features)[0])
            tree_preds = np.array([
                tree[0].predict(features)[0]
                for tree in self.model.estimators_
            ])
            pred_std = float(np.std(tree_preds))
        else:
            pred = 0.0
            pred_std = 0.0

        # Combine: direction from classifier, confidence from both
        if self.classifier is not None:
            raw_signal = direction * classifier_confidence
            confidence = classifier_confidence
        else:
            # Legacy fallback for old regressor-only models
            raw_signal = np.clip(pred * 100, -1, 1)
            confidence = max(0.0, 1.0 - pred_std / (abs(pred) + 1e-8))
            confidence = min(confidence, 1.0)

        return {
            "prediction": pred,
            "confidence": round(float(confidence), 4),
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
                "classifier": self.classifier,
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
        self.classifier = data.get("classifier")  # None for old models
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
    # Map timeframe to target column — targets are candle-offsets
    # On 1h data: target_1h = 1 candle = 1h ahead
    # On 4h data: target_1h = 1 candle = 4h ahead (good for 4h models)
    # On 1d data: target_1h = 1 candle = 1d ahead
    target_col = "target_1h"
    results = {}

    for symbol in symbols:
        predictor = LightweightPredictor(target_horizon=target_col)
        try:
            stats = predictor.train(symbol=symbol, timeframe=timeframe)
            predictor.save(CHECKPOINT_DIR / f"lightweight_{symbol.replace('/', '_')}_{timeframe}.pkl")
            results[symbol] = stats
        except FileNotFoundError:
            logger.warning("no_data_for_training", symbol=symbol, timeframe=timeframe)
            results[symbol] = {"error": "No historical data. Run download first."}
        except Exception as err:
            logger.warning("training_failed", symbol=symbol, error=repr(err))
            results[symbol] = {"error": repr(err)}

    return results
