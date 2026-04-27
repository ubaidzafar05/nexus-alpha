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
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score

from nexus_alpha.learning.historical_data import (
    build_features,
    load_ohlcv,
    prepare_training_data,
)
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = Path("data/checkpoints")


def _build_learning_matrix(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = dataset["features"]
    directions = dataset["directions"]
    return np.concatenate([features, directions.reshape(-1, 1)], axis=1), dataset["targets"]


def _dataset_kwargs(
    *,
    min_trades: int,
    target_mode: str,
    strong_move_pct: float,
    min_quality_score: float,
    balanced: bool,
    target_metric: str,
    target_threshold: float | None,
    regime_slice: str | None,
) -> dict[str, Any]:
    return {
        "min_trades": min_trades,
        "target_mode": target_mode,
        "strong_move_pct": strong_move_pct,
        "min_quality_score": min_quality_score,
        "balanced": balanced,
        "target_metric": target_metric,
        "target_threshold": target_threshold,
        "regime_slice": regime_slice,
    }


def _split_chronological(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(X)
    split = int(n * 0.8)
    if split <= 0 or split >= n:
        return None
    return X[:split], X[split:], y[:split], y[split:]


def _baseline_stats(y_val: np.ndarray) -> dict[str, float]:
    classes, counts = np.unique(y_val, return_counts=True)
    majority_idx = int(np.argmax(counts))
    majority_class = int(classes[majority_idx])
    baseline_pred = np.full(len(y_val), majority_class)
    return {
        "majority_class": majority_class,
        "accuracy": float(accuracy_score(y_val, baseline_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, baseline_pred)),
        "macro_f1": float(f1_score(y_val, baseline_pred, average="macro", zero_division=0)),
    }


def _prediction_edge_from_proba(proba: np.ndarray, target_mode: str) -> tuple[float, float]:
    if target_mode == "binary":
        win_probability = float(proba[1]) if len(proba) > 1 else 0.5
        edge_score = (win_probability - 0.5) * 2.0
        return edge_score, win_probability
    if target_mode == "ternary":
        loss_p = float(proba[0]) if len(proba) > 0 else 0.0
        neutral_p = float(proba[1]) if len(proba) > 1 else 0.0
        win_p = float(proba[2]) if len(proba) > 2 else 0.0
        edge_score = (win_p - loss_p) + 0.1 * neutral_p
        return float(np.clip(edge_score, -1.0, 1.0)), win_p
    loss_strong = float(proba[0]) if len(proba) > 0 else 0.0
    loss_weak = float(proba[1]) if len(proba) > 1 else 0.0
    win_weak = float(proba[2]) if len(proba) > 2 else 0.0
    win_strong = float(proba[3]) if len(proba) > 3 else 0.0
    edge_score = (1.25 * win_strong + 0.5 * win_weak) - (0.5 * loss_weak + 1.25 * loss_strong)
    win_probability = win_weak + win_strong
    return float(np.clip(edge_score, -1.0, 1.0)), float(np.clip(win_probability, 0.0, 1.0))


def _evaluate_classifier(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    results = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "pred_class_counts": np.bincount(pred, minlength=max(int(np.max(y_train)), int(np.max(y_val))) + 1).tolist(),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)
        results["mae"] = float(mean_absolute_error(y_val, np.argmax(proba, axis=1)))
    return results


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
            "all_importances": [(n, round(v, 6)) for n, v in importances],
            "training_time_seconds": round(elapsed, 1),
        }

        # G4: Log feature importances to SQLite for trend analysis
        try:
            from nexus_alpha.learning.trade_logger import TradeLogger
            tl = TradeLogger()
            tl.log_feature_importances(symbol, timeframe, importances)
        except Exception:
            pass  # Non-critical

        logger.info("training_complete", **{k: v for k, v in self.training_stats.items() if k != "all_importances"})
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

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch predict: returns (signals, confidences) arrays. Much faster than per-row."""
        n = len(X)
        signals = np.zeros(n)
        confidences = np.zeros(n)
        if self.classifier is None and self.model is None:
            return signals, confidences
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.classifier is not None:
            proba = self.classifier.predict_proba(X)
            up_prob = proba[:, 1] if proba.shape[1] > 1 else np.full(n, 0.5)
            direction = np.where(up_prob > 0.5, 1.0, -1.0)
            clf_conf = np.abs(up_prob - 0.5) * 2
            signals = direction * clf_conf
            confidences = clf_conf

        return signals, confidences

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
        min_total_trades: int = 50,
        min_direction_accuracy: float = 0.52,
        min_balanced_accuracy: float = 0.45,
        target_mode: str = "binary",
        target_metric: str = "pnl_pct",
        target_threshold: float | None = None,
        strong_move_pct: float = 0.02,
        min_quality_score: float = 0.0,
        balanced_replay: bool = False,
        regime_slice: str | None = None,
        model_path: Path | None = None,
    ):
        self.predictor = predictor or LightweightPredictor()
        self.retrain_interval = retrain_interval_hours * 3600
        self.min_new_trades = min_new_trades
        self.min_total_trades = min_total_trades
        self.min_direction_accuracy = min_direction_accuracy
        self.min_balanced_accuracy = min_balanced_accuracy
        self.target_mode = target_mode
        self.target_metric = target_metric
        self.target_threshold = target_threshold
        self.strong_move_pct = strong_move_pct
        self.min_quality_score = min_quality_score
        self.balanced_replay = balanced_replay
        self.regime_slice = regime_slice
        self.model_path = model_path or CHECKPOINT_DIR / "lightweight_online_reward.pkl"
        self._last_retrain = 0.0
        self.predictor.load(self.model_path)

    def _augment_features(self, features: np.ndarray, directions: np.ndarray | float) -> np.ndarray:
        """Append trade direction so reward learning distinguishes longs from shorts."""
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        direction_arr = np.asarray(directions, dtype=np.float32)
        if direction_arr.ndim == 0:
            direction_arr = np.full((len(X), 1), float(direction_arr), dtype=np.float32)
        elif direction_arr.ndim == 1:
            direction_arr = direction_arr.reshape(-1, 1)
        if len(direction_arr) != len(X):
            raise ValueError("Direction count must match feature rows")
        return np.concatenate([X, direction_arr], axis=1)

    def record_outcome(self, features: np.ndarray, actual_return: float) -> None:
        """Record a trade outcome for future retraining."""
        return None

    def should_retrain(self, trade_logger: Any) -> bool:
        """Check if enough new closed-trade evidence has accumulated."""
        elapsed = time.time() - self._last_retrain
        total_closed_trades = trade_logger.count_closed_trades()
        if total_closed_trades < self.min_total_trades:
            return False
        last_seen_metric = trade_logger.get_latest_metric("online_retrain_closed_trades")
        last_seen_closed = int(last_seen_metric["metric_value"]) if last_seen_metric else 0
        new_trades = max(0, total_closed_trades - last_seen_closed)
        return (
            elapsed >= self.retrain_interval
            and new_trades >= self.min_new_trades
        )

    def retrain_from_journal(self, trade_logger) -> dict | None:
        """
        Retrain using closed trades from the trade journal.
        Blends historical base knowledge with live trade experience.
        """
        dataset = trade_logger.build_learning_dataset(
            **_dataset_kwargs(
                min_trades=self.min_total_trades,
                target_mode=self.target_mode,
                strong_move_pct=self.strong_move_pct,
                min_quality_score=self.min_quality_score,
                balanced=self.balanced_replay,
                target_metric=self.target_metric,
                target_threshold=self.target_threshold,
                regime_slice=self.regime_slice,
            ),
        )
        if dataset is None:
            return None

        logger.info(
            "online_retrain_started",
            n_trades=dataset["n_trades"],
            target_mode=self.target_mode,
            target_metric=self.target_metric,
            min_quality_score=self.min_quality_score,
            balanced_replay=self.balanced_replay,
        )

        X, y = _build_learning_matrix(dataset)
        n = len(X)
        split_data = _split_chronological(X, y)
        if split_data is None:
            return None
        X_train, X_val, y_train, y_val = split_data
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            total_closed = trade_logger.count_closed_trades()
            latest_marker = trade_logger.get_latest_metric("online_retrain_closed_trades")
            last_closed = int(latest_marker["metric_value"]) if latest_marker else 0
            stats = {
                "n_trades": n,
                "new_trades": max(0, total_closed - last_closed),
                "val_mae": 1.0,
                "val_direction_accuracy": 0.0,
                "val_balanced_accuracy": 0.0,
                "baseline_direction_accuracy": 1.0,
                "baseline_balanced_accuracy": 1.0,
                "target_type": self.target_mode,
                "target_metric": self.target_metric,
                "updated": False,
                "reason": "single_class_outcomes",
            }
            trade_logger.log_metric("retrain_direction_accuracy", 0.0)
            trade_logger.log_metric("retrain_mae", 1.0)
            trade_logger.log_metric(
                "online_retrain_closed_trades",
                float(total_closed),
                details=json.dumps(stats),
            )
            logger.info("online_retrain_rejected_single_class", **stats)
            return stats

        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
        )
        classifier.fit(X_train, y_train)

        val_pred = classifier.predict(X_val)
        val_dir_acc = float(accuracy_score(y_val, val_pred))
        val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        val_mae = float(mean_absolute_error(y_val, val_pred))
        baseline = _baseline_stats(y_val)
        total_closed = trade_logger.count_closed_trades()
        latest_marker = trade_logger.get_latest_metric("online_retrain_closed_trades")
        last_closed = int(latest_marker["metric_value"]) if latest_marker else 0

        stats = {
            "n_trades": n,
            "new_trades": max(0, total_closed - last_closed),
            "val_mae": round(val_mae, 4),
            "val_direction_accuracy": round(val_dir_acc, 4),
            "val_balanced_accuracy": round(val_bal_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "baseline_direction_accuracy": round(baseline["accuracy"], 4),
            "baseline_balanced_accuracy": round(baseline["balanced_accuracy"], 4),
            "target_type": self.target_mode,
            "target_metric": self.target_metric,
            "class_counts": dataset["class_counts"],
            "slice_counts": dataset["slice_counts"],
            "min_quality_score": self.min_quality_score,
            "balanced_replay": self.balanced_replay,
            "regime_slice": self.regime_slice,
        }

        min_required_accuracy = max(self.min_direction_accuracy, baseline["accuracy"] + 0.02)
        min_required_balanced_accuracy = max(
            self.min_balanced_accuracy,
            baseline["balanced_accuracy"] + 0.02,
        )
        if val_dir_acc >= min_required_accuracy and val_bal_acc >= min_required_balanced_accuracy:
            self.predictor.model = None
            self.predictor.classifier = classifier
            self.predictor.feature_names = [f"f{i}" for i in range(X.shape[1])]
            self.predictor.training_stats = stats
            self.predictor.save(self.model_path)
            stats["updated"] = True
            stats["min_required_accuracy"] = round(min_required_accuracy, 4)
            stats["min_required_balanced_accuracy"] = round(min_required_balanced_accuracy, 4)
            logger.info("online_retrain_accepted", **stats)
        else:
            stats["updated"] = False
            stats["min_required_accuracy"] = round(min_required_accuracy, 4)
            stats["min_required_balanced_accuracy"] = round(min_required_balanced_accuracy, 4)
            logger.info("online_retrain_rejected_low_accuracy", **stats)

        self._last_retrain = time.time()

        trade_logger.log_metric("retrain_direction_accuracy", val_dir_acc)
        trade_logger.log_metric("retrain_mae", val_mae)
        trade_logger.log_metric(
            "online_retrain_closed_trades",
            float(total_closed),
            details=json.dumps(stats),
        )

        return stats

    def predict_reward(self, features: np.ndarray, trade_direction: float) -> dict[str, float] | None:
        """Predict expected reward of a proposed long/short setup."""
        target_type = self.predictor.training_stats.get("target_type")
        if target_type in {"binary", "ternary", "quaternary", "profit_classification"} and self.predictor.classifier is not None:
            X = self._augment_features(features, float(np.sign(trade_direction) or 0.0))
            proba = self.predictor.classifier.predict_proba(X)[0]
            effective_mode = "binary" if target_type == "profit_classification" else target_type
            edge_score, win_probability = _prediction_edge_from_proba(proba, effective_mode)
            confidence = abs(edge_score)
            return {
                "reward_prediction": round(edge_score, 4),
                "confidence": round(float(confidence), 4),
                "std": round(float(1.0 - confidence), 6),
                "win_probability": round(win_probability, 4),
            }

        if self.predictor.model is None:
            return None
        X = self._augment_features(features, float(np.sign(trade_direction) or 0.0))
        reward_pred = float(self.predictor.model.predict(X)[0])
        tree_preds = np.array([
            tree[0].predict(X)[0]
            for tree in self.predictor.model.estimators_
        ])
        pred_std = float(np.std(tree_preds))
        confidence = max(0.0, 1.0 - pred_std / (abs(reward_pred) + 1e-6))
        confidence = min(confidence, 1.0)
        return {
            "reward_prediction": round(reward_pred, 4),
            "confidence": round(float(confidence), 4),
            "std": round(pred_std, 6),
        }


def benchmark_trade_outcome_models(
    trade_logger: Any,
    min_trades: int = 30,
    min_quality_score: float = 0.0,
    balanced: bool = False,
    target_metric: str = "pnl_pct",
    target_threshold: float | None = None,
    regime_slice: str | None = None,
) -> dict[str, Any] | None:
    """Benchmark simple outcome models on filtered journal data."""
    dataset = trade_logger.build_learning_dataset(
        **_dataset_kwargs(
            min_trades=min_trades,
            target_mode="binary",
            strong_move_pct=0.02,
            min_quality_score=min_quality_score,
            balanced=balanced,
            target_metric=target_metric,
            target_threshold=target_threshold,
            regime_slice=regime_slice,
        ),
    )
    if dataset is None:
        return None

    X, y = _build_learning_matrix(dataset)
    split_data = _split_chronological(X, y)
    if split_data is None:
        return None
    X_train, X_val, y_train, y_val = split_data
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None

    baseline = _baseline_stats(y_val)

    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        ),
    }

    results: dict[str, Any] = {
        "n_trades": len(X),
        "baseline_accuracy": round(baseline["accuracy"], 4),
        "baseline_balanced_accuracy": round(baseline["balanced_accuracy"], 4),
        "baseline_macro_f1": round(baseline["macro_f1"], 4),
        "class_counts": dataset["class_counts"],
        "slice_counts": dataset["slice_counts"],
        "quality_mean": round(float(np.mean(dataset["quality_scores"])), 4),
        "balanced_dataset": balanced,
        "target_metric": target_metric,
        "target_threshold": dataset.get("target_threshold"),
        "regime_slice": regime_slice or "all",
        "models": {},
    }

    best_name = ""
    best_score = -1.0
    for name, model in candidates.items():
        metrics = _evaluate_classifier(model, X_train, y_train, X_val, y_val)
        results["models"][name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        score = metrics["balanced_accuracy"] + metrics["macro_f1"]
        if score > best_score:
            best_name = name
            best_score = score

    results["best_model"] = best_name
    return results


def benchmark_trade_bucket_models(
    trade_logger: Any,
    min_trades: int = 30,
    strong_move_pct: float = 0.02,
    min_quality_score: float = 0.0,
    balanced: bool = False,
    target_metric: str = "pnl_pct",
    target_threshold: float | None = None,
    regime_slice: str | None = None,
) -> dict[str, Any] | None:
    """Benchmark multiclass bucketed-outcome models on filtered journal data."""
    dataset = trade_logger.build_learning_dataset(
        **_dataset_kwargs(
            min_trades=min_trades,
            target_mode="quaternary",
            strong_move_pct=strong_move_pct,
            min_quality_score=min_quality_score,
            balanced=balanced,
            target_metric=target_metric,
            target_threshold=target_threshold,
            regime_slice=regime_slice,
        ),
    )
    if dataset is None:
        return None

    X, y = _build_learning_matrix(dataset)
    split_data = _split_chronological(X, y)
    if split_data is None:
        return None
    X_train, X_val, y_train, y_val = split_data
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None

    class_count_len = max(int(np.max(y_train)), int(np.max(y_val))) + 1
    class_counts = np.bincount(y_val, minlength=class_count_len)
    baseline = _baseline_stats(y_val)

    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }

    results: dict[str, Any] = {
        "n_trades": len(X),
        "strong_move_pct": strong_move_pct,
        "train_class_counts": np.bincount(y_train, minlength=class_count_len).tolist(),
        "val_class_counts": class_counts.tolist(),
        "majority_class_accuracy": round(baseline["accuracy"], 4),
        "majority_class_balanced_accuracy": round(baseline["balanced_accuracy"], 4),
        "majority_class_macro_f1": round(baseline["macro_f1"], 4),
        "class_counts": dataset["class_counts"],
        "slice_counts": dataset["slice_counts"],
        "quality_mean": round(float(np.mean(dataset["quality_scores"])), 4),
        "balanced_dataset": balanced,
        "target_metric": target_metric,
        "target_threshold": dataset.get("target_threshold"),
        "regime_slice": regime_slice or "all",
        "models": {},
    }

    best_name = ""
    best_score = -1.0
    for name, model in candidates.items():
        metrics = _evaluate_classifier(model, X_train, y_train, X_val, y_val)
        results["models"][name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        score = metrics["balanced_accuracy"] + metrics["macro_f1"]
        if score > best_score:
            best_name = name
            best_score = score

    results["best_model"] = best_name
    return results


def benchmark_learning_targets(
    trade_logger: Any,
    min_trades: int = 30,
    strong_move_pct: float = 0.02,
    min_quality_score: float = 0.0,
    target_metric: str = "pnl_pct",
    target_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Compare binary, ternary, and quaternary targets on chronological vs balanced datasets."""
    variants = {
        "binary_chronological": benchmark_trade_outcome_models(
            trade_logger,
            min_trades=min_trades,
            min_quality_score=min_quality_score,
            balanced=False,
            target_metric=target_metric,
            target_threshold=target_threshold,
        ),
        "binary_balanced": benchmark_trade_outcome_models(
            trade_logger,
            min_trades=min_trades,
            min_quality_score=min_quality_score,
            balanced=True,
            target_metric=target_metric,
            target_threshold=target_threshold,
        ),
        "quaternary_chronological": benchmark_trade_bucket_models(
            trade_logger,
            min_trades=min_trades,
            strong_move_pct=strong_move_pct,
            min_quality_score=min_quality_score,
            balanced=False,
            target_metric=target_metric,
            target_threshold=target_threshold,
        ),
        "quaternary_balanced": benchmark_trade_bucket_models(
            trade_logger,
            min_trades=min_trades,
            strong_move_pct=strong_move_pct,
            min_quality_score=min_quality_score,
            balanced=True,
            target_metric=target_metric,
            target_threshold=target_threshold,
        ),
    }
    if not any(result is not None for result in variants.values()):
        return None
    return {
        "strong_move_pct": strong_move_pct,
        "min_quality_score": min_quality_score,
        "target_metric": target_metric,
        "target_threshold": target_threshold,
        "variants": variants,
    }


def diagnose_learning_features(
    trade_logger: Any,
    min_trades: int = 30,
    target_mode: str = "quaternary",
    strong_move_pct: float = 0.02,
    min_quality_score: float = 0.0,
    top_n: int = 10,
    target_metric: str = "pnl_pct",
    target_threshold: float | None = None,
    regime_slice: str | None = None,
) -> dict[str, Any] | None:
    """Inspect feature separability for a target scheme on the journal dataset."""
    dataset = trade_logger.build_learning_dataset(
        **_dataset_kwargs(
            min_trades=min_trades,
            target_mode=target_mode,
            strong_move_pct=strong_move_pct,
            min_quality_score=min_quality_score,
            balanced=False,
            target_metric=target_metric,
            target_threshold=target_threshold,
            regime_slice=regime_slice,
        ),
    )
    if dataset is None:
        return None

    X, y = _build_learning_matrix(dataset)
    if len(np.unique(y)) < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X, y)

    summaries = []
    feature_names = list(dataset.get("feature_names", []))
    for idx in range(X.shape[1]):
        feature_values = X[:, idx]
        class_means = {
            int(cls): round(float(np.mean(feature_values[y == cls])), 4)
            for cls in sorted(np.unique(y))
        }
        separation = 0.0
        if class_means:
            mean_values = list(class_means.values())
            spread = max(mean_values) - min(mean_values)
            denom = float(np.std(feature_values) + 1e-6)
            separation = spread / denom
        summaries.append(
            {
                "feature": feature_names[idx] if idx < len(feature_names) else (f"f{idx}" if idx < X.shape[1] - 1 else "trade_direction"),
                "importance": round(float(model.feature_importances_[idx]), 4),
                "separation": round(float(separation), 4),
                "class_means": class_means,
            }
        )

    top_importance = sorted(summaries, key=lambda item: item["importance"], reverse=True)[:top_n]
    top_separation = sorted(summaries, key=lambda item: item["separation"], reverse=True)[:top_n]
    slice_target_counts: dict[str, dict[int, int]] = {}
    for slice_name, target in zip(dataset["regime_slices"], dataset["targets"]):
        target_counts = slice_target_counts.setdefault(slice_name, {})
        target_counts[int(target)] = target_counts.get(int(target), 0) + 1

    return {
        "target_mode": target_mode,
        "target_metric": target_metric,
        "target_threshold": dataset.get("target_threshold"),
        "regime_slice": regime_slice or "all",
        "n_trades": dataset["n_trades"],
        "class_counts": dataset["class_counts"],
        "slice_counts": dataset["slice_counts"],
        "slice_target_counts": slice_target_counts,
        "top_importance_features": top_importance,
        "top_separation_features": top_separation,
    }


def benchmark_regime_slices(
    trade_logger: Any,
    min_trades: int = 10,
    target_mode: str = "binary",
    strong_move_pct: float = 0.02,
    min_quality_score: float = 0.0,
    target_metric: str = "pnl_pct",
    target_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Benchmark learnability separately for each regime slice."""
    seed_dataset = trade_logger.build_learning_dataset(
        **_dataset_kwargs(
            min_trades=1,
            target_mode=target_mode,
            strong_move_pct=strong_move_pct,
            min_quality_score=min_quality_score,
            balanced=False,
            target_metric=target_metric,
            target_threshold=target_threshold,
            regime_slice=None,
        ),
    )
    if seed_dataset is None:
        return None

    slice_names = list(dict.fromkeys(seed_dataset["regime_slices"]))
    results: dict[str, Any] = {}
    for slice_name in slice_names:
        if target_mode == "binary":
            result = benchmark_trade_outcome_models(
                trade_logger,
                min_trades=min_trades,
                min_quality_score=min_quality_score,
                balanced=False,
                target_metric=target_metric,
                target_threshold=target_threshold,
                regime_slice=slice_name,
            )
        else:
            result = benchmark_trade_bucket_models(
                trade_logger,
                min_trades=min_trades,
                strong_move_pct=strong_move_pct,
                min_quality_score=min_quality_score,
                balanced=False,
                target_metric=target_metric,
                target_threshold=target_threshold,
                regime_slice=slice_name,
            )
        if result is not None:
            results[slice_name] = result

    if not results:
        return None
    return {
        "target_mode": target_mode,
        "target_metric": target_metric,
        "target_threshold": target_threshold,
        "slices": results,
    }


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
