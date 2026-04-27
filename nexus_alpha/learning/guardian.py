"""
Guardian AI (Meta-Labeling) — The secondary ML layer for NEXUS-V3.

This module fits a classifier to the *outcomes* of the primary trading signals.
Its job is to predict if a signal is likely to lead to a profit (label=1) or a loss (label=0)
based on the market context (features) at decision time.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("data/models")
GUARDIAN_PATH = MODEL_DIR / "guardian_v1.pkl"

class GuardianAI:
    def __init__(self, shadow_mode: bool = True, min_trades_required: int = 50):
        self.shadow_mode = shadow_mode
        self.min_trades_required = min_trades_required
        self.model: HistGradientBoostingClassifier | None = None
        self.is_fitted = False
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.load_model()

    def load_model(self) -> None:
        """Load the latest guardian model from disk."""
        if GUARDIAN_PATH.exists():
            try:
                with open(GUARDIAN_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self.is_fitted = True
                logger.info("guardian_model_loaded", path=str(GUARDIAN_PATH))
            except Exception as e:
                logger.error("guardian_load_failed", error=str(e))

    def save_model(self) -> None:
        """Persist the guardian model."""
        if self.model:
            with open(GUARDIAN_PATH, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("guardian_model_saved", path=str(GUARDIAN_PATH))

    def fit_on_history(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        Train the guardian on historical outcomes.
        dataset should contain 'features' and 'pnl_pcts'.
        """
        features = dataset.get("features")
        pnl_pcts = dataset.get("pnl_pcts")
        
        if features is None or pnl_pcts is None or len(features) < self.min_trades_required:
            logger.warning("insufficient_data_for_guardian_training", 
                           count=len(features) if features is not None else 0)
            return {"status": "skipped", "reason": "insufficient_data"}

        # Labels: 1 if profitable, 0 if loss
        labels = (pnl_pcts > 0).astype(int)
        
        # We need a balanced-ish dataset or class weights
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            class_weight="balanced"
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        
        self.is_fitted = True
        self.save_model()
        
        logger.info("guardian_trained", accuracy=f"{score:.4f}", n_samples=len(features))
        return {
            "status": "success",
            "accuracy": score,
            "n_samples": len(features)
        }

    def predict_safety(self, feature_vector: list[float] | np.ndarray) -> dict[str, float]:
        """
        Evaluate a single trade opportunity.
        Returns: {probability_of_success: float, is_safe: bool}
        """
        if not self.is_fitted or self.model is None:
            return {"probability": 1.0, "is_safe": True, "shadow": True}

        # Reshape for single prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        try:
            proba = self.model.predict_proba(X)[0][1] # Probability of class 1 (win)
            
            # Policy: Safety threshold
            is_safe = proba >= 0.55 # Requirement: > 55% conviction of success
            
            if self.shadow_mode:
                return {"probability": float(proba), "is_safe": True, "shadow": True}
            
            return {"probability": float(proba), "is_safe": is_safe, "shadow": False}
        except Exception as e:
            logger.error("guardian_inference_failed", error=str(e))
            return {"probability": 1.0, "is_safe": True, "shadow": True}
