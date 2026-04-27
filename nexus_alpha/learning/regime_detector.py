"""
Regime Detector — Uses unsupervised learning (GMM) to classify market states.

Identifies:
1. Low-volatility sideways (Chop)
2. Trending-Up
3. Trending-Down
4. High-volatility / Panic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


class RegimeDetector:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._regime_map = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility and trend features for clustering."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Volatility (ATR-like)
        high_low = (df["high"] - df["low"]) / df["close"]
        features["volatility"] = high_low.rolling(window=14).mean()
        
        # 2. Trend (Momentum)
        features["momentum"] = df["close"].pct_change(periods=24) # 24h momentum
        
        # 3. Variance of returns
        features["returns_std"] = df["close"].pct_change().rolling(window=24).std()

        return features.dropna()

    def fit(self, df: pd.DataFrame) -> None:
        """Train the GMM on historical data."""
        features = self.prepare_features(df)
        scaled_features = self.scaler.fit_transform(features)
        
        self.model.fit(scaled_features)
        self.is_fitted = True
        
        # Map clusters to human-readable regimes based on mean momentum/vol
        means = self.model.means_
        # Note: This is a simplification; in production we'd use more robust mapping
        for i in range(self.n_regimes):
            vol = means[i, 0]
            mom = means[i, 1]
            
            if vol > 1.0: # High relative vol
                self._regime_map[i] = "panicked"
            elif abs(mom) < 0.2:
                self._regime_map[i] = "sideways"
            elif mom > 0.2:
                self._regime_map[i] = "trending_up"
            else:
                self._regime_map[i] = "trending_down"

        logger.info("regime_detector_fitted", regimes=self._regime_map)

    def predict_current(self, df: pd.DataFrame) -> str:
        """Predict the regime for the latest candle."""
        if not self.is_fitted:
            return "unknown"
            
        features = self.prepare_features(df).tail(1)
        if features.empty:
            return "unknown"
            
        scaled = self.scaler.transform(features)
        cluster = self.model.predict(scaled)[0]
        return self._regime_map.get(cluster, "unknown")

    def get_regime_multiplier(self, regime: str) -> float:
        """
        Policy: Scale confidence by market state to optimize efficiency.
        - trending_up / trending_down: 1.0 (Full confidence)
        - panicked: 0.6 (Cautious but microstructure thrives here)
        - sideways: 0.3 (Dampened to avoid fee-death; only ultra-alpha passes)
        - unknown: 0.5
        """
        multipliers = {
            "trending_up": 1.0,
            "trending_down": 1.0,
            "panicked": 0.6,
            "sideways": 0.3,
            "unknown": 0.5
        }
        return multipliers.get(regime, 0.5)
