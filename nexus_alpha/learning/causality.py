"""
Causal Intelligence Layer.
Uses DoWhy to verify if signals have a causal relationship with price movements.
Prunes spurious correlations (noise) to increase win rate.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class CausalSignalValidator:
    """
    Validates signals using causal inference to prune noise-driven trades.
    Focuses on 'Average Treatment Effect' (ATE) of a signal on future returns.
    
    Includes a TTL-based cache to prevent performance bottlenecks in the production loop.
    """

    def __init__(self, significance_level: float = 0.05, cache_ttl_minutes: int = 60):
        self.significance_level = significance_level
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, tuple[bool, datetime]] = {} # {signal_name: (is_valid, timestamp)}

    def validate_signal_causality(
        self, 
        df: pd.DataFrame, 
        signal_col: str, 
        target_col: str = "future_return",
        confounder_cols: List[str] | None = None
    ) -> bool:
        """
        Perform a causal check: Does the signal 'cause' the target movement?
        Uses cache to avoid redundant heavy estimation.
        """
        now = datetime.utcnow()
        if signal_col in self._cache:
            result, timestamp = self._cache[signal_col]
            if now - timestamp < self.cache_ttl:
                return result

        if df.empty or len(df) < 50:
            # Not enough data for causal inference, default to true (don't prune yet)
            return True

        # Fill missing values to prevent do_why from dropping entire rows or failing with NaN
        df = df.fillna(0.0)

        # Treat signal as a binary treatment (triggered or not)
        # In practice, signals can be continuous, but v1 uses binary thresholding
        df["treatment"] = (df[signal_col].abs() > 0.5).astype(int)
        
        # Confounders: variables that affect BOTH signal and return (Market Vol, BTC Trend, etc.)
        if confounder_cols is None:
            confounder_cols = ["volatility", "market_regime"]
        
        # Ensure confounders exist and have enough variance to avoid singular matrices
        valid_confounders = []
        for c in confounder_cols:
            if c not in df.columns:
                df[c] = 0.0 
            
            # Injection: add tiny epsilon noise to handle static (zero-variance) columns
            # This prevents Statsmodels singular matrix errors
            if df[c].std() < 1e-10:
                df[c] += np.random.normal(0, 1e-9, size=len(df))
            
            valid_confounders.append(c)
        
        # Safeguard: Treatment must not be static
        if df["treatment"].std() < 1e-10:
            logger.debug("causal_skipped_static_treatment", signal=signal_col)
            return True

        try:
            # 1. Model: Define the causal graph
            model = CausalModel(
                data=df,
                treatment="treatment",
                outcome=target_col,
                common_causes=confounder_cols
            )

            # 2. Identify: Identify the causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # 3. Estimate: Estimate the causal effect using linear regression (Backdoor Adjustment)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )

            # 4. Refute: Refute the estimate (Placebo test)
            # This is the "Pruning" logic: if a random placebo also 'causes' the movement, 
            # our signal isn't special.
            refutation = model.refute_estimate(
                identified_estimand, 
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute"
            )

            is_valid = bool(estimate.value > 0 and refutation.refutation_result.get("p_value", 1.0) > self.significance_level)
            
            # Update cache
            self._cache[signal_col] = (is_valid, now)
            
            logger.info(
                "causal_validation_complete",
                signal=signal_col,
                estimate=f"{estimate.value:.6f}",
                p_value=f"{refutation.refutation_result.get('p_value', 1.0):.3f}",
                is_valid=is_valid
            )
            
            return is_valid

        except Exception as e:
            logger.warning("causal_validation_failed_handled", signal=signal_col, error=str(e))
            return True

    def batch_prune_signals(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Utility to prune a batch of signals at decision time."""
        # Implementation depends on how signals are structured in the main engine
        # For NEXUS-ULTRA, this will be integrated into the SignalFusionEngine.
        return signals
